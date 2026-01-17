import asyncio
import importlib
import logging
import os
import shutil
import subprocess
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv

# Load .env file from project root
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from asr_service.postprocess import postprocess, merge_speaker_segments

APP_ROOT = Path(__file__).resolve().parent
STATIC_DIR = APP_ROOT / "static"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("asr_service")

MODEL_NAME = os.getenv("ASR_MODEL_NAME", "paraformer-zh")
VAD_MODEL = os.getenv("ASR_VAD_MODEL", "fsmn-vad")
PUNC_MODEL = os.getenv("ASR_PUNC_MODEL", "ct-punc")
SPK_MODEL = os.getenv("ASR_SPK_MODEL", "cam++")
ENABLE_SPK = os.getenv("ASR_ENABLE_SPK", "true").lower() == "true"
DEVICE = os.getenv("ASR_DEVICE", "cpu")
MAX_FILE_SIZE_MB = int(os.getenv("ASR_MAX_FILE_SIZE_MB", "50"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ENABLE_POSTPROCESS = os.getenv("ASR_ENABLE_POSTPROCESS", "true").lower() == "true"
ENABLE_DENOISE = os.getenv("ASR_AUDIO_DENOISE", "true").lower() == "true"


ALLOWED_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".mp4"}


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    global model
    logger.info("Loading FunASR model: %s (spk_model=%s, enabled=%s)", MODEL_NAME, SPK_MODEL, ENABLE_SPK)
    funasr_module = importlib.import_module("funasr")
    auto_model = getattr(funasr_module, "AutoModel")
    model = auto_model(
        model=MODEL_NAME,
        vad_model=VAD_MODEL,
        punc_model=PUNC_MODEL,
        spk_model=SPK_MODEL if ENABLE_SPK else None,
        device=DEVICE
    )
    logger.info("Model loaded")
    yield


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
model = None


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    hotwords: str = Form(default=""),
    enable_postprocess: bool = Form(default=True)
) -> JSONResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not ready")
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing file")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    tmp_dir = Path(tempfile.mkdtemp(prefix="asr_upload_"))
    input_path = tmp_dir / f"input{suffix}"
    output_path = tmp_dir / "audio.wav"

    try:
        size = 0
        with input_path.open("wb") as output_file:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_FILE_SIZE_BYTES:
                    raise HTTPException(status_code=413, detail="File too large")
                output_file.write(chunk)

        await _run_ffmpeg(input_path, output_path)
        
        # 解析热词列表
        hotword_list = [w.strip() for w in hotwords.split(",") if w.strip()] if hotwords else None
        
        result = await _run_asr(output_path, hotword_list)
        text = result.get("text", "").strip()
        if not text:
            raise HTTPException(status_code=500, detail="Empty transcription result")
        
        # 应用后处理
        if enable_postprocess and ENABLE_POSTPROCESS:
            text = postprocess(text)
            result["text_processed"] = text
        
        return JSONResponse({"text": text, "result": result})
    finally:
        await file.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)


async def _run_ffmpeg(input_path: Path, output_path: Path) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        "16000",
    ]
    
    # 添加降噪滤波
    if ENABLE_DENOISE:
        command.extend(["-af", "highpass=f=80,lowpass=f=8000,afftdn=nf=-20"])
    
    command.extend(["-f", "wav", str(output_path)])
    
    logger.info("Running ffmpeg conversion (denoise=%s)", ENABLE_DENOISE)
    await asyncio.to_thread(_run_command, command, "ffmpeg")


async def _run_asr(audio_path: Path, hotwords: list[str] | None = None) -> dict:
    logger.info("Starting ASR (hotwords=%s)", hotwords[:3] if hotwords else None)
    result = await asyncio.to_thread(_run_model, audio_path, hotwords)
    logger.info("ASR finished")
    return result


def _run_command(command: list[str], name: str) -> None:
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        raise HTTPException(status_code=500, detail=f"{name} failed") from exc


def _run_model(audio_path: Path, hotwords: list[str] | None = None) -> dict:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not ready")
    
    # 构建生成参数
    generate_kwargs = {"input": str(audio_path)}
    if hotwords:
        generate_kwargs["hotword"] = " ".join(hotwords)
    
    result = model.generate(**generate_kwargs)
    if isinstance(result, list) and result:
        if isinstance(result[0], dict):
            return result[0]
        return {"text": str(result[0])}
    if isinstance(result, dict):
        return result
    return {"text": str(result)}
