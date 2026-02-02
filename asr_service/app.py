import asyncio
import importlib
import logging
import os
import shutil
import subprocess
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Optional, Literal

import aiohttp
from dotenv import load_dotenv

# Load .env file from project root
try:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:
    pass

async def download_audio(url: str, save_path: Path):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                raise Exception(f"Failed to download audio: {response.status}")
            with open(save_path, 'wb') as f:
                while True:
                    chunk = await response.content.read(1024*1024)
                    if not chunk:
                        break
                    f.write(chunk)

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from asr_service.postprocess import postprocess, merge_speaker_segments

APP_ROOT = Path(__file__).resolve().parent
STATIC_DIR = APP_ROOT / "static"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("asr_service")

# Configuration
defaults = {
    "model": "paraformer-zh",
    "vad": "fsmn-vad",
    "punc": "ct-punc",
    "spk": "cam++",
}

MODEL_NAME = os.getenv("ASR_MODEL_NAME", defaults["model"])
VAD_MODEL = os.getenv("ASR_VAD_MODEL", defaults["vad"])
PUNC_MODEL = os.getenv("ASR_PUNC_MODEL", defaults["punc"])
SPK_MODEL = os.getenv("ASR_SPK_MODEL", defaults["spk"])
# Default enable flags
ENABLE_SPK = os.getenv("ASR_ENABLE_SPK", "true").lower() == "true"
ENABLE_POSTPROCESS = os.getenv("ASR_ENABLE_POSTPROCESS", "true").lower() == "true"
ENABLE_DENOISE = os.getenv("ASR_AUDIO_DENOISE", "true").lower() == "true"

MAX_FILE_SIZE_MB = int(os.getenv("ASR_MAX_FILE_SIZE_MB", "100"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# Device selection
_device_env = os.getenv("ASR_DEVICE", "auto").lower()
if _device_env == "auto":
    try:
        import torch
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        DEVICE = "cpu"
else:
    DEVICE = _device_env
    
logger.info(f"Using device: {DEVICE} (env={_device_env})")

ALLOWED_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".mp4"}

model = None

@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    global model
    logger.info("Loading FunASR model...")
    logger.info(f"ASR: {MODEL_NAME}, VAD: {VAD_MODEL}, PUNC: {PUNC_MODEL}, SPK: {SPK_MODEL}")
    
    funasr_module = importlib.import_module("funasr")
    auto_model = getattr(funasr_module, "AutoModel")
    
    # Initialize model with globally configured device
    model = auto_model(
        model=MODEL_NAME,
        vad_model=VAD_MODEL,
        punc_model=PUNC_MODEL,
        spk_model=SPK_MODEL if ENABLE_SPK else None,
        device=DEVICE
    )
    logger.info("Model loaded successfully.")
    yield


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/models")
async def get_models():
    """Return loaded model information"""
    return JSONResponse({
        "asr_model": MODEL_NAME,
        "vad_model": VAD_MODEL,
        "punc_model": PUNC_MODEL,
        "spk_model": SPK_MODEL if ENABLE_SPK else None,
        "device": DEVICE,
        "enable_spk": ENABLE_SPK
    })

@app.get("/health")
async def health_check():
    """Health check endpoint to verify service status"""
    status = "ready" if model is not None else "loading"
    return JSONResponse({"status": status, "device": DEVICE})


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/asr/transcribe")
async def transcribe(
    file: Optional[UploadFile] = File(None),
    audio_url: Optional[str] = Form(None),
    hotword: Annotated[str, Form()] = "",
    use_itn: bool = Form(True),
    merge_vad: bool = Form(True),
    merge_length_s: int = Form(15),
    batch_size_s: int = Form(300),
    language: str = Form("auto"),
    device: str = Form(None),
) -> JSONResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not ready")
        
    if not file and not audio_url:
        raise HTTPException(status_code=400, detail="Either 'file' or 'audio_url' must be provided")

    # Temp file handling
    tmp_dir = Path(tempfile.mkdtemp(prefix="asr_upload_"))
    try:
        # 1. Get Audio File
        if file:
            suffix = Path(file.filename).suffix.lower() if file.filename else ".wav"
            input_path = tmp_dir / f"input{suffix}"
            if suffix not in ALLOWED_SUFFIXES:
                 # Try to accept it anyway, ffmpeg might handle it
                 pass
            
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
        else:
            # Download from URL
            input_path = tmp_dir / "input_url.wav" # aiohttp doesn't easily guess ext
            try:
                await download_audio(audio_url, input_path)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to download audio: {str(e)}")

        # 2. Convert to 16k mono wav
        audio_wav_path = tmp_dir / "audio.wav"
        await _run_ffmpeg(input_path, audio_wav_path)
        
        # 3. Prepare options
        hotword_list = [w.strip() for w in hotword.split(",") if w.strip()] if hotword else None
        hotword_str = " ".join(hotword_list) if hotword_list else None
        
        # 4. Run ASR
        # Note: We pass the device param if specified, but AutoModel.generate might ignore it if fixed.
        # We rely on 'batch_size_int' name correction if needed. FunASR API varies.
        # Standard generate args: input, hotword, batch_size_s, merge_vad, merge_length_s, key_param_dict
        
        generate_kwargs = {
            "input": str(audio_wav_path),
            "batch_size_s": batch_size_s,
            "merge_vad": merge_vad,
            "merge_length_s": merge_length_s,
        }
        if hotword_str:
            generate_kwargs["hotword"] = hotword_str
        if language != "auto":
            generate_kwargs["language"] = language
        
        # Log request
        logger.info(f"Processing params: {generate_kwargs}, use_itn={use_itn}")

        result = await asyncio.to_thread(model.generate, **generate_kwargs)
        
        # Handle result format (can be list or dict)
        raw_text = ""
        res_dict = {}
        if isinstance(result, list) and result:
            res_dict = result[0] if isinstance(result[0], dict) else {"text": str(result[0])}
        elif isinstance(result, dict):
            res_dict = result
        else:
            res_dict = {"text": str(result)}
            
        raw_text = res_dict.get("text", "")
        
        # 5. Post-process
        processed_text = raw_text
        if ENABLE_POSTPROCESS:
            processed_text = postprocess(raw_text, apply_itn_flag=use_itn)
            res_dict["text_processed"] = processed_text
            # If user wanted ITN, usually they want the processed text as main return?
            # We return both.
        
        return JSONResponse({
            "text": processed_text if use_itn else raw_text,
            "raw_text": raw_text,
            "result": res_dict
        })

    finally:
        if file:
            await file.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)


async def _run_ffmpeg(input_path: Path, output_path: Path) -> None:
    command = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        "16000",
    ]
    
    if ENABLE_DENOISE:
        command.extend(["-af", "highpass=f=80,lowpass=f=8000,afftdn=nf=-20"])
    
    command.extend(["-f", "wav", str(output_path)])
    
    # logger.info("Running ffmpeg...")
    await asyncio.to_thread(_run_command, command, "ffmpeg")


def _run_command(command: list[str], name: str) -> None:
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        raise HTTPException(status_code=500, detail=f"{name} failed: {exc.stderr.decode()}") from exc
