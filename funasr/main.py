import os
import shutil
import tempfile
import logging
import asyncio
from typing import Optional, Annotated
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Query
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# Import funasr
try:
    from funasr import AutoModel
except ImportError:
    # Build fallback or just fail if not installed
    print("Warning: funasr not installed. Service will fail to load model.")
    AutoModel = None

# Import local utils
from utils import download_audio, run_ffmpeg, postprocess


def parse_hotwords(hotword_str: str) -> Optional[str]:
    """
    解析热词字符串，支持多种格式：
    - 逗号分隔: "关联交易,净利润,应收账款"
    - 空格分隔: "关联交易 净利润 应收账款"
    - 带权重格式: "关联交易:5 净利润:3 应收账款"
    
    返回 FunASR 所需的热词格式（空格分隔）
    """
    if not hotword_str or not hotword_str.strip():
        return None
    
    # 统一分隔符：逗号替换为空格
    normalized = hotword_str.replace(",", " ").replace("，", " ")
    
    words = []
    for item in normalized.split():
        item = item.strip()
        if item:
            words.append(item)
    
    return " ".join(words) if words else None

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("funasr_service")

# Load environment variables
load_dotenv()

# Configuration
DEFAULTS = {
    "model": "paraformer-zh",
    "vad": "fsmn-vad",
    "punc": "ct-punc",
    "spk": "cam++",
}

MODEL_NAME = os.getenv("ASR_MODEL_NAME", DEFAULTS["model"])
VAD_MODEL = os.getenv("ASR_VAD_MODEL", DEFAULTS["vad"])
PUNC_MODEL = os.getenv("ASR_PUNC_MODEL", DEFAULTS["punc"])
SPK_MODEL = os.getenv("ASR_SPK_MODEL", DEFAULTS["spk"])
# Default enable flags
ENABLE_SPK = os.getenv("ASR_ENABLE_SPK", "true").lower() == "true"
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

# Global model instance
model_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_instance
    logger.info("Loading FunASR model...")
    logger.info(f"ASR: {MODEL_NAME}, VAD: {VAD_MODEL}, PUNC: {PUNC_MODEL}, SPK: {SPK_MODEL}")
    
    if AutoModel:
        model_instance = AutoModel(
            model=MODEL_NAME,
            vad_model=VAD_MODEL,
            punc_model=PUNC_MODEL,
            spk_model=SPK_MODEL if ENABLE_SPK else None,
            device=DEVICE
        )
        logger.info("Model loaded successfully.")
    else:
        logger.error("FunASR module not found. Model not loaded.")
    
    yield
    # Clean up if needed

from pydantic import BaseModel, Field
from typing import List, Union, Dict, Any

# ... (imports remain)

# =============================================================================
# Pydantic Models for Swagger / OpenAPI
# =============================================================================

class SentenceInfo(BaseModel):
    text: str = Field(..., description="The content of the sentence")
    start: int = Field(..., description="Start time in milliseconds")
    end: int = Field(..., description="End time in milliseconds")
    spk: Optional[Union[int, str]] = Field(None, description="Speaker ID")
    timestamp: Optional[List[List[int]]] = Field(None, description="Word-level timestamps [[start, end], ...]")
    confidence: Optional[float] = Field(None, description="Confidence score")

class ResultCore(BaseModel):
    key: str = Field("audio", description="Identifier key")
    text: str = Field(..., description="Original transcription text")
    text_processed: str = Field(..., description="Post-processed text")
    sentence_info: List[SentenceInfo] = Field(default_factory=list, description="Detailed sentence information")

class MetaInfo(BaseModel):
    time_unit: str = Field("ms", description="Time unit used in timestamps")
    model: str = Field(..., description="Name of the ASR model used")
    device: str = Field(..., description="Device used for inference (cuda/cpu)")

class AsrResponse(BaseModel):
    text: str = Field(..., description="Final text content (usually processed)")
    raw_text: str = Field(..., description="Raw transcription text before post-processing")
    result: ResultCore = Field(..., description="Detailed result structure")
    meta: MetaInfo = Field(..., description="Metadata about the inference")

class HealthResponse(BaseModel):
    status: str
    device: str

class ModelInfoResponse(BaseModel):
    asr_model: str
    vad_model: str
    punc_model: str
    spk_model: Optional[str]
    device: str
    enable_spk: bool

# ... (lifespan and app definition remain)

app = FastAPI(
    title="FunASR Service",
    description="Production-grade ASR service using FunASR models.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint to verify if the model is loaded and service is ready."""
    status = "ready" if model_instance is not None else "loading_or_failed"
    return JSONResponse({"status": status, "device": DEVICE})

@app.get("/models", response_model=ModelInfoResponse, tags=["System"])
async def get_models():
    """Return information about the loaded models and configuration."""
    return JSONResponse({
        "asr_model": MODEL_NAME,
        "vad_model": VAD_MODEL,
        "punc_model": PUNC_MODEL,
        "spk_model": SPK_MODEL if ENABLE_SPK else None,
        "device": DEVICE,
        "enable_spk": ENABLE_SPK
    })

@app.post("/asr/transcribe", response_model=AsrResponse, tags=["ASR"])
async def transcribe(
    file: Optional[UploadFile] = File(None, description="Audio file to upload (wav, mp3, etc.)"),
    audio_url: Optional[str] = Form(None, description="URL of the audio file to download"),
    hotword: Annotated[str, Form(description="热词，逗号或空格分隔，支持权重格式如 '关联交易:5 净利润:3'")] = "",
    use_itn: bool = Form(True, description="Enable Inverse Text Normalization (convert numbers to digits, etc.)"),
    merge_vad: bool = Form(True, description="Merge VAD segments"),
    merge_length_s: int = Form(8, description="VAD 合并窗口（秒），会议场景建议 5-10"),
    batch_size_s: int = Form(600, description="批处理窗口（秒），长音频建议 600"),
    language: str = Form("auto", description="Language code"),
    # 会议场景优化参数
    spk_thresh: float = Form(0.7, description="说话人分离阈值 (0.5-0.9)，会议场景建议 0.65-0.75"),
    sentence_timestamp: bool = Form(True, description="输出句子级时间戳"),
):
    """
    Transcribe an audio file or URL.
    
    - **file**: Upload an audio file directly.
    - **audio_url**: Provide a URL to download the audio from (mutually exclusive with file, file takes precedence).
    - **hotword**: Specific words to boost in recognition.
    """
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model is not ready")
        
    if not file and not audio_url:
        raise HTTPException(status_code=400, detail="Either 'file' or 'audio_url' must be provided")

    # Temp file handling
    tmp_dir = Path(tempfile.mkdtemp(prefix="funasr_req_"))
    try:
        # 1. Get Audio File
        if file:
            suffix = Path(file.filename).suffix.lower() if file.filename else ".wav"
            input_path = tmp_dir / f"input{suffix}"
            
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
            input_path = tmp_dir / "input_url.wav"
            try:
                await download_audio(audio_url, input_path)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to download audio: {str(e)}")

        # 2. Convert to 16k mono wav
        audio_wav_path = tmp_dir / "audio.wav"
        try:
            await run_ffmpeg(input_path, audio_wav_path, enable_denoise=ENABLE_DENOISE)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Audio conversion failed: {str(e)}")
        
        # 3. Prepare options
        start_time_ms = 0 # Base offset if we were slicing, here it's 0
        
        # 解析热词（支持逗号/空格分隔，支持权重格式 "词:权重"）
        hotword_str = parse_hotwords(hotword) if hotword else None
        
        generate_kwargs = {
            "input": str(audio_wav_path),
            "batch_size_s": batch_size_s,
            "merge_vad": merge_vad,
            "merge_length_s": merge_length_s,
            "sentence_timestamp": sentence_timestamp,
        }
        if hotword_str:
            generate_kwargs["hotword"] = hotword_str
        if language != "auto":
            generate_kwargs["language"] = language
        # 说话人分离参数
        if ENABLE_SPK:
            generate_kwargs["spk_thresh"] = spk_thresh
            
        # 4. Run ASR
        # Run in thread pool to avoid blocking event loop
        result = await asyncio.to_thread(model_instance.generate, **generate_kwargs)
        
        # 5. Parse Result
        # FunASR returns List[Dict] usually.
        # Structure: [{"key": "...", "text": "...", "timestamp": ...}]
        
        res_dict = {}
        if isinstance(result, list) and result:
            res_dict = result[0] if isinstance(result[0], dict) else {"text": str(result[0])}
        elif isinstance(result, dict):
            res_dict = result
        else:
            res_dict = {"text": str(result)}
            
        raw_text = res_dict.get("text", "")
        
        # 6. Post-process
        processed_text = raw_text
        if use_itn: # We can always run postprocess, but use_itn controls the ITN part
            processed_text = postprocess(raw_text, apply_itn_flag=use_itn)
        else:
            # Still run basic cleanup without ITN if needed? 
            # The prompt implies "use_itn" might just toggle ITN.
            # Let's run full postprocess with flag
             processed_text = postprocess(raw_text, apply_itn_flag=False)

        # 7. Construct Final Response (adhering to result.md)
        
        # Extract sentence info if available
        # FunASR result typically has 'sentence_info' or 'timestamp' that we can parse
        # If 'sentence_info' is already there, use it.
        # Note: The raw result from generate might differ based on version.
        # Assuming standard FunASR output.
        
        sentence_info = res_dict.get("sentence_info", [])
        if not sentence_info and "timestamp" in res_dict:
             # If no sentence_info but we have timestamps, maybe we can construct it?
             # For now, let's just pass what we have or empty list if missing
             pass
        
        # Ensure sentence_info items have required fields if we want strict compliance?
        # The user provided result.md as "Recommended Norm".
        # We will wrap the result.
        
        final_result_struct = {
            "key": "audio",
            "text": raw_text,
            "text_processed": processed_text,
            "sentence_info": sentence_info
        }
        
        response_data = {
            "text": processed_text, # Main text
            "raw_text": raw_text,
            "result": final_result_struct,
            "meta": {
                "time_unit": "ms",
                "model": MODEL_NAME,
                "device": DEVICE
            }
        }
        
        return JSONResponse(response_data)

    finally:
        if file:
            await file.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
