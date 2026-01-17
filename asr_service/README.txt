Local ASR Service (FunASR)

Setup (macOS):
1) Install ffmpeg: brew install ffmpeg
2) Create venv: python3 -m venv .venv
3) Activate: source .venv/bin/activate
4) Install deps: pip install -r requirements.txt
5) Run server: uvicorn asr_service.app:app --host 0.0.0.0 --port 8000

Setup (Windows PowerShell):
1) Install ffmpeg: choco install ffmpeg -y
2) Create venv: python -m venv .venv
3) Activate: .venv\Scripts\activate
4) Install deps: pip install -r requirements.txt
5) Run server: uvicorn asr_service.app:app --host 0.0.0.0 --port 8000

Open http://localhost:8000

Config:
ASR_MODEL_NAME: FunASR model id
ASR_VAD_MODEL: VAD model id
ASR_PUNC_MODEL: Punctuation model id
ASR_DEVICE: cpu
ASR_MAX_FILE_SIZE_MB: upload limit (default 50)
