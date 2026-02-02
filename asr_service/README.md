# FunASR HTTP Service

This is a configurable HTTP service wrapping FunASR for speech-to-text transcription.

## Requirements

- Python 3.10+
- FFmpeg (must be installed on the system)

Install python dependencies:

```bash
pip install -r requirements.txt
```

## Running the Service

You can start the service using the provided script:

```bash
./start_service.sh
```

This will run the server on `http://0.0.0.0:8000`.

## Configuration

You can configure the service via environment variables in `.env` or passed to the shell:

- `ASR_DEVICE`: `auto` (default), `cuda`, `cpu`, or `mps` (Apple Silicon).
- `ASR_MODEL_NAME`: Default `paraformer-zh`.
- `ASR_VAD_MODEL`: Default `fsmn-vad`.
- `ASR_PUNC_MODEL`: Default `ct-punc`.
- `ASR_SPK_MODEL`: Default `cam++`.

## API Usage

### Health Check

`GET /health`

Returns the service status and current device.

**Response:**

```json
{
  "status": "ready",
  "device": "mps"
}
```

### Models Info

`GET /models`

Returns configuration of loaded models.

### Transcribe Audio

`POST /asr/transcribe`

Uploads an audio file OR provides a URL for transcription.

**Parameters:**

- `file`: Audio file (wav, mp3, m4a, etc.) [Optional if audio_url provided]
- `audio_url`: Public URL to download audio from [Optional if file provided]
- `hotword`: Comma-separated list of hotwords (e.g. `语音,测试`).
- `use_itn`: Boolean (true/false) to enable Inverse Text Normalization (number conversion). Default: `true`.
- `merge_vad`: Boolean (true/false) to merge short VAD segments. Default: `true`.
- `batch_size_s`: Batch size in seconds. Default `300`.
- `language`: Language code (default: `auto` or `zh-cn`).
- `device`: Device hint (optional).

**Example (using curl with file):**

```bash
curl -X POST "http://localhost:8000/asr/transcribe" \
  -F "file=@test.wav" \
  -F "hotword=语音识别,测试" \
  -F "use_itn=true"
```

**Example (using Python with URL):**

```bash
python client_test.py --audio-url http://example.com/audio.wav --hotword "测试"
```

**Testing:**

Run the included test script:

```bash
python client_test.py --file path/to/audio.wav
# OR
python client_test.py --audio-url http://localhost:8001/audio.wav
```
