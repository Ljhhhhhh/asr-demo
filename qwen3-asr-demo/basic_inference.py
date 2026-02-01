import torch
from qwen_asr import Qwen3ASRModel
import sys

def main():
    print("Initializing Qwen3-ASR model...")
    
    # Check for MPS availability
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS (Metal Performance Shaders) acceleration.")
    else:
        print("MPS not available, using CPU. This might be slow.")

    try:
        # Check if local model exists
        import os
        model_path = "Qwen/Qwen3-ASR-1.7B"
        local_path = "./models/Qwen3-ASR-1.7B"
        if os.path.exists(local_path):
            print(f"Loading model from local path: {local_path}")
            model_path = local_path
        else:
            print(f"Loading model from Hugging Face Hub: {model_path}")

        # Load model using bfloat16 for efficiency on Apple Silicon
        model = Qwen3ASRModel.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map=device,
            max_inference_batch_size=32,
            max_new_tokens=256,
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Sample audio URL (Chinese)
    audio_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_zh.wav"
    print(f"Transcribing audio from: {audio_url}")

    try:
        results = model.transcribe(
            audio=audio_url,
            language=None, # Auto-detect
        )
        
        print("\n--- Transcription Result ---")
        for i, res in enumerate(results):
            print(f"Result {i+1}:")
            print(f"  Language: {res.language}")
            print(f"  Text: {res.text}")
            
    except Exception as e:
        print(f"Error during transcription: {e}")

if __name__ == "__main__":
    main()
