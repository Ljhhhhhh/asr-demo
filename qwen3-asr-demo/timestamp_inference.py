import torch
from qwen_asr import Qwen3ASRModel
import json

def main():
    print("Initializing Qwen3-ASR model with Forced Aligner...")
    
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS (Metal Performance Shaders) acceleration.")

    try:
        import os
        
        # Setup Aligner Path
        aligner_path = "Qwen/Qwen3-ForcedAligner-0.6B"
        local_aligner_path = "./models/Qwen3-ForcedAligner-0.6B"
        if os.path.exists(local_aligner_path):
             print(f"Using local aligner: {local_aligner_path}")
             aligner_path = local_aligner_path

        # Setup Model Path
        model_path = "Qwen/Qwen3-ASR-1.7B"
        local_model_path = "./models/Qwen3-ASR-1.7B"
        if os.path.exists(local_model_path):
             print(f"Loading model from local path: {local_model_path}")
             model_path = local_model_path

        model = Qwen3ASRModel.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map=device,
            forced_aligner=aligner_path,
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # audio_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav"
    import sys
    audio_path = sys.argv[1] if len(sys.argv) > 1 else "demo_5min.m4a"
    print(f"Transcribing audio from: {audio_path}")

    try:
        # Enable timestamp return with forced aligner
        results = model.transcribe(
            audio=audio_path,
            return_time_stamps=True,
        )
        
        # Process into sentence-level segments
        raw_tokens = []
        for res in results:
            if res.time_stamps:
                raw_tokens.extend(res.time_stamps.items)
            else:
                 # Fallback if no timestamps
                pass

        sentences = []
        current_sentence_tokens = []
        
        # Punctuation marks to split on
        punctuations = ['。', '？', '！', '.', '?', '!']
        
        for token in raw_tokens:
            current_sentence_tokens.append(token)
            txt = token.text.strip()
            if txt and txt[-1] in punctuations:
                # End of sentence
                start_t = current_sentence_tokens[0].start_time
                end_t = current_sentence_tokens[-1].end_time
                full_text = "".join([t.text for t in current_sentence_tokens])
                
                sentences.append({
                    "text": full_text,
                    "timestamps": [
                        {"text": t.text, "start": t.start_time, "end": t.end_time} 
                        for t in current_sentence_tokens
                    ],
                    "start": start_t,
                    "end": end_t
                })
                current_sentence_tokens = []
        
        # Add remaining tokens
        if current_sentence_tokens:
            start_t = current_sentence_tokens[0].start_time
            end_t = current_sentence_tokens[-1].end_time
            full_text = "".join([t.text for t in current_sentence_tokens])
            sentences.append({
                "text": full_text,
                "timestamps": [
                    {"text": t.text, "start": t.start_time, "end": t.end_time} 
                    for t in current_sentence_tokens
                ],
                "start": start_t,
                "end": end_t
            })

        print("\n--- Transcription with Timestamps (Sentence Split) ---")
        json_str = json.dumps(sentences, indent=2, ensure_ascii=False)
        print(json_str)
        
        with open("result.json", "w", encoding="utf-8") as f:
            f.write(json_str)
            print("\nResult saved to result.json")
            
    except Exception as e:
        print(f"Error during transcription: {e}")

if __name__ == "__main__":
    main()
