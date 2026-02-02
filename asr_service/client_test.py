import requests
import sys
import os
import time
import json
import argparse

def test_transcribe(args, host="http://localhost:8000"):
    """
    Test the transcription API with a local audio file or URL.
    """
    url = f"{host}/asr/transcribe"
    
    data = {
        'hotword': args.hotword,
        'use_itn': str(args.use_itn).lower(),
        'merge_vad': str(args.merge_vad).lower(),
        'language': args.language,
    }
    
    if args.device:
        data['device'] = args.device

    print(f"Requesting: {url}")
    print(f"Params: {data}")

    try:
        start_time = time.time()
        
        if args.audio_url:
            print(f"Using Audio URL: {args.audio_url}")
            data['audio_url'] = args.audio_url
            response = requests.post(url, data=data)
        elif args.file:
            if not os.path.exists(args.file):
                 print(f"Error: File '{args.file}' not found.")
                 return
            print(f"Uploading file: {args.file}")
            with open(args.file, 'rb') as f:
                files = {'file': f}
                response = requests.post(url, files=files, data=data)
        else:
            print("Error: Either --file or --audio-url must be provided.")
            return

        end_time = time.time()
        
        print(f"Status Code: {response.status_code}")
        print(f"Time Taken: {end_time - start_time:.2f}s")
        
        if response.status_code == 200:
            result = response.json()
            print("-" * 30)
            print("Transcription Result:")
            print(result.get('text', 'No text found'))
            print("-" * 30)
            # print("Full Response:", json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print("Error Response:", response.text)
                
    except Exception as e:
        print(f"Request failed: {e}")

def check_models(host="http://localhost:8000"):
    try:
        resp = requests.get(f"{host}/models")
        if resp.status_code == 200:
            print(f"Models Info: {json.dumps(resp.json(), indent=2)}")
            return True
        else:
            print(f"Models endpoint failed: {resp.status_code}")
            return False
    except Exception as e:
        print(f"Models check failed: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test FunASR Service")
    parser.add_argument("--file", help="Path to local audio file")
    parser.add_argument("--audio-url", help="URL of audio file")
    parser.add_argument("--host", default="http://localhost:8000", help="Service host")
    parser.add_argument("--hotword", default="", help="Comma separated hotwords")
    parser.add_argument("--use-itn", action="store_true", default=True, help="Enable ITN")
    parser.add_argument("--no-itn", action="store_false", dest="use_itn", help="Disable ITN")
    parser.add_argument("--merge-vad", action="store_true", default=True, help="Merge VAD segments")
    parser.add_argument("--language", default="auto", help="Language")
    parser.add_argument("--device", help="Device hint")
    
    args = parser.parse_args()
    
    if check_models(args.host):
        if args.file or args.audio_url:
            test_transcribe(args, args.host)
        else:
             print("Please provide --file or --audio-url to run transcription test.")

