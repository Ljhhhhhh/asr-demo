import requests
import json
import os
import sys

def test_demo_audio():
    # File path provided by user
    file_path = "/Users/pipilu/Documents/MaDun/GuoFu/asr-demo/demo_5min.m4a"
    url = "http://127.0.0.1:8000/asr/transcribe"
    
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    print(f"Sending {file_path} to {url}...")
    print("This may take some time depending on your machine (approx. 1/10 to 1/2 of audio duration)...")

    try:
        with open(file_path, 'rb') as f:
            files = {'file': f}
            # Optional parameters
            data = {
                'use_itn': 'true',
                'hotword': '测试',
            }
            # Set a long timeout (e.g. 10 mins) because 5min audio might take a while on CPU
            response = requests.post(url, files=files, data=data, timeout=600)
            
        if response.status_code == 200:
            print("\nTranscription Success!")
            result = response.json()
            # Print a summary
            print(f"Text (preview): {result.get('text', '')[:200]}...")
            print(f"Raw Text (preview): {result.get('raw_text', '')[:200]}...")
            
            # Save full result to a file for inspection
            output_file = "demo_result.json"
            with open(output_file, "w", encoding="utf-8") as f_out:
                json.dump(result, f_out, ensure_ascii=False, indent=2)
            print(f"\nFull result saved to {output_file}")
        else:
            print(f"\nError: {response.status_code}")
            print(response.text)

    except Exception as e:
        print(f"\nException occurred: {str(e)}")

if __name__ == "__main__":
    test_demo_audio()
