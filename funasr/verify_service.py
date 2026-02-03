import requests
import time
import subprocess
import sys
import os
from pathlib import Path

def test_service():
    print("Starting service...")
    # Start the service in the background
    # We assume we are in the root directory relative to 'funasr'
    # The user asked to build in 'funasr' directory.
    # PYTHONPATH needs to include current dir to resolve imports if running from inside funasr
    
    # Let's assume we run this script from the project root or we fix paths.
    # The service file is at funasr/main.py.
    
    # Ensure dependencies are satisfied. For this test we assume environment includes them OR we just test logic if possible.
    # Since we can't easily install deps in this environment without permission, 
    # and the user might not have them installed yet, we rely on existing env.
    # We will try to start it.
    
    cwd = Path(__file__).parent
    
    # IMPORTANT: We use a different port to avoid conflicts
    params = [sys.executable, "-m", "uvicorn", "main:app", "--port", "18000", "--host", "127.0.0.1"]
    
    # We need to set PYTHONPATH to include current directory so imports work
    env = os.environ.copy()
    env["PYTHONPATH"] = str(cwd)
    
    proc = subprocess.Popen(params, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    try:
        # Wait for service to start
        print("Waiting for service to start (up to 60s)...")
        for i in range(60):
            if i % 10 == 0:
                print(f"Waiting... {i}s")
            try:
                resp = requests.get("http://127.0.0.1:18000/health", timeout=1)
                if resp.status_code == 200:
                    print("Service is ready!")
                    break
            except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout):
                pass
            
            # Check if process is still alive
            if proc.poll() is not None:
                print("Service process died unexpectedly!")
                break
                
            time.sleep(1)
        else:
            print("Timeout waiting for service.")
            # Print stderr
            _, stderr = proc.communicate(timeout=1)
            print(f"Service stderr: {stderr.decode()}")
            return

        # Test GET /models
        print("\nTesting /models...")
        resp = requests.get("http://127.0.0.1:18000/models")
        print(f"Status: {resp.status_code}")
        print(f"Response: {resp.json()}")
        assert resp.status_code == 200

        # Test POST /asr/transcribe
        # Use a dummy URL or small file if possible.
        # Since we might not have internet or a file, let's try a URL that is likely to fail safely or just check validation.
        print("\nTesting /asr/transcribe (validation)...")
        resp = requests.post("http://127.0.0.1:18000/asr/transcribe")
        print(f"Status (expect 400): {resp.status_code}") # Expect 400 because no file
        assert resp.status_code == 400
        
        # Checking result format logic requires a real inference which requires model loading (heavy).
        # We trust the code structure for now, unless we have a small audio file.
        
    finally:
        print("\nStopping service...")
        proc.terminate()
        proc.wait()

if __name__ == "__main__":
    test_service()
