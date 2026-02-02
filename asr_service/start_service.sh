#!/bin/bash
set -e

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Go to parent directory (project root) so asr_service can be imported
cd "$SCRIPT_DIR/.."

# 检查是否安装了依赖
if ! python3 -c "import fastapi" &> /dev/null; then
    echo "Dependencies not found. Installing..."
    pip install -r asr_service/requirements.txt
fi

echo "Starting ASR Service on port 8000..."
# 使用 uvicorn 启动
# --reload: 开发模式，代码变动自动重启
# --host 0.0.0.0: 允许外部访问
uvicorn asr_service.app:app --host 0.0.0.0 --port 8000 --reload
