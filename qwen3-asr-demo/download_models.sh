#!/bin/bash

# Install hf_transfer for faster downloads
pip install hf_transfer

# Enable hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "Downloading Qwen3-ASR-1.7B to ./models/Qwen3-ASR-1.7B..."
huggingface-cli download Qwen/Qwen3-ASR-1.7B \
    --local-dir ./models/Qwen3-ASR-1.7B \
    --local-dir-use-symlinks False \
    --resume-download

echo "Downloading Qwen3-ForcedAligner-0.6B to ./models/Qwen3-ForcedAligner-0.6B..."
huggingface-cli download Qwen/Qwen3-ForcedAligner-0.6B \
    --local-dir ./models/Qwen3-ForcedAligner-0.6B \
    --local-dir-use-symlinks False \
    --resume-download

echo "Download complete!"
