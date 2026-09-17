#!/bin/bash
# Download sentence-transformers model using China-friendly mirrors
# Run this on a machine WITH internet, then SCP the model to your offline Mac

MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"
CACHE_DIR="models/all-MiniLM-L6-v2"

# Option 1: hf-mirror.com (often works in China without VPN)
export HF_ENDPOINT=https://hf-mirror.com

# Option 2: modelscope.cn (Alibaba's mirror, very reliable in China)
# pip install modelscope
# python -c "from modelscope import snapshot_download; snapshot_download('$MODEL_NAME', cache_dir='models/')"

mkdir -p "$CACHE_DIR"

# Download using huggingface-cli with mirror
python -c "
from huggingface_hub import snapshot_download
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
snapshot_download(repo_id='$MODEL_NAME', local_dir='$CACHE_DIR', local_dir_use_symlinks=False)
" || echo "hf-mirror failed, try modelscope instead"

echo "Model downloaded to $CACHE_DIR"
echo "SCP this folder to your Mac: scp -r $CACHE_DIR mac@macbook:~/yellow_phoenix/models/"
