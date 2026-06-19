#!/bin/bash

set -e

# download models and datasets from Hugging Face.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prc-estimator-llm

# Stable Diffusion 1.5 (For GIM ablation experiments)
hf download stable-diffusion-v1-5/stable-diffusion-v1-5 --revision 451f4fe16113bff5a5d2269ed5ad43b0592e9a14  --exclude "*.bin" "*.ckpt" "*.fp16.safetensors" "*pruned*.safetensors" --local-dir .models/stable-diffusion-v1-5/stable-diffusion-v1-5

# Stable Diffusion 2 Base (For GIM ablation experiments)
hf download Manojb/stable-diffusion-2-base --revision 64bf7b4f10eee35494b38d55c06c0c78cf8b44d0  --exclude "*.bin" "*.ckpt" "*.fp16.safetensors" "*pruned*.safetensors" --local-dir .models/Manojb/stable-diffusion-2-base
