#!/bin/bash

set -e

# download models and datasets from Hugging Face.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prc-estimator-llm

# Stable Diffusion 2.1 Base (For GIM main experiments)
hf download Manojb/stable-diffusion-2-1-base --revision 0094d483a120f3f33dafbd187ea4aa60d10de75c --exclude "*.bin" "*.ckpt" "*.fp16.safetensors" "*pruned.safetensors" --local-dir .models/Manojb/stable-diffusion-2-1-base

# Stable Diffusion Prompts (For GIM experiments)
hf download Gustavosta/Stable-Diffusion-Prompts --repo-type dataset --revision d816d4a05cb89bde39dd99284c459801e1e7e69a --local-dir .models/Gustavosta/Stable-Diffusion-Prompts