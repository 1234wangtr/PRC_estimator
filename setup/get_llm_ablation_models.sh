#!/bin/bash

set -e

# download models and datasets from Hugging Face.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prc-estimator-llm

# Qwen3 8B (For LLM ablation experiments)
hf download Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 --local-dir .models/Qwen/Qwen3-8B

