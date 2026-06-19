#!/bin/bash

set -e

# download models and datasets from Hugging Face.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prc-estimator-llm

# DeepSeek Qwen2.5 7B (For LLM main experiments)
hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --revision 916b56a44061fd5cd7d6a8fb632557ed4f724f60 --local-dir .models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
