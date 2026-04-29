"""Download stabilityai/stable-diffusion-3.5-medium in diffusers format.

IMPORTANT: This is a gated model. Before running:
  1. Accept the license at https://huggingface.co/stabilityai/stable-diffusion-3.5-medium
  2. Log in:  huggingface-cli login
     (paste your HF token from https://huggingface.co/settings/tokens)
"""

from pathlib import Path
from huggingface_hub import snapshot_download

REPO_ID = "stabilityai/stable-diffusion-3.5-medium"
LOCAL_DIR = Path("models/stable-diffusion-3.5-medium")

ALLOW_PATTERNS = [
    "model_index.json",
    "scheduler/**",
    "text_encoder/**",    # CLIP-L
    "text_encoder_2/**",  # CLIP-G
    "text_encoder_3/**",  # T5-XXL
    "tokenizer/**",
    "tokenizer_2/**",
    "tokenizer_3/**",
    "transformer/**",
    "vae/**",
]

print(f"Downloading {REPO_ID}  (~17 GB, takes a few minutes)...")
snapshot_download(
    repo_id=REPO_ID,
    local_dir=str(LOCAL_DIR),
    allow_patterns=ALLOW_PATTERNS,
)
print(f"\nDone. Model at {LOCAL_DIR}")
print("Use --model-dir models/stable-diffusion-3.5-medium in train/inference scripts.")
