#!/usr/bin/env python3
"""Download guarded Hugging Face model repos with a token.

Usage:
  python scripts/download_guarded_hf_models.py

Optional environment variables:
  HF_TOKEN=hf_xxx...                      # preferred over editing this file
  HF_MODEL_IDS=org/sam-audio-small,org/sam-audio-large
"""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download

# Prefer setting HF_TOKEN in your shell, but you can paste it here if needed.
HF_TOKEN = os.getenv("HF_TOKEN", "PASTE_YOUR_HF_TOKEN_HERE")

# Replace with full repo IDs if needed, e.g. "my-org/sam-audio-small".
# You can also override with HF_MODEL_IDS env var.
DEFAULT_MODEL_IDS = [
    "facebook/sam-audio-small",
    "facebook/sam-audio-large",
]


def parse_model_ids() -> list[str]:
    raw = os.getenv("HF_MODEL_IDS", "")
    if not raw.strip():
        return DEFAULT_MODEL_IDS
    return [item.strip() for item in raw.split(",") if item.strip()]


def main() -> None:
    if HF_TOKEN == "PASTE_YOUR_HF_TOKEN_HERE":
        raise ValueError(
            "Set HF_TOKEN in your environment or replace HF_TOKEN placeholder in this file."
        )

    model_ids = parse_model_ids()
    base_dir = Path("models")
    base_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {len(model_ids)} model(s) into {base_dir.resolve()}...")

    for repo_id in model_ids:
        local_dir = base_dir / repo_id.replace("/", "__")
        print(f"\n-> Downloading {repo_id} to {local_dir}...")
        snapshot_download(
            repo_id=repo_id,
            token=HF_TOKEN,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"✓ Done: {repo_id}")

    print("\nAll downloads finished.")


if __name__ == "__main__":
    main()
