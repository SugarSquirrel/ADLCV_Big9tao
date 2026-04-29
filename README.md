# ADLCV Ex1 - SD3.5 DreamBooth-LoRA Rockfall Generation

This repository contains the important code and small training dataset for the Ex1 rockfall image generation experiment.

The main experiment uses Stable Diffusion 3.5 Medium with LoRA / DreamBooth-LoRA fine-tuning to generate realistic road rockfall scenes.

## Repository Layout

```text
.
├── data/
│   ├── raw/                 # 15 original rockfall training images
│   └── train/               # 512x512 preprocessed images and metadata_v10.jsonl
├── evaluation/              # CLIP score and KID evaluation scripts
├── models/
│   └── stable-diffusion-3.5-medium/
│       └── .gitkeep         # placeholder only; model weights are not uploaded
├── outputs/
│   ├── lora/                # placeholder for trained LoRA weights
│   ├── runs/                # placeholder for generated images
│   └── results/             # placeholder for curated results
└── scripts/
    ├── sd35_config.py
    ├── sd35_download_model.py
    ├── sd35_preprocess.py
    ├── sd35_train_lora.py
    ├── sd35_train_dreambooth_lora.py
    ├── sd35_infer_combined.py
    └── sd35_evaluate.py
```

## Setup

Use the `adlcv_sd` conda environment used for the experiment.

Download SD3.5 Medium locally before training or inference:

```bash
huggingface-cli login
python scripts/sd35_download_model.py
```

The model will be placed at:

```text
models/stable-diffusion-3.5-medium/
```

Model weights are intentionally ignored by Git.

## Workflow

Preprocess the 15 training images:

```bash
python scripts/sd35_preprocess.py
```

Train DreamBooth-LoRA:

```bash
python scripts/sd35_train_dreambooth_lora.py --force-recompute
```

Run a smoke inference:

```bash
python scripts/sd35_infer_combined.py \
  --lora-dir outputs/lora/sksrockfall-v10-db \
  --output-root-a outputs/runs/sd35_v10db_settingA_smoke \
  --output-root-b outputs/runs/sd35_v10db_settingB_smoke \
  --num-images-per-prompt 2
```

Run evaluation:

```bash
python evaluation/clip_score.py \
  --image-root outputs/runs/sd35_v10db_settingA_smoke \
  --prompt-source sd35 \
  --output outputs/runs/sd35_v10db_settingA_smoke/clip_score.txt

python evaluation/kid_score.py \
  --real-dir data/train \
  --fake-root outputs/runs/sd35_v10db_settingA_smoke \
  --bootstrap 1000 \
  --output outputs/runs/sd35_v10db_settingA_smoke/kid_score.txt
```

## Notes

- Stable Diffusion model files, LoRA checkpoints, generated images, and cache files are not uploaded.
- The GitHub version keeps only placeholder folders for `models/` and `outputs/`.
- `data/raw/` and `data/train/` are included because the dataset contains only 15 images.
