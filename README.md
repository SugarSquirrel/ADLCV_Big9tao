# ADLCV Ex1 — SD 3.5 DreamBooth-LoRA Rockfall Generation

Generates synthetic road rockfall images using **Stable Diffusion 3.5 Medium** fine-tuned with **DreamBooth + LoRA** on 15 real images.
Learned trigger token: `<rfblk>`.

Three experimental conditions are evaluated:

| | Base Model | Setting A | Setting B |
|---|:---:|:---:|:---:|
| LoRA | ✗ | ✓ | ✓ |
| Steps | 35 | 35 | 8 |
| CFG | 5.5 | 5.5 | 3.5 |
| Images | 300 | 3,000 | 3,000 |

## Repository Layout

```
.
├── assets/
│   ├── base/          # Base Model qualitative samples (p1–p10.png)
│   ├── eval/          # Evaluation charts (clip_per_prompt.png, summary.png)
│   ├── settingA/      # Setting A qualitative samples
│   ├── settingB/      # Setting B qualitative samples
│   └── train/         # Training image examples
├── data/
│   ├── raw/           # 15 original rockfall photos (not uploaded)
│   └── train/         # 512×512 preprocessed images + metadata_v10.jsonl
├── models/
│   └── stable-diffusion-3.5-medium/   # not uploaded; see Setup
├── outputs/           # all generated files; not uploaded
│   ├── eval/          # CLIP/KID scores and auto-generated charts
│   ├── lora/          # LoRA weights and concept token embeddings
│   ├── results/v10db_final_infer/
│   │   ├── base/      # Base Model outputs
│   │   ├── settingA/  # Setting A outputs
│   │   └── settingB/  # Setting B outputs
│   └── runs/          # smoke test outputs
├── scripts/
│   ├── sd35_config.py                 # prompts and negative prompts
│   ├── sd35_download_model.py         # download SD 3.5 Medium from HuggingFace
│   ├── sd35_preprocess.py             # center-crop + resize training images
│   ├── sd35_train_dreambooth_lora.py  # DreamBooth + LoRA training
│   ├── sd35_train_lora.py             # concept token injection helper
│   ├── sd35_infer_combined.py         # inference for all three conditions
│   └── sd35_evaluate.py               # CLIP Score + KID evaluation + charts
├── ex1_report.md
└── README.md
```

## Setup

```bash
conda create -n adlcv_sd python=3.10 -y
conda activate adlcv_sd
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers accelerate peft pillow
pip install torchmetrics torch-fidelity matplotlib
```

Download SD 3.5 Medium (requires HuggingFace login with model access):

```bash
huggingface-cli login
conda run -n adlcv_sd python scripts/sd35_download_model.py
```

## Workflow

### 1 · Preprocess training data

Place 15 raw images in `data/raw/`, then:

```bash
conda run -n adlcv_sd python scripts/sd35_preprocess.py
```

Outputs 512×512 PNG files and `data/train/metadata_v10.jsonl`.

### 2 · Train DreamBooth + LoRA

```bash
conda run -n adlcv_sd python scripts/sd35_train_dreambooth_lora.py
```

Add `--force-recompute` to discard the cached prior-image latents and regenerate them (needed if training data changes).

Key parameters:

| Flag | Default | Description |
|---|:---:|---|
| `--max-train-steps` | 600 | Total training steps |
| `--learning-rate` | 3e-5 | Learning rate |
| `--rank` | 8 | LoRA rank |
| `--prior-weight` | 0.1 | Prior preservation loss weight λ |
| `--save-every` | 50 | Checkpoint save interval |
| `--force-recompute` | — | Re-generate prior image latent cache |

Outputs to `outputs/lora/sksrockfall-v10-db/`:
`pytorch_lora_weights.safetensors`, `concept_token_embeddings.pt`, and `checkpoint-*/` folders.

### 3 · Inference

**Setting A + B** (LoRA model, 300 images/prompt each):

```bash
conda run -n adlcv_sd python scripts/sd35_infer_combined.py
```

**Base Model** (no LoRA, 30 images/prompt):

```bash
conda run -n adlcv_sd python scripts/sd35_infer_combined.py \
    --no-lora --num-images-per-prompt 30
```

Generation is resumable — already-existing images are skipped automatically.

Use `--skip-a` / `--skip-b` to run only one LoRA setting.
Use `--checkpoint checkpoint-400` to evaluate a mid-training checkpoint.

### 4 · Evaluate (CLIP Score + KID)

**Full evaluation** (all three conditions):

```bash
conda run -n adlcv_sd python scripts/sd35_evaluate.py \
    --dir-base outputs/results/v10db_final_infer/base
```

**LoRA only** (Setting A + B):

```bash
conda run -n adlcv_sd python scripts/sd35_evaluate.py
```

**Re-plot only** (no recompute, uses saved `outputs/eval/eval_results.json`):

```bash
conda run -n adlcv_sd python scripts/sd35_evaluate.py --plot-only
```

**Skip CLIP recompute** (reload from JSON, recompute KID only):

```bash
conda run -n adlcv_sd python scripts/sd35_evaluate.py \
    --dir-base outputs/results/v10db_final_infer/base --skip-clip
```

Charts are saved to `outputs/eval/`; copy to `assets/eval/` to include in the report.

## Notes

- Model weights (`*.safetensors`, `*.pt`), generated images, and LoRA checkpoints are excluded from this repository.
- `data/train/` (15 images + captions) is included because the dataset is small.
- `data/raw/` is not uploaded.
