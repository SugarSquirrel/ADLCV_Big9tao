# Scripts

This folder contains the active SD3.5 experiment scripts for rockfall image generation.

## SD3.5 Files

- `sd35_download_model.py`  
  Download/cache SD3.5 Medium into `models/stable-diffusion-3.5-medium`.

- `sd35_preprocess.py`  
  Build `data/train/metadata_v10.jsonl` and SD3.5 training images/captions.

- `sd35_train_lora.py`  
  Train SD3.5 transformer LoRA. Default output: `outputs/lora/sksrockfall-v10`.

- `sd35_train_dreambooth_lora.py`  
  Train SD3.5 DreamBooth-LoRA with prior preservation. Default output: `outputs/lora/sksrockfall-v10-db`.

- `sd35_infer_combined.py`  
  Combined SD3.5 Setting A and Setting B inference. Each run writes `run_config.json` and `RUN_SUMMARY.txt`.

- `sd35_config.py`  
  Shared SD3.5 trigger token, prompts, and negative prompts.

- `sd35_evaluate.py`  
  Helper wrapper for SD3.5 evaluation workflow.

## Typical Commands

```bash
python scripts/sd35_preprocess.py
python scripts/sd35_train_dreambooth_lora.py --force-recompute
python scripts/sd35_infer_combined.py \
  --lora-dir outputs/lora/sksrockfall-v10-db \
  --output-root-a outputs/runs/sd35_v10db_settingA \
  --output-root-b outputs/runs/sd35_v10db_settingB \
  --num-images-per-prompt 10
```
