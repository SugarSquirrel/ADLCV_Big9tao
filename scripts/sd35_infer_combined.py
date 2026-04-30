"""Combined inference — Base model / Setting A (normal steps) / Setting B (few steps).

Normal mode (LoRA):
  Loads SD3.5 + DreamBooth LoRA, injects concept token, generates Setting A and B.

Base model mode (--no-lora):
  Loads SD3.5 base only (no LoRA, no concept token).
  Uses prompts stripped of <rfblk> for a fair comparison baseline.
  Recommended: --num-images-per-prompt 30

Seed formula (shared across all modes): seed = prompt_idx * 1000 + image_idx
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers")
warnings.filterwarnings("ignore", category=FutureWarning, module="peft")
warnings.filterwarnings("ignore", message=".*weights_only.*")
warnings.filterwarnings("ignore", message=".*has no attribute.*xpu.*")

import torch
from diffusers import StableDiffusion3Pipeline

sys.path.insert(0, os.path.dirname(__file__))
from sd35_config import (
    FEW_STEP_NEGATIVE_PROMPT,
    NEGATIVE_PROMPT,
    PLACEHOLDER_TOKEN,
    PROMPTS,
)
from sd35_train_lora import load_concept_token

# Prompts with <rfblk> stripped — used for base model (no concept token)
BASE_PROMPTS = [
    p[len("<rfblk>, "):] if p.startswith("<rfblk>, ") else p
    for p in PROMPTS
]

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--model-dir",            default="models/stable-diffusion-3.5-medium")
parser.add_argument("--lora-dir",             default="outputs/lora/sksrockfall-v10-db")
parser.add_argument("--checkpoint",           default=None,
                    help="Checkpoint sub-dir (e.g. checkpoint-600). "
                         "None = load final pytorch_lora_weights.safetensors.")
parser.add_argument("--output-root-a",        default="outputs/results/v10db_final_infer/settingA")
parser.add_argument("--output-root-b",        default="outputs/results/v10db_final_infer/settingB")
parser.add_argument("--output-root-base",     default="outputs/results/v10db_final_infer/base",
                    help="Output directory for Base Model run (--no-lora mode)")
parser.add_argument("--num-images-per-prompt", type=int,   default=300)
parser.add_argument("--steps-a",              type=int,   default=35)
parser.add_argument("--guidance-a",           type=float, default=5.5)
parser.add_argument("--steps-b",              type=int,   default=8)
parser.add_argument("--guidance-b",           type=float, default=4.0)
parser.add_argument("--lora-scale",           type=float, default=0.8)
parser.add_argument("--height",               type=int,   default=768)
parser.add_argument("--width",                type=int,   default=768)
parser.add_argument("--skip-a",               action="store_true", help="Skip Setting A")
parser.add_argument("--skip-b",               action="store_true", help="Skip Setting B")
parser.add_argument("--no-lora",              action="store_true",
                    help="Base model mode: skip LoRA/concept-token, use stripped prompts")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Load pipeline
# ---------------------------------------------------------------------------

print("Loading SD3.5 Medium pipeline …")
pipe = StableDiffusion3Pipeline.from_pretrained(
    args.model_dir,
    torch_dtype=torch.float16,
)
pipe = pipe.to(device)

# ── LoRA + concept token (skipped in base model mode) ──────────────────────
if args.no_lora:
    lora_path = None
    lora_weight_path = None
    ok = False
    print("Base model mode — LoRA and concept token loading skipped.")
else:
    lora_path = (
        os.path.join(args.lora_dir, args.checkpoint)
        if args.checkpoint else args.lora_dir
    )
    lora_weight_path = os.path.join(lora_path, "pytorch_lora_weights.safetensors")
    if not os.path.isfile(lora_weight_path):
        raise FileNotFoundError(
            f"LoRA weight file not found: {lora_weight_path}\n"
            "Check --lora-dir / --checkpoint."
        )
    print(f"Loading LoRA weights from {lora_path} …")
    pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors")
    pipe.fuse_lora(lora_scale=args.lora_scale)

    print(f"Injecting concept token '{PLACEHOLDER_TOKEN}' …")
    ok = load_concept_token(args.lora_dir, pipe, PLACEHOLDER_TOKEN)
    if not ok:
        print(f"  WARNING: concept_token_embeddings.pt not found in {args.lora_dir}")

pipe.enable_model_cpu_offload()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_run_metadata(output_root, label, steps, guidance, negative_prompt, prompts):
    payload = {
        "created_at":            datetime.now().isoformat(timespec="seconds"),
        "label":                 label,
        "model_dir":             args.model_dir,
        "no_lora":               args.no_lora,
        "lora_dir":              args.lora_dir if not args.no_lora else None,
        "checkpoint":            args.checkpoint,
        "resolved_lora_path":    lora_path,
        "concept_token":         PLACEHOLDER_TOKEN if not args.no_lora else None,
        "concept_embedding_loaded": ok,
        "num_images_per_prompt": args.num_images_per_prompt,
        "steps":                 steps,
        "guidance":              guidance,
        "lora_scale":            args.lora_scale if not args.no_lora else None,
        "height":                args.height,
        "width":                 args.width,
        "seed_formula":          "seed = prompt_index * 1000 + image_index (0-based)",
        "negative_prompt":       negative_prompt,
        "prompts":               prompts,
    }
    os.makedirs(output_root, exist_ok=True)
    with open(os.path.join(output_root, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")

    lines = [label, ""]
    for k, v in payload.items():
        if k not in ("prompts", "negative_prompt"):
            lines.append(f"{k}: {v}")
    lines += ["", "negative_prompt:", negative_prompt, "", "prompts:"]
    lines += [f"{i+1}. {p}" for i, p in enumerate(prompts)]
    with open(os.path.join(output_root, "RUN_SUMMARY.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_setting(output_root, steps, guidance, neg_prompt, label, prompts):
    write_run_metadata(output_root, label, steps, guidance, neg_prompt, prompts)
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  {steps} steps, guidance={guidance}")
    print(f"  Output → {output_root}")
    print(f"  {len(prompts)} prompts × {args.num_images_per_prompt} images")
    print(f"{'='*60}")

    for i, prompt in enumerate(prompts):
        prompt_dir = os.path.join(output_root, f"prompt_{i+1}")
        os.makedirs(prompt_dir, exist_ok=True)
        print(f"\nPrompt {i+1}/{len(prompts)}: {prompt[:80]}…")

        for j in range(args.num_images_per_prompt):
            out_path = os.path.join(prompt_dir, f"{j:04d}.png")
            if os.path.exists(out_path):
                continue

            seed = i * 1000 + j
            generator = torch.Generator(device="cpu").manual_seed(seed)

            image = pipe(
                prompt=prompt,
                negative_prompt=neg_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                height=args.height,
                width=args.width,
                generator=generator,
            ).images[0]

            image.save(out_path)

            if j % 50 == 0:
                print(f"  [{i+1}/{len(prompts)}] {j+1}/{args.num_images_per_prompt} done")

    print(f"\n{label} complete → {output_root}")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if args.no_lora:
    # Base model baseline — same inference params as Setting A for fair comparison
    run_setting(
        output_root=args.output_root_base,
        steps=args.steps_a,
        guidance=args.guidance_a,
        neg_prompt=NEGATIVE_PROMPT,
        label=f"Base Model ({args.steps_a} steps, guidance={args.guidance_a}, no LoRA)",
        prompts=BASE_PROMPTS,
    )
else:
    if not args.skip_a:
        run_setting(
            output_root=args.output_root_a,
            steps=args.steps_a,
            guidance=args.guidance_a,
            neg_prompt=NEGATIVE_PROMPT,
            label=f"Setting A ({args.steps_a} steps, guidance={args.guidance_a})",
            prompts=PROMPTS,
        )
    if not args.skip_b:
        run_setting(
            output_root=args.output_root_b,
            steps=args.steps_b,
            guidance=args.guidance_b,
            neg_prompt=FEW_STEP_NEGATIVE_PROMPT,
            label=f"Setting B ({args.steps_b} steps, guidance={args.guidance_b})",
            prompts=PROMPTS,
        )

print("\nAll done.")
