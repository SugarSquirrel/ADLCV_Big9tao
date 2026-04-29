"""Combined inference — Setting A (normal steps) + Setting B (few steps).

Loads SD3.5 + LoRA once, injects the saved concept token embeddings, then
generates all images for both settings sequentially.  Using the same seed
formula (prompt_idx * 1000 + image_idx) across both settings ensures a fair
comparison required by the assignment.

Default output layout:
  outputs/runs/sd35_v10_settingA/prompt_1/ … prompt_10/
  outputs/runs/sd35_v10_settingB/prompt_1/ … prompt_10/
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

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--model-dir",            default="models/stable-diffusion-3.5-medium")
parser.add_argument("--lora-dir",             default="outputs/lora/sksrockfall-v10")
parser.add_argument("--checkpoint",           default=None,
                    help="Checkpoint sub-dir to load (e.g. checkpoint-600). "
                         "If None, loads final pytorch_lora_weights.safetensors.")
parser.add_argument("--output-root-a",        default="outputs/runs/sd35_v10_settingA")
parser.add_argument("--output-root-b",        default="outputs/runs/sd35_v10_settingB")
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
args = parser.parse_args()

os.makedirs(args.output_root_a, exist_ok=True)
os.makedirs(args.output_root_b, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Load pipeline (once, shared by both settings)
# ---------------------------------------------------------------------------

print("Loading SD3.5 Medium pipeline …")
pipe = StableDiffusion3Pipeline.from_pretrained(
    args.model_dir,
    torch_dtype=torch.float16,
)
pipe = pipe.to(device)

lora_path = (
    os.path.join(args.lora_dir, args.checkpoint)
    if args.checkpoint else args.lora_dir
)
lora_weight_path = os.path.join(lora_path, "pytorch_lora_weights.safetensors")
if not os.path.isfile(lora_weight_path):
    raise FileNotFoundError(
        f"LoRA weight file not found: {lora_weight_path}\n"
        f"Check --lora-dir/--checkpoint. The directory must contain "
        f"pytorch_lora_weights.safetensors."
    )

print(f"Loading LoRA weights from {lora_path} …")
pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors")
pipe.fuse_lora(lora_scale=args.lora_scale)

print(f"Injecting concept token '{PLACEHOLDER_TOKEN}' embeddings …")
ok = load_concept_token(args.lora_dir, pipe, PLACEHOLDER_TOKEN)
if not ok:
    print(f"  WARNING: concept_token_embeddings.pt not found in {args.lora_dir} — "
          "trigger token will be random; results may be poor.")

pipe.enable_model_cpu_offload()

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def write_run_metadata(output_root, label, steps, guidance, negative_prompt):
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "label": label,
        "model_dir": args.model_dir,
        "lora_dir": args.lora_dir,
        "checkpoint": args.checkpoint,
        "resolved_lora_path": lora_path,
        "lora_weight_path": lora_weight_path,
        "concept_token": PLACEHOLDER_TOKEN,
        "concept_embedding_loaded": ok,
        "num_images_per_prompt": args.num_images_per_prompt,
        "steps": steps,
        "guidance": guidance,
        "lora_scale": args.lora_scale,
        "height": args.height,
        "width": args.width,
        "seed_formula": "seed = prompt_index * 1000 + image_index, prompt_index is zero-based",
        "negative_prompt": negative_prompt,
        "prompts": PROMPTS,
    }
    os.makedirs(output_root, exist_ok=True)
    with open(os.path.join(output_root, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")

    summary = [
        f"{label}",
        "",
        f"created_at: {payload['created_at']}",
        f"model_dir: {args.model_dir}",
        f"lora_dir: {args.lora_dir}",
        f"checkpoint: {args.checkpoint}",
        f"resolved_lora_path: {lora_path}",
        f"concept_token: {PLACEHOLDER_TOKEN}",
        f"concept_embedding_loaded: {ok}",
        f"num_images_per_prompt: {args.num_images_per_prompt}",
        f"steps: {steps}",
        f"guidance: {guidance}",
        f"lora_scale: {args.lora_scale}",
        f"height: {args.height}",
        f"width: {args.width}",
        "",
        "negative_prompt:",
        negative_prompt,
        "",
        "prompts:",
    ]
    summary.extend(f"{i + 1}. {prompt}" for i, prompt in enumerate(PROMPTS))
    with open(os.path.join(output_root, "RUN_SUMMARY.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary) + "\n")

def run_setting(output_root, steps, guidance, neg_prompt, label):
    write_run_metadata(output_root, label, steps, guidance, neg_prompt)
    print(f"\n{'='*60}")
    print(f"  {label}: {steps} steps, guidance={guidance}")
    print(f"  Output → {output_root}")
    print(f"  {len(PROMPTS)} prompts × {args.num_images_per_prompt} images")
    print(f"{'='*60}")

    for i, prompt in enumerate(PROMPTS):
        # PROMPTS already contain <rfblk> — do NOT prepend again
        prompt_dir = os.path.join(output_root, f"prompt_{i+1}")
        os.makedirs(prompt_dir, exist_ok=True)
        print(f"\nPrompt {i+1}/{len(PROMPTS)}: {prompt[:80]}…")

        for j in range(args.num_images_per_prompt):
            out_path = os.path.join(prompt_dir, f"{j:04d}.png")
            if os.path.exists(out_path):
                continue  # resume: skip already generated

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
                print(f"  [{i+1}/{len(PROMPTS)}] {j+1}/{args.num_images_per_prompt} done")

    print(f"\n{label} complete → {output_root}")


# ---------------------------------------------------------------------------
# Run both settings
# ---------------------------------------------------------------------------

if not args.skip_a:
    run_setting(
        output_root=args.output_root_a,
        steps=args.steps_a,
        guidance=args.guidance_a,
        neg_prompt=NEGATIVE_PROMPT,
        label=f"Setting A ({args.steps_a} steps, guidance={args.guidance_a})",
    )

if not args.skip_b:
    run_setting(
        output_root=args.output_root_b,
        steps=args.steps_b,
        guidance=args.guidance_b,
        neg_prompt=FEW_STEP_NEGATIVE_PROMPT,
        label=f"Setting B ({args.steps_b} steps, guidance={args.guidance_b})",
    )

print("\nAll done.")
