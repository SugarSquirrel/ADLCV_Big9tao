"""SD3.5 Medium DreamBooth-LoRA for rockfall concept — v10.

Extends plain LoRA with prior preservation loss:

    total_loss = instance_loss + prior_weight * class_loss

Prior images are generated from the base SD3.5 using a class prompt that
has NO trigger token but keeps the same broad event class: road rockfall.
These act as a light regularisation signal for realistic rockfall scenes
while leaving the <rfblk> trigger responsible for the specific learned
training distribution.

Three-phase execution
---------------------
Phase 1  Generate N prior images from base SD3.5 (class prompt, no LoRA).
Phase 2  Precompute VAE latents + text embeddings for both instance and prior
         images and save to cache dirs (fast; skipped if cache exists).
Phase 3  Train LoRA with combined instance + prior loss.

Usage
-----
python scripts/sd35_train_dreambooth_lora.py               # default paths
python scripts/sd35_train_dreambooth_lora.py --force-recompute --num-prior-images 15
"""

import argparse
import json
import math
import random
import shutil
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers")
warnings.filterwarnings("ignore", category=FutureWarning, module="peft")
warnings.filterwarnings("ignore", message=".*weights_only.*")
warnings.filterwarnings("ignore", message=".*has no attribute.*xpu.*")

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
)
from peft import LoraConfig, get_peft_model_state_dict
from PIL import Image
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from sd35_config import INITIALIZER_TOKEN, PLACEHOLDER_TOKEN, TOKEN_EMBEDDING_FILE
from sd35_train_lora import (   # noqa: E402
    _encoder_triples,
    collate_fn,
    get_sigmas,
    prepare_concept_token,
    save_concept_token,
)

# ---------------------------------------------------------------------------
# Class (prior) prompt — NO trigger token
# ---------------------------------------------------------------------------
CLASS_PROMPT = (
    "realistic photograph of a rockfall event on an asphalt mountain road, "
    "fallen boulders and small rock fragments scattered on the driving lane, "
    "visible lane markings, rocky hillside, outdoor daylight"
)

CLASS_NEGATIVE_PROMPT = (
    "empty road, clean road, no rocks on road, rocks only on roadside, "
    "rocks only on hillside, traffic cone, construction cone, people, vehicle, "
    "cartoon, anime, painting, 3d render, cgi, blurry, low quality, text, watermark"
)


# ---------------------------------------------------------------------------
# Phase 1: generate prior images from base SD3.5
# ---------------------------------------------------------------------------

def generate_prior_images(args, prior_img_dir: Path):
    if args.force_recompute and prior_img_dir.exists():
        shutil.rmtree(prior_img_dir)
        print(f"Deleted prior images ({prior_img_dir})")

    if prior_img_dir.exists() and len(list(prior_img_dir.glob("*.png"))) >= args.num_prior_images:
        print(f"Prior images already exist ({len(list(prior_img_dir.glob('*.png')))} images) — skipping generation.")
        return

    prior_img_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(prior_img_dir.glob("*.png")))
    needed = args.num_prior_images - existing
    if needed <= 0:
        return

    print(f"\nPhase 1: Generating {needed} prior images from base SD3.5 …")
    print(f"  Class prompt: {CLASS_PROMPT}")
    print(f"  Class negative prompt: {CLASS_NEGATIVE_PROMPT}")
    device = torch.device("cuda")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model_dir, torch_dtype=torch.float16
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    for i in range(existing, args.num_prior_images):
        generator = torch.Generator(device="cpu").manual_seed(args.seed + i + 10000)
        image = pipe(
            prompt=CLASS_PROMPT,
            negative_prompt=CLASS_NEGATIVE_PROMPT,
            num_inference_steps=28,
            guidance_scale=7.0,
            height=args.resolution,
            width=args.resolution,
            generator=generator,
        ).images[0]
        image.save(prior_img_dir / f"{i:03d}.png")
        if (i - existing + 1) % 5 == 0:
            print(f"  Generated {i - existing + 1}/{needed} prior images …")

    del pipe
    torch.cuda.empty_cache()
    print(f"  Prior images saved to {prior_img_dir}")


# ---------------------------------------------------------------------------
# Phase 2: precompute latents + embeddings for instance and prior
# ---------------------------------------------------------------------------

def precompute_instance(args, cache_dir: Path):
    """Identical to train_lora_v9.precompute — reuse its logic."""
    if args.force_recompute and cache_dir.exists():
        shutil.rmtree(cache_dir)
        print(f"Deleted instance cache ({cache_dir})")
    if cache_dir.exists() and any(cache_dir.glob("*_latent.pt")):
        print(f"Instance cache exists at {cache_dir} — skipping.")
        return

    cache_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    dtype = torch.float16
    records = [json.loads(l) for l in open(args.metadata, encoding="utf-8") if l.strip()]
    print(f"\nPhase 2a: Precomputing instance cache ({len(records)} images) …")

    # VAE
    vae = AutoencoderKL.from_pretrained(args.model_dir, subfolder="vae").to(device, dtype=dtype)
    vae.eval()
    for idx, rec in enumerate(records):
        img = Image.open(rec["file_name"]).convert("RGB")
        w, h = img.size
        m = min(w, h)
        img = img.crop(((w-m)//2, (h-m)//2, (w+m)//2, (h+m)//2)).resize(
            (args.resolution, args.resolution), Image.BICUBIC)
        px = torch.from_numpy(np.array(img)).float().permute(2,0,1) / 255.0
        px = (px * 2 - 1).unsqueeze(0).to(device, dtype=dtype)
        with torch.no_grad():
            latent = vae.encode(px).latent_dist.sample()
            latent = (latent - vae.config.shift_factor) * vae.config.scaling_factor
        torch.save(latent.cpu(), cache_dir / f"{idx:03d}_latent.pt")
    del vae; torch.cuda.empty_cache()

    # Text encoders
    pipe = StableDiffusion3Pipeline.from_pretrained(args.model_dir, torch_dtype=dtype)
    for te in [pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3]:
        te.to(device)
    prepare_concept_token(pipe, args.placeholder_token, args.initializer_token)
    for idx, rec in enumerate(records):
        with torch.no_grad():
            pe, _, ppe, _ = pipe.encode_prompt(
                prompt=rec["caption"], prompt_2=rec["caption"], prompt_3=rec["caption"],
                device=device, num_images_per_prompt=1, do_classifier_free_guidance=False,
            )
        torch.save({"prompt_embeds": pe.cpu(), "pooled_prompt_embeds": ppe.cpu()},
                   cache_dir / f"{idx:03d}_embeds.pt")

    save_concept_token(cache_dir.parent, pipe, args.placeholder_token)
    del pipe; torch.cuda.empty_cache()
    print(f"  Instance cache done → {cache_dir}")


def precompute_prior(args, prior_img_dir: Path, prior_cache_dir: Path):
    if args.force_recompute and prior_cache_dir.exists():
        shutil.rmtree(prior_cache_dir)
        print(f"Deleted prior cache ({prior_cache_dir})")
    existing_cache = sorted(prior_cache_dir.glob("*_latent.pt")) if prior_cache_dir.exists() else []
    if len(existing_cache) == args.num_prior_images:
        print(f"Prior cache exists at {prior_cache_dir} — skipping.")
        return
    if existing_cache:
        shutil.rmtree(prior_cache_dir)
        print(
            f"Deleted prior cache ({len(existing_cache)} items) because "
            f"num_prior_images={args.num_prior_images}"
        )

    prior_cache_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    dtype = torch.float16
    prior_files = sorted(prior_img_dir.glob("*.png"))[:args.num_prior_images]
    print(f"\nPhase 2b: Precomputing prior cache ({len(prior_files)} images) …")

    # VAE
    vae = AutoencoderKL.from_pretrained(args.model_dir, subfolder="vae").to(device, dtype=dtype)
    vae.eval()
    for idx, f in enumerate(prior_files):
        img = Image.open(f).convert("RGB")
        px = torch.from_numpy(np.array(img)).float().permute(2,0,1) / 255.0
        px = (px * 2 - 1).unsqueeze(0).to(device, dtype=dtype)
        with torch.no_grad():
            latent = vae.encode(px).latent_dist.sample()
            latent = (latent - vae.config.shift_factor) * vae.config.scaling_factor
        torch.save(latent.cpu(), prior_cache_dir / f"{idx:03d}_latent.pt")
    del vae; torch.cuda.empty_cache()

    # Text encoders — class prompt (NO trigger token)
    pipe = StableDiffusion3Pipeline.from_pretrained(args.model_dir, torch_dtype=dtype)
    for te in [pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3]:
        te.to(device)
    with torch.no_grad():
        pe, _, ppe, _ = pipe.encode_prompt(
            prompt=CLASS_PROMPT, prompt_2=CLASS_PROMPT, prompt_3=CLASS_PROMPT,
            device=device, num_images_per_prompt=1, do_classifier_free_guidance=False,
        )
    pe_cpu, ppe_cpu = pe.cpu(), ppe.cpu()
    for idx in range(len(prior_files)):
        torch.save({"prompt_embeds": pe_cpu, "pooled_prompt_embeds": ppe_cpu},
                   prior_cache_dir / f"{idx:03d}_embeds.pt")
    del pipe; torch.cuda.empty_cache()
    print(f"  Prior cache done → {prior_cache_dir}")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DreamBoothDataset(Dataset):
    """Returns one instance item and one prior item per index.
    The shorter side is cycled so all available instance/prior items are used.
    """
    def __init__(self, instance_cache_dir: Path, prior_cache_dir: Path):
        self.instance = sorted(instance_cache_dir.glob("*_latent.pt"))
        self.prior    = sorted(prior_cache_dir.glob("*_latent.pt"))
        if not self.instance:
            raise RuntimeError(f"No instance latents in {instance_cache_dir}")
        if not self.prior:
            raise RuntimeError(f"No prior latents in {prior_cache_dir}")

    def __len__(self):
        return max(len(self.instance), len(self.prior))

    def _load(self, latent_path: Path):
        stem = latent_path.stem.replace("_latent", "")
        cache_dir = latent_path.parent
        latent = torch.load(cache_dir / f"{stem}_latent.pt", map_location="cpu", weights_only=True)
        embeds = torch.load(cache_dir / f"{stem}_embeds.pt", map_location="cpu", weights_only=True)
        return {
            "latent": latent.squeeze(0),
            "prompt_embeds": embeds["prompt_embeds"].squeeze(0),
            "pooled_prompt_embeds": embeds["pooled_prompt_embeds"].squeeze(0),
        }

    def __getitem__(self, idx):
        inst  = self._load(self.instance[idx % len(self.instance)])
        prior = self._load(self.prior[idx % len(self.prior)])
        return inst, prior


def dreambooth_collate(batch):
    inst_list, prior_list = zip(*batch)
    def stack(items):
        return {k: torch.stack([b[k] for b in items]) for k in items[0]}
    return stack(inst_list), stack(prior_list)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir",    type=Path, default=Path("models/stable-diffusion-3.5-medium"))
    p.add_argument("--metadata",     type=Path, default=Path("data/train/metadata_v10.jsonl"))
    p.add_argument("--output-dir",   type=Path, default=Path("outputs/lora/sksrockfall-v10-db"))
    p.add_argument("--resolution",   type=int,  default=512)
    p.add_argument("--rank",         type=int,  default=8)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--learning-rate", type=float, default=3e-5)
    p.add_argument("--max-train-steps", type=int, default=600)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--train-batch-size", type=int, default=1)
    p.add_argument("--save-every",   type=int,  default=50)
    p.add_argument("--seed",         type=int,  default=42)
    p.add_argument("--num-prior-images", type=int, default=15)
    p.add_argument("--prior-weight", type=float, default=0.1)
    p.add_argument("--placeholder-token", default=PLACEHOLDER_TOKEN)
    p.add_argument("--initializer-token", default=INITIALIZER_TOKEN)
    p.add_argument("--force-recompute", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float16
    args.output_dir.mkdir(parents=True, exist_ok=True)

    inst_cache_dir  = args.output_dir / ".cache_instance"
    prior_img_dir   = args.output_dir / ".prior_images"
    prior_cache_dir = args.output_dir / ".cache_prior"

    # Phase 1: generate prior images
    generate_prior_images(args, prior_img_dir)

    # Phase 2: precompute both caches
    precompute_instance(args, inst_cache_dir)
    precompute_prior(args, prior_img_dir, prior_cache_dir)

    # Phase 3: train
    print(f"\nPhase 3: DreamBooth-LoRA training …")
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.model_dir, subfolder="scheduler")
    noise_scheduler.set_timesteps(1000, device=device)

    transformer = SD3Transformer2DModel.from_pretrained(
        args.model_dir, subfolder="transformer")
    transformer.requires_grad_(False)
    transformer.to(device, dtype=weight_dtype)

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        lora_dropout=args.lora_dropout,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    transformer.add_adapter(lora_config)

    lora_params = []
    for name, param in transformer.named_parameters():
        if "lora_" in name:
            param.data = param.data.float()
            param.requires_grad_(True)
            lora_params.append(param)
    print(f"  Trainable LoRA parameters: {sum(p.numel() for p in lora_params):,}")

    dataset = DreamBoothDataset(inst_cache_dir, prior_cache_dir)
    dataloader = DataLoader(
        dataset, batch_size=args.train_batch_size,
        shuffle=True, num_workers=0, collate_fn=dreambooth_collate,
    )

    optimizer = torch.optim.AdamW(
        lora_params, lr=args.learning_rate,
        betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-8,
    )

    steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    num_epochs = math.ceil(args.max_train_steps / max(steps_per_epoch, 1))

    progress_bar = tqdm(total=args.max_train_steps, desc="DreamBooth SD3.5 LoRA")
    global_step = 0
    accum_step = 0

    def compute_loss(latents, prompt_embeds, pooled_prompt_embeds):
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        u = torch.rand((bsz,), device=device)
        indices = (u * noise_scheduler.config.num_train_timesteps).long()
        timesteps = noise_scheduler.timesteps[indices].to(device)
        sigmas = get_sigmas(timesteps, noise_scheduler, device,
                            n_dim=latents.ndim, dtype=weight_dtype)
        noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
        model_pred = transformer(
            hidden_states=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            return_dict=False,
        )[0]
        target = noise - latents  # velocity prediction
        return F.mse_loss(model_pred.float(), target.float(), reduction="mean")

    for _ in range(num_epochs):
        for inst_batch, prior_batch in dataloader:
            inst_latents = inst_batch["latent"].to(device, dtype=weight_dtype)
            inst_pe      = inst_batch["prompt_embeds"].to(device, dtype=weight_dtype)
            inst_ppe     = inst_batch["pooled_prompt_embeds"].to(device, dtype=weight_dtype)

            prior_latents = prior_batch["latent"].to(device, dtype=weight_dtype)
            prior_pe      = prior_batch["prompt_embeds"].to(device, dtype=weight_dtype)
            prior_ppe     = prior_batch["pooled_prompt_embeds"].to(device, dtype=weight_dtype)

            inst_loss  = compute_loss(inst_latents,  inst_pe,  inst_ppe)
            prior_loss = compute_loss(prior_latents, prior_pe, prior_ppe)
            loss = inst_loss + args.prior_weight * prior_loss

            (loss / args.gradient_accumulation_steps).backward()

            accum_step += 1
            if accum_step % args.gradient_accumulation_steps != 0:
                continue

            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            progress_bar.update(1)
            progress_bar.set_postfix(
                inst=f"{inst_loss.item():.4f}",
                prior=f"{prior_loss.item():.4f}",
            )

            if args.save_every > 0 and global_step % args.save_every == 0:
                ckpt = args.output_dir / f"checkpoint-{global_step}"
                ckpt.mkdir(parents=True, exist_ok=True)
                save_file(get_peft_model_state_dict(transformer),
                          ckpt / "pytorch_lora_weights.safetensors")

            if global_step >= args.max_train_steps:
                break

    progress_bar.close()
    save_file(get_peft_model_state_dict(transformer),
              args.output_dir / "pytorch_lora_weights.safetensors")
    print(f"\nLoRA weights saved to {args.output_dir}")
    print(f"Concept token embeddings saved to {args.output_dir / TOKEN_EMBEDDING_FILE}")


if __name__ == "__main__":
    main()
