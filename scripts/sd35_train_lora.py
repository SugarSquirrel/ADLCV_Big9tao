"""SD3.5 Medium LoRA training for rockfall concept — v10.

Key design decisions
--------------------
Trigger token  <rfblk> is added to all three tokenizers (CLIP-L, CLIP-G,
               T5-XXL) and initialised from the mean embedding of "rockfall".
               The token embedding is FROZEN during training — only the
               transformer LoRA is updated.  The initialised embeddings are
               saved alongside the LoRA weights so inference is reproducible.

Precompute     Image latents (VAE) and text embeddings (3 encoders) are
               computed once and cached on disk.  This lets us unload the
               heavy text encoders (~11 GB) before the training loop so
               the 4090 24 GB has plenty of headroom for the transformer.

Loss           Standard SD3 flow-matching velocity prediction loss:
               target = noise - latent  (NOT x0-prediction).
"""

import argparse
import json
import math
import random
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


# ---------------------------------------------------------------------------
# Trigger token helpers
# ---------------------------------------------------------------------------

def _encoder_triples(pipe):
    """Yield (name, tokenizer, text_encoder) for all three SD3 encoders."""
    return [
        ("clip_l", pipe.tokenizer,   pipe.text_encoder),
        ("clip_g", pipe.tokenizer_2, pipe.text_encoder_2),
        ("t5",     pipe.tokenizer_3, pipe.text_encoder_3),
    ]


def prepare_concept_token(pipe, placeholder_token, initializer_token):
    """Add trigger token to all three tokenizers / text encoders.

    The new token is initialised from the mean embedding of `initializer_token`
    in each encoder so the model immediately has a sensible starting point.
    Token embeddings are NOT trained — they are frozen during LoRA training.
    """
    results = {}
    for name, tokenizer, text_encoder in _encoder_triples(pipe):
        n_added = tokenizer.add_tokens([placeholder_token])
        text_encoder.resize_token_embeddings(len(tokenizer))

        placeholder_id = tokenizer.convert_tokens_to_ids(placeholder_token)
        init_ids = tokenizer(
            initializer_token, add_special_tokens=False
        ).input_ids
        if not init_ids:
            raise ValueError(f"[{name}] initializer_token '{initializer_token}' has no token ids")

        embeddings = text_encoder.get_input_embeddings().weight
        with torch.no_grad():
            init_embed = embeddings[init_ids].mean(dim=0)
            embeddings[placeholder_id].copy_(init_embed.to(embeddings.dtype))

        results[name] = {"placeholder_id": placeholder_id, "n_added": n_added}
        print(f"  [{name}] added '{placeholder_token}' id={placeholder_id} "
              f"(init from ids={init_ids}, n_added={n_added})")

    return results


def save_concept_token(output_dir, pipe, placeholder_token):
    """Save the token embedding vectors from all three encoders."""
    payload = {"placeholder_token": placeholder_token, "embeddings": {}}
    for name, tokenizer, text_encoder in _encoder_triples(pipe):
        pid = tokenizer.convert_tokens_to_ids(placeholder_token)
        embed = text_encoder.get_input_embeddings().weight[pid].detach().cpu().float()
        payload["embeddings"][name] = embed
    torch.save(payload, Path(output_dir) / TOKEN_EMBEDDING_FILE)


def load_concept_token(lora_dir, pipe, placeholder_token):
    """Load saved token embeddings and inject into all three encoders."""
    path = Path(lora_dir) / TOKEN_EMBEDDING_FILE
    if not path.exists():
        return False
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["placeholder_token"] == placeholder_token, \
        f"Token mismatch: stored={payload['placeholder_token']}, expected={placeholder_token}"

    for name, tokenizer, text_encoder in _encoder_triples(pipe):
        # Add token if not already registered in this tokenizer instance
        if tokenizer.convert_tokens_to_ids(placeholder_token) == getattr(tokenizer, "unk_token_id", -1):
            tokenizer.add_tokens([placeholder_token])
            text_encoder.resize_token_embeddings(len(tokenizer))

        pid = tokenizer.convert_tokens_to_ids(placeholder_token)
        weight = text_encoder.get_input_embeddings().weight
        with torch.no_grad():
            weight[pid].copy_(payload["embeddings"][name].to(weight.device, weight.dtype))
    return True


# ---------------------------------------------------------------------------
# Sigma helper for flow matching
# ---------------------------------------------------------------------------

def get_sigmas(timesteps, scheduler, device, n_dim=4, dtype=torch.float32):
    sigmas = scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = scheduler.timesteps.to(device)
    step_indices = [schedule_timesteps.tolist().index(t) for t in timesteps.tolist()]
    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


# ---------------------------------------------------------------------------
# Dataset (reads precomputed cache)
# ---------------------------------------------------------------------------

class CachedLatentDataset(Dataset):
    def __init__(self, cache_dir: Path, augment_flip: bool = True):
        self.cache_dir = cache_dir
        self.augment_flip = augment_flip
        self.items = sorted(cache_dir.glob("*_latent.pt"))
        if not self.items:
            raise RuntimeError(f"No cached latents in {cache_dir}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        stem = self.items[idx].stem.replace("_latent", "")
        latent = torch.load(self.cache_dir / f"{stem}_latent.pt", map_location="cpu", weights_only=True)
        embeds = torch.load(self.cache_dir / f"{stem}_embeds.pt", map_location="cpu", weights_only=True)
        if self.augment_flip and random.random() < 0.5:
            latent = latent.flip(-1)
        return {
            "latent": latent.squeeze(0),
            "prompt_embeds": embeds["prompt_embeds"].squeeze(0),
            "pooled_prompt_embeds": embeds["pooled_prompt_embeds"].squeeze(0),
        }


def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


# ---------------------------------------------------------------------------
# Precompute latents + embeddings
# ---------------------------------------------------------------------------

def precompute(args, cache_dir: Path):
    if args.force_recompute and cache_dir.exists():
        import shutil
        shutil.rmtree(cache_dir)
        print(f"Deleted old cache at {cache_dir} (--force-recompute)")
    if cache_dir.exists() and any(cache_dir.glob("*_latent.pt")):
        print(f"Cache already exists at {cache_dir} — skipping.")
        return

    cache_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    dtype = torch.float16

    records = [json.loads(l) for l in open(args.metadata, encoding="utf-8") if l.strip()]
    print(f"Precomputing for {len(records)} images …")

    # ---- VAE: encode images ----
    print("  Loading VAE …")
    vae = AutoencoderKL.from_pretrained(args.model_dir, subfolder="vae").to(device, dtype=dtype)
    vae.eval()
    for idx, rec in enumerate(records):
        img = Image.open(rec["file_name"]).convert("RGB")
        w, h = img.size
        m = min(w, h)
        img = img.crop(((w-m)//2, (h-m)//2, (w+m)//2, (h+m)//2)).resize(
            (args.resolution, args.resolution), Image.BICUBIC)
        px = torch.from_numpy(np.array(img)).float().permute(2,0,1) / 255.0
        px = (px * 2.0 - 1.0).unsqueeze(0).to(device, dtype=dtype)
        with torch.no_grad():
            latent = vae.encode(px).latent_dist.sample()
            latent = (latent - vae.config.shift_factor) * vae.config.scaling_factor
        torch.save(latent.cpu(), cache_dir / f"{idx:03d}_latent.pt")
    del vae
    torch.cuda.empty_cache()

    # ---- Text encoders: encode captions ----
    print("  Loading pipeline for text encoding (T5 + CLIP-L + CLIP-G) …")
    pipe = StableDiffusion3Pipeline.from_pretrained(args.model_dir, torch_dtype=dtype)
    pipe.text_encoder  = pipe.text_encoder.to(device)
    pipe.text_encoder_2 = pipe.text_encoder_2.to(device)
    pipe.text_encoder_3 = pipe.text_encoder_3.to(device)

    # Add trigger token BEFORE encoding so embeddings include it
    print(f"  Adding trigger token '{args.placeholder_token}' to all encoders …")
    prepare_concept_token(pipe, args.placeholder_token, args.initializer_token)

    for idx, rec in enumerate(records):
        caption = rec["caption"]

        if args.placeholder_token not in caption:
            caption = f"{args.placeholder_token}, {caption}"

        with torch.no_grad():
            pe, _, ppe, _ = pipe.encode_prompt(
                prompt=caption, prompt_2=caption, prompt_3=caption,
                device=device, num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
        torch.save({"prompt_embeds": pe.cpu(), "pooled_prompt_embeds": ppe.cpu()},
                   cache_dir / f"{idx:03d}_embeds.pt")
        print(f"  [{idx+1}/{len(records)}] {caption[:70]}…")

    # Save token embeddings for reproducible inference
    save_concept_token(cache_dir.parent, pipe, args.placeholder_token)
    del pipe
    torch.cuda.empty_cache()
    print(f"Precomputation done. Cache: {cache_dir}")


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir",   type=Path, default=Path("models/stable-diffusion-3.5-medium"))
    p.add_argument("--metadata",    type=Path, default=Path("data/train/metadata_v10.jsonl"))
    p.add_argument("--output-dir",  type=Path, default=Path("outputs/lora/sksrockfall-v10"))
    p.add_argument("--cache-dir",   type=Path, default=Path("outputs/lora/sksrockfall-v10/.cache"))
    p.add_argument("--resolution",  type=int,   default=512)
    p.add_argument("--train-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--max-train-steps", type=int, default=1200)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--rank",        type=int,   default=4) # ori: 16
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--save-every",  type=int,   default=300)
    p.add_argument("--num-workers", type=int,   default=2)
    p.add_argument("--augment-flip", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--placeholder-token", default=PLACEHOLDER_TOKEN)
    p.add_argument("--initializer-token", default=INITIALIZER_TOKEN)
    p.add_argument("--force-recompute", action="store_true",
                   help="Delete existing cache and recompute embeddings (needed after caption/token changes)")
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

    precompute(args, args.cache_dir)

    print("Loading scheduler and SD3 transformer …")
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
    print(f"Trainable LoRA parameters: {sum(p.numel() for p in lora_params):,}")

    dataset = CachedLatentDataset(args.cache_dir, augment_flip=args.augment_flip)
    dataloader = DataLoader(dataset, batch_size=args.train_batch_size,
                            shuffle=True, num_workers=args.num_workers,
                            collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(
        lora_params, lr=args.learning_rate,
        betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-8)

    steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    num_epochs = math.ceil(args.max_train_steps / steps_per_epoch)

    progress_bar = tqdm(total=args.max_train_steps, desc="Training SD3.5 LoRA")
    global_step = 0
    accum_step = 0

    for _ in range(num_epochs):
        for batch in dataloader:
            latents = batch["latent"].to(device, dtype=weight_dtype)
            prompt_embeds = batch["prompt_embeds"].to(device, dtype=weight_dtype)
            pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(device, dtype=weight_dtype)

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

            # Standard SD3 flow-matching loss: predict velocity (noise - x0)
            target = noise - latents
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            (loss / args.gradient_accumulation_steps).backward()

            accum_step += 1
            if accum_step % args.gradient_accumulation_steps != 0:
                continue

            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            progress_bar.update(1)
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

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
    print(f"LoRA weights saved to {args.output_dir}")
    print(f"Concept token embeddings saved to {args.output_dir / TOKEN_EMBEDDING_FILE}")


if __name__ == "__main__":
    main()
