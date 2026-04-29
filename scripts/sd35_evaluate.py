"""CLIP Score + KID evaluation for Setting A and B.

Required packages (install if missing):
    pip install torchmetrics matplotlib

Usage:
    # Full evaluation (both settings, CLIP + KID)
    python scripts/sd35_evaluate.py

    # Smoke test (quick check with fewer images)
    python scripts/sd35_evaluate.py --max-per-prompt 5 --skip-kid

    # Only Setting A
    python scripts/sd35_evaluate.py --skip-b

Outputs (in --out-dir, default outputs/eval/):
    eval_results.json       — all scores in machine-readable form
    clip_per_prompt.png     — grouped bar chart: CLIP score per prompt, A vs B
    summary.png             — overall CLIP + KID side-by-side
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from sd35_config import PROMPTS

# Strip <rfblk> — CLIP has never seen this custom token
EVAL_PROMPTS = [
    p[len("<rfblk>, "):] if p.startswith("<rfblk>, ") else p
    for p in PROMPTS
]
PROMPT_LABELS = [f"P{i+1}" for i in range(len(PROMPTS))]

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Evaluate generated images: CLIP Score + KID")
parser.add_argument("--dir-a",           default="outputs/runs/sd35_v10db_settingA",
                    help="Setting A output root (prompt_1/ … prompt_10/ inside)")
parser.add_argument("--dir-b",           default="outputs/runs/sd35_v10db_settingB",
                    help="Setting B output root")
parser.add_argument("--real-dir",        default="data/train",
                    help="Real training images directory (KID reference set)")
parser.add_argument("--out-dir",         default="outputs/eval")
parser.add_argument("--clip-model",      default="openai/clip-vit-base-patch32",
                    help="CLIP model ID (must match across all experiments)")
parser.add_argument("--batch-size",      type=int, default=64,
                    help="Images per CLIP forward pass")
parser.add_argument("--max-per-prompt",  type=int, default=0,
                    help="Cap images per prompt (0 = use all; useful for smoke tests)")
parser.add_argument("--kid-subset",      type=int, default=10,
                    help="KID subset_size; auto-capped to min(#real, #gen//2)")
parser.add_argument("--kid-num-subsets", type=int, default=100)
parser.add_argument("--skip-kid",        action="store_true", help="Skip KID (faster)")
parser.add_argument("--skip-b",          action="store_true", help="Skip Setting B")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device : {device}")
print(f"Out dir: {args.out_dir}")

# ── CLIP Score ────────────────────────────────────────────────────────────────

print(f"\nLoading CLIP model: {args.clip_model} …")
from transformers import CLIPModel, CLIPProcessor

clip_model     = CLIPModel.from_pretrained(args.clip_model).to(device).eval()
clip_processor = CLIPProcessor.from_pretrained(args.clip_model)


@torch.no_grad()
def _clip_scores_batch(images_pil: list, text: str) -> np.ndarray:
    """Cosine similarity × 100 for PIL images against a single text string."""
    inputs = clip_processor(
        text=[text] * len(images_pil),
        images=images_pil,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77,
    ).to(device)
    out    = clip_model(**inputs)
    i_feat = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
    t_feat = out.text_embeds  / out.text_embeds.norm(dim=-1,  keepdim=True)
    return (i_feat * t_feat).sum(-1).cpu().numpy() * 100


def compute_clip(root_dir: str, label: str) -> dict:
    print(f"\n{'─'*56}")
    print(f"CLIP Score  {label}")
    print(f"{'─'*56}")

    per_prompt    = []
    all_scores    = []

    for i, prompt_text in enumerate(EVAL_PROMPTS):
        prompt_dir = os.path.join(root_dir, f"prompt_{i+1}")
        if not os.path.isdir(prompt_dir):
            print(f"  P{i+1:02d}: MISSING — skipped")
            per_prompt.append({"prompt_idx": i + 1, "mean": None, "std": None, "n": 0})
            continue

        paths = sorted(Path(prompt_dir).glob("*.png"))
        if args.max_per_prompt:
            paths = paths[: args.max_per_prompt]
        images = [Image.open(p).convert("RGB") for p in paths]

        if not images:
            per_prompt.append({"prompt_idx": i + 1, "mean": None, "std": None, "n": 0})
            continue

        scores = []
        for start in range(0, len(images), args.batch_size):
            batch = images[start : start + args.batch_size]
            scores.extend(_clip_scores_batch(batch, prompt_text).tolist())

        m  = float(np.mean(scores))
        sd = float(np.std(scores))
        per_prompt.append({"prompt_idx": i + 1, "mean": m, "std": sd, "n": len(scores)})
        all_scores.extend(scores)
        print(f"  P{i+1:02d}: {m:6.3f} ± {sd:5.3f}  (n={len(scores)})")

    if all_scores:
        total_mean = float(np.mean(all_scores))
        total_std  = float(np.std(all_scores))
    else:
        total_mean = total_std = float("nan")

    print(f"  {'─'*40}")
    print(f"  Overall : {total_mean:6.3f} ± {total_std:5.3f}  (N={len(all_scores)})")

    return {
        "label":        label,
        "per_prompt":   per_prompt,
        "overall_mean": total_mean,
        "overall_std":  total_std,
        "n_total":      len(all_scores),
    }


# ── KID ───────────────────────────────────────────────────────────────────────

def _pil_to_uint8_tensor(img: Image.Image, size: int = 299) -> torch.Tensor:
    """Resize to (size, size) and return uint8 CHW tensor."""
    img = img.resize((size, size), Image.BICUBIC)
    arr = np.array(img, dtype=np.uint8)        # H W C
    return torch.from_numpy(arr).permute(2, 0, 1)  # C H W


def compute_kid(gen_dir: str, real_dir: str, label: str) -> dict | None:
    try:
        from torchmetrics.image.kid import KernelInceptionDistance
    except ImportError:
        print("  [WARN] torchmetrics not installed — skipping KID")
        print("         pip install torchmetrics")
        return None

    print(f"\n{'─'*56}")
    print(f"KID  {label}")
    print(f"{'─'*56}")

    real_paths = sorted(Path(real_dir).glob("*.png")) + \
                 sorted(Path(real_dir).glob("*.jpg"))
    gen_paths  = []
    for i in range(len(PROMPTS)):
        d = os.path.join(gen_dir, f"prompt_{i+1}")
        if os.path.isdir(d):
            pp = sorted(Path(d).glob("*.png"))
            if args.max_per_prompt:
                pp = pp[: args.max_per_prompt]
            gen_paths.extend(pp)

    n_real, n_gen = len(real_paths), len(gen_paths)
    subset = min(args.kid_subset, n_real, max(1, n_gen // 2))
    print(f"  Real images     : {n_real}")
    print(f"  Generated images: {n_gen}")
    print(f"  subset_size     : {subset}  (num_subsets={args.kid_num_subsets})")

    if n_real < 2 or n_gen < 2 or subset < 2:
        print("  Insufficient images — skipping KID")
        return None

    if n_real < 30:
        print(f"  [NOTE] Only {n_real} real images → KID variance will be high.")

    kid_metric = KernelInceptionDistance(
        subset_size=subset,
        normalize=False,
    ).to(device)

    # Feed real images
    for p in real_paths:
        t = _pil_to_uint8_tensor(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
        kid_metric.update(t, real=True)

    # Feed generated images in batches
    bs = args.batch_size
    for start in range(0, n_gen, bs):
        batch_paths = gen_paths[start : start + bs]
        tensors = [_pil_to_uint8_tensor(Image.open(p).convert("RGB")) for p in batch_paths]
        t = torch.stack(tensors).to(device)
        kid_metric.update(t, real=False)
        if (start // bs) % 10 == 0:
            print(f"  [{min(start + bs, n_gen):>5}/{n_gen}] generated images fed …")

    kid_mean, kid_std = kid_metric.compute()
    km, ks = float(kid_mean), float(kid_std)
    print(f"  KID : {km:.6f} ± {ks:.6f}")
    return {
        "label":  label,
        "mean":   km,
        "std":    ks,
        "n_real": n_real,
        "n_gen":  n_gen,
        "subset_size":   subset,
        "num_subsets":   args.kid_num_subsets,
    }


# ── Run evaluations ───────────────────────────────────────────────────────────

clip_a = compute_clip(args.dir_a, "Setting A (35 steps, guidance=5.5)")
clip_b = None if args.skip_b else compute_clip(args.dir_b, "Setting B (8 steps, guidance=3.5)")

kid_a = kid_b = None
if not args.skip_kid:
    kid_a = compute_kid(args.dir_a, args.real_dir, "Setting A (35 steps, guidance=5.5)")
    if not args.skip_b:
        kid_b = compute_kid(args.dir_b, args.real_dir, "Setting B (8 steps, guidance=3.5)")

# ── Save JSON ─────────────────────────────────────────────────────────────────

results = {
    "clip_model": args.clip_model,
    "clip": {"setting_a": clip_a, "setting_b": clip_b},
    "kid":  {"setting_a": kid_a,  "setting_b": kid_b},
}
json_path = os.path.join(args.out_dir, "eval_results.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nResults saved → {json_path}")

# ── Plotting ──────────────────────────────────────────────────────────────────

COLORS = {"A": "#4C72B0", "B": "#DD8452"}   # blue / orange


# ── Chart 1: CLIP Score per prompt ──────────────────────────────────────────

fig, ax = plt.subplots(figsize=(14, 5))
x = np.arange(len(PROMPTS))
w = 0.35 if clip_b else 0.55

def _safe(rec, key):
    return rec[key] if rec[key] is not None else 0.0

means_a = [_safe(r, "mean") for r in clip_a["per_prompt"]]
stds_a  = [_safe(r, "std")  for r in clip_a["per_prompt"]]

offset_a = -w / 2 if clip_b else 0
bars_a = ax.bar(
    x + offset_a, means_a, w, yerr=stds_a, capsize=3,
    color=COLORS["A"], label="Setting A (35 steps)", alpha=0.85, error_kw={"elinewidth": 1},
)

if clip_b:
    means_b = [_safe(r, "mean") for r in clip_b["per_prompt"]]
    stds_b  = [_safe(r, "std")  for r in clip_b["per_prompt"]]
    ax.bar(
        x + w / 2, means_b, w, yerr=stds_b, capsize=3,
        color=COLORS["B"], label="Setting B (8 steps)", alpha=0.85, error_kw={"elinewidth": 1},
    )

ax.set_xticks(x)
ax.set_xticklabels(PROMPT_LABELS, fontsize=9)
ax.set_xlabel("Prompt", fontsize=11)
ax.set_ylabel("CLIP Score (cosine similarity × 100)", fontsize=11)
ax.set_title("CLIP Score per Prompt — Setting A vs Setting B", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.4)

ymin = max(0, min(means_a) - 3)
ax.set_ylim(ymin, max(means_a + (means_b if clip_b else [])) + 4)

fig.tight_layout()
p1 = os.path.join(args.out_dir, "clip_per_prompt.png")
fig.savefig(p1, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Chart saved → {p1}")


# ── Chart 2: Summary (overall CLIP + KID) ────────────────────────────────────

has_kid = bool(kid_a or kid_b)
fig, axes = plt.subplots(1, 1 + has_kid, figsize=(6 * (1 + has_kid), 5))
if not has_kid:
    axes = [axes]

# Panel A: overall CLIP
ax = axes[0]
rows = []
if clip_a: rows.append(("A", "Setting A\n(35 steps)", clip_a["overall_mean"], clip_a["overall_std"]))
if clip_b: rows.append(("B", "Setting B\n(8 steps)",  clip_b["overall_mean"], clip_b["overall_std"]))

bar_labels = [r[1] for r in rows]
bar_means  = [r[2] for r in rows]
bar_stds   = [r[3] for r in rows]
bar_colors = [COLORS[r[0]] for r in rows]

bars = ax.bar(bar_labels, bar_means, yerr=bar_stds, capsize=7,
              color=bar_colors, alpha=0.85, width=0.45, error_kw={"elinewidth": 1.5})
for bar, m in zip(bars, bar_means):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + bar.get_height() * 0.01 + 0.2,
        f"{m:.3f}",
        ha="center", va="bottom", fontsize=11, fontweight="bold",
    )
ax.set_ylabel("CLIP Score (cosine × 100)", fontsize=11)
ax.set_title("Overall CLIP Score", fontsize=12, fontweight="bold")
ax.grid(axis="y", linestyle="--", alpha=0.4)
ymin = max(0, min(bar_means) - 5)
ax.set_ylim(ymin, max(bar_means) + 6)

# Panel B: KID
if has_kid:
    ax = axes[1]
    kid_rows = []
    if kid_a: kid_rows.append(("A", "Setting A\n(35 steps)", kid_a["mean"], kid_a["std"]))
    if kid_b: kid_rows.append(("B", "Setting B\n(8 steps)",  kid_b["mean"], kid_b["std"]))

    k_labels = [r[1] for r in kid_rows]
    k_means  = [r[2] for r in kid_rows]
    k_stds   = [r[3] for r in kid_rows]
    k_colors = [COLORS[r[0]] for r in kid_rows]

    bars_k = ax.bar(k_labels, k_means, yerr=k_stds, capsize=7,
                    color=k_colors, alpha=0.85, width=0.45, error_kw={"elinewidth": 1.5})
    for bar, m in zip(bars_k, k_means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(k_means) * 0.02,
            f"{m:.5f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    n_real = kid_a["n_real"] if kid_a else (kid_b["n_real"] if kid_b else "?")
    subset = kid_a["subset_size"] if kid_a else (kid_b["subset_size"] if kid_b else "?")
    ax.set_ylabel("KID (↓ better)", fontsize=11)
    ax.set_title(
        f"KID vs Real Images\n(n_real={n_real}, subset_size={subset})",
        fontsize=12, fontweight="bold",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_ylim(0, max(k_means) * 1.35)

fig.suptitle("Evaluation Summary — Setting A vs Setting B", fontsize=13, fontweight="bold")
fig.tight_layout()
p2 = os.path.join(args.out_dir, "summary.png")
fig.savefig(p2, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Chart saved → {p2}")

# ── Print final summary ───────────────────────────────────────────────────────

print("\n" + "=" * 56)
print("  FINAL EVALUATION SUMMARY")
print("=" * 56)
for tag, clip_res, kid_res in [("Setting A", clip_a, kid_a), ("Setting B", clip_b, kid_b)]:
    if clip_res is None:
        continue
    kid_str = f"{kid_res['mean']:.6f} ± {kid_res['std']:.6f}" if kid_res else "—"
    print(f"  {tag}")
    print(f"    CLIP Score : {clip_res['overall_mean']:.3f} ± {clip_res['overall_std']:.3f}  "
          f"(N={clip_res['n_total']})")
    print(f"    KID        : {kid_str}")
print("=" * 56)
print("\nDone.")
