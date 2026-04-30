"""CLIP Score + KID evaluation for Base Model / Setting A / Setting B.

Required packages:
    pip install torchmetrics torch-fidelity matplotlib

Usage:
    # Full evaluation (all 3 conditions)
    python scripts/sd35_evaluate.py --dir-base outputs/results/v10db_final_infer/base

    # LoRA only (Setting A + B)
    python scripts/sd35_evaluate.py

    # Smoke test
    python scripts/sd35_evaluate.py --max-per-prompt 5 --skip-kid

    # Re-run only KID (skip CLIP recompute, load saved JSON)
    python scripts/sd35_evaluate.py --skip-clip --dir-base outputs/results/v10db_final_infer/base

    # Re-plot only, using saved outputs/eval/eval_results.json
    python scripts/sd35_evaluate.py --plot-only

Outputs (in --out-dir, default outputs/eval/):
    eval_results.json       — all scores in machine-readable form
    clip_per_prompt.png     — CLIP score per prompt, Base vs A vs B
    summary.png             — overall CLIP + KID comparison
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
import matplotlib.font_manager as fm

# ── Chinese font setup ────────────────────────────────────────────────────────

def _setup_chinese_font():
    preferred = [
        "Noto Sans CJK JP", "Noto Sans CJK SC", "Noto Sans CJK TC",
        "WenQuanYi Micro Hei", "SimHei", "Microsoft YaHei",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in preferred:
        if font in available:
            matplotlib.rcParams["font.sans-serif"] = [font, "DejaVu Sans"]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return font
    return None

_zh_font = _setup_chinese_font()

sys.path.insert(0, os.path.dirname(__file__))
from sd35_config import PROMPTS

# Strip <rfblk> — CLIP has never seen this custom token
EVAL_PROMPTS = [
    p[len("<rfblk>, "):] if p.startswith("<rfblk>, ") else p
    for p in PROMPTS
]
PROMPT_LABELS = [f"P{i+1}" for i in range(len(PROMPTS))]

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Evaluate generated images: CLIP + KID")
parser.add_argument("--dir-a",            default="outputs/results/v10db_final_infer/settingA")
parser.add_argument("--dir-b",            default="outputs/results/v10db_final_infer/settingB")
parser.add_argument("--dir-base",         default=None,
                    help="Base model (no LoRA) output dir; omit to skip base evaluation")
parser.add_argument("--real-dir",         default="data/train",
                    help="Real training images (KID reference set)")
parser.add_argument("--out-dir",          default="outputs/eval")
parser.add_argument("--clip-model",       default="openai/clip-vit-base-patch32")
parser.add_argument("--batch-size",       type=int, default=64)
parser.add_argument("--max-per-prompt",   type=int, default=0,
                    help="Cap images per prompt (0 = all; use 5 for smoke test)")
parser.add_argument("--kid-subset",       type=int, default=10,
                    help="KID subset_size; auto-capped to min(#real, #gen//2)")
parser.add_argument("--kid-num-subsets",  type=int, default=100)
parser.add_argument("--skip-clip",        action="store_true",
                    help="Load previous CLIP results from JSON instead of recomputing")
parser.add_argument("--skip-kid",         action="store_true")
parser.add_argument("--skip-b",           action="store_true")
parser.add_argument("--plot-only",        action="store_true",
                    help="Only reload eval_results.json and regenerate plots; no CLIP/KID recompute")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device : {device}")
print(f"Out dir: {args.out_dir}")
if _zh_font:
    print(f"中文字型 : {_zh_font}")
else:
    print("[WARN] 未找到中文字型，圖表文字可能顯示為方塊")

# ── CLIP Score ────────────────────────────────────────────────────────────────

clip_model = clip_processor = None
if not args.plot_only and not args.skip_clip:
    print(f"\nLoading CLIP model: {args.clip_model} …")
    from transformers import CLIPModel, CLIPProcessor

    clip_model     = CLIPModel.from_pretrained(args.clip_model).to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained(args.clip_model)


@torch.no_grad()
def _clip_scores_batch(images_pil: list, text: str) -> np.ndarray:
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

    per_prompt = []
    all_scores = []

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

    total_mean = float(np.mean(all_scores)) if all_scores else float("nan")
    total_std  = float(np.std(all_scores))  if all_scores else float("nan")
    print(f"  {'─'*40}")
    print(f"  Overall : {total_mean:6.3f} ± {total_std:5.3f}  (N={len(all_scores)})")

    return {
        "label":        label,
        "per_prompt":   per_prompt,
        "overall_mean": total_mean,
        "overall_std":  total_std,
        "n_total":      len(all_scores),
    }


# ── Shared image loader ───────────────────────────────────────────────────────

def _pil_to_uint8_tensor(img: Image.Image, size: int = 299) -> torch.Tensor:
    """Resize to (size, size), return uint8 CHW tensor [0, 255]."""
    img = img.resize((size, size), Image.BICUBIC)
    arr = np.array(img, dtype=np.uint8)
    return torch.from_numpy(arr).permute(2, 0, 1)


def _collect_gen_paths(gen_dir: str) -> list:
    paths = []
    for i in range(len(PROMPTS)):
        d = os.path.join(gen_dir, f"prompt_{i+1}")
        if os.path.isdir(d):
            pp = sorted(Path(d).glob("*.png"))
            if args.max_per_prompt:
                pp = pp[: args.max_per_prompt]
            paths.extend(pp)
    return paths


def _collect_real_paths(real_dir: str) -> list:
    return (sorted(Path(real_dir).glob("*.png")) +
            sorted(Path(real_dir).glob("*.jpg")))


# ── KID ───────────────────────────────────────────────────────────────────────

def compute_kid(gen_dir: str, real_dir: str, label: str) -> dict | None:
    try:
        from torchmetrics.image.kid import KernelInceptionDistance
    except ImportError:
        print("  [WARN] pip install torchmetrics torch-fidelity")
        return None

    print(f"\n{'─'*56}")
    print(f"KID  {label}")
    print(f"{'─'*56}")

    real_paths = _collect_real_paths(real_dir)
    gen_paths  = _collect_gen_paths(gen_dir)
    n_real, n_gen = len(real_paths), len(gen_paths)
    subset = min(args.kid_subset, n_real, max(1, n_gen // 2))
    print(f"  Real: {n_real}  Generated: {n_gen}  subset_size: {subset}")

    if n_real < 2 or n_gen < 2 or subset < 2:
        print("  Insufficient images — skipping KID")
        return None
    if n_real < 30:
        print(f"  [NOTE] Only {n_real} real images → KID variance will be high.")

    kid_metric = KernelInceptionDistance(subset_size=subset, normalize=False).to(device)

    for p in real_paths:
        t = _pil_to_uint8_tensor(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
        kid_metric.update(t, real=True)

    bs = args.batch_size
    for start in range(0, n_gen, bs):
        tensors = [_pil_to_uint8_tensor(Image.open(p).convert("RGB"))
                   for p in gen_paths[start : start + bs]]
        kid_metric.update(torch.stack(tensors).to(device), real=False)
        if (start // bs) % 10 == 0:
            print(f"  [{min(start + bs, n_gen):>5}/{n_gen}] fed …")

    km, ks = (float(v) for v in kid_metric.compute())
    print(f"  KID : {km:.6f} ± {ks:.6f}")
    return {"label": label, "mean": km, "std": ks,
            "n_real": n_real, "n_gen": n_gen,
            "subset_size": subset, "num_subsets": args.kid_num_subsets}


# ── Run evaluations ───────────────────────────────────────────────────────────

clip_base = clip_a = clip_b = None
kid_base = kid_a = kid_b = None

def _get_result_group(results: dict, section: str, name: str):
    group = results.get(section, {})
    return group.get(name) or group.get(f"setting_{name}")

if args.plot_only:
    json_path_prev = os.path.join(args.out_dir, "eval_results.json")
    if not os.path.isfile(json_path_prev):
        raise FileNotFoundError(f"--plot-only requires existing result JSON: {json_path_prev}")
    with open(json_path_prev, encoding="utf-8") as f:
        _prev = json.load(f)
    clip_a    = _get_result_group(_prev, "clip", "a")
    clip_b    = _get_result_group(_prev, "clip", "b")
    clip_base = _get_result_group(_prev, "clip", "base")
    kid_a     = _get_result_group(_prev, "kid", "a")
    kid_b     = _get_result_group(_prev, "kid", "b")
    kid_base  = _get_result_group(_prev, "kid", "base")
    args.skip_clip = True
    args.skip_kid = True
    print(f"Loaded previous results from {json_path_prev}")

# CLIP — optionally reload from saved JSON
if args.skip_clip and not args.plot_only:
    json_path_prev = os.path.join(args.out_dir, "eval_results.json")
    if os.path.isfile(json_path_prev):
        with open(json_path_prev) as f:
            _prev = json.load(f)
        clip_a    = _get_result_group(_prev, "clip", "a")
        clip_b    = _get_result_group(_prev, "clip", "b")
        clip_base = _get_result_group(_prev, "clip", "base")
        print(f"Loaded previous CLIP results from {json_path_prev}")
    else:
        print("[WARN] --skip-clip: no previous JSON found, recomputing CLIP")
        args.skip_clip = False

if not args.skip_clip:
    if args.dir_base and os.path.isdir(args.dir_base):
        clip_base = compute_clip(args.dir_base, "Base Model (35 steps, no LoRA)")
    clip_a = compute_clip(args.dir_a, "Setting A (35 steps, guidance=5.5)")
    clip_b = None if args.skip_b else compute_clip(args.dir_b, "Setting B (8 steps, guidance=3.5)")

if not args.skip_kid:
    if args.dir_base and os.path.isdir(args.dir_base):
        kid_base = compute_kid(args.dir_base, args.real_dir, "Base Model (35 steps, no LoRA)")
    kid_a = compute_kid(args.dir_a, args.real_dir, "Setting A (35 steps, guidance=5.5)")
    if not args.skip_b:
        kid_b = compute_kid(args.dir_b, args.real_dir, "Setting B (8 steps, guidance=3.5)")

# ── Save JSON ─────────────────────────────────────────────────────────────────

results = {
    "clip_model": args.clip_model,
    "clip": {"setting_base": clip_base, "setting_a": clip_a, "setting_b": clip_b},
    "kid":  {"setting_base": kid_base,  "setting_a": kid_a,  "setting_b": kid_b},
}
json_path = os.path.join(args.out_dir, "eval_results.json")
if args.plot_only:
    print(f"\nResults loaded → {json_path}")
else:
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved → {json_path}")

# ── Plotting ──────────────────────────────────────────────────────────────────

COLORS = {"Base": "#55A868", "A": "#4C72B0", "B": "#DD8452"}


def _safe(rec, key):
    return rec[key] if rec and rec.get(key) is not None else 0.0


# ── Chart 1: CLIP Score per prompt ───────────────────────────────────────────

# Build ordered list of available groups
clip_groups = []
if clip_base: clip_groups.append(("Base", "Base Model（無 LoRA）", clip_base))
if clip_a:    clip_groups.append(("A",    "Setting A（35 步）", clip_a))
if clip_b:    clip_groups.append(("B",    "Setting B（8 步）",  clip_b))

n_groups = len(clip_groups)
BAR_W_PP = 0.22 if n_groups == 3 else (0.30 if n_groups == 2 else 0.50)
offsets  = np.linspace(-(n_groups - 1) / 2 * BAR_W_PP,
                        (n_groups - 1) / 2 * BAR_W_PP,
                        n_groups)

fig, ax = plt.subplots(figsize=(14, 5.5))
x = np.arange(len(PROMPTS))

for (key, legend_label, clip_res), offset in zip(clip_groups, offsets):
    means = [_safe(r, "mean") for r in clip_res["per_prompt"]]
    stds  = [_safe(r, "std")  for r in clip_res["per_prompt"]]
    ax.bar(x + offset, means, BAR_W_PP, yerr=stds, capsize=3,
           color=COLORS[key], label=legend_label, alpha=0.85,
           error_kw={"elinewidth": 1})

ax.set_xticks(x)
ax.set_xticklabels(PROMPT_LABELS, fontsize=9)
ax.set_xlabel("Prompt", fontsize=11)
ax.set_ylabel("CLIP 分數（cosine × 100）", fontsize=11)
ax.set_title("各Prompt CLIP 分數比較（越大越好 ↑）", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.4)

all_means = []
for _, _, clip_res in clip_groups:
    all_means += [_safe(r, "mean") for r in clip_res["per_prompt"]]
if all_means:
    ax.set_ylim(max(0, min(all_means) - 3), max(all_means) + 4)

fig.tight_layout()
p1 = os.path.join(args.out_dir, "clip_per_prompt.png")
fig.savefig(p1, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Chart saved → {p1}")


# ── Chart 2: Summary (CLIP overall + KID) ────────────────────────────────────

has_kid = bool(kid_base or kid_a or kid_b)
n_panels = 1 + has_kid

fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
if n_panels == 1:
    axes = [axes]

BAR_W   = 0.20
BAR_XLIM = (-0.6, 2.6)

# ── Panel 1: Overall CLIP ──
ax = axes[0]
rows = []
if clip_base: rows.append(("Base", "Base Model\n（無 LoRA）",  clip_base["overall_mean"], clip_base["overall_std"]))
if clip_a:    rows.append(("A",    "Setting A\n（35 步）",     clip_a["overall_mean"],    clip_a["overall_std"]))
if clip_b:    rows.append(("B",    "Setting B\n（8 步）",      clip_b["overall_mean"],    clip_b["overall_std"]))

bars = ax.bar([r[1] for r in rows], [r[2] for r in rows],
              yerr=[r[3] for r in rows], capsize=7,
              color=[COLORS[r[0]] for r in rows],
              alpha=0.85, width=BAR_W, error_kw={"elinewidth": 1.5})
ax.set_xlim(*BAR_XLIM)
for bar, m in zip(bars, [r[2] for r in rows]):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + bar.get_height() * 0.01 + 0.2,
            f"{m:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

ax.set_ylabel("CLIP 分數（cosine × 100）", fontsize=10)
ax.set_title("整體 CLIP 分數\n（越大越好 ↑）", fontsize=12, fontweight="bold")
ax.grid(axis="y", linestyle="--", alpha=0.4)
ms = [r[2] for r in rows]
if ms:
    ax.set_ylim(max(0, min(ms) - 5), max(ms) + 6)

# ── Panel 2: KID ──
if has_kid:
    ax = axes[1]
    kid_rows = []
    if kid_base: kid_rows.append(("Base", "Base Model\n（無 LoRA）", kid_base["mean"], kid_base["std"]))
    if kid_a:    kid_rows.append(("A",    "Setting A\n（35 步）",    kid_a["mean"],    kid_a["std"]))
    if kid_b:    kid_rows.append(("B",    "Setting B\n（8 步）",     kid_b["mean"],    kid_b["std"]))

    bars_k = ax.bar([r[1] for r in kid_rows], [r[2] for r in kid_rows],
                    yerr=[r[3] for r in kid_rows], capsize=7,
                    color=[COLORS[r[0]] for r in kid_rows],
                    alpha=0.85, width=BAR_W, error_kw={"elinewidth": 1.5})
    ax.set_xlim(*BAR_XLIM)
    for bar, m in zip(bars_k, [r[2] for r in kid_rows]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(r[2] for r in kid_rows) * 0.02,
                f"{m:.5f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    n_real = (kid_base or kid_a or kid_b)["n_real"]
    subset = (kid_base or kid_a or kid_b)["subset_size"]
    ax.set_ylabel("KID", fontsize=10)
    ax.set_title(f"KID（vs 真實圖，越小越好 ↓）\n"
                 f"n_real={n_real}，subset_size={subset}，num_subsets={args.kid_num_subsets}",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_ylim(0, max(r[2] for r in kid_rows) * 1.35)

title_parts = []
if clip_base or kid_base: title_parts.append("Base Model")
if clip_a    or kid_a:    title_parts.append("Setting A")
if clip_b    or kid_b:    title_parts.append("Setting B")
fig.suptitle(" vs ".join(title_parts), fontsize=13, fontweight="bold")
fig.tight_layout()
p2 = os.path.join(args.out_dir, "summary.png")
fig.savefig(p2, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Chart saved → {p2}")

# ── Final summary ─────────────────────────────────────────────────────────────

print("\n" + "=" * 56)
print("  FINAL EVALUATION SUMMARY")
print("=" * 56)
for tag, clip_res, kid_res in [
    ("Base Model",  clip_base, kid_base),
    ("Setting A",  clip_a,    kid_a),
    ("Setting B",  clip_b,    kid_b),
]:
    if clip_res is None:
        continue
    kid_str = f"{kid_res['mean']:.6f} ± {kid_res['std']:.6f}" if kid_res else "—"
    print(f"  {tag}")
    print(f"    CLIP Score : {clip_res['overall_mean']:.3f} ± {clip_res['overall_std']:.3f}"
          f"  (N={clip_res['n_total']})")
    print(f"    KID        : {kid_str}")
print("=" * 56)
print("\nDone.")
