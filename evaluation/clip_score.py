"""
CLIP Text-Image Score 評估腳本

計算生成圖片與對應 prompt 的 cosine similarity，
數值越高代表生成圖片與文字描述越相符。

期望的生成圖片資料夾結構：
  <image-root>/
    prompt_1/  0000.png, 0001.png, ...
    prompt_2/  ...
    ...
    prompt_N/ ...

用法：
  python evaluation/clip_score.py \
    --image-root outputs/runs/sd35_v10db_settingA \
    --prompt-source sd35

  # 或直接指定 CLIP 模型路徑（預設會從 HuggingFace 下載）
  python evaluation/clip_score.py \
    --image-root outputs/runs/sd35_v10db_settingA \
    --clip-model openai/clip-vit-large-patch14
"""

import argparse
import os
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))


def load_default_prompts(prompt_source: str) -> list[str]:
    """Load prompts from the active SD config file."""
    if prompt_source == "sd35":
        from sd35_config import PROMPTS
    elif prompt_source == "sd15":
        from sd15_config import PROMPTS
    else:
        raise ValueError(f"Unknown prompt source: {prompt_source}")

    return [clean_prompt(prompt) for prompt in PROMPTS]


def clean_prompt(prompt: str) -> str:
    """將 trigger token（如 <sksrockfall>）替換為 rockfall，CLIP 才能理解。"""
    return re.sub(r"<\w+>", "rockfall", prompt)


def load_images_from_dir(folder: Path, processor, device, batch_size=32):
    """載入資料夾內所有 PNG/JPG，回傳 batched pixel values tensor。"""
    paths = sorted(folder.glob("*.png")) + sorted(folder.glob("*.jpg"))
    if not paths:
        return None

    all_features = []
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i : i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
        all_features.append(inputs["pixel_values"])

    return torch.cat(all_features, dim=0)


@torch.no_grad()
def compute_clip_score(model, processor, image_root: Path, prompts: list, device, batch_size=32):
    results = {}

    prompt_dirs = sorted(
        [d for d in image_root.iterdir() if d.is_dir() and d.name.startswith("prompt_")],
        key=lambda d: int(d.name.split("_")[1]),
    )

    if not prompt_dirs:
        raise ValueError(f"在 {image_root} 找不到 prompt_* 子資料夾")

    if len(prompt_dirs) != len(prompts):
        print(
            f"[警告] prompt 資料夾數量 ({len(prompt_dirs)}) "
            f"與 prompts 清單數量 ({len(prompts)}) 不同，取較小值。"
        )

    all_scores = []

    for idx, prompt_dir in enumerate(tqdm(prompt_dirs, desc="評估 prompts")):
        if idx >= len(prompts):
            break

        prompt_text = clean_prompt(prompts[idx])

        # --- 計算文字 embedding ---
        text_inputs = processor(
            text=[prompt_text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        ).to(device)
        text_feat = model.get_text_features(**text_inputs)
        text_feat = F.normalize(text_feat, dim=-1)

        # --- 分批計算圖片 embedding ---
        image_paths = sorted(prompt_dir.glob("*.png")) + sorted(prompt_dir.glob("*.jpg"))
        if not image_paths:
            print(f"[警告] {prompt_dir} 內無圖片，跳過")
            continue

        batch_scores = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            img_inputs = processor(images=images, return_tensors="pt").to(device)
            img_feat = model.get_image_features(**img_inputs)
            img_feat = F.normalize(img_feat, dim=-1)

            scores = (img_feat @ text_feat.T).squeeze(-1)  # (B,)
            batch_scores.append(scores)

        prompt_scores = torch.cat(batch_scores)
        mean_score = prompt_scores.mean().item()
        results[prompt_dir.name] = {
            "mean": mean_score,
            "std": prompt_scores.std().item(),
            "n": len(image_paths),
            "prompt": prompts[idx],
        }
        all_scores.extend(prompt_scores.tolist())

    overall_mean = sum(all_scores) / len(all_scores) if all_scores else 0.0
    return results, overall_mean


def main():
    parser = argparse.ArgumentParser(description="CLIP Text-Image Score 評估")
    parser.add_argument(
        "--image-root",
        type=Path,
        required=True,
        help="生成圖片根目錄（底下有 prompt_1/, prompt_2/, ... 子資料夾）",
    )
    parser.add_argument(
        "--prompts-file",
        type=Path,
        default=None,
        help="每行一個 prompt 的文字檔（行數需對應 prompt 資料夾數量）",
    )
    parser.add_argument(
        "--prompt-source",
        choices=["sd35", "sd15"],
        default="sd35",
        help="未指定 --prompts-file 時使用哪一組 scripts/*_config.py prompts（預設：sd35）",
    )
    parser.add_argument(
        "--clip-model",
        default="openai/clip-vit-large-patch14",
        help="CLIP 模型名稱或本地路徑（預設：openai/clip-vit-large-patch14）",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, default=None, help="結果輸出路徑（.txt）")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用裝置：{device}")
    print(f"載入 CLIP 模型：{args.clip_model}")

    model = CLIPModel.from_pretrained(args.clip_model).to(device).eval()
    processor = CLIPProcessor.from_pretrained(args.clip_model)

    # 載入 prompts
    if args.prompts_file and args.prompts_file.exists():
        prompts = [
            line.strip()
            for line in args.prompts_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        print(f"從 {args.prompts_file} 載入 {len(prompts)} 個 prompts")
    else:
        prompts = load_default_prompts(args.prompt_source)
        print(f"使用 {args.prompt_source} 預設 {len(prompts)} 個 prompts")

    results, overall = compute_clip_score(
        model, processor, args.image_root, prompts, device, args.batch_size
    )

    # --- 輸出結果 ---
    lines = []
    lines.append(f"CLIP Score 評估結果：{args.image_root}")
    lines.append("=" * 60)
    for name, info in results.items():
        lines.append(f"{name:12s}  mean={info['mean']:.4f}  std={info['std']:.4f}  n={info['n']}")
    lines.append("-" * 60)
    lines.append(f"{'整體平均':12s}  mean={overall:.4f}")

    output_str = "\n".join(lines)
    print("\n" + output_str)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_str, encoding="utf-8")
        print(f"\n結果已儲存至 {args.output}")


if __name__ == "__main__":
    main()
