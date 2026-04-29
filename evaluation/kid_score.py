"""
KID (Kernel Inception Distance) 評估腳本

KID 衡量真實圖片與生成圖片之間的特徵分佈距離，
數值越低代表生成圖片的分佈越接近真實圖片。

計算方式：
  1. 用 InceptionV3（pool3 層，2048-dim 特徵）萃取所有圖片的特徵向量
  2. 以 polynomial kernel MMD 計算兩個特徵集之間的距離

注意：本資料集訓練圖only 15 張，KID 估計的變異數較大，
建議搭配 bootstrap 信賴區間一起看。

用法：
  python evaluation/kid_score.py \
    --real-dir data/train \
    --fake-root outputs/runs/sd35_v10db_settingA

  # 指定特定 prompt 資料夾（只評估部分生成圖片）
  python evaluation/kid_score.py \
    --real-dir data/train \
    --fake-root outputs/runs/sd35_v10db_settingA/prompt_1

  # 開啟 bootstrap 信賴區間估計
  python evaluation/kid_score.py \
    --real-dir data/train \
    --fake-root outputs/runs/sd35_v10db_settingA \
    --bootstrap 1000
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)


# ========== InceptionV3 特徵萃取器 ==========

class InceptionFeatureExtractor(torch.nn.Module):
    """取 InceptionV3 avgpool 層輸出（2048-dim）作為特徵。"""

    def __init__(self):
        super().__init__()
        inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        inception.eval()

        # 只保留到 avgpool 層
        self.layers = torch.nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            torch.nn.MaxPool2d(3, stride=2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            torch.nn.MaxPool2d(3, stride=2),
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
            torch.nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.layers(x)
        return x.flatten(1)  # (B, 2048)


# ========== 圖片載入 ==========

def collect_image_paths(root: Path) -> list:
    """遞迴收集目錄下所有 PNG/JPG 路徑。"""
    exts = {".png", ".jpg", ".jpeg"}
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in exts)


def extract_features(
    paths: list,
    model: InceptionFeatureExtractor,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """萃取所有圖片的 InceptionV3 特徵，回傳 (N, 2048) numpy array。"""
    preprocess = transforms.Compose([
        transforms.Resize((299, 299), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    all_feats = []
    with torch.no_grad():
        for i in tqdm(range(0, len(paths), batch_size), desc="萃取特徵", leave=False):
            batch_paths = paths[i : i + batch_size]
            imgs = torch.stack([
                preprocess(Image.open(p).convert("RGB")) for p in batch_paths
            ]).to(device)
            feats = model(imgs)
            all_feats.append(feats.cpu())

    return torch.cat(all_feats, dim=0).numpy()


# ========== KID 計算 ==========

def polynomial_kernel(X: np.ndarray, Y: np.ndarray, degree=3, gamma=None, coef=1.0) -> np.ndarray:
    """Polynomial kernel：k(x,y) = (gamma * x·y + coef)^degree。"""
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    K = (gamma * X @ Y.T + coef) ** degree
    return K


def compute_kid(real_feats: np.ndarray, fake_feats: np.ndarray) -> float:
    """
    計算 KID（Kernel Inception Distance）。
    使用 unbiased MMD² 估計量。
    """
    n = real_feats.shape[0]
    m = fake_feats.shape[0]

    Kxx = polynomial_kernel(real_feats, real_feats)
    Kyy = polynomial_kernel(fake_feats, fake_feats)
    Kxy = polynomial_kernel(real_feats, fake_feats)

    # Unbiased MMD²：對角線不計入（i≠j）
    np.fill_diagonal(Kxx, 0)
    np.fill_diagonal(Kyy, 0)

    mmd2 = (
        Kxx.sum() / (n * (n - 1))
        + Kyy.sum() / (m * (m - 1))
        - 2.0 * Kxy.mean()
    )
    return float(mmd2)


def bootstrap_kid(
    real_feats: np.ndarray,
    fake_feats: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> tuple:
    """Bootstrap 估計 KID 的均值與 95% 信賴區間。"""
    rng = np.random.default_rng(seed)
    n, m = len(real_feats), len(fake_feats)
    kid_samples = []

    for _ in tqdm(range(n_bootstrap), desc="Bootstrap", leave=False):
        r_idx = rng.integers(0, n, size=n)
        f_idx = rng.integers(0, m, size=m)
        kid_samples.append(compute_kid(real_feats[r_idx], fake_feats[f_idx]))

    mean = float(np.mean(kid_samples))
    ci_low = float(np.percentile(kid_samples, 2.5))
    ci_high = float(np.percentile(kid_samples, 97.5))
    return mean, ci_low, ci_high


# ========== 主程式 ==========

def main():
    parser = argparse.ArgumentParser(description="KID（Kernel Inception Distance）評估")
    parser.add_argument(
        "--real-dir",
        type=Path,
        default=Path("data/train"),
        help="真實訓練圖片資料夾（預設：data/train）",
    )
    parser.add_argument(
        "--fake-root",
        type=Path,
        required=True,
        help="生成圖片根目錄（會遞迴收集底下所有 PNG/JPG）",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        help="Bootstrap 次數（0 表示關閉，建議 1000）",
    )
    parser.add_argument("--output", type=Path, default=None, help="結果輸出路徑（.txt）")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置：{device}")

    # 載入 InceptionV3
    print("載入 InceptionV3...")
    model = InceptionFeatureExtractor().to(device).eval()

    # 收集圖片路徑
    real_paths = collect_image_paths(args.real_dir)
    fake_paths = collect_image_paths(args.fake_root)

    if not real_paths:
        raise FileNotFoundError(f"在 {args.real_dir} 找不到圖片")
    if not fake_paths:
        raise FileNotFoundError(f"在 {args.fake_root} 找不到圖片")

    print(f"真實圖片：{len(real_paths)} 張（來自 {args.real_dir}）")
    print(f"生成圖片：{len(fake_paths)} 張（來自 {args.fake_root}）")

    if len(real_paths) < 10:
        print(
            f"[警告] 真實圖片只有 {len(real_paths)} 張，KID 估計值的變異數較大，"
            "建議搭配 bootstrap 信賴區間解讀。"
        )

    # 萃取特徵
    print("\n萃取真實圖片特徵...")
    real_feats = extract_features(real_paths, model, device, args.batch_size)
    print("萃取生成圖片特徵...")
    fake_feats = extract_features(fake_paths, model, device, args.batch_size)

    # 計算 KID
    print("\n計算 KID...")
    kid = compute_kid(real_feats, fake_feats)

    lines = []
    lines.append(f"KID 評估結果")
    lines.append("=" * 60)
    lines.append(f"真實圖片：{len(real_paths)} 張  ← {args.real_dir}")
    lines.append(f"生成圖片：{len(fake_paths)} 張  ← {args.fake_root}")
    lines.append(f"KID（point estimate）：{kid:.6f}")

    if args.bootstrap > 0:
        print(f"執行 Bootstrap（{args.bootstrap} 次）...")
        mean, ci_low, ci_high = bootstrap_kid(real_feats, fake_feats, args.bootstrap)
        lines.append(f"KID Bootstrap mean  ：{mean:.6f}")
        lines.append(f"95% CI              ：[{ci_low:.6f}, {ci_high:.6f}]")

    lines.append(
        "\n[說明] KID 越低代表生成圖片分佈越接近真實圖片分佈，"
        "理想值接近 0。"
    )

    output_str = "\n".join(lines)
    print("\n" + output_str)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_str, encoding="utf-8")
        print(f"\n結果已儲存至 {args.output}")


if __name__ == "__main__":
    main()
