"""
热力图样式本地测试脚本

生成连续平滑的热力区域，基于眼位坐标点做高斯模糊后上伪彩。
"""
import argparse
import base64
import json
import sys
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from PIL import Image

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from mcp_server.dota2_fastmcp import WardDataExtractor  # noqa: E402

BASE_URL = "https://api.opendota.com/api"
TIMEOUT = 30
MAPS_DIR = ROOT_DIR / "maps"
MAP_VERSION = "740"


def load_match_from_file(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Unexpected JSON format, expected object.")
    return data


def load_match_from_api(match_id: int) -> Dict[str, Any]:
    url = f"{BASE_URL}/matches/{match_id}"
    response = requests.get(url, timeout=TIMEOUT)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError("Unexpected response format, expected object.")
    return data


def load_map_image() -> Optional[Image.Image]:
    for ext in [".jpeg", ".jpg", ".png"]:
        p = MAPS_DIR / f"{MAP_VERSION}{ext}"
        if p.exists():
            return Image.open(p)
    return None


def generate_heatmap(
    points: List[Tuple[float, float]],
    map_image: Image.Image,
    sigma: float = 15.0,
    alpha: float = 0.7,
    point_weight: float = 1.0,
) -> Image.Image:
    """
    生成平滑连续的热力图。

    Args:
        points: 眼位坐标列表 [(x, y), ...], 坐标范围 0-128
        map_image: 底图 PIL Image
        sigma: 高斯模糊 sigma, 在 0-128 坐标系下的单位 (越大越平滑)
        alpha: 热力图最大透明度 0-1
        point_weight: 每个点的初始权重

    Returns:
        叠加后的 PIL Image
    """
    width, height = map_image.size

    # 缩放 sigma: 让 sigma 与图像尺寸成正比
    # sigma 在 0-128 坐标系, 转换到像素坐标
    sigma_px = sigma * (width / 128.0)

    # 创建累积 canvas
    canvas = np.zeros((height, width), dtype=np.float32)

    x_scale = (width - 1) / 128.0
    y_scale = (height - 1) / 128.0

    for x_val, y_val in points:
        if not (np.isfinite(x_val) and np.isfinite(y_val)):
            continue
        x_val = np.clip(x_val, 0, 128)
        y_val = np.clip(y_val, 0, 128)
        px = int(round(x_val * x_scale))
        # Y 轴翻转: 游戏坐标 y=0 在底部, 图像 y=0 在顶部
        py = int(round((128 - y_val) * y_scale))
        if 0 <= px < width and 0 <= py < height:
            canvas[py, px] += point_weight

    # 高斯模糊
    if HAS_OPENCV:
        # kernel size 需要是奇数, 大约为 6*sigma 以涵盖 99.7% 的高斯分布
        kernel_size = int(round(sigma_px * 6))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = max(3, min(kernel_size, min(width, height) | 1))
        if kernel_size % 2 == 0:
            kernel_size -= 1
        blurred = cv2.GaussianBlur(canvas, (kernel_size, kernel_size), sigma_px)
    else:
        # 用 FFT 实现高斯模糊
        blurred = _gaussian_blur_fft(canvas, sigma_px)

    # 归一化到 0-1
    max_val = blurred.max()
    if max_val > 0:
        blurred = blurred / max_val

    # 伪彩 (JET colormap)
    if HAS_OPENCV:
        heat_color = cv2.applyColorMap(
            (blurred * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
    else:
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap("jet")
        heat_color = (cmap(blurred)[:, :, :3] * 255).astype(np.uint8)

    # 叠加: 只在有热度的地方叠加, 热度越高越不透明
    base_arr = np.array(map_image.convert("RGB"))
    alpha_map = np.clip(blurred * alpha, 0, alpha)
    overlay = (
        base_arr * (1 - alpha_map[..., None]) + heat_color * alpha_map[..., None]
    ).astype(np.uint8)

    return Image.fromarray(overlay)


def _gaussian_blur_fft(data: np.ndarray, sigma: float) -> np.ndarray:
    """用 FFT 实现高斯模糊 (无 OpenCV 时的后备方案)"""
    if sigma <= 0:
        return data
    size = int(max(3, round(sigma * 6)))
    if size % 2 == 0:
        size += 1
    half = size // 2
    ax = np.arange(-half, half + 1, dtype=float)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()

    # zero-pad kernel to data shape
    pad_kernel = np.zeros(data.shape, dtype=float)
    kh, kw = kernel.shape
    pad_kernel[:kh, :kw] = kernel
    pad_kernel = np.roll(pad_kernel, -half, axis=0)
    pad_kernel = np.roll(pad_kernel, -half, axis=1)

    fdata = np.fft.rfft2(data)
    fkernel = np.fft.rfft2(pad_kernel)
    blurred = np.fft.irfft2(fdata * fkernel, data.shape)
    return np.clip(blurred, 0, None).astype(np.float32)


def extract_ward_points(match_data: Dict[str, Any]) -> List[Tuple[float, float]]:
    """
    提取眼位坐标，返回 (0-128, 0-128) 范围的坐标列表。

    API 返回的坐标大约在 64-192 范围 (以 128 为中心)，
    需要减去 64 转换到 0-128 范围。
    """
    extractor = WardDataExtractor()
    if not extractor.extract_from_match(match_data):
        return []

    df_obs, df_sen = extractor.get_dataframes()

    points = []
    for df in [df_obs, df_sen]:
        if df.empty:
            continue
        for _, row in df.iterrows():
            x = row.get("x")
            y = row.get("y")
            if x is not None and y is not None:
                # API 坐标范围约 64-192, 减 64 变成 0-128
                points.append((float(x) - 64, float(y) - 64))
    return points


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate ward heatmap preview image.")
    parser.add_argument("--match-id", type=int, help="Fetch match data from OpenDota API.")
    parser.add_argument(
        "--match-file",
        type=Path,
        default=ROOT_DIR / "api_samples" / "match_details.json",
        help="Path to local match_details.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT_DIR / "ward_analysis" / "heatmap_preview.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=8.0,
        help="Gaussian sigma in 0-128 coordinate units. Larger = smoother/wider spread. Try 5-20.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.65,
        help="Max heatmap overlay alpha 0-1. Higher = more opaque heatmap.",
    )
    args = parser.parse_args()

    try:
        if args.match_id:
            print(f"Fetching match {args.match_id} from API...")
            match_data = load_match_from_api(args.match_id)
        else:
            print(f"Loading match from {args.match_file}...")
            match_data = load_match_from_file(args.match_file)
    except Exception as exc:
        print(f"Failed to load match data: {exc}")
        return 1

    map_image = load_map_image()
    if map_image is None:
        print(f"Map image not found in {MAPS_DIR}")
        return 1

    points = extract_ward_points(match_data)
    if not points:
        print("No ward data found.")
        return 1

    print(f"Found {len(points)} ward points, generating heatmap (sigma={args.sigma}, alpha={args.alpha})...")

    heatmap_image = generate_heatmap(points, map_image, sigma=args.sigma, alpha=args.alpha)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    heatmap_image.save(args.output, format="PNG")
    print(f"Heatmap saved to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
