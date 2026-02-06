"""
Local heatmap test script using MCP heatmap logic.

This script calls WardAnalyzer._generate_heatmap_base64 to keep the output
identical to the heatmap shown in the vision analysis webpage.
"""
import argparse
import base64
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import requests

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from mcp_server.dota2_fastmcp import WardDataExtractor, WardAnalyzer  # noqa: E402

BASE_URL = "https://api.opendota.com/api"
TIMEOUT = 30


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
        raise ValueError("Unexpected JSON format, expected object.")
    return data


def generate_heatmap_base64(
    match_data: Dict[str, Any], sigma: float, alpha: float
) -> Optional[str]:
    extractor = WardDataExtractor()
    if not extractor.extract_from_match(match_data):
        return None

    df_obs, df_sen = extractor.get_dataframes()
    if df_obs.empty and df_sen.empty:
        return None

    analyzer = WardAnalyzer(
        df_obs,
        df_sen,
        match_duration=match_data.get("duration"),
    )
    return analyzer._generate_heatmap_base64(sigma=sigma, alpha=alpha)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate heatmap using MCP heatmap logic."
    )
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
        default=ROOT_DIR / "ward_analysis" / "heatmap_from_mcp.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=5.0,
        help="Gaussian sigma in 0-128 coordinate units (same as MCP).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.65,
        help="Max heatmap overlay alpha 0-1 (same as MCP).",
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

    heatmap_b64 = generate_heatmap_base64(match_data, sigma=args.sigma, alpha=args.alpha)
    if not heatmap_b64:
        print("No ward data found or heatmap generation failed.")
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(base64.b64decode(heatmap_b64))
    print(f"Heatmap saved to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
