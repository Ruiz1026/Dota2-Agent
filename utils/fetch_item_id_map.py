import argparse
import json
import os
import sys
from typing import Any, Dict

import requests

BASE_URL = "https://api.opendota.com/api"
ENDPOINT = "constants/items"
DEFAULT_OUTPUT = os.path.join("api_samples", "constants_items_map.json")
TIMEOUT = 30


def fetch_constants() -> Dict[str, Any]:
    url = f"{BASE_URL}/{ENDPOINT}"
    response = requests.get(url, timeout=TIMEOUT)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError("Unexpected response format, expected a JSON object.")
    return data


def build_item_map(data: Dict[str, Any]) -> Dict[str, Any]:
    items = []
    by_id: Dict[str, Dict[str, Any]] = {}

    for key, info in data.items():
        if not isinstance(info, dict):
            continue
        item_id = info.get("id")
        if item_id is None:
            continue
        try:
            item_id_int = int(item_id)
        except (TypeError, ValueError):
            continue
        name = info.get("dname") or info.get("name") or key
        entry = {"id": item_id_int, "key": key, "name": name}
        items.append(entry)
        by_id[str(item_id_int)] = {"key": key, "name": name}

    items.sort(key=lambda x: x["id"])
    return {
        "source": f"{BASE_URL}/{ENDPOINT}",
        "count": len(items),
        "items": items,
        "by_id": by_id,
    }


def write_json(path: str, payload: Dict[str, Any]) -> None:
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch OpenDota constants/items and build an item id map."
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    try:
        data = fetch_constants()
        item_map = build_item_map(data)
        write_json(args.output, item_map)
    except Exception as exc:
        print(f"Failed to build item id map: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote item id map to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
