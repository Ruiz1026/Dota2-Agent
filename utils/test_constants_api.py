"""
OpenDota constants API test script.

Endpoint:
  - GET /constants/{resource}
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Iterable, Optional

import requests

BASE_URL = "https://api.opendota.com/api"
TIMEOUT = 30
DEFAULT_RESOURCES = [
    "heroes",
    "items",
    "game_mode",
    "lobby_type",
    "patch",
    "abilities",
]


def make_request(resource: str) -> Dict[str, Any]:
    url = f"{BASE_URL}/constants/{resource}"
    print(f"\nGET {url}")
    try:
        response = requests.get(url, timeout=TIMEOUT)
        response.raise_for_status()
        return {"success": True, "data": response.json()}
    except requests.exceptions.RequestException as exc:
        print(f"  error: {exc}")
        return {"success": False, "error": str(exc)}


def _describe_payload(payload: Any) -> str:
    if isinstance(payload, list):
        return f"List[{len(payload)}]"
    if isinstance(payload, dict):
        return f"Dict[{len(payload)}]"
    return type(payload).__name__


def _sample_preview(payload: Any) -> str:
    if isinstance(payload, list) and payload:
        first = payload[0]
        if isinstance(first, dict):
            return "fields: " + ", ".join(list(first.keys())[:12])
        return f"sample: {str(first)[:60]}"
    if isinstance(payload, dict) and payload:
        keys = list(payload.keys())[:12]
        return "keys: " + ", ".join(keys)
    return "empty"


def save_sample(name: str, data: Any, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  saved: {path}")


def parse_resources(values: Iterable[str]) -> list[str]:
    items: list[str] = []
    for value in values:
        for part in value.split(","):
            cleaned = part.strip()
            if cleaned:
                items.append(cleaned)
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenDota constants API tests")
    parser.add_argument(
        "--resource",
        action="append",
        default=[],
        help="Resource name (can repeat or use comma-separated list)",
    )
    parser.add_argument(
        "--output-dir",
        default="api_samples",
        help="Directory to save JSON outputs",
    )
    args = parser.parse_args()

    resources = parse_resources(args.resource) or DEFAULT_RESOURCES

    success_count = 0
    for resource in resources:
        print(f"\n[resource] {resource}")
        result = make_request(resource)
        if not result["success"]:
            continue

        data = result["data"]
        print(f"  type: {_describe_payload(data)}")
        print(f"  preview: {_sample_preview(data)}")

        save_sample(f"constants_{resource}", data, args.output_dir)
        success_count += 1

    print(f"\nDone: {success_count}/{len(resources)} ok")


if __name__ == "__main__":
    main()
