"""
OpenDota scenarios API test script.

Endpoints:
  - GET /scenarios/itemTimings
  - GET /scenarios/laneRoles
  - GET /scenarios/misc
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Optional

import requests

BASE_URL = "https://api.opendota.com/api"
TIMEOUT = 30


def make_request(endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    url = f"{BASE_URL}/{endpoint}"
    print(f"\nGET {url}")
    if params:
        print(f"  params: {params}")
    try:
        response = requests.get(url, params=params, timeout=TIMEOUT)
        response.raise_for_status()
        return {"success": True, "data": response.json()}
    except requests.exceptions.RequestException as exc:
        print(f"  error: {exc}")
        return {"success": False, "error": str(exc)}


def save_sample(name: str, data: Any, output_dir: str, sample_limit: int) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.json")
    payload = data
    if isinstance(data, list):
        payload = data[:sample_limit]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"  saved: {path}")


def test_item_timings(
    output_dir: str,
    sample_limit: int,
    item: Optional[str],
    hero_id: Optional[int],
) -> bool:
    print("\n[1] /scenarios/itemTimings")
    params: Dict[str, Any] = {}
    if item:
        params["item"] = item
    if hero_id is not None:
        params["hero_id"] = hero_id
    result = make_request("scenarios/itemTimings", params=params or None)
    if not result["success"]:
        return False
    data = result["data"] or []
    print(f"  rows: {len(data)}")
    for row in data[:sample_limit]:
        print(
            f"  - hero_id={row.get('hero_id')} item={row.get('item')} "
            f"time={row.get('time')} games={row.get('games')} wins={row.get('wins')}"
        )
    save_sample("scenarios_item_timings_sample", data, output_dir, sample_limit)
    return True


def test_lane_roles(
    output_dir: str,
    sample_limit: int,
    lane_role: Optional[int],
    hero_id: Optional[int],
) -> bool:
    print("\n[2] /scenarios/laneRoles")
    params: Dict[str, Any] = {}
    if lane_role is not None:
        params["lane_role"] = lane_role
    if hero_id is not None:
        params["hero_id"] = hero_id
    result = make_request("scenarios/laneRoles", params=params or None)
    if not result["success"]:
        return False
    data = result["data"] or []
    print(f"  rows: {len(data)}")
    for row in data[:sample_limit]:
        print(
            f"  - hero_id={row.get('hero_id')} lane_role={row.get('lane_role')} "
            f"time={row.get('time')} games={row.get('games')} wins={row.get('wins')}"
        )
    save_sample("scenarios_lane_roles_sample", data, output_dir, sample_limit)
    return True


def test_misc(
    output_dir: str,
    sample_limit: int,
    scenario: Optional[str],
) -> bool:
    print("\n[3] /scenarios/misc")
    params: Dict[str, Any] = {}
    if scenario:
        params["scenario"] = scenario
    result = make_request("scenarios/misc", params=params or None)
    if not result["success"]:
        return False
    data = result["data"] or []
    print(f"  rows: {len(data)}")
    for row in data[:sample_limit]:
        print(
            f"  - scenario={row.get('scenario')} is_radiant={row.get('is_radiant')} "
            f"region={row.get('region')} games={row.get('games')} wins={row.get('wins')}"
        )
    save_sample("scenarios_misc_sample", data, output_dir, sample_limit)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenDota scenarios API tests")
    parser.add_argument("--item", default="", help="item name e.g. spirit_vessel")
    parser.add_argument("--hero-id", type=int, default=None, help="hero id for filtering")
    parser.add_argument("--lane-role", type=int, default=None, help="lane role 1-4")
    parser.add_argument("--scenario", default="", help="scenario name for /scenarios/misc")
    parser.add_argument("--sample-limit", type=int, default=5, help="max rows to print/save")
    parser.add_argument(
        "--output-dir",
        default="api_samples",
        help="directory to save sample json outputs",
    )
    args = parser.parse_args()

    item = args.item.strip() or None
    scenario = args.scenario.strip() or None

    results = [
        test_item_timings(args.output_dir, args.sample_limit, item, args.hero_id),
        test_lane_roles(args.output_dir, args.sample_limit, args.lane_role, args.hero_id),
        test_misc(args.output_dir, args.sample_limit, scenario),
    ]

    success_count = sum(1 for ok in results if ok)
    print(f"\nDone: {success_count}/{len(results)} ok")


if __name__ == "__main__":
    main()
