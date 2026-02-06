"""
OpenDota Heroes API test script.

Endpoints:
  - GET /heroes
  - GET /heroes/{hero_id}/matches
  - GET /heroes/{hero_id}/matchups
  - GET /heroes/{hero_id}/durations
  - GET /heroes/{hero_id}/players
  - GET /heroes/{hero_id}/itemPopularity
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import requests

BASE_URL = "https://api.opendota.com/api"
TIMEOUT = 30
DEFAULT_HERO_ID = 1


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


def save_sample(name: str, data: Any, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  saved: {path}")


def resolve_hero_id(hero_id: Optional[int]) -> int:
    if hero_id is not None:
        return hero_id
    result = make_request("heroes")
    if not result["success"]:
        return DEFAULT_HERO_ID
    heroes = result["data"] or []
    for hero in heroes:
        try:
            candidate = int(hero.get("id", 0))
        except (TypeError, ValueError):
            continue
        if candidate > 0:
            return candidate
    return DEFAULT_HERO_ID


def test_heroes(output_dir: str, sample_limit: int) -> bool:
    print("\n[1] /heroes")
    result = make_request("heroes")
    if not result["success"]:
        return False
    data = result["data"] or []
    print(f"  heroes: {len(data)}")
    for hero in data[:sample_limit]:
        print(
            f"  - id={hero.get('id')} "
            f"name={hero.get('localized_name')} "
            f"attr={hero.get('primary_attr')} "
            f"attack={hero.get('attack_type')}"
        )
    save_sample("heroes_list_sample", data[:sample_limit], output_dir)
    return True


def test_hero_matches(hero_id: int, output_dir: str, sample_limit: int) -> bool:
    print(f"\n[2] /heroes/{hero_id}/matches")
    result = make_request(f"heroes/{hero_id}/matches")
    if not result["success"]:
        return False
    data = result["data"] or []
    print(f"  matches: {len(data)}")
    for match in data[:sample_limit]:
        print(
            f"  - match_id={match.get('match_id')} "
            f"duration={match.get('duration')} "
            f"radiant_win={match.get('radiant_win')}"
        )
    save_sample(f"heroes_{hero_id}_matches_sample", data[:sample_limit], output_dir)
    return True


def test_hero_matchups(hero_id: int, output_dir: str, sample_limit: int) -> bool:
    print(f"\n[3] /heroes/{hero_id}/matchups")
    result = make_request(f"heroes/{hero_id}/matchups")
    if not result["success"]:
        return False
    data = result["data"] or []
    print(f"  matchups: {len(data)}")
    for row in data[:sample_limit]:
        print(
            f"  - hero_id={row.get('hero_id')} "
            f"games={row.get('games_played')} wins={row.get('wins')}"
        )
    save_sample(f"heroes_{hero_id}_matchups_sample", data[:sample_limit], output_dir)
    return True


def test_hero_durations(hero_id: int, output_dir: str, sample_limit: int) -> bool:
    print(f"\n[4] /heroes/{hero_id}/durations")
    result = make_request(f"heroes/{hero_id}/durations")
    if not result["success"]:
        return False
    data = result["data"] or []
    print(f"  durations: {len(data)}")
    for row in data[:sample_limit]:
        print(
            f"  - duration_bin={row.get('duration_bin')} "
            f"games={row.get('games_played')} wins={row.get('wins')}"
        )
    save_sample(f"heroes_{hero_id}_durations_sample", data[:sample_limit], output_dir)
    return True


def test_hero_players(hero_id: int, output_dir: str, sample_limit: int) -> bool:
    print(f"\n[5] /heroes/{hero_id}/players")
    result = make_request(f"heroes/{hero_id}/players")
    if not result["success"]:
        return False
    data = result["data"] or []
    print(f"  players: {len(data)}")
    for row in data[:sample_limit]:
        print(
            f"  - account_id={row.get('account_id')} "
            f"name={row.get('name') or row.get('personaname')} "
            f"is_pro={row.get('is_pro')}"
        )
    save_sample(f"heroes_{hero_id}_players_sample", data[:sample_limit], output_dir)
    return True


def test_hero_item_popularity(hero_id: int, output_dir: str) -> bool:
    print(f"\n[6] /heroes/{hero_id}/itemPopularity")
    result = make_request(f"heroes/{hero_id}/itemPopularity")
    if not result["success"]:
        return False
    data = result["data"] or {}
    print(f"  sections: {list(data.keys())}")
    save_sample(f"heroes_{hero_id}_item_popularity_sample", data, output_dir)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenDota Heroes API tests")
    parser.add_argument("--hero-id", type=int, default=None, help="Hero ID for hero endpoints")
    parser.add_argument("--sample-limit", type=int, default=5, help="Max rows to print/save")
    parser.add_argument(
        "--output-dir",
        default="api_samples",
        help="Directory to save sample JSON outputs",
    )
    args = parser.parse_args()

    hero_id = resolve_hero_id(args.hero_id)
    print(f"\nUsing hero_id={hero_id}")

    results = [
        test_heroes(args.output_dir, args.sample_limit),
        test_hero_matches(hero_id, args.output_dir, args.sample_limit),
        test_hero_matchups(hero_id, args.output_dir, args.sample_limit),
        test_hero_durations(hero_id, args.output_dir, args.sample_limit),
        test_hero_players(hero_id, args.output_dir, args.sample_limit),
        test_hero_item_popularity(hero_id, args.output_dir),
    ]

    success_count = sum(1 for ok in results if ok)
    print(f"\nDone: {success_count}/{len(results)} ok")


if __name__ == "__main__":
    main()
