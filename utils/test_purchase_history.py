import argparse
import sys
from typing import Any, Dict, List, Tuple

import requests

BASE_URL = "https://api.opendota.com/api"
TIMEOUT = 30


def fetch_match(match_id: int) -> Dict[str, Any]:
    url = f"{BASE_URL}/matches/{match_id}"
    response = requests.get(url, timeout=TIMEOUT)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError("Unexpected response format, expected a JSON object.")
    return data


def _get_player_name(player: Dict[str, Any]) -> str:
    name = player.get("name") or player.get("personaname") or player.get("account_id")
    return str(name) if name is not None else "Unknown"


def _get_time_samples(purchase_time: Dict[str, Any], limit: int = 5) -> List[Tuple[str, int]]:
    items = []
    for key, value in purchase_time.items():
        try:
            t = int(value)
        except (TypeError, ValueError):
            continue
        items.append((key, t))
    items.sort(key=lambda x: x[1])
    return items[:limit]


def print_player_summary(player: Dict[str, Any], idx: int) -> None:
    name = _get_player_name(player)
    hero_id = player.get("hero_id")
    slot = player.get("player_slot")

    purchase_log = player.get("purchase_log") or []
    purchase_time = player.get("purchase_time") or {}
    first_purchase_time = player.get("first_purchase_time") or {}

    print(f"\n[{idx}] player={name} hero_id={hero_id} slot={slot}")
    print(f"  purchase_log entries: {len(purchase_log)}")
    print(f"  purchase_time entries: {len(purchase_time)}")
    print(f"  first_purchase_time entries: {len(first_purchase_time)}")

    if purchase_log:
        print("  purchase_log sample:")
        for entry in purchase_log[:5]:
            print(f"    - time={entry.get('time')} key={entry.get('key')}")
    else:
        print("  purchase_log sample: (none)")

    if purchase_time:
        print("  purchase_time sample (earliest):")
        for key, t in _get_time_samples(purchase_time, limit=5):
            print(f"    - time={t} key={key}")
    else:
        print("  purchase_time sample: (none)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check if OpenDota match API provides purchase time/log."
    )
    parser.add_argument("match_id", type=int, help="Dota 2 match ID")
    args = parser.parse_args()

    try:
        data = fetch_match(args.match_id)
    except Exception as exc:
        print(f"Failed to fetch match: {exc}", file=sys.stderr)
        return 1

    players = data.get("players", [])
    if not players:
        print("No players found in match data.")
        return 1

    print(f"Match ID: {data.get('match_id')} players: {len(players)}")
    for idx, player in enumerate(players, 1):
        print_player_summary(player, idx)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
