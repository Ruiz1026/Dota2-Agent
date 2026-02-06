import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen


API_BASE = "https://api.opendota.com/api"


def _request_json(path: str, timeout: int = 20) -> Any:
    url = f"{API_BASE}/{path.lstrip('/')}"
    try:
        req = Request(url, headers={"User-Agent": "opendota-check"})
        with urlopen(req, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
        return json.loads(payload)
    except URLError as exc:
        return {"error": f"network error: {exc}"}
    except json.JSONDecodeError:
        return {"error": "invalid json response"}


def _inspect_log(log_path: str) -> None:
    if not log_path or not os.path.exists(log_path):
        print(f"[log] file not found: {log_path}")
        return

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        print(f"[log] invalid json: {exc}")
        return

    iterations = data.get("iterations") or []
    matches_actions = [it for it in iterations if it.get("action") == "get_player_matches"]
    if not matches_actions:
        print("[log] no get_player_matches actions found")
        return

    print(f"[log] get_player_matches calls: {len(matches_actions)}")
    for idx, item in enumerate(matches_actions, start=1):
        obs = item.get("observation") or ""
        lines = [line.strip() for line in obs.splitlines() if line.strip()]
        data_rows = [
            line
            for line in lines
            if line.startswith("|")
            and "Match ID" not in line
            and "---" not in line
        ]
        print(f"  - call {idx}: rows={len(data_rows)}")
        if data_rows:
            print(f"    sample: {data_rows[0]}")


def _summarize_matches(matches: List[Dict[str, Any]], limit: int) -> None:
    if not matches:
        print("[api] recentMatches returned 0 items")
        print("      possible reasons: account private, no recent matches, or data not ingested yet.")
        return

    trimmed = matches[:limit] if limit > 0 else matches
    print(f"[api] recentMatches items: {len(matches)} (showing {len(trimmed)})")
    for i, match in enumerate(trimmed, start=1):
        match_id = match.get("match_id")
        hero_id = match.get("hero_id")
        duration = match.get("duration")
        player_slot = match.get("player_slot")
        print(f"  {i}. match_id={match_id} hero_id={hero_id} duration={duration} slot={player_slot}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check why get_player_matches has no rows")
    parser.add_argument("--account-id", type=int, required=True, help="Steam 32-bit account ID")
    parser.add_argument("--limit", type=int, default=10, help="Show first N matches")
    parser.add_argument(
        "--log",
        default=os.path.join("logs", "conversation_20260120_225211_001.json"),
        help="Optional conversation log to inspect",
    )
    args = parser.parse_args()

    print(f"[log] inspecting: {args.log}")
    _inspect_log(args.log)

    print(f"[api] fetching /players/{args.account_id}/recentMatches")
    result = _request_json(f"players/{args.account_id}/recentMatches")

    if isinstance(result, dict) and "error" in result:
        print(f"[api] error: {result['error']}")
        return 1
    if not isinstance(result, list):
        print("[api] unexpected response type")
        return 1

    _summarize_matches(result, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
