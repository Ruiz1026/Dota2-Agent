"""
测试 analyze_multi_match_wards 多场比赛视野分析
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from mcp_server.dota2_fastmcp import analyze_multi_match_wards


def _parse_match_ids(raw: str) -> List[int]:
    if not raw:
        return []
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    result: List[int] = []
    for part in parts:
        try:
            result.append(int(part))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"非法比赛ID: {part}") from exc
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="测试多场比赛视野分析（analyze_multi_match_wards）",
        epilog=(
            "示例: python utils/test_analyze_multi_match_wards.py --team-id 9572001 --limit 5 --debug\n"
            "示例: python utils/test_analyze_multi_match_wards.py --match-ids 8636761702,8636668127 --no-debug"
        ),
    )
    parser.add_argument("--team-id", type=int, help="战队ID")
    parser.add_argument("--account-id", type=int, help="玩家账号ID")
    parser.add_argument(
        "--match-ids",
        type=_parse_match_ids,
        help="指定比赛ID列表，用英文逗号分隔",
    )
    parser.add_argument("--limit", type=int, default=10, help="自动获取比赛数量")
    parser.add_argument("--sigma", type=float, default=5.0, help="热力图高斯 sigma")
    parser.add_argument("--alpha", type=float, default=0.65, help="热力图最大透明度")
    parser.add_argument("--debug", dest="debug", action="store_true", help="输出调试日志")
    parser.add_argument("--no-debug", dest="debug", action="store_false", help="不输出调试日志")
    parser.set_defaults(debug=None)

    args = parser.parse_args()

    if not args.team_id and not args.account_id and not args.match_ids:
        parser.error("必须提供 --team-id、--account-id 或 --match-ids 之一")

    kwargs = {
        "team_id": args.team_id,
        "account_id": args.account_id,
        "match_ids": args.match_ids,
        "limit": args.limit,
        "sigma": args.sigma,
        "alpha": args.alpha,
    }
    if args.debug is not None:
        kwargs["debug"] = args.debug

    result = analyze_multi_match_wards(**kwargs)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
