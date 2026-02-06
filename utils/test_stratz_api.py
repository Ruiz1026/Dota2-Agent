import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional

try:
    import requests
except ImportError as exc:
    raise SystemExit("Missing dependency: pip install requests") from exc


DEFAULT_URL = "https://api.stratz.com/graphql"
DEFAULT_UA = "STRATZ_API"
DEFAULT_ARG_VALUES = {
    "id": 8657014616,
    "matchId": 8657014616,
    "steamAccountId": 355962940,
    "accountId": 1,
    "heroId": 1,
    "leagueId": 1,
    "teamId": 1,
    "regionId": 1,
    "page": 1,
    "pageSize": 1,
    "take": 1,
    "skip": 0,
    "limit": 1,
    "isParsed": False,
    "isPro": False,
    "from": 0,
    "to": 0,
}
MATCH_FIELD_ORDER = [
    "id",
    "didRadiantWin",
    "durationSeconds",
    "startDateTime",
    "endDateTime",
    "towerStatusRadiant",
    "towerStatusDire",
    "barracksStatusRadiant",
    "barracksStatusDire",
    "clusterId",
    "firstBloodTime",
    "lobbyType",
    "numHumanPlayers",
    "gameMode",
    "replaySalt",
    "isStats",
    "tournamentId",
    "tournamentRound",
    "actualRank",
    "averageRank",
    "averageImp",
    "parsedDateTime",
    "statsDateTime",
    "leagueId",
    "league",
    "radiantTeamId",
    "radiantTeam",
    "direTeamId",
    "direTeam",
    "seriesId",
    "series",
    "gameVersionId",
    "regionId",
    "sequenceNum",
    "rank",
    "bracket",
    "analysisOutcome",
    "predictedOutcomeWeight",
    "players",
    "radiantNetworthLeads",
    "radiantExperienceLeads",
    "radiantKills",
    "direKills",
    "pickBans",
    "towerStatus",
    "laneReport",
    "winRates",
    "predictedWinRates",
    "chatEvents",
    "towerDeaths",
    "playbackData",
    "spectators",
    "bottomLaneOutcome",
    "midLaneOutcome",
    "topLaneOutcome",
    "didRequestDownload",
]
SCHEMA_QUERY = """
query SchemaOverview {
  __schema {
    queryType {
      name
      fields {
        name
        type {
          kind
          name
          ofType {
            kind
            name
            ofType {
              kind
              name
              ofType {
                kind
                name
              }
            }
          }
        }
        args {
          name
          type {
            kind
            name
            ofType {
              kind
              name
              ofType {
                kind
                name
                ofType {
                  kind
                  name
                }
              }
            }
          }
        }
      }
    }
  }
}
""".strip()


def _type_string(type_info: Optional[Dict[str, Any]]) -> str:
    if not type_info:
        return "String"
    kind = type_info.get("kind")
    name = type_info.get("name")
    of_type = type_info.get("ofType")
    if kind == "NON_NULL":
        return f"{_type_string(of_type)}!"
    if kind == "LIST":
        return f"[{_type_string(of_type)}]"
    return name or "String"


def _base_type(type_info: Optional[Dict[str, Any]]) -> Dict[str, Optional[str]]:
    info = {"kind": None, "name": None}
    current = type_info or {}
    while current:
        kind = current.get("kind")
        name = current.get("name")
        if kind not in ("NON_NULL", "LIST"):
            info["kind"] = kind
            info["name"] = name
            return info
        current = current.get("ofType") or {}
    return info


def _is_required(type_info: Optional[Dict[str, Any]]) -> bool:
    return bool(type_info) and type_info.get("kind") == "NON_NULL"


def _find_query_field(schema: Dict[str, Any], field_name: str) -> Optional[Dict[str, Any]]:
    fields = schema.get("data", {}).get("__schema", {}).get("queryType", {}).get("fields", [])
    for field in fields:
        if field.get("name") == field_name:
            return field
    return None


def _pick_arg(field: Dict[str, Any], preferred: list[str]) -> Optional[Dict[str, Any]]:
    args = field.get("args", []) if field else []
    for name in preferred:
        for arg in args:
            if arg.get("name") == name:
                return arg
    return args[0] if args else None


def _default_value(arg_name: str, type_name: Optional[str]) -> Optional[Any]:
    if arg_name in DEFAULT_ARG_VALUES:
        return DEFAULT_ARG_VALUES[arg_name]
    if type_name in ("Int", "Long"):
        return 1
    if type_name == "Float":
        return 1.0
    if type_name == "Boolean":
        return False
    if type_name in ("String", "ID"):
        return "1"
    return None


def _fetch_type_definition(
    session: requests.Session,
    url: str,
    type_name: str,
    token: Optional[str],
    cookie: Optional[str],
    user_agent: str,
    timeout: int,
    cache: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if type_name in cache:
        return cache[type_name]
    payload = {
        "query": """
        query TypeDef($name: String!) {
          __type(name: $name) {
            kind
            name
            enumValues { name }
            inputFields {
              name
              type {
                kind
                name
                ofType {
                  kind
                  name
                  ofType {
                    kind
                    name
                    ofType {
                      kind
                      name
                    }
                  }
                }
              }
            }
            fields {
              name
              args {
                name
                type {
                  kind
                  name
                  ofType {
                    kind
                    name
                    ofType {
                      kind
                      name
                      ofType {
                        kind
                        name
                      }
                    }
                  }
                }
              }
              type {
                kind
                name
                ofType {
                  kind
                  name
                  ofType {
                    kind
                    name
                    ofType {
                      kind
                      name
                    }
                  }
                }
              }
            }
          }
        }
        """,
        "variables": {"name": type_name},
    }
    response = _request(session, url, payload, token, cookie, user_agent, timeout)
    type_def = (response.get("json") or {}).get("data", {}).get("__type")
    cache[type_name] = type_def
    return type_def


def _value_from_type(
    type_info: Optional[Dict[str, Any]],
    arg_name: str,
    session: requests.Session,
    url: str,
    token: Optional[str],
    cookie: Optional[str],
    user_agent: str,
    timeout: int,
    cache: Dict[str, Any],
    path: Optional[str] = None,
) -> tuple[Optional[Any], list[str]]:
    label = path or arg_name
    if not type_info:
        return None, [label]

    kind = type_info.get("kind")
    name = type_info.get("name")
    of_type = type_info.get("ofType")

    if kind == "NON_NULL":
        value, missing = _value_from_type(
            of_type,
            arg_name,
            session,
            url,
            token,
            cookie,
            user_agent,
            timeout,
            cache,
            label,
        )
        if value is None and label not in missing:
            missing.append(label)
        return value, missing

    if kind == "LIST":
        value, missing = _value_from_type(
            of_type,
            arg_name,
            session,
            url,
            token,
            cookie,
            user_agent,
            timeout,
            cache,
            label,
        )
        if value is None:
            return None, missing
        return [value], []

    if kind == "SCALAR":
        value = _default_value(arg_name, name)
        if value is None:
            return None, [label]
        return value, []

    if kind == "ENUM":
        definition = _fetch_type_definition(session, url, name, token, cookie, user_agent, timeout, cache)
        enum_values = (definition or {}).get("enumValues") or []
        if enum_values:
            return enum_values[0].get("name"), []
        return None, [label]

    if kind == "INPUT_OBJECT":
        definition = _fetch_type_definition(session, url, name, token, cookie, user_agent, timeout, cache)
        if not definition:
            return None, [label]
        obj: Dict[str, Any] = {}
        missing: list[str] = []
        for field in definition.get("inputFields") or []:
            field_name = field.get("name")
            field_type = field.get("type")
            field_value, field_missing = _value_from_type(
                field_type,
                field_name,
                session,
                url,
                token,
                cookie,
                user_agent,
                timeout,
                cache,
                f"{label}.{field_name}",
            )
            if field_missing and _is_required(field_type):
                missing.extend(field_missing)
                continue
            if field_value is not None:
                obj[field_name] = field_value
        if missing:
            return None, missing
        return obj, []

    return None, [label]


def _scalar_fields_for_type(
    session: requests.Session,
    url: str,
    type_name: str,
    token: Optional[str],
    cookie: Optional[str],
    user_agent: str,
    timeout: int,
    cache: Dict[str, Any],
    preferred: Optional[list[str]] = None,
    max_fields: int = 8,
) -> list[str]:
    definition = _fetch_type_definition(session, url, type_name, token, cookie, user_agent, timeout, cache)
    fields = (definition or {}).get("fields") or []
    scalar_fields: list[str] = []
    for field in fields:
        base = _base_type(field.get("type"))
        if base.get("kind") in ("SCALAR", "ENUM"):
            name = field.get("name")
            if name:
                scalar_fields.append(name)
    if preferred:
        ordered = [name for name in preferred if name in scalar_fields]
        for name in scalar_fields:
            if name not in ordered:
                ordered.append(name)
        scalar_fields = ordered
    if max_fields > 0:
        scalar_fields = scalar_fields[:max_fields]
    return scalar_fields


def _field_has_required_args(field: Dict[str, Any]) -> bool:
    for arg in field.get("args") or []:
        if _is_required(arg.get("type")):
            return True
    return False


def _selection_for_type(
    session: requests.Session,
    url: str,
    type_name: str,
    token: Optional[str],
    cookie: Optional[str],
    user_agent: str,
    timeout: int,
    cache: Dict[str, Any],
    depth: int,
    max_fields: int = 20,
    seen: Optional[set[str]] = None,
) -> str:
    if depth < 0:
        return ""
    seen = seen or set()
    if type_name in seen:
        return ""
    seen.add(type_name)
    try:
        definition = _fetch_type_definition(session, url, type_name, token, cookie, user_agent, timeout, cache)
        fields = (definition or {}).get("fields") or []
        selections: list[str] = []
        for field in fields:
            if _field_has_required_args(field):
                continue
            field_name = field.get("name")
            if not field_name:
                continue
            base = _base_type(field.get("type"))
            kind = base.get("kind")
            if kind in ("SCALAR", "ENUM"):
                selections.append(field_name)
            elif depth > 0 and kind in ("OBJECT", "INTERFACE", "UNION") and base.get("name"):
                sub_selection = _selection_for_type(
                    session,
                    url,
                    base["name"],
                    token,
                    cookie,
                    user_agent,
                    timeout,
                    cache,
                    depth - 1,
                    max_fields=max_fields,
                    seen=seen,
                )
                if sub_selection:
                    selections.append(f"{field_name} {{ {sub_selection} }}")
        if max_fields > 0:
            selections = selections[:max_fields]
        return " ".join(selections)
    finally:
        seen.discard(type_name)


def _build_match_selection_and_vars(
    session: requests.Session,
    url: str,
    token: Optional[str],
    cookie: Optional[str],
    user_agent: str,
    timeout: int,
    cache: Dict[str, Any],
    fields_order: list[str],
    depth: int,
    max_fields: int,
) -> tuple[str, Dict[str, Any], list[str]]:
    definition = _fetch_type_definition(session, url, "MatchType", token, cookie, user_agent, timeout, cache)
    match_fields = (definition or {}).get("fields") or []
    fields_by_name = {field.get("name"): field for field in match_fields if field.get("name")}
    selections: list[str] = []
    variables: Dict[str, Any] = {}
    var_defs: Dict[str, str] = {}

    for field_name in fields_order:
        field = fields_by_name.get(field_name)
        if not field:
            continue
        args_expr = []
        skip_field = False
        for arg in field.get("args") or []:
            arg_name = arg.get("name")
            if not arg_name:
                continue
            value, missing = _value_from_type(
                arg.get("type"),
                arg_name,
                session,
                url,
                token,
                cookie,
                user_agent,
                timeout,
                cache,
            )
            if missing and _is_required(arg.get("type")):
                skip_field = True
                break
            if value is None:
                continue
            var_type = _type_string(arg.get("type"))
            if arg_name in var_defs and var_defs[arg_name] != var_type:
                skip_field = True
                break
            var_defs[arg_name] = var_type
            variables.setdefault(arg_name, value)
            args_expr.append(f"{arg_name}: ${arg_name}")
        if skip_field:
            continue

        args_section = f"({', '.join(args_expr)})" if args_expr else ""
        field_expr = f"{field_name}{args_section}"
        base = _base_type(field.get("type"))
        kind = base.get("kind")
        if kind in ("SCALAR", "ENUM"):
            selections.append(field_expr)
            continue
        if kind in ("OBJECT", "INTERFACE", "UNION"):
            sub_selection = _selection_for_type(
                session,
                url,
                base.get("name") or "",
                token,
                cookie,
                user_agent,
                timeout,
                cache,
                depth=depth,
                max_fields=max_fields,
            )
            if not sub_selection:
                sub_selection = "__typename"
            selections.append(f"{field_expr} {{ {sub_selection} }}")
            continue

    return " ".join(selections), variables, [f"${name}: {var_type}" for name, var_type in var_defs.items()]


def _build_field_query(field: Dict[str, Any], arg_values: Dict[str, Any]) -> Dict[str, Any]:
    field_name = field.get("name")
    args = field.get("args", [])
    vars_def = []
    args_expr = []
    variables: Dict[str, Any] = {}
    for arg in args:
        arg_name = arg.get("name")
        arg_type = arg.get("type")
        arg_type_str = _type_string(arg_type)
        vars_def.append(f"${arg_name}: {arg_type_str}")
        args_expr.append(f"{arg_name}: ${arg_name}")
        variables[arg_name] = arg_values[arg_name]

    base = _base_type(field.get("type"))
    needs_selection = base.get("kind") in ("OBJECT", "INTERFACE", "UNION")
    selection = " { __typename }" if needs_selection else ""

    vars_section = f"({', '.join(vars_def)})" if vars_def else ""
    args_section = f"({', '.join(args_expr)})" if args_expr else ""
    query = f"query FieldProbe{vars_section} {{ {field_name}{args_section}{selection} }}"
    return {"query": query, "variables": variables}


def _cloudflare_blocked(text: str) -> bool:
    lowered = text.lower()
    return "just a moment" in lowered or "cf_chl" in lowered or "cloudflare" in lowered


def _request(
    session: requests.Session,
    url: str,
    payload: Dict[str, Any],
    token: Optional[str],
    cookie: Optional[str],
    user_agent: str,
    timeout: int,
) -> Dict[str, Any]:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": user_agent,
        "Origin": "https://api.stratz.com",
        "Referer": "https://api.stratz.com/graphiql",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if cookie:
        headers["Cookie"] = cookie

    response = session.post(url, headers=headers, json=payload, timeout=timeout)
    text = response.text or ""
    return {
        "status_code": response.status_code,
        "ok": response.ok,
        "cloudflare_blocked": _cloudflare_blocked(text),
        "text_snippet": text[:1000],
        "json": response.json() if response.headers.get("content-type", "").startswith("application/json") else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Test STRATZ GraphQL API and save response")
    parser.add_argument("--url", default=DEFAULT_URL, help="GraphQL endpoint URL")
    parser.add_argument("--out", default="stratz_api_response.json", help="Output JSON file")
    parser.add_argument("--token", default="", help="API token (or set STRATZ_API_TOKEN)")
    parser.add_argument("--cookie", default="", help="Cookie header value (or set STRATZ_API_COOKIE)")
    parser.add_argument("--user-agent", default=DEFAULT_UA, help="User-Agent header")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout seconds")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of fields to test (0 = all)")
    parser.add_argument("--delay", type=float, default=0.0, help="Delay seconds between requests")
    args = parser.parse_args()

    dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(dotenv_path):
        with open(dotenv_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                os.environ.setdefault(key, value)

    token = args.token or os.getenv("STRATZ_API_TOKEN")
    cookie = args.cookie or os.getenv("STRATZ_API_COOKIE")

    results: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "url": args.url,
        "used_token": bool(token),
        "used_cookie": bool(cookie),
        "queries": [],
    }

    session = requests.Session()
    type_cache: Dict[str, Any] = {}
    match_selection, match_vars, match_var_defs = _build_match_selection_and_vars(
        session,
        args.url,
        token,
        cookie,
        args.user_agent,
        args.timeout,
        type_cache,
        MATCH_FIELD_ORDER,
        depth=1,
        max_fields=12,
    )
    if not match_selection:
        match_selection = "id"
    player_selection = _selection_for_type(
        session,
        args.url,
        "PlayerType",
        token,
        cookie,
        args.user_agent,
        args.timeout,
        type_cache,
        depth=1,
        max_fields=25,
    )
    if not player_selection:
        player_selection = "steamAccountId"
    match_var_defs = ["$id: Long!"] + match_var_defs
    match_vars = {"id": DEFAULT_ARG_VALUES.get("id"), **match_vars}
    probes = [
        {
            "name": "field_match",
            "payload": {
                "query": f"query MatchProbe({', '.join(match_var_defs)}) {{ match(id: $id) {{ {match_selection} }} }}",
                "variables": match_vars,
            },
            "required": ["id"],
        },
        {
            "name": "field_player",
            "payload": {
                "query": f"query PlayerProbe($steamAccountId: Long!) {{ player(steamAccountId: $steamAccountId) {{ {player_selection} }} }}",
                "variables": {"steamAccountId": DEFAULT_ARG_VALUES.get("steamAccountId")},
            },
            "required": ["steamAccountId"],
        },
    ]
    if args.limit > 0:
        probes = probes[: args.limit]

    for probe in probes:
        missing_required = [key for key in probe["required"] if probe["payload"]["variables"].get(key) is None]
        if missing_required:
            results["queries"].append({
                "name": probe["name"],
                "payload": None,
                "response": {
                    "skipped": True,
                    "reason": f"Missing required args: {', '.join(missing_required)}",
                },
            })
            continue

        payload = probe["payload"]
        results["queries"].append({
            "name": probe["name"],
            "payload": payload,
            "response": _request(
                session,
                args.url,
                payload,
                token,
                cookie,
                args.user_agent,
                args.timeout,
            ),
        })
        if args.delay > 0:
            import time

            time.sleep(args.delay)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Saved response to {args.out}")
    if any(
        (entry.get("response") or {}).get("cloudflare_blocked")
        for entry in results["queries"]
    ):
        print("Cloudflare challenge detected. Use a browser cookie (cf_clearance) or approved network.", file=sys.stderr)
        return 1
    if any(
        (entry.get("response") or {}).get("ok") is False
        for entry in results["queries"]
    ):
        print("One or more queries failed. Check token or endpoint.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
