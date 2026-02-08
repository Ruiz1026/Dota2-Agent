# dota2_fastmcp.py
"""
Dota 2 MCP Server - FastMCP 版本
使用 FastMCP 简化 MCP Server 的实现
"""

import os
import re
import json
import base64
import html
from html.parser import HTMLParser
import requests
import time
from io import BytesIO
from difflib import SequenceMatcher
from typing import Optional, Dict, Any, List, Tuple
from functools import lru_cache

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import math

from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 创建 FastMCP Server ====================

mcp = FastMCP("Dota2 Assistant")

# 读取 .env（用于 SerpApi 等第三方配置）
_ENV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv(_ENV_PATH, override=False)

# ==================== 配置 ====================

BASE_URL = "https://api.opendota.com/api"
TIMEOUT = 30

# 地图配置
MAPS_DIR = "maps"
MAP_VERSION = "740"  # 统一使用的地图版本
REGION_TEMPLATE_PATH = os.path.join("api_samples", f"ward_region_template_{MAP_VERSION}.json")
ITEMS_CONSTANTS_PATH = os.path.join("api_samples", "constants_items.json")
ITEMS_MAP_PATH = os.path.join("api_samples", "constants_items_map.json")

# OpenDota constants 资源列表（支持的资源名）
CONSTANT_RESOURCES = [
    "abilities",
    "ability_ids",
    "aghs_desc",
    "ancients",
    "chat_wheel",
    "cluster",
    "countries",
    "game_mode",
    "hero_abilities",
    "hero_lore",
    "heroes",
    "item_colors",
    "item_ids",
    "items",
    "lobby_type",
    "neutral_abilities",
    "order_types",
    "patch",
    "patchnotes",
    "permanent_buffs",
    "player_colors",
    "region",
    "skillshots",
    "xp_level",
]

# SerpApi 搜索
SERPAPI_ENDPOINT = "https://serpapi.com/search.json"

# 眼位分析输出目录
WARD_OUTPUT_DIR = "ward_analysis"

# 战队常用英雄分析输出目录
HERO_REPORT_DIR = "hero_analysis"

# 多场比赛详情可视化输出目录
MATCH_REPORT_DIR = "match_analysis"

# 英雄文本介绍目录（本地知识库）
HEROES_TXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "heroes_txt"))

# 英雄中文名映射
HERO_CN_NAMES = {
    "Anti-Mage": "敌法师", "Axe": "斧王", "Bane": "祸乱之源",
    "Bloodseeker": "血魔", "Crystal Maiden": "水晶室女", "Drow Ranger": "卓尔游侠",
    "Earthshaker": "撼地者", "Juggernaut": "主宰", "Mirana": "米拉娜",
    "Morphling": "变体精灵", "Shadow Fiend": "影魔", "Phantom Lancer": "幻影长矛手",
    "Puck": "帕克", "Pudge": "帕吉", "Razor": "电魂",
    "Sand King": "沙王", "Storm Spirit": "风暴之灵", "Sven": "斯温",
    "Tiny": "小小", "Vengeful Spirit": "复仇之魂", "Windranger": "风行者",
    "Zeus": "宙斯", "Kunkka": "昆卡", "Lina": "莉娜",
    "Lion": "莱恩", "Shadow Shaman": "暗影萨满", "Slardar": "斯拉达",
    "Tidehunter": "潮汐猎人", "Witch Doctor": "巫医", "Lich": "巫妖",
    "Riki": "力丸", "Enigma": "谜团", "Tinker": "修补匠",
    "Sniper": "狙击手", "Necrophos": "瘟疫法师", "Warlock": "术士",
    "Beastmaster": "兽王", "Queen of Pain": "痛苦女王", "Venomancer": "剧毒术士",
    "Faceless Void": "虚空假面", "Wraith King": "冥魂大帝", "Death Prophet": "死亡先知",
    "Phantom Assassin": "幻影刺客", "Pugna": "帕格纳", "Templar Assassin": "圣堂刺客",
    "Viper": "冥界亚龙", "Luna": "露娜", "Dragon Knight": "龙骑士",
    "Dazzle": "戴泽", "Clockwerk": "发条技师", "Leshrac": "拉席克",
    "Nature's Prophet": "先知", "Lifestealer": "噬魂鬼", "Dark Seer": "黑暗贤者",
    "Clinkz": "克林克兹", "Omniknight": "全能骑士", "Enchantress": "魅惑魔女",
    "Huskar": "哈斯卡", "Night Stalker": "暗夜魔王", "Broodmother": "育母蜘蛛",
    "Bounty Hunter": "赏金猎人", "Weaver": "编织者", "Jakiro": "杰奇洛",
    "Batrider": "蝙蝠骑士", "Chen": "陈", "Spectre": "幽鬼",
    "Ancient Apparition": "远古冰魄", "Doom": "末日使者", "Ursa": "熊战士",
    "Spirit Breaker": "裂魂人", "Gyrocopter": "矮人直升机", "Alchemist": "炼金术士",
    "Invoker": "祈求者", "Silencer": "沉默术士", "Outworld Destroyer": "殁境神蚀者",
    "Lycan": "狼人", "Brewmaster": "酒仙", "Shadow Demon": "暗影恶魔",
    "Lone Druid": "德鲁伊", "Chaos Knight": "混沌骑士", "Meepo": "米波",
    "Treant Protector": "树精卫士", "Ogre Magi": "食人魔魔法师", "Undying": "不朽尸王",
    "Rubick": "拉比克", "Disruptor": "干扰者", "Nyx Assassin": "司夜刺客",
    "Naga Siren": "娜迦海妖", "Keeper of the Light": "光之守卫", "Io": "艾欧",
    "Visage": "维萨吉", "Slark": "斯拉克", "Medusa": "美杜莎",
    "Troll Warlord": "巨魔战将", "Centaur Warrunner": "半人马战行者", "Magnus": "马格纳斯",
    "Timbersaw": "伐木机", "Bristleback": "钢背兽", "Tusk": "巨牙海民",
    "Skywrath Mage": "天怒法师", "Abaddon": "亚巴顿", "Elder Titan": "上古巨神",
    "Legion Commander": "军团指挥官", "Techies": "工程师", "Ember Spirit": "灰烬之灵",
    "Earth Spirit": "大地之灵", "Underlord": "孽主", "Terrorblade": "恐怖利刃",
    "Phoenix": "凤凰", "Oracle": "神谕者", "Winter Wyvern": "寒冬飞龙",
    "Arc Warden": "天穹守望者", "Monkey King": "齐天大圣", "Dark Willow": "邪影芳灵",
    "Pangolier": "石鳞剑士", "Grimstroke": "天涯墨客", "Hoodwink": "森海飞霞",
    "Void Spirit": "虚无之灵", "Snapfire": "电炎绝手", "Mars": "玛尔斯",
    "Dawnbreaker": "破晓辰星", "Marci": "玛西", "Primal Beast": "獸",
    "Muerta": "琼英碧灵", "Ringmaster": "百戏大王", "Kez": "凯", "Largo": "郎戈",
}

RANK_TIER_MAP: Dict[int, Tuple[str, str]] = {
    1: ("Herald", "先锋"),
    2: ("Guardian", "卫士"),
    3: ("Crusader", "中军"),
    4: ("Archon", "统帅"),
    5: ("Legend", "传奇"),
    6: ("Ancient", "万古"),
    7: ("Divine", "超凡"),
    8: ("Immortal", "冠绝"),
}


# ==================== 辅助函数 ====================

def _make_request(endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
    """发起 API 请求"""
    url = f"{BASE_URL}/{endpoint}"
    try:
        response = requests.get(url, params=params, timeout=TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def _make_post_request(endpoint: str, payload: Optional[Dict] = None) -> Dict[str, Any]:
    """发起 API POST 请求"""
    url = f"{BASE_URL}/{endpoint}"
    try:
        response = requests.post(url, json=payload, timeout=TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def _truncate_text(text: str, max_len: int = 160) -> str:
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._chunks: List[str] = []
        self._skip_depth = 0
        self._block_tags = {
            "p", "br", "div", "li", "tr", "section", "article",
            "header", "footer", "h1", "h2", "h3", "h4", "h5", "h6", "pre",
        }
        self._skip_tags = {"script", "style", "noscript"}

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag in self._skip_tags:
            self._skip_depth += 1
            return
        if tag in self._block_tags:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._skip_tags:
            if self._skip_depth > 0:
                self._skip_depth -= 1
            return
        if tag in self._block_tags:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and data:
            self._chunks.append(data)

    def get_text(self) -> str:
        return "".join(self._chunks)


def _extract_text_from_html(html_text: str) -> str:
    if not html_text:
        return ""
    parser = _HTMLTextExtractor()
    try:
        parser.feed(html_text)
        parser.close()
    except Exception:
        return ""
    text = html.unescape(parser.get_text() or "")
    text = text.replace("\r", "")
    text = re.sub(r"[ \t]+", " ", text)
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def _fetch_fulltext(url: str, max_chars: int = 8000) -> Tuple[Optional[str], Optional[str], bool]:
    if not url or not isinstance(url, str):
        return None, "invalid url", False
    if not url.startswith("http"):
        return None, "unsupported url scheme", False
    try:
        response = requests.get(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            },
            timeout=TIMEOUT,
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        return None, str(exc), False

    content_type = response.headers.get("Content-Type", "")
    if content_type and "html" not in content_type.lower():
        # 尝试解析，但标记为非 HTML 内容
        pass

    text = _extract_text_from_html(response.text)
    if not text:
        return None, "no text extracted", False

    truncated = False
    if max_chars and max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars].rstrip() + "\n...[truncated]"
        truncated = True

    return text, None, truncated


@lru_cache(maxsize=1)
def _get_heroes_cached() -> tuple:
    """获取英雄列表（带缓存，返回 tuple 以支持缓存）"""
    result = _make_request("heroes")
    return tuple(result) if isinstance(result, list) else ()


def _build_hero_map() -> Dict[int, str]:
    """构建英雄ID到名称的映射"""
    heroes = _get_heroes_cached()
    return {h["id"]: h.get("localized_name", f"Hero {h['id']}") for h in heroes}


def _load_constants_resource(resource: str) -> Tuple[Optional[Any], str, str, Optional[str]]:
    output_dir = "api_samples"
    filename = f"constants_{resource}.json"
    output_path = os.path.join(output_dir, filename)

    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data, "local", output_path, None
        except Exception as exc:
            return None, "local", output_path, f"读取失败: {exc}"

    data = _make_request(f"constants/{resource}")
    if isinstance(data, dict) and "error" in data:
        return None, "api", output_path, data.get("error")

    try:
        os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    return data, "api", output_path, None

@lru_cache(maxsize=1)
def _load_items_map() -> Dict[int, Dict[str, str]]:
    """加载物品常量表，返回 item_id -> {key, name, qual}"""
    data: Optional[Dict[str, Any]] = None
    if os.path.exists(ITEMS_CONSTANTS_PATH):
        try:
            with open(ITEMS_CONSTANTS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = None
    if not data:
        data = _make_request("constants/items")
        if isinstance(data, dict) and "error" in data:
            data = None
    if not isinstance(data, dict):
        return {}

    item_map: Dict[int, Dict[str, str]] = {}
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
        item_map[item_id_int] = {
            "key": str(key),
            "name": str(name),
            "qual": info.get("qual"),
        }
    return item_map


@lru_cache(maxsize=1)
def _load_item_id_map_file() -> Dict[str, Dict[str, str]]:
    """加载本地物品ID映射文件，返回 by_id 字典"""
    if not os.path.exists(ITEMS_MAP_PATH):
        return {}
    try:
        with open(ITEMS_MAP_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    by_id = data.get("by_id") if isinstance(data, dict) else None
    if not isinstance(by_id, dict):
        return {}
    return {str(k): v for k, v in by_id.items() if isinstance(v, dict)}


def _build_item_entry(item_id: Any, item_map: Dict[int, Dict[str, str]]) -> Optional[Dict[str, Any]]:
    """构建单个物品信息（为空时返回 None）"""
    if item_id is None:
        return None
    try:
        item_id_int = int(item_id)
    except (TypeError, ValueError):
        return None
    if item_id_int <= 0:
        return None
    info = item_map.get(item_id_int)
    if info:
        return {"id": item_id_int, "key": info.get("key"), "name": info.get("name")}
    return {"id": item_id_int, "key": None, "name": None}


def _get_cn_name(en_name: str) -> str:
    """获取英雄中文名"""
    return HERO_CN_NAMES.get(en_name, en_name)


def _format_rank_tier(rank_tier: Any) -> Optional[str]:
    if rank_tier is None:
        return None
    try:
        rank_float = float(rank_tier)
        rank_int = int(round(rank_float))
    except (TypeError, ValueError):
        return str(rank_tier)
    if rank_int <= 0:
        return None
    tier = rank_int // 10
    star = rank_int % 10
    if tier >= 8:
        en, cn = RANK_TIER_MAP.get(8, ("Immortal", "冠绝"))
        return f"{en}（{cn}）"
    entry = RANK_TIER_MAP.get(tier)
    if not entry:
        return str(rank_tier)
    en, cn = entry
    if star <= 0:
        return f"{en}（{cn}）"
    return f"{en} {star}（{cn}{star}星）"


def _format_rank_bin(bin_name: Any, bin_id: Any = None) -> str:
    if bin_id is not None:
        try:
            bin_int = int(bin_id)
        except (TypeError, ValueError):
            bin_int = None
        if bin_int in RANK_TIER_MAP:
            en, cn = RANK_TIER_MAP[bin_int]
            return f"{en}（{cn}）"
    name_str = str(bin_name) if bin_name is not None else "N/A"
    for en, cn in RANK_TIER_MAP.values():
        if en.lower() in name_str.lower():
            return f"{en}（{cn}）"
    return name_str



def _normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = text.replace("_", " ").replace("-", " ").replace("'", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    text = _normalize_text(text)
    tokens = re.findall(r"[a-z0-9]+", text)
    tokens.extend(re.findall(r"[\u4e00-\u9fff]", text))
    return tokens


def _build_tfidf_vector(tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
    if not tokens or not idf:
        return {}
    tf: Dict[str, int] = {}
    for tok in tokens:
        tf[tok] = tf.get(tok, 0) + 1
    vector: Dict[str, float] = {}
    for tok, count in tf.items():
        weight = idf.get(tok)
        if weight is None:
            continue
        vector[tok] = (1.0 + math.log(count)) * weight
    return vector


def _build_tfidf_dense(
    tokens: List[str],
    vocab: Dict[str, int],
    idf: Dict[str, float],
    dim: int,
) -> "np.ndarray":
    if not tokens or not vocab or dim <= 0:
        return np.zeros(dim, dtype=np.float32)
    tf: Dict[str, int] = {}
    for tok in tokens:
        tf[tok] = tf.get(tok, 0) + 1
    vector = np.zeros(dim, dtype=np.float32)
    for tok, count in tf.items():
        idx = vocab.get(tok)
        if idx is None:
            continue
        weight = idf.get(tok)
        if weight is None:
            continue
        vector[idx] = (1.0 + math.log(count)) * float(weight)
    return vector


def _cosine_similarity(
    vector_a: Dict[str, float],
    norm_a: float,
    vector_b: Dict[str, float],
    norm_b: float,
) -> float:
    if not vector_a or not vector_b:
        return 0.0
    if norm_a <= 0 or norm_b <= 0:
        return 0.0
    if len(vector_a) > len(vector_b):
        vector_a, vector_b = vector_b, vector_a
    dot = 0.0
    for tok, val in vector_a.items():
        other = vector_b.get(tok)
        if other is not None:
            dot += val * other
    return dot / (norm_a * norm_b)


def _parse_hero_names_from_text(text: str) -> Tuple[Optional[str], Optional[str]]:
    if not text:
        return None, None
    first_line = text.strip().splitlines()[0].strip()
    match = re.search(r"英雄名称[:：]\s*([A-Za-z0-9' \-]+?)\s+([\u4e00-\u9fff]+)", first_line)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    match = re.search(r"英雄名称[:：]\s*([^\n]+)", text)
    if match:
        parts = match.group(1).strip().split()
        if len(parts) >= 2:
            return " ".join(parts[:-1]).strip(), parts[-1].strip()
    return None, None


@lru_cache(maxsize=1)
def _load_hero_documents() -> List[Dict[str, Any]]:
    if not os.path.isdir(HEROES_TXT_DIR):
        return []
    docs: List[Dict[str, Any]] = []
    for filename in sorted(os.listdir(HEROES_TXT_DIR)):
        if not filename.endswith(".txt"):
            continue
        path = os.path.join(HEROES_TXT_DIR, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            continue
        name_en, name_cn = _parse_hero_names_from_text(content)
        if not name_en:
            name_en = os.path.splitext(filename)[0].replace("_", " ")
        if not name_cn and name_en:
            name_cn = HERO_CN_NAMES.get(name_en)
        name_keys = set()
        for candidate in (name_en, name_cn, os.path.splitext(filename)[0], filename.replace(".txt", "")):
            if candidate:
                name_keys.add(_normalize_text(str(candidate)))
        name_token_list = _tokenize(f"{name_en or ''} {name_cn or ''}")
        content_token_list = _tokenize(content)
        name_tokens = set(name_token_list)
        content_tokens = set(content_token_list)
        all_token_list = name_token_list + content_token_list
        docs.append({
            "name_en": name_en or "",
            "name_cn": name_cn or "",
            "path": path,
            "content": content,
            "name_keys": name_keys,
            "name_tokens": name_tokens,
            "content_tokens": content_tokens,
            "name_token_list": name_token_list,
            "content_token_list": content_token_list,
            "all_token_list": all_token_list,
        })
    return docs


@lru_cache(maxsize=1)
def _load_hero_vector_index() -> Dict[str, Any]:
    docs = _load_hero_documents()
    if not docs:
        return {
            "docs": [],
            "idf": {},
            "vocab": {},
            "doc_vectors": None,
            "faiss_index": None,
            "dim": 0,
        }

    doc_count = len(docs)
    df: Dict[str, int] = {}
    for doc in docs:
        for tok in set(doc.get("all_token_list") or []):
            df[tok] = df.get(tok, 0) + 1
    idf = {
        tok: (math.log((doc_count + 1) / (count + 1)) + 1.0)
        for tok, count in df.items()
    }

    vocab_tokens = sorted(idf.keys())
    vocab = {tok: idx for idx, tok in enumerate(vocab_tokens)}
    dim = len(vocab_tokens)
    doc_vectors = np.zeros((doc_count, dim), dtype=np.float32) if dim > 0 else None

    for i, doc in enumerate(docs):
        vector = _build_tfidf_dense(doc.get("all_token_list") or [], vocab, idf, dim)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        if doc_vectors is not None:
            doc_vectors[i] = vector
        doc["vector"] = vector

    faiss_index = None
    if HAS_FAISS and doc_vectors is not None and dim > 0:
        faiss_index = faiss.IndexFlatIP(dim)
        faiss_index.add(doc_vectors)

    return {
        "docs": docs,
        "idf": idf,
        "vocab": vocab,
        "doc_vectors": doc_vectors,
        "faiss_index": faiss_index,
        "dim": dim,
    }


def _rank_hero_documents(query: str, top_k: int) -> List[Dict[str, Any]]:
    index = _load_hero_vector_index()
    docs = index.get("docs") or []
    idf = index.get("idf") or {}
    vocab = index.get("vocab") or {}
    doc_vectors = index.get("doc_vectors")
    faiss_index = index.get("faiss_index")
    dim = int(index.get("dim") or 0)
    if not docs:
        return []
    query_norm = _normalize_text(query)
    query_tokens = _tokenize(query)
    query_vector = _build_tfidf_dense(query_tokens, vocab, idf, dim)
    query_norm_value = float(np.linalg.norm(query_vector)) if dim > 0 else 0.0
    if query_norm_value > 0:
        query_vector = query_vector / query_norm_value

    vector_scores: Dict[int, float] = {}
    if faiss_index is not None and query_norm_value > 0:
        vector_k = max(10, top_k * 5)
        vector_k = min(len(docs), vector_k)
        distances, indices = faiss_index.search(
            query_vector.reshape(1, -1).astype(np.float32),
            vector_k,
        )
        for idx, score in zip(indices[0], distances[0]):
            if idx < 0:
                continue
            vector_scores[int(idx)] = float(score)

    keyword_candidates: Dict[int, int] = {}
    for idx, doc in enumerate(docs):
        name_score = 0
        content_score = 0
        if query_norm:
            for key in doc["name_keys"]:
                if not key:
                    continue
                if query_norm == key:
                    name_score += 200
                elif query_norm in key or key in query_norm:
                    name_score += 120
        if doc["name_cn"] and doc["name_cn"] in query:
            name_score += 200
        if doc["name_en"] and _normalize_text(doc["name_en"]) in query_norm:
            name_score += 120
        for token in query_tokens:
            if token in doc["name_tokens"]:
                name_score += 15
            elif token in doc["content_tokens"]:
                content_score += 1

        score = name_score + content_score
        if name_score > 0 or content_score >= 4:
            keyword_candidates[idx] = score

    candidate_ids = set(keyword_candidates.keys()) | set(vector_scores.keys())

    if not candidate_ids and query_norm and docs:
        for idx, doc in enumerate(docs):
            best_ratio = 0.0
            for key in doc["name_keys"]:
                if not key:
                    continue
                best_ratio = max(best_ratio, SequenceMatcher(None, query_norm, key).ratio())
            if best_ratio >= 0.5:
                keyword_candidates[idx] = int(best_ratio * 100)
        candidate_ids = set(keyword_candidates.keys())

    if not candidate_ids:
        return []

    scored: List[Dict[str, Any]] = []
    for idx in candidate_ids:
        doc = docs[idx]
        keyword_score = keyword_candidates.get(idx, 0)
        vector_score = vector_scores.get(idx, 0.0)
        if vector_score == 0.0 and query_norm_value > 0 and doc_vectors is not None:
            vector_score = float(np.dot(query_vector, doc_vectors[idx]))
        scored.append({
            "doc": doc,
            "keyword_score": keyword_score,
            "vector_score": vector_score,
        })

    max_kw = max(item["keyword_score"] for item in scored) if scored else 0
    max_vec = max(item["vector_score"] for item in scored) if scored else 0.0
    kw_weight = 0.7
    vec_weight = 0.3
    for item in scored:
        kw_norm = (item["keyword_score"] / max_kw) if max_kw > 0 else 0.0
        vec_norm = (item["vector_score"] / max_vec) if max_vec > 0 else 0.0
        item["kw_norm"] = kw_norm
        item["vec_norm"] = vec_norm
        item["hybrid_score"] = kw_weight * kw_norm + vec_weight * vec_norm

    scored.sort(key=lambda item: item["hybrid_score"], reverse=True)
    if top_k <= 0:
        top_k = 1
    return scored[: min(top_k, len(scored))]


def _format_hero_rag_output(query: str, results: List[Dict[str, Any]], max_chars: int) -> str:
    lines = ["# RAG 英雄介绍检索结果", f"query: {query}", ""]
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    for idx, item in enumerate(results, 1):
        doc = item["doc"]
        hybrid_score = item.get("hybrid_score", 0.0)
        kw_norm = item.get("kw_norm", 0.0)
        vec_norm = item.get("vec_norm", 0.0)
        name_en = doc.get("name_en") or "Unknown"
        name_cn = doc.get("name_cn") or ""
        display_name = f"{name_cn} ({name_en})" if name_cn else name_en
        source_rel = os.path.relpath(doc["path"], repo_root).replace("\\", "/")
        lines.append(f"## Top {idx}: {display_name}")
        lines.append(f"score: {hybrid_score:.3f} (kw={kw_norm:.3f}, vec={vec_norm:.3f})")
        lines.append(f"source: {source_rel}")
        lines.append("")
        content = doc.get("content", "").strip()
        if max_chars and max_chars > 0 and len(content) > max_chars:
            content = content[:max_chars].rstrip() + "\n...[truncated]"
        lines.append(content)
        lines.append("")
    return "\n".join(lines).strip()


def _format_time_mmss(seconds: int) -> str:
    """格式化时间为 M:SS（支持负数）"""
    sign = "-" if seconds < 0 else ""
    seconds = abs(int(seconds))
    minutes, secs = divmod(seconds, 60)
    return f"{sign}{minutes}:{secs:02d}"


@lru_cache(maxsize=1)
def _load_region_template() -> List[Dict[str, Any]]:
    """加载地图区域模板"""
    if not os.path.exists(REGION_TEMPLATE_PATH):
        return []
    try:
        with open(REGION_TEMPLATE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    regions = data.get("regions")
    if regions is not None:
        return regions
    merged_regions = []
    for side in ("radiant", "dire"):
        for key, info in data.get(side, {}).items():
            merged = dict(info)
            merged.setdefault("key", key)
            merged.setdefault("side", side)
            merged_regions.append(merged)
    return merged_regions


def _point_in_bbox(x: float, y: float, area: Dict[str, Any]) -> bool:
    return (
        x >= float(area.get("x_min", 0))
        and x <= float(area.get("x_max", 0))
        and y >= float(area.get("y_min", 0))
        and y <= float(area.get("y_max", 0))
    )


def _point_in_polygon(x: float, y: float, points: List[List[float]]) -> bool:
    inside = False
    if not points:
        return False
    j = len(points) - 1
    for i in range(len(points)):
        xi, yi = points[i]
        xj, yj = points[j]
        intersect = (yi > y) != (yj > y) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi
        )
        if intersect:
            inside = not inside
        j = i
    return inside


def _polygon_area(points: List[List[float]]) -> float:
    if len(points) < 3:
        return float("inf")
    area = 0.0
    j = len(points) - 1
    for i in range(len(points)):
        xi, yi = points[i]
        xj, yj = points[j]
        area += (xj + xi) * (yj - yi)
        j = i
    return abs(area) / 2.0


def _distance_point_to_segment(
    px: float,
    py: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> float:
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(px - x1, py - y1)
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.hypot(px - proj_x, py - proj_y)


def _distance_to_bbox(x: float, y: float, area: Dict[str, Any]) -> float:
    x_min = float(area.get("x_min", 0))
    x_max = float(area.get("x_max", 0))
    y_min = float(area.get("y_min", 0))
    y_max = float(area.get("y_max", 0))
    dx = 0.0
    if x < x_min:
        dx = x_min - x
    elif x > x_max:
        dx = x - x_max
    dy = 0.0
    if y < y_min:
        dy = y_min - y
    elif y > y_max:
        dy = y - y_max
    return math.hypot(dx, dy)


def _distance_to_polygon(x: float, y: float, points: List[List[float]]) -> float:
    if not points:
        return float("inf")
    if _point_in_polygon(x, y, points):
        return 0.0
    min_dist = float("inf")
    j = len(points) - 1
    for i in range(len(points)):
        x1, y1 = points[j]
        x2, y2 = points[i]
        min_dist = min(min_dist, _distance_point_to_segment(x, y, x1, y1, x2, y2))
        j = i
    return min_dist


def _match_region_with_distance(
    x: float,
    y: float,
    regions: List[Dict[str, Any]],
    allow_nearest: bool = True,
) -> Tuple[Optional[str], Optional[str], List[str], float, bool]:
    matches: List[Tuple[float, str, str]] = []
    for region in regions:
        label = str(region.get("label") or region.get("key") or "未知区域")
        key = str(region.get("key") or label)
        for area in region.get("areas", []):
            area_type = area.get("type")
            if area_type == "bbox":
                if _point_in_bbox(x, y, area):
                    size = abs(
                        (float(area.get("x_max", 0)) - float(area.get("x_min", 0)))
                        * (float(area.get("y_max", 0)) - float(area.get("y_min", 0)))
                    )
                    matches.append((size, key, label))
            elif area_type == "polygon":
                points = area.get("points") or []
                if _point_in_polygon(x, y, points):
                    size = _polygon_area(points)
                    matches.append((size, key, label))
    if matches:
        matches.sort(key=lambda item: item[0])
        primary_key = matches[0][1]
        primary_label = matches[0][2]
        labels = [item[2] for item in matches]
        return primary_key, primary_label, labels, 0.0, True
    if not allow_nearest:
        return None, None, [], float("inf"), False
    nearest_key = None
    nearest_label = None
    nearest_dist = float("inf")
    for region in regions:
        label = str(region.get("label") or region.get("key") or "未知区域")
        key = str(region.get("key") or label)
        for area in region.get("areas", []):
            area_type = area.get("type")
            if area_type == "bbox":
                dist = _distance_to_bbox(x, y, area)
            elif area_type == "polygon":
                points = area.get("points") or []
                dist = _distance_to_polygon(x, y, points)
            else:
                continue
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_key = key
                nearest_label = label
    if nearest_key is None:
        return None, None, [], float("inf"), False
    return nearest_key, nearest_label, [nearest_label], nearest_dist, False


def _gaussian_blur(data: np.ndarray, sigma: float) -> np.ndarray:
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


def _match_region(x: float, y: float, regions: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str], List[str]]:
    primary_key, primary_label, labels, _dist, _matched = _match_region_with_distance(
        x,
        y,
        regions,
        allow_nearest=True,
    )
    return primary_key, primary_label, labels


def _parse_tower_key(key: Optional[str]) -> Dict[str, Optional[str]]:
    info: Dict[str, Optional[str]] = {"team": None, "lane": None, "tier": None}
    if not key:
        return info
    match = re.match(r"npc_dota_(goodguys|badguys)_tower(\d)_(top|mid|bot)", key)
    if not match:
        return info
    side = "radiant" if match.group(1) == "goodguys" else "dire"
    info["team"] = side
    info["tier"] = match.group(2)
    info["lane"] = match.group(3)
    return info


def _build_ward_report_data(
    df_obs: pd.DataFrame,
    df_sen: pd.DataFrame,
    radiant_name: str,
    dire_name: str,
    match_duration: Optional[int],
    radiant_players: List[Dict[str, Any]],
    dire_players: List[Dict[str, Any]],
    objectives: Optional[List[Dict[str, Any]]] = None,
    tower_status: Optional[Dict[str, Optional[int]]] = None,
    kill_events: Optional[List[Dict[str, Any]]] = None,
    bucket_seconds: int = 300,
) -> Dict[str, Any]:
    """构建视野分析报告所需的结构化数据"""
    # 玩家映射（hero_id -> 玩家信息）
    player_by_hero_id: Dict[int, Dict[str, str]] = {}
    for p in radiant_players:
        hero_id = int(p.get("hero_id", 0))
        player_by_hero_id[hero_id] = {
            "team": "radiant",
            "team_name": radiant_name,
            "player": str(p.get("player", "Unknown")),
            "hero": str(p.get("hero", "Unknown")),
        }
    for p in dire_players:
        hero_id = int(p.get("hero_id", 0))
        player_by_hero_id[hero_id] = {
            "team": "dire",
            "team_name": dire_name,
            "player": str(p.get("player", "Unknown")),
            "hero": str(p.get("hero", "Unknown")),
        }

    # 统计玩家插眼贡献
    player_stats_map: Dict[str, Dict[str, Any]] = {}

    def _ensure_player_stat(hero_id: int, is_radiant: int) -> Dict[str, Any]:
        info = player_by_hero_id.get(hero_id)
        team_name = radiant_name if is_radiant == 1 else dire_name
        player_name = info.get("player") if info else "Unknown"
        hero_name = info.get("hero") if info else f"Hero {hero_id}"
        key = f"{team_name}:{hero_id}:{player_name}"
        if key not in player_stats_map:
            player_stats_map[key] = {
                "team": team_name,
                "player": player_name,
                "hero": hero_name,
                "obs": 0,
                "sen": 0,
                "total": 0,
                "first_time": None,
                "last_time": None,
            }
        return player_stats_map[key]

    if not df_obs.empty:
        for _, row in df_obs.iterrows():
            hero_id = int(row.get("hero_id", 0))
            is_radiant = int(row.get("is_radiant", 0))
            stat = _ensure_player_stat(hero_id, is_radiant)
            stat["obs"] += 1
            stat["total"] += 1
            t = int(row.get("time", 0))
            stat["first_time"] = t if stat["first_time"] is None else min(stat["first_time"], t)
            stat["last_time"] = t if stat["last_time"] is None else max(stat["last_time"], t)

    if not df_sen.empty:
        for _, row in df_sen.iterrows():
            hero_id = int(row.get("hero_id", 0))
            is_radiant = int(row.get("is_radiant", 0))
            stat = _ensure_player_stat(hero_id, is_radiant)
            stat["sen"] += 1
            stat["total"] += 1
            t = int(row.get("time", 0))
            stat["first_time"] = t if stat["first_time"] is None else min(stat["first_time"], t)
            stat["last_time"] = t if stat["last_time"] is None else max(stat["last_time"], t)

    player_stats = list(player_stats_map.values())
    for stat in player_stats:
        if stat["first_time"] is not None:
            stat["first_time"] = _format_time_mmss(stat["first_time"])
        if stat["last_time"] is not None:
            stat["last_time"] = _format_time_mmss(stat["last_time"])

    # 地图区域统计
    regions = _load_region_template()
    region_stats_map: Dict[str, Dict[str, Any]] = {}
    ward_regions: List[Dict[str, Any]] = []

    def _ensure_region_stat(label: str, key: Optional[str]) -> Dict[str, Any]:
        if label not in region_stats_map:
            region_stats_map[label] = {
                "key": key,
                "label": label,
                "obs_radiant": 0,
                "obs_dire": 0,
                "sen_radiant": 0,
                "sen_dire": 0,
                "total": 0,
            }
        return region_stats_map[label]

    def _collect_region_for_row(row: pd.Series, is_obs: bool) -> None:
        x = row.get("x")
        y = row.get("y")
        if x is None or y is None:
            return
        try:
            x_val = float(x)
            y_val = float(y)
        except (TypeError, ValueError):
            return

        primary_key, primary_label, labels = _match_region(x_val, y_val, regions)
        label = primary_label or "未知区域"
        key = primary_key

        is_radiant = int(row.get("is_radiant", 0)) == 1
        stat = _ensure_region_stat(label, key)
        if is_obs:
            if is_radiant:
                stat["obs_radiant"] += 1
            else:
                stat["obs_dire"] += 1
        else:
            if is_radiant:
                stat["sen_radiant"] += 1
            else:
                stat["sen_dire"] += 1
        stat["total"] += 1

        ward_regions.append({
            "x": x_val,
            "y": y_val,
            "time": int(row.get("time", 0)),
            "is_obs": is_obs,
            "team": "radiant" if is_radiant else "dire",
            "region": label,
            "region_key": key,
            "region_labels": labels,
        })

    if not df_obs.empty:
        for _, row in df_obs.iterrows():
            _collect_region_for_row(row, is_obs=True)
    if not df_sen.empty:
        for _, row in df_sen.iterrows():
            _collect_region_for_row(row, is_obs=False)

    region_summary = sorted(region_stats_map.values(), key=lambda x: x.get("total", 0), reverse=True)

    # 防御塔事件与击杀事件
    tower_events: List[Dict[str, Any]] = []
    tower_summary = {"radiant": 0, "dire": 0}
    for obj in objectives or []:
        if obj.get("type") != "building_kill":
            continue
        key = str(obj.get("key", ""))
        if "tower" not in key:
            continue
        info = _parse_tower_key(key)
        tower_team = info.get("team")
        if tower_team in tower_summary:
            tower_summary[tower_team] += 1
        tower_events.append({
            "time": int(obj.get("time", 0)),
            "key": key,
            "tower_team": tower_team,
            "lane": info.get("lane"),
            "tier": info.get("tier"),
            "unit": obj.get("unit"),
            "player_slot": obj.get("player_slot"),
            "slot": obj.get("slot"),
        })
    tower_events.sort(key=lambda x: x.get("time", 0))

    kill_events_list = list(kill_events or [])
    kill_events_list.sort(key=lambda x: x.get("time", 0))

    # 时间桶统计
    times = []
    if not df_obs.empty:
        times.extend(df_obs["time"].dropna().tolist())
    if not df_sen.empty:
        times.extend(df_sen["time"].dropna().tolist())

    time_buckets: List[Dict[str, Any]] = []
    if times:
        min_time = int(min(times))
        max_time = int(max(times))
        start_time = min_time if min_time < 0 else 0
        if match_duration and match_duration > 0:
            max_time = max(max_time, int(match_duration))

        bucket_start = int(np.floor(start_time / bucket_seconds)) * bucket_seconds
        bucket_end = int(np.floor(max_time / bucket_seconds)) * bucket_seconds

        for t in range(bucket_start, bucket_end + bucket_seconds, bucket_seconds):
            t_end = t + bucket_seconds
            obs_bucket = df_obs[(df_obs["time"] >= t) & (df_obs["time"] < t_end)] if not df_obs.empty else pd.DataFrame()
            sen_bucket = df_sen[(df_sen["time"] >= t) & (df_sen["time"] < t_end)] if not df_sen.empty else pd.DataFrame()

            rad_obs = len(obs_bucket[obs_bucket["is_radiant"] == 1]) if not obs_bucket.empty else 0
            dir_obs = len(obs_bucket[obs_bucket["is_radiant"] == 0]) if not obs_bucket.empty else 0
            rad_sen = len(sen_bucket[sen_bucket["is_radiant"] == 1]) if not sen_bucket.empty else 0
            dir_sen = len(sen_bucket[sen_bucket["is_radiant"] == 0]) if not sen_bucket.empty else 0
            total = rad_obs + dir_obs + rad_sen + dir_sen

            time_buckets.append({
                "start": t,
                "end": t_end,
                "label": f"{_format_time_mmss(t)}-{_format_time_mmss(t_end)}",
                "radiant_obs": rad_obs,
                "dire_obs": dir_obs,
                "radiant_sen": rad_sen,
                "dire_sen": dir_sen,
                "total": total,
            })

    # 队伍总计
    obs_rad = len(df_obs[df_obs["is_radiant"] == 1]) if not df_obs.empty else 0
    obs_dir = len(df_obs[df_obs["is_radiant"] == 0]) if not df_obs.empty else 0
    sen_rad = len(df_sen[df_sen["is_radiant"] == 1]) if not df_sen.empty else 0
    sen_dir = len(df_sen[df_sen["is_radiant"] == 0]) if not df_sen.empty else 0

    def _team_first_time(df: pd.DataFrame, is_radiant: int) -> Optional[str]:
        if df.empty:
            return None
        subset = df[df["is_radiant"] == is_radiant]
        if subset.empty:
            return None
        return _format_time_mmss(int(subset["time"].min()))

    first_time_radiant = _team_first_time(pd.concat([df_obs, df_sen], ignore_index=True), 1)
    first_time_dire = _team_first_time(pd.concat([df_obs, df_sen], ignore_index=True), 0)

    top_windows = sorted(time_buckets, key=lambda x: x.get("total", 0), reverse=True)[:3]

    match_id = None
    if not df_obs.empty and "match_id" in df_obs.columns:
        match_id = int(df_obs["match_id"].iloc[0])
    elif not df_sen.empty and "match_id" in df_sen.columns:
        match_id = int(df_sen["match_id"].iloc[0])

    return {
        "match_id": match_id,
        "duration": int(match_duration) if match_duration else None,
        "bucket_seconds": bucket_seconds,
        "radiant_name": radiant_name,
        "dire_name": dire_name,
        "ward_totals": {
            "radiant": {"obs": obs_rad, "sen": sen_rad, "total": obs_rad + sen_rad},
            "dire": {"obs": obs_dir, "sen": sen_dir, "total": obs_dir + sen_dir},
        },
        "first_ward_time": {
            "radiant": first_time_radiant,
            "dire": first_time_dire,
        },
        "region_template": REGION_TEMPLATE_PATH if regions else None,
        "region_summary": region_summary,
        "ward_regions": ward_regions,
        "tower_status": tower_status or {"radiant": None, "dire": None},
        "tower_summary": tower_summary,
        "tower_events": tower_events,
        "kill_events": kill_events_list,
        "time_buckets": time_buckets,
        "top_windows": top_windows,
        "player_stats": sorted(player_stats, key=lambda x: x.get("total", 0), reverse=True),
    }


def _build_multi_match_region_summary(
    obs_rows: List[Dict[str, Any]],
    sen_rows: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    regions = _load_region_template()
    region_stats_map: Dict[str, Dict[str, Any]] = {}

    def _ensure_region_stat(label: str, key: Optional[str]) -> Dict[str, Any]:
        if label not in region_stats_map:
            region_stats_map[label] = {
                "key": key,
                "label": label,
                "obs_radiant": 0,
                "obs_dire": 0,
                "sen_radiant": 0,
                "sen_dire": 0,
                "total": 0,
            }
        return region_stats_map[label]

    def _collect_row(row: Dict[str, Any], is_obs: bool) -> None:
        x = row.get("x")
        y = row.get("y")
        if x is None or y is None:
            return
        try:
            x_val = float(x)
            y_val = float(y)
        except (TypeError, ValueError):
            return

        primary_key, primary_label, _labels = _match_region(x_val, y_val, regions)
        label = primary_label or "未知区域"
        key = primary_key
        is_radiant = int(row.get("is_radiant", 0)) == 1

        stat = _ensure_region_stat(label, key)
        if is_obs:
            if is_radiant:
                stat["obs_radiant"] += 1
            else:
                stat["obs_dire"] += 1
        else:
            if is_radiant:
                stat["sen_radiant"] += 1
            else:
                stat["sen_dire"] += 1
        stat["total"] += 1

    for row in obs_rows:
        _collect_row(row, is_obs=True)
    for row in sen_rows:
        _collect_row(row, is_obs=False)

    region_summary = sorted(region_stats_map.values(), key=lambda x: x.get("total", 0), reverse=True)
    template_path = REGION_TEMPLATE_PATH if regions else None
    return region_summary, template_path


# ==================== 眼位分析辅助函数 ====================

def _get_map_path(version: str) -> Optional[str]:
    """获取地图文件路径"""
    map_file = os.path.join(MAPS_DIR, f"{version}.jpeg")
    if os.path.exists(map_file):
        return map_file
    
    # 尝试其他扩展名
    for ext in [".jpg", ".png"]:
        alt_file = os.path.join(MAPS_DIR, f"{version}{ext}")
        if os.path.exists(alt_file):
            return alt_file
    
    return None


class WardDataExtractor:
    """从比赛数据中提取眼位信息"""
    
    def __init__(self):
        self.obs_data = []  # 假眼数据
        self.sen_data = []  # 真眼数据
    
    def extract_from_match(self, match_data: Dict) -> bool:
        """从单场比赛数据中提取眼位"""
        if not match_data:
            return False
        
        match_id = match_data.get("match_id")
        start_time = match_data.get("start_time", 0)
        patch = match_data.get("patch", 0)
        
        # 获取地图版本
        map_version = MAP_VERSION
        
        # 检查是否有解析数据
        if not match_data.get("players"):
            return False
        
        # 提取目标事件时间
        objectives = match_data.get("objectives", [])
        obj_times = self._extract_objectives(match_id, objectives)
        
        # 从每个玩家提取眼位
        for player in match_data.get("players", []):
            hero_id = player.get("hero_id")
            player_slot = player.get("player_slot", 0)
            is_radiant = 1 if player_slot < 128 else 0

            obs_left_map: Dict[int, int] = {}
            for left_entry in player.get("obs_left_log", []) or []:
                ehandle = left_entry.get("ehandle")
                if ehandle is None:
                    continue
                left_time = int(left_entry.get("time", 0))
                prev_time = obs_left_map.get(ehandle)
                if prev_time is None or left_time < prev_time:
                    obs_left_map[ehandle] = left_time

            sen_left_map: Dict[int, int] = {}
            for left_entry in player.get("sen_left_log", []) or []:
                ehandle = left_entry.get("ehandle")
                if ehandle is None:
                    continue
                left_time = int(left_entry.get("time", 0))
                prev_time = sen_left_map.get(ehandle)
                if prev_time is None or left_time < prev_time:
                    sen_left_map[ehandle] = left_time
            
            # 提取假眼
            obs_log = player.get("obs_log", [])
            for ward in obs_log:
                ehandle = ward.get("ehandle")
                left_time = obs_left_map.get(ehandle) if ehandle is not None else None
                self.obs_data.append({
                    "match_id": match_id,
                    "start_time": start_time,
                    "patch": patch,
                    "map_version": map_version,
                    "hero_id": hero_id,
                    "is_radiant": is_radiant,
                    "time": ward.get("time", 0),
                    "x": ward.get("x", 0),
                    "y": ward.get("y", 0),
                    "z": ward.get("z", 0),
                    "ehandle": ehandle,
                    "left_time": left_time,
                    **obj_times,
                })
            
            # 提取真眼
            sen_log = player.get("sen_log", [])
            for ward in sen_log:
                ehandle = ward.get("ehandle")
                left_time = sen_left_map.get(ehandle) if ehandle is not None else None
                self.sen_data.append({
                    "match_id": match_id,
                    "start_time": start_time,
                    "patch": patch,
                    "map_version": map_version,
                    "hero_id": hero_id,
                    "is_radiant": is_radiant,
                    "time": ward.get("time", 0),
                    "x": ward.get("x", 0),
                    "y": ward.get("y", 0),
                    "z": ward.get("z", 0),
                    "ehandle": ehandle,
                    "left_time": left_time,
                    **obj_times,
                })
        
        obs_count = len([w for w in self.obs_data if w["match_id"] == match_id])
        sen_count = len([w for w in self.sen_data if w["match_id"] == match_id])
        
        return obs_count > 0 or sen_count > 0
    
    def get_dataframes(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """返回眼位数据的 DataFrame"""
        df_obs = pd.DataFrame(self.obs_data) if self.obs_data else pd.DataFrame()
        df_sen = pd.DataFrame(self.sen_data) if self.sen_data else pd.DataFrame()
        return df_obs, df_sen

    def _extract_objectives(self, match_id: int, objectives: List) -> Dict[str, Any]:
        """提取目标事件时间"""
        result: Dict[str, Any] = {"match_id": match_id}
        
        # 最大时间（用于未发生的事件）
        max_time = 3 * 60 * 60
        
        # 塔的列名
        towers = [
            "radiant_tower1_top", "radiant_tower2_top", "radiant_tower3_top",
            "radiant_tower1_mid", "radiant_tower2_mid", "radiant_tower3_mid",
            "radiant_tower1_bot", "radiant_tower2_bot", "radiant_tower3_bot",
            "dire_tower1_top", "dire_tower2_top", "dire_tower3_top",
            "dire_tower1_mid", "dire_tower2_mid", "dire_tower3_mid",
            "dire_tower1_bot", "dire_tower2_bot", "dire_tower3_bot",
        ]
        
        # 初始化所有塔为最大时间
        for tower in towers:
            result[tower] = max_time
        
        # 肉山击杀
        rosh_count = 0
        for i in range(4):
            result[f"ROSHAN_{i}"] = max_time
        
        # 解析目标事件
        for obj in objectives:
            obj_type = obj.get("type", "")
            key = obj.get("key", "")
            time = obj.get("time", max_time)
            
            if obj_type == "building_kill":
                col_name = key.replace("npc_dota_goodguys", "radiant")
                col_name = col_name.replace("npc_dota_badguys", "dire")
                if col_name in result:
                    result[col_name] = time
            elif obj_type == "CHAT_MESSAGE_ROSHAN_KILL":
                if rosh_count < 4:
                    result[f"ROSHAN_{rosh_count}"] = time
                    rosh_count += 1
        
        return result


class WardAnalyzer:
    """眼位分析和可视化"""
    
    def __init__(self, df_obs: pd.DataFrame, df_sen: pd.DataFrame,
                 radiant_name: str = "天辉 Radiant", dire_name: str = "夜魇 Dire",
                 match_duration: Optional[int] = None,
                 radiant_players: Optional[List[Dict[str, Any]]] = None,
                 dire_players: Optional[List[Dict[str, Any]]] = None):
        self.df_obs = df_obs.copy()
        self.df_sen = df_sen.copy()
        self.radiant_name = radiant_name
        self.dire_name = dire_name
        self.match_duration = match_duration
        self.radiant_players = radiant_players or []
        self.dire_players = dire_players or []
        
        # 坐标转换 (64,64) -> (0,0)
        if not self.df_obs.empty:
            self.df_obs["x"] = self.df_obs["x"] - 64
            self.df_obs["y"] = self.df_obs["y"] - 64
        
        if not self.df_sen.empty:
            self.df_sen["x"] = self.df_sen["x"] - 64
            self.df_sen["y"] = self.df_sen["y"] - 64
        
        # 加载地图图片
        self.map_image = None
        map_path = _get_map_path(MAP_VERSION)
        if map_path:
            try:
                self.map_image = Image.open(map_path)
            except Exception:
                pass
        
        # 加载眼位图标
        self.ward_icons = {}
        icon_dir = "figure"
        icon_files = {
            "obs_radiant": "goodguys_observer.png",
            "obs_dire": "badguys_observer.png",
            "sen_radiant": "goodguys_sentry.png",
            "sen_dire": "badguys_sentry.png",
        }
        for key, filename in icon_files.items():
            icon_path = os.path.join(icon_dir, filename)
            if os.path.exists(icon_path):
                try:
                    self.ward_icons[key] = plt.imread(icon_path)
                except Exception:
                    pass
        
        self.icon_zoom = 0.55
    
    def _add_ward_icon(self, ax, x: float, y: float, icon_key: str):
        """在指定位置添加眼位图标"""
        if icon_key in self.ward_icons:
            img = OffsetImage(self.ward_icons[icon_key], zoom=self.icon_zoom)
            ab = AnnotationBbox(img, (x, y), frameon=False)
            ax.add_artist(ab)
    
    def _create_icon_legend(self, ax, counts: dict):
        """创建带图标的自定义图例"""
        legend_items = []
        labels = []
        
        legend_config = [
            ("obs_radiant", f"{self.radiant_name} 假眼 ({{}})"),
            ("obs_dire", f"{self.dire_name} 假眼 ({{}})"),
            ("sen_radiant", f"{self.radiant_name} 真眼 ({{}})"),
            ("sen_dire", f"{self.dire_name} 真眼 ({{}})"),
        ]
        
        for icon_key, label_template in legend_config:
            count = counts.get(icon_key, 0)
            if icon_key in self.ward_icons:
                img = OffsetImage(self.ward_icons[icon_key], zoom=0.25)
                legend_items.append(img)
                labels.append(label_template.format(count))
        
        legend_y = 1.12
        legend_x_start = 0.1
        legend_spacing = 0.22
        
        for i, (item, label) in enumerate(zip(legend_items, labels)):
            x_pos = legend_x_start + i * legend_spacing
            ab = AnnotationBbox(item, (x_pos, legend_y), 
                              xycoords='axes fraction', frameon=False)
            ax.add_artist(ab)
            ax.text(x_pos + 0.03, legend_y, label, transform=ax.transAxes,
                   fontsize=9, verticalalignment='center')
    
    def generate_scatter_plot(self, save_path: str, figsize: Tuple = (12, 12)) -> bool:
        """生成眼位散点图"""
        try:
            fig, ax = plt.subplots(figsize=figsize)
            
            # 显示地图
            if self.map_image:
                ax.imshow(self.map_image, extent=[0, 128, 0, 128])
            else:
                ax.set_facecolor("gray")
            
            # 统计各类眼位数量
            counts = {"obs_radiant": 0, "obs_dire": 0, "sen_radiant": 0, "sen_dire": 0}
            
            # 绘制假眼
            if not self.df_obs.empty:
                obs_rad = self.df_obs[self.df_obs["is_radiant"] == 1]
                obs_dir = self.df_obs[self.df_obs["is_radiant"] == 0]
                counts["obs_radiant"] = len(obs_rad)
                counts["obs_dire"] = len(obs_dir)
                
                for _, row in obs_rad.iterrows():
                    self._add_ward_icon(ax, row["x"], row["y"], "obs_radiant")
                for _, row in obs_dir.iterrows():
                    self._add_ward_icon(ax, row["x"], row["y"], "obs_dire")
            
            # 绘制真眼
            if not self.df_sen.empty:
                sen_rad = self.df_sen[self.df_sen["is_radiant"] == 1]
                sen_dir = self.df_sen[self.df_sen["is_radiant"] == 0]
                counts["sen_radiant"] = len(sen_rad)
                counts["sen_dire"] = len(sen_dir)
                
                for _, row in sen_rad.iterrows():
                    self._add_ward_icon(ax, row["x"], row["y"], "sen_radiant")
                for _, row in sen_dir.iterrows():
                    self._add_ward_icon(ax, row["x"], row["y"], "sen_dire")
            
            ax.set_xlim(0, 128)
            ax.set_ylim(0, 128)
            
            # 获取比赛ID并添加队伍信息
            if not self.df_obs.empty and "match_id" in self.df_obs.columns:
                match_id = self.df_obs["match_id"].iloc[0]
                title = f"Dota 2 眼位分布图 - 比赛 {match_id}\n{self.radiant_name} vs {self.dire_name}"
            else:
                title = f"Dota 2 眼位分布图\n{self.radiant_name} vs {self.dire_name}"
            
            ax.set_title(title, pad=60)
            
            # 创建图例
            self._create_icon_legend(ax, counts)
            
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            
            return True
        except Exception as e:
            print(f"生成散点图失败: {e}")
            return False
    
    def generate_interactive_html(self, save_path: str, obs_duration: int = 360, 
                                   sen_duration: int = 420) -> bool:
        """生成交互式 HTML 页面"""
        try:
            # 将地图图片转为 base64
            map_base64 = ""
            if self.map_image:
                buffered = BytesIO()
                self.map_image.save(buffered, format="JPEG")
                map_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            heatmap_base64 = self._generate_heatmap_base64()
            if heatmap_base64:
                heatmap_html = (
                    "<div class=\"heatmap-container\">"
                    "<div class=\"heatmap-title\">视野热力图</div>"
                    f"<img src=\"data:image/png;base64,{heatmap_base64}\" class=\"heatmap-image\">"
                    "</div>"
                )
            else:
                heatmap_html = (
                    "<div class=\"heatmap-container\">"
                    "<div class=\"heatmap-title\">视野热力图</div>"
                    "<p class=\"placeholder\">热力图生成失败</p>"
                    "</div>"
                )
            
            # 将眼位图标转为 base64
            icon_base64 = {}
            icon_dir = "figure"
            icon_files = {
                "obs_radiant": "goodguys_observer.png",
                "obs_dire": "badguys_observer.png",
                "sen_radiant": "goodguys_sentry.png",
                "sen_dire": "badguys_sentry.png",
            }
            for key, filename in icon_files.items():
                icon_path = os.path.join(icon_dir, filename)
                if os.path.exists(icon_path):
                    with open(icon_path, "rb") as f:
                        icon_base64[key] = base64.b64encode(f.read()).decode()
            
            # 生成双方阵容信息
            def _format_roster(players: List[Dict[str, Any]]) -> str:
                if not players:
                    return '<li class="empty">暂无数据</li>'
                items = []
                for p in players:
                    hero_name = html.escape(str(p.get("hero", "Unknown")))
                    player_name = html.escape(str(p.get("player", "Unknown")))
                    items.append(f"<li><span class=\"hero\">{hero_name}</span><span class=\"player\">{player_name}</span></li>")
                return "\n".join(items)
            
            radiant_roster_html = _format_roster(self.radiant_players)
            dire_roster_html = _format_roster(self.dire_players)
            
            # 英雄ID -> 选手名映射（用于眼位提示）
            player_by_hero_id: Dict[int, str] = {}
            for p in self.radiant_players + self.dire_players:
                hero_id = p.get("hero_id")
                if hero_id is not None:
                    player_by_hero_id[int(hero_id)] = str(p.get("player", "Unknown"))
            
            # 准备眼位数据
            wards_data = []
            obs_range = 1600 / 128
            sen_range = 1000 / 128
            
            # 构建英雄ID到名称的映射
            hero_map = _build_hero_map()

            def _resolve_end_time(row: pd.Series, time_val: int, default_duration: int) -> int:
                end_time_val = None
                left_raw = row.get("left_time")
                if left_raw is not None and not pd.isna(left_raw):
                    try:
                        candidate = int(left_raw)
                    except (TypeError, ValueError):
                        candidate = None
                    if candidate is not None and candidate >= time_val:
                        end_time_val = candidate
                if end_time_val is None:
                    end_time_val = time_val + default_duration
                return end_time_val
            
            # 处理假眼数据
            if not self.df_obs.empty:
                for _, row in self.df_obs.iterrows():
                    ward_type = "obs_radiant" if row["is_radiant"] == 1 else "obs_dire"
                    hero_id = int(row.get("hero_id", 0))
                    hero_en = hero_map.get(hero_id, f"Hero {hero_id}")
                    hero_cn = _get_cn_name(hero_en)
                    team_name = self.radiant_name if row["is_radiant"] == 1 else self.dire_name
                    player_name = player_by_hero_id.get(hero_id, "Unknown")
                    time_val = int(row.get("time", 0))
                    end_time = _resolve_end_time(row, time_val, obs_duration)
                    wards_data.append({
                        "x": float(row["x"]),
                        "y": float(row["y"]),
                        "time": time_val,
                        "duration": obs_duration,
                        "end_time": end_time,
                        "type": ward_type,
                        "is_obs": True,
                        "range": obs_range,
                        "hero": hero_cn,
                        "team": team_name,
                        "player": player_name
                    })
            
            # 处理真眼数据
            if not self.df_sen.empty:
                for _, row in self.df_sen.iterrows():
                    ward_type = "sen_radiant" if row["is_radiant"] == 1 else "sen_dire"
                    hero_id = int(row.get("hero_id", 0))
                    hero_en = hero_map.get(hero_id, f"Hero {hero_id}")
                    hero_cn = _get_cn_name(hero_en)
                    team_name = self.radiant_name if row["is_radiant"] == 1 else self.dire_name
                    player_name = player_by_hero_id.get(hero_id, "Unknown")
                    time_val = int(row.get("time", 0))
                    end_time = _resolve_end_time(row, time_val, sen_duration)
                    wards_data.append({
                        "x": float(row["x"]),
                        "y": float(row["y"]),
                        "time": time_val,
                        "duration": sen_duration,
                        "end_time": end_time,
                        "type": ward_type,
                        "is_obs": False,
                        "range": sen_range,
                        "hero": hero_cn,
                        "team": team_name,
                        "player": player_name
                    })
            
            # 计算时间范围
            all_times = [w["time"] for w in wards_data]
            min_time = -90
            if self.match_duration and self.match_duration > 0:
                max_time = int(self.match_duration)
            else:
                max_time = max(all_times) + max(obs_duration, sen_duration) if all_times else 3600
            
            # 获取比赛ID
            match_id = ""
            if not self.df_obs.empty and "match_id" in self.df_obs.columns:
                match_id = str(self.df_obs["match_id"].iloc[0])

            # 选手筛选列表
            player_list = sorted({w.get("player", "Unknown") for w in wards_data})
            player_filter_html = "\n".join(
                f"<label class=\"filter-item\"><input type=\"checkbox\" name=\"playerFilter\" value=\"{html.escape(player)}\" checked> {html.escape(player)}</label>"
                for player in player_list
            )
            
            # 生成 HTML (简化版本，只包含核心功能)
            html_content = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dota 2 眼位时间线 - 比赛 {match_id}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Microsoft YaHei', Arial, sans-serif; background: #0b0b0b; min-height: 100vh; padding: 20px; color: #e6e6e6; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ text-align: center; margin-bottom: 10px; color: #f0f0f0; letter-spacing: 0.5px; }}
        .teams {{ text-align: center; margin-bottom: 10px; font-size: 18px; color: #d7c27a; font-weight: 600; }}
        .map-layout {{ display: flex; flex-direction: column; gap: 12px; align-items: stretch; }}
        .team-row {{ display: flex; gap: 12px; align-items: stretch; flex-wrap: nowrap; overflow-x: auto; }}
        .team-card {{ background: #121212; border-radius: 12px; padding: 10px 12px; border: 1px solid #1f1f1f; min-width: 0; flex: 1 1 0; }}
        .team-title {{ font-size: 13px; font-weight: 600; margin-bottom: 6px; color: #f0f0f0; }}
        .roster-list {{ list-style: none; font-size: 11px; }}
        .roster-list li {{ display: flex; justify-content: space-between; align-items: center; padding: 3px 0; border-bottom: 1px dashed rgba(255,255,255,0.08); gap: 10px; min-width: 0; }}
        .roster-list li:last-child {{ border-bottom: none; }}
        .roster-list .hero {{ color: #d7c27a; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
        .roster-list .player {{ color: #bdbdbd; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
        .roster-list .empty {{ color: #999; justify-content: center; }}
        .team-card.radiant {{ border-left: 4px solid #2e7d32; }}
        .team-card.dire {{ border-left: 4px solid #b23b3b; }}
        .team-card .badge {{ font-size: 12px; padding: 2px 8px; border-radius: 999px; background: #1a1a1a; color: #bdbdbd; margin-left: 6px; border: 1px solid #2a2a2a; }}
        @media (max-width: 980px) {{
            .team-row {{ gap: 8px; }}
        }}
        .map-container {{ position: relative; width: 100%; max-width: 800px; margin: 0 auto; border: 2px solid #2a2a2a; border-radius: 10px; overflow: hidden; box-shadow: 0 6px 18px rgba(0,0,0,0.45); }}
        .map-image {{ width: 100%; display: block; }}
        .ward {{ position: absolute; transform: translate(-50%, -50%); transition: opacity 0.2s ease; pointer-events: auto; z-index: 10; cursor: pointer; }}
        .ward img {{ width: 26px; height: 26px; }}
        .ward.hidden {{ opacity: 0; pointer-events: none; }}
        .ward-range {{ position: absolute; transform: translate(-50%, -50%); border-radius: 50%; pointer-events: none; z-index: 5; opacity: 0.95; }}
        .ward-range.range-obs {{ border: 1px dashed rgba(88, 166, 255, 0.75); background: rgba(88, 166, 255, 0.16); }}
        .ward-range.range-sen {{ border: 1px dashed rgba(255, 122, 122, 0.75); background: rgba(255, 122, 122, 0.16); }}
        .ward-range.hidden {{ opacity: 0; }}
        .tooltip {{ position: fixed; background: rgba(12,12,12,0.95); color: #f0f0f0; padding: 10px 14px; border-radius: 8px; font-size: 13px; z-index: 1000; pointer-events: none; box-shadow: 0 4px 12px rgba(0,0,0,0.6); border: 1px solid #2a2a2a; white-space: nowrap; }}
        .tooltip .hero {{ color: #d7c27a; font-weight: 600; font-size: 14px; }}
        .tooltip .player {{ color: #ddd; font-size: 12px; }}
        .tooltip .team {{ color: #aaa; font-size: 12px; }}
        .tooltip .time {{ color: #9ecbff; }}
        .tooltip .ward-type {{ color: #98c379; }}
        .controls {{ width: 100%; max-width: 800px; margin: 16px auto; background: #121212; padding: 10px 12px; border-radius: 10px; border: 1px solid #1f1f1f; }}
        .time-display {{ text-align: center; font-size: 18px; font-weight: 600; margin-bottom: 8px; color: #d7c27a; }}
        .slider-container {{ display: flex; align-items: center; gap: 8px; }}
        .slider {{ flex: 1; -webkit-appearance: none; height: 6px; border-radius: 6px; background: #2a2a2a; outline: none; cursor: pointer; }}
        .slider::-webkit-slider-thumb {{ -webkit-appearance: none; width: 14px; height: 14px; border-radius: 50%; background: #d7c27a; cursor: pointer; box-shadow: 0 2px 6px rgba(0,0,0,0.5); }}
        .time-label {{ font-size: 11px; color: #aaa; min-width: 46px; }}
        .stats {{ display: flex; justify-content: center; gap: 40px; margin-top: 15px; font-size: 14px; }}
        .stat-item {{ text-align: center; }}
        .stat-value {{ font-size: 24px; font-weight: 600; color: #d7c27a; }}
        .filters {{ max-width: 800px; margin: 16px auto 0; background: #121212; padding: 12px 16px; border-radius: 10px; border: 1px solid #1f1f1f; }}
        .filters-header {{ display: flex; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 8px; }}
        .filters-title {{ font-size: 14px; color: #f0f0f0; font-weight: 600; }}
        .filter-actions {{ display: flex; gap: 8px; }}
        .filter-button {{ background: #1a1a1a; color: #cfcfcf; border: 1px solid #2a2a2a; padding: 4px 10px; border-radius: 999px; font-size: 12px; cursor: pointer; }}
        .filter-button:hover {{ background: #242424; }}
        .filter-list {{ display: flex; flex-wrap: wrap; gap: 8px 12px; }}
        .filter-item {{ font-size: 12px; color: #ddd; display: flex; align-items: center; gap: 6px; }}
        .filter-item input {{ accent-color: #d7c27a; }}
        .report {{ margin-top: 22px; background: #121212; border: 1px solid #1f1f1f; border-radius: 12px; padding: 16px 18px; }}
        .report h2 {{ font-size: 18px; margin-bottom: 10px; color: #f6f6f6; }}
        .report .report-section {{ margin-bottom: 12px; }}
        .report .report-section:last-child {{ margin-bottom: 0; }}
        .report p {{ line-height: 1.6; color: #e0e0e0; }}
        .report .tag {{ display: inline-block; padding: 2px 8px; border-radius: 999px; background: #1a1a1a; font-size: 12px; color: #d7c27a; margin-right: 6px; border: 1px solid #2a2a2a; }}
        .report table {{ width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 13px; }}
        .report th, .report td {{ padding: 8px 6px; border-bottom: 1px solid rgba(255,255,255,0.08); text-align: left; }}
        .report th {{ color: #d7c27a; font-weight: 600; }}
        .report .placeholder {{ color: #aaa; }}
        .heatmap-container {{ max-width: 800px; margin: 14px auto 0; background: #121212; padding: 10px 12px; border-radius: 10px; border: 1px solid #1f1f1f; }}
        .heatmap-title {{ font-size: 13px; color: #f0f0f0; margin-bottom: 8px; }}
        .heatmap-image {{ width: 100%; display: block; border-radius: 8px; border: 1px solid #2a2a2a; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Dota 2 眼位时间线</h1>
        <div class="teams">🟢 {self.radiant_name} vs {self.dire_name} 🔴</div>
        <p style="text-align: center; margin-bottom: 15px; color: #aaa;">比赛 ID: {match_id}</p>
        
        <div class="map-layout">
            <div class="team-row">
                <div class="team-card radiant">
                    <div class="team-title">🟢 {self.radiant_name} 阵容<span class="badge">Radiant</span></div>
                    <ul class="roster-list">
                        {radiant_roster_html}
                    </ul>
                </div>
                <div class="team-card dire">
                    <div class="team-title">🔴 {self.dire_name} 阵容<span class="badge">Dire</span></div>
                    <ul class="roster-list">
                        {dire_roster_html}
                    </ul>
                </div>
            </div>
            <div class="map-container" id="mapContainer">
                <img src="data:image/jpeg;base64,{map_base64}" class="map-image" id="mapImage">
            </div>
            <div class="controls">
                <div class="time-display" id="timeDisplay">00:00</div>
                
                <div class="slider-container">
                    <span class="time-label" id="minTimeLabel">{min_time // 60}:{min_time % 60:02d}</span>
                    <input type="range" class="slider" id="timeSlider" min="{min_time}" max="{max_time}" value="{min_time}">
                    <span class="time-label" id="maxTimeLabel">{max_time // 60}:{max_time % 60:02d}</span>
                </div>
                
                <div class="stats">
                    <div class="stat-item">
                        <div class="stat-value" id="activeObs">0</div>
                        <div>当前假眼</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="activeSen">0</div>
                        <div>当前真眼</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="totalWards">{len(wards_data)}</div>
                        <div>总眼位数</div>
                    </div>
                </div>
            </div>
            {heatmap_html}
        </div>

        <div class="filters">
            <div class="filters-header">
                <div class="filters-title">按选手筛选眼位</div>
                <div class="filter-actions">
                    <button class="filter-button" id="selectAllPlayers">全选</button>
                    <button class="filter-button" id="clearAllPlayers">清空</button>
                </div>
            </div>
            <div class="filter-list" id="playerFilters">
                {player_filter_html}
            </div>
        </div>

        <section class="report" id="analysisReport">
            <h2>视野分析报告</h2>
            <div class="report-content" id="reportContent">
                <!-- WARD_REPORT_START -->
                <p class="placeholder">视野分析报告将在生成后显示。</p>
                <!-- WARD_REPORT_END -->
            </div>
        </section>
    </div>

    <script>
        const wardsData = {json.dumps(wards_data)};
        const MIN_TIME = {min_time};
        const icons = {{
            'obs_radiant': 'data:image/png;base64,{icon_base64.get("obs_radiant", "")}',
            'obs_dire': 'data:image/png;base64,{icon_base64.get("obs_dire", "")}',
            'sen_radiant': 'data:image/png;base64,{icon_base64.get("sen_radiant", "")}',
            'sen_dire': 'data:image/png;base64,{icon_base64.get("sen_dire", "")}'
        }};
        
        const mapContainer = document.getElementById('mapContainer');
        const timeSlider = document.getElementById('timeSlider');
        const timeDisplay = document.getElementById('timeDisplay');
        const activeObs = document.getElementById('activeObs');
        const activeSen = document.getElementById('activeSen');
        const selectAllPlayers = document.getElementById('selectAllPlayers');
        const clearAllPlayers = document.getElementById('clearAllPlayers');
        const playerFilterInputs = document.querySelectorAll('input[name="playerFilter"]');
        
        let wardElements = [];
        let rangeElements = [];
        
        // 创建 tooltip 元素
        const tooltip = document.createElement('div');
        tooltip.className = 'tooltip';
        tooltip.style.display = 'none';
        document.body.appendChild(tooltip);
        
        function createWardElements() {{
            wardsData.forEach((ward, index) => {{
                const xPercent = (ward.x / 128) * 100;
                const yPercent = (1 - ward.y / 128) * 100;
                const rangePercent = (ward.range / 128) * 100;
                
                const range = document.createElement('div');
                range.className = `ward-range ${{ward.is_obs ? 'range-obs' : 'range-sen'}} hidden`;
                range.style.left = xPercent + '%';
                range.style.top = yPercent + '%';
                range.style.width = (rangePercent * 2) + '%';
                range.style.height = (rangePercent * 2) + '%';
                mapContainer.appendChild(range);
                rangeElements.push(range);

                const div = document.createElement('div');
                div.className = 'ward hidden';
                div.dataset.index = index;
                
                const img = document.createElement('img');
                img.src = icons[ward.type];
                div.appendChild(img);
                
                div.style.left = xPercent + '%';
                div.style.top = yPercent + '%';
                
                // 添加鼠标悬停事件
                div.addEventListener('mouseenter', (e) => {{
                    const wardType = ward.is_obs ? '假眼 (Observer)' : '真眼 (Sentry)';
                    const timeStr = formatTime(ward.time);
                    tooltip.innerHTML = `
                        <div class="hero">${{ward.hero}}</div>
                        <div class="player">选手: ${{ward.player}}</div>
                        <div class="team">${{ward.team}}</div>
                        <div class="ward-type">${{wardType}}</div>
                        <div class="time">放置时间: ${{timeStr}}</div>
                    `;
                    tooltip.style.display = 'block';
                }});
                
                div.addEventListener('mousemove', (e) => {{
                    tooltip.style.left = (e.clientX + 15) + 'px';
                    tooltip.style.top = (e.clientY + 15) + 'px';
                }});
                
                div.addEventListener('mouseleave', () => {{
                    tooltip.style.display = 'none';
                }});
                
                mapContainer.appendChild(div);
                wardElements.push(div);
            }});
        }}
        
        function getSelectedPlayers() {{
            const selected = [];
            playerFilterInputs.forEach((input) => {{
                if (input.checked) {{
                    selected.push(input.value);
                }}
            }});
            return selected;
        }}

        function updateWards(currentTime) {{
            let obsCount = 0;
            let senCount = 0;
            const showAll = currentTime <= MIN_TIME;
            const selectedPlayers = getSelectedPlayers();
            const hasPlayerFilter = selectedPlayers.length > 0;
            const playerSet = new Set(selectedPlayers);
            
            wardsData.forEach((ward, index) => {{
                const endTime = (ward.end_time !== undefined && ward.end_time !== null)
                    ? ward.end_time
                    : ((ward.duration !== undefined && ward.duration !== null) ? ward.time + ward.duration : null);
                const isActive = showAll || (currentTime >= ward.time && (endTime === null || currentTime < endTime));
                const matchesPlayer = !hasPlayerFilter || playerSet.has(ward.player);
                
                if (isActive && matchesPlayer) {{
                    wardElements[index].classList.remove('hidden');
                    rangeElements[index].classList.remove('hidden');
                    if (ward.is_obs) obsCount++;
                    else senCount++;
                }} else {{
                    wardElements[index].classList.add('hidden');
                    rangeElements[index].classList.add('hidden');
                }}
            }});
            
            activeObs.textContent = obsCount;
            activeSen.textContent = senCount;
        }}
        
        function formatTime(seconds) {{
            const sign = seconds < 0 ? '-' : '';
            const absSeconds = Math.abs(seconds);
            const mins = Math.floor(absSeconds / 60);
            const secs = absSeconds % 60;
            return sign + mins + ':' + secs.toString().padStart(2, '0');
        }}
        
        timeSlider.addEventListener('input', function() {{
            const currentTime = parseInt(this.value);
            timeDisplay.textContent = currentTime <= MIN_TIME ? '全部' : formatTime(currentTime);
            updateWards(currentTime);
        }});


        playerFilterInputs.forEach((input) => {{
            input.addEventListener('change', () => {{
                updateWards(parseInt(timeSlider.value));
            }});
        }});

        selectAllPlayers.addEventListener('click', () => {{
            playerFilterInputs.forEach((input) => {{
                input.checked = true;
            }});
            updateWards(parseInt(timeSlider.value));
        }});

        clearAllPlayers.addEventListener('click', () => {{
            playerFilterInputs.forEach((input) => {{
                input.checked = false;
            }});
            updateWards(parseInt(timeSlider.value));
        }});
        
        createWardElements();
        updateWards(parseInt(timeSlider.value));
        timeDisplay.textContent = parseInt(timeSlider.value) <= MIN_TIME ? '全部' : formatTime(parseInt(timeSlider.value));
    </script>
</body>
</html>'''
            
            # 保存 HTML 文件
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return True
        except Exception as e:
            print(f"生成HTML失败: {e}")
            return False

    def _generate_heatmap_base64(
        self,
        sigma: float = 5.0,
        alpha: float = 0.65,
        ward_type: Optional[str] = None,
    ) -> str:
        """
        生成热力图并返回 base64 编码的 PNG。

        Args:
            sigma: 高斯模糊 sigma, 在 0-128 坐标系下的单位 (越大越平滑)
            alpha: 热力图最大透明度 0-1
            ward_type: "obs"/"sen" 用于筛选眼位类型，None 表示全部

        Returns:
            base64 编码的 PNG 字符串, 失败返回空字符串
        """
        if self.map_image is None:
            return ""

        width, height = self.map_image.size
        canvas = np.zeros((height, width), dtype=np.float32)
        point_weight = 1.0

        # 收集眼位坐标 (WardAnalyzer 已将坐标转为 0-128 范围)
        if ward_type not in (None, "obs", "sen"):
            return ""

        points = []
        if ward_type in (None, "obs") and not self.df_obs.empty:
            points.extend(self.df_obs[["x", "y"]].to_numpy())
        if ward_type in (None, "sen") and not self.df_sen.empty:
            points.extend(self.df_sen[["x", "y"]].to_numpy())
        if not points:
            return ""

        # 转换为像素坐标并累积
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

        # 高斯模糊 (sigma 转换到像素尺度)
        sigma_px = sigma * (width / 128.0)

        if HAS_OPENCV:
            kernel_size = int(round(sigma_px * 6))
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel_size = max(3, min(kernel_size, min(width, height) - 1))
            if kernel_size % 2 == 0:
                kernel_size -= 1
            blurred = cv2.GaussianBlur(canvas, (kernel_size, kernel_size), sigma_px)
        else:
            blurred = _gaussian_blur(canvas, sigma_px)

        # 归一化到 0-1
        max_val = blurred.max()
        if max_val > 0:
            blurred = blurred / max_val

        # 伪彩 (JET colormap)
        if HAS_OPENCV:
            heat_color = cv2.applyColorMap(
                (blurred * 255).astype(np.uint8),
                cv2.COLORMAP_JET,
            )
            heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
        else:
            cmap = plt.get_cmap("jet")
            heat_color = (cmap(blurred)[:, :, :3] * 255).astype(np.uint8)

        # 叠加: 只在有热度的地方叠加
        base_image = np.array(self.map_image.convert("RGB"))
        alpha_map = np.clip(blurred * alpha, 0, alpha)
        overlay = (
            base_image * (1 - alpha_map[..., None]) + heat_color * alpha_map[..., None]
        ).astype(np.uint8)

        try:
            output = Image.fromarray(overlay)
            buffered = BytesIO()
            output.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode()
        except Exception:
            return ""

    def _generate_ward_points_base64(self) -> str:
        """
        生成所有眼位点位图（叠加在地图上），返回 base64 编码 PNG。

        Returns:
            base64 编码的 PNG 字符串, 失败返回空字符串
        """
        if self.map_image is None:
            return ""

        try:
            base = self.map_image.convert("RGBA")
            width, height = base.size
            overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))

            icon_dir = "figure"
            icon_files = {
                "obs_radiant": "goodguys_observer.png",
                "obs_dire": "badguys_observer.png",
                "sen_radiant": "goodguys_sentry.png",
                "sen_dire": "badguys_sentry.png",
            }

            # 图标大小随地图大小调整（以 800px 宽度为基准）
            icon_size = max(16, int(round(width * 26 / 800)))

            icon_cache: Dict[str, Image.Image] = {}
            for key, filename in icon_files.items():
                icon_path = os.path.join(icon_dir, filename)
                if os.path.exists(icon_path):
                    try:
                        icon = Image.open(icon_path).convert("RGBA")
                        if icon_size > 0:
                            icon = icon.resize((icon_size, icon_size), Image.LANCZOS)
                        icon_cache[key] = icon
                    except Exception:
                        pass

            def _paste_icon(x_val: float, y_val: float, icon_key: str) -> None:
                icon = icon_cache.get(icon_key)
                if icon is None:
                    return
                x_val = float(np.clip(x_val, 0, 128))
                y_val = float(np.clip(y_val, 0, 128))
                px = x_val * (width - 1) / 128.0
                py = (128 - y_val) * (height - 1) / 128.0
                left = int(round(px - icon.width / 2))
                top = int(round(py - icon.height / 2))
                overlay.alpha_composite(icon, (left, top))

            if not self.df_obs.empty:
                for _, row in self.df_obs.iterrows():
                    x_val = row.get("x")
                    y_val = row.get("y")
                    if not (np.isfinite(x_val) and np.isfinite(y_val)):
                        continue
                    icon_key = "obs_radiant" if int(row.get("is_radiant", 0)) == 1 else "obs_dire"
                    _paste_icon(float(x_val), float(y_val), icon_key)

            if not self.df_sen.empty:
                for _, row in self.df_sen.iterrows():
                    x_val = row.get("x")
                    y_val = row.get("y")
                    if not (np.isfinite(x_val) and np.isfinite(y_val)):
                        continue
                    icon_key = "sen_radiant" if int(row.get("is_radiant", 0)) == 1 else "sen_dire"
                    _paste_icon(float(x_val), float(y_val), icon_key)

            combined = Image.alpha_composite(base, overlay).convert("RGB")
            buffered = BytesIO()
            combined.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode()
        except Exception:
            return ""
    
    def get_stats_summary(self) -> str:
        """获取统计摘要"""
        lines = ["# 📊 眼位数据统计\n"]
        
        # 显示比赛ID和队伍信息
        if not self.df_obs.empty and "match_id" in self.df_obs.columns:
            match_id = self.df_obs["match_id"].iloc[0]
            lines.append(f"🏆 比赛ID: {match_id}")
            lines.append(f"🟢 {self.radiant_name} vs {self.dire_name} 🔴\n")
        
        # 按队伍统计
        if not self.df_obs.empty:
            obs_rad = len(self.df_obs[self.df_obs["is_radiant"] == 1])
            obs_dir = len(self.df_obs[self.df_obs["is_radiant"] == 0])
            lines.append(f"\n假眼总计: {len(self.df_obs)}")
            lines.append(f"   {self.radiant_name}: {obs_rad} 个")
            lines.append(f"   {self.dire_name}: {obs_dir} 个")
        
        if not self.df_sen.empty:
            sen_rad = len(self.df_sen[self.df_sen["is_radiant"] == 1])
            sen_dir = len(self.df_sen[self.df_sen["is_radiant"] == 0])
            lines.append(f"\n真眼总计: {len(self.df_sen)}")
            lines.append(f"   {self.radiant_name}: {sen_rad} 个")
            lines.append(f"   {self.dire_name}: {sen_dir} 个")
        
        # 时间分布
        if not self.df_obs.empty:
            early_wards = len(self.df_obs[self.df_obs["time"] <= 600])
            mid_wards = len(self.df_obs[(self.df_obs["time"] > 600) & (self.df_obs["time"] <= 1800)])
            late_wards = len(self.df_obs[self.df_obs["time"] > 1800])
            
            lines.append(f"\n⏰ 眼位时间分布:")
            lines.append(f"   前10分钟: {early_wards} 个")
            lines.append(f"   10-30分钟: {mid_wards} 个")
            lines.append(f"   30分钟后: {late_wards} 个")
        
        return "\n".join(lines)


# ==================== MCP 工具定义 ====================

@mcp.tool()
def get_match_details(match_id: int) -> str:
    """
    获取 Dota 2 比赛信息摘要（matches/{match_id}）

    Args:
        match_id: Dota 2 比赛ID，例如 8650430843

    Returns:
        比赛摘要 + 双方阵容/数据概览（不包含原始 JSON）
    """
    data = _make_request(f"matches/{match_id}")

    if isinstance(data, dict) and "error" in data:
        return f"❌ API 错误: {data['error']}"

    if not isinstance(data, dict):
        return "❌ 获取比赛详情失败"

    hero_map = _build_hero_map()
    item_map = _load_items_map()

    def _count_bits(value: Any) -> Optional[int]:
        try:
            return bin(int(value)).count("1")
        except (TypeError, ValueError):
            return None

    def _format_items_from_ids(item_ids: List[Any]) -> str:
        names: List[str] = []
        for item_id in item_ids:
            entry = _build_item_entry(item_id, item_map)
            if entry:
                item_name = entry.get("name") or str(entry.get("id"))
            else:
                item_name = "-"
            names.append(str(item_name))
        return " / ".join(names)

    def _format_items(p: Dict[str, Any]) -> str:
        return _format_items_from_ids([p.get(f"item_{slot}") for slot in range(6)])

    def _format_backpack(p: Dict[str, Any]) -> str:
        backpack_ids = [p.get(f"backpack_{slot}") for slot in range(3)]
        if not any(backpack_ids):
            return "-"
        return _format_items_from_ids(backpack_ids)

    def _format_neutral(p: Dict[str, Any]) -> str:
        neutral_id = p.get("item_neutral") if p.get("item_neutral") is not None else p.get("item_neutral_id")
        if neutral_id is None:
            return "-"
        entry = _build_item_entry(neutral_id, item_map)
        if entry:
            return str(entry.get("name") or entry.get("id"))
        return str(neutral_id)

    def _lookup_const_name(resource: str, key: Any) -> str:
        if key is None:
            return "N/A"
        data, _, _, _ = _load_constants_resource(resource)
        key_str = str(key)
        if isinstance(data, dict):
            entry = data.get(key_str)
            if entry is None:
                try:
                    entry = data.get(int(key))
                except (TypeError, ValueError):
                    entry = None
            if isinstance(entry, dict):
                for field in ("name", "localized_name", "desc", "date"):
                    if entry.get(field):
                        return str(entry.get(field))
                if entry.get("id") is not None:
                    return str(entry.get("id"))
            if isinstance(entry, str):
                return entry
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                if str(item.get("id")) == key_str or str(item.get("patch")) == key_str:
                    for field in ("name", "localized_name", "desc", "date"):
                        if item.get(field):
                            return str(item.get(field))
        return str(key)

    def _skill_display(value: Any) -> str:
        mapping = {1: "Normal", 2: "High", 3: "Very High"}
        try:
            value_int = int(value)
        except (TypeError, ValueError):
            return "N/A" if value is None else str(value)
        return mapping.get(value_int, str(value_int))

    def _series_display(value: Any) -> str:
        mapping = {0: "Single Game", 1: "Bo1", 2: "Bo3", 3: "Bo5"}
        try:
            value_int = int(value)
        except (TypeError, ValueError):
            return "N/A" if value is None else str(value)
        return mapping.get(value_int, str(value_int))

    def _sum_int(players: List[Dict[str, Any]], field: str) -> int:
        total = 0
        for p in players:
            try:
                total += int(p.get(field) or 0)
            except (TypeError, ValueError):
                continue
        return total

    def _sum_float(players: List[Dict[str, Any]], field: str) -> float:
        total = 0.0
        for p in players:
            try:
                total += float(p.get(field) or 0)
            except (TypeError, ValueError):
                continue
        return total

    duration = int(data.get("duration") or 0)
    minutes, seconds = divmod(duration, 60)
    radiant_win = data.get("radiant_win")
    if radiant_win is True:
        winner = "天辉 (Radiant)"
    elif radiant_win is False:
        winner = "夜魇 (Dire)"
    else:
        winner = "未知"

    start_time = data.get("start_time")
    start_time_str = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(int(start_time))) if start_time else "N/A"
    first_blood_time = data.get("first_blood_time")
    first_blood_str = _format_time_mmss(int(first_blood_time)) if first_blood_time is not None else "N/A"

    game_mode = data.get("game_mode")
    lobby_type = data.get("lobby_type")
    region = data.get("region")
    patch = data.get("patch")
    skill = data.get("skill")

    game_mode_name = _lookup_const_name("game_mode", game_mode)
    lobby_name = _lookup_const_name("lobby_type", lobby_type)
    region_name = _lookup_const_name("region", region)
    patch_name = _lookup_const_name("patch", patch)

    radiant_team = data.get("radiant_team") or {}
    dire_team = data.get("dire_team") or {}
    league = data.get("league") or {}

    tower_radiant = data.get("tower_status_radiant")
    tower_dire = data.get("tower_status_dire")
    barracks_radiant = data.get("barracks_status_radiant")
    barracks_dire = data.get("barracks_status_dire")

    tower_radiant_alive = _count_bits(tower_radiant)
    tower_dire_alive = _count_bits(tower_dire)
    barracks_radiant_alive = _count_bits(barracks_radiant)
    barracks_dire_alive = _count_bits(barracks_dire)

    lines = [
        f"# 比赛详情 - Match ID: {data.get('match_id')}",
        "",
        "## 基本信息",
        f"- 时间: {start_time_str}",
        f"- 时长: {minutes}分{seconds}秒 ({duration}s)",
        f"- 获胜方: {winner}",
        f"- 比分: 天辉 {data.get('radiant_score', 0)} - {data.get('dire_score', 0)} 夜魇",
        f"- 首杀时间: {first_blood_str}",
        f"- 模式: {game_mode_name} ({game_mode})",
        f"- 房间: {lobby_name} ({lobby_type})",
        f"- 地区: {region_name} ({region})",
        f"- 段位: {_skill_display(skill)}",
        f"- 联赛: {league.get('name') or data.get('leagueid', 'N/A')} (ID: {data.get('leagueid', 'N/A')})",
        f"- 系列赛: {_series_display(data.get('series_type'))} (series_id: {data.get('series_id', 'N/A')})",
        f"- Patch: {patch_name} ({patch})",
        f"- Cluster: {data.get('cluster', 'N/A')}",
        f"- 人类玩家数: {data.get('human_players', 'N/A')}",
    ]

    if data.get("replay_url"):
        lines.append(f"- 回放: {data.get('replay_url')}")

    if radiant_team or dire_team:
        lines.append("")
        lines.append("## 队伍信息")
        if radiant_team:
            lines.append(f"- 天辉: {radiant_team.get('name', 'Unknown')} (ID: {radiant_team.get('team_id', 'N/A')})")
        if dire_team:
            lines.append(f"- 夜魇: {dire_team.get('name', 'Unknown')} (ID: {dire_team.get('team_id', 'N/A')})")

    if tower_radiant is not None or tower_dire is not None:
        lines.append("")
        lines.append("## 建筑状态")
        if tower_radiant is not None or tower_dire is not None:
            tr = f"{tower_radiant_alive}/11" if tower_radiant_alive is not None else "N/A"
            td = f"{tower_dire_alive}/11" if tower_dire_alive is not None else "N/A"
            lines.append(f"- 防御塔剩余: 天辉 {tr} (mask {tower_radiant}) / 夜魇 {td} (mask {tower_dire})")
        if barracks_radiant is not None or barracks_dire is not None:
            br = f"{barracks_radiant_alive}/6" if barracks_radiant_alive is not None else "N/A"
            bd = f"{barracks_dire_alive}/6" if barracks_dire_alive is not None else "N/A"
            lines.append(f"- 兵营剩余: 天辉 {br} (mask {barracks_radiant}) / 夜魇 {bd} (mask {barracks_dire})")

    gold_adv = data.get("radiant_gold_adv") or []
    xp_adv = data.get("radiant_xp_adv") or []
    if gold_adv or xp_adv:
        lines.append("")
        lines.append("## 经济/经验走势")
        if gold_adv:
            lines.append(
                f"- 经济优势(天辉视角): 最佳 {max(gold_adv)}, 最差 {min(gold_adv)}, 终局 {gold_adv[-1]}"
            )
        if xp_adv:
            lines.append(
                f"- 经验优势(天辉视角): 最佳 {max(xp_adv)}, 最差 {min(xp_adv)}, 终局 {xp_adv[-1]}"
            )

    players = data.get("players") or []
    if players:
        def _is_radiant(p: Dict[str, Any]) -> bool:
            if p.get("isRadiant") is not None:
                return bool(p.get("isRadiant"))
            return p.get("player_slot", 128) < 128

        radiant_players = [p for p in players if _is_radiant(p)]
        dire_players = [p for p in players if not _is_radiant(p)]

        def _team_summary(title: str, team_players: List[Dict[str, Any]]) -> List[str]:
            if not team_players:
                return []
            kills = _sum_int(team_players, "kills")
            deaths = _sum_int(team_players, "deaths")
            assists = _sum_int(team_players, "assists")
            net_worth = _sum_int(team_players, "net_worth")
            hero_damage = _sum_int(team_players, "hero_damage")
            tower_damage = _sum_int(team_players, "tower_damage")
            hero_healing = _sum_int(team_players, "hero_healing")
            gpm = _sum_int(team_players, "gold_per_min")
            xpm = _sum_int(team_players, "xp_per_min")
            last_hits = _sum_int(team_players, "last_hits")
            denies = _sum_int(team_players, "denies")
            obs = _sum_int(team_players, "obs_placed")
            sen = _sum_int(team_players, "sen_placed")
            stuns = _sum_float(team_players, "stuns")
            return [
                f"- {title}: K/D/A={kills}/{deaths}/{assists}, 净资产={net_worth}, GPM={gpm}, XPM={xpm}, LH/DN={last_hits}/{denies}",
                f"  伤害(英雄/塔/治疗)={hero_damage}/{tower_damage}/{hero_healing}, 视野(真/假)={obs}/{sen}, 控制时长={stuns:.1f}s",
            ]

        lines.append("")
        lines.append("## 阵营总览")
        lines.extend(_team_summary("天辉", radiant_players))
        lines.extend(_team_summary("夜魇", dire_players))

        def _append_player_table(title: str, team_players: List[Dict[str, Any]]) -> None:
            if not team_players:
                return
            lines.append("")
            lines.append(title)
            lines.append("| 英雄 | 选手 | 选手ID | K/D/A | 等级 | LH/DN | GPM/XPM | 净资产 | 伤害(英雄/塔/治疗) | 视野(真/假) | 位置 | 装备 | 背包 | 中立 |")
            lines.append("|------|------|--------|-------|------|------|---------|-------|------------------|-----------|------|------|------|------|")
            for p in team_players:
                hero_en = hero_map.get(p.get("hero_id"), f"Hero {p.get('hero_id')}")
                hero_cn = _get_cn_name(hero_en)
                player_name = p.get("name") or p.get("personaname") or "Unknown"
                player_id = p.get("account_id") if p.get("account_id") is not None else "Unknown"
                kda = f"{p.get('kills', 0)}/{p.get('deaths', 0)}/{p.get('assists', 0)}"
                level = p.get("level", 0)
                lh_dn = f"{p.get('last_hits', 0)}/{p.get('denies', 0)}"
                gpm_xpm = f"{p.get('gold_per_min', 0)}/{p.get('xp_per_min', 0)}"
                net_worth = p.get("net_worth", 0)
                damage_block = f"{p.get('hero_damage', 0)}/{p.get('tower_damage', 0)}/{p.get('hero_healing', 0)}"
                wards_block = f"{p.get('obs_placed', 0)}/{p.get('sen_placed', 0)}"
                lane = p.get("lane")
                lane_role = p.get("lane_role")
                lane_display = "-"
                if lane is not None or lane_role is not None:
                    lane_display = f"{lane if lane is not None else '-'} / {lane_role if lane_role is not None else '-'}"
                items_display = _format_items(p)
                backpack_display = _format_backpack(p)
                neutral_display = _format_neutral(p)
                lines.append(
                    f"| {hero_cn} | {player_name} | {player_id} | {kda} | {level} | {lh_dn} | {gpm_xpm} | {net_worth} | "
                    f"{damage_block} | {wards_block} | {lane_display} | {items_display} | {backpack_display} | {neutral_display} |"
                )

        _append_player_table("## 🟦 天辉阵容 (Radiant)", radiant_players)
        _append_player_table("## 🟥 夜魇阵容 (Dire)", dire_players)

    picks_bans = data.get("picks_bans") or []
    if picks_bans:
        lines.append("")
        lines.append("## BP 阶段")
        lines.append("| 顺序 | 选择 | 阵营 | 英雄 |")
        lines.append("|------|------|------|------|")
        for entry in picks_bans:
            hero_id = entry.get("hero_id")
            hero_en = hero_map.get(hero_id, f"Hero {hero_id}")
            hero_cn = _get_cn_name(hero_en)
            is_pick = entry.get("is_pick")
            pick_label = "Pick" if is_pick else "Ban"
            team_val = entry.get("team")
            if team_val == 0:
                team_label = "天辉"
            elif team_val == 1:
                team_label = "夜魇"
            else:
                team_label = str(team_val)
            lines.append(f"| {entry.get('order', '-')} | {pick_label} | {team_label} | {hero_cn} |")

    objectives = data.get("objectives") or []
    if objectives:
        lines.append("")
        lines.append("## 关键事件统计")
        counts: Dict[str, int] = {}
        for obj in objectives:
            obj_type = obj.get("type") or "unknown"
            counts[obj_type] = counts.get(obj_type, 0) + 1
        sorted_counts = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
        lines.append(", ".join([f"{k}: {v}" for k, v in sorted_counts[:15]]))

    return "\n".join(lines)
@mcp.tool()
def get_match_items(match_id: int) -> str:
    """
    提取比赛中所有玩家的购买记录（基于 purchase_log / purchase_time）

    Args:
        match_id: Dota 2 比赛ID，例如 8650430843

    Returns:
        所有玩家的购买记录(JSON)，仅保留非消耗品装备
    """
    match_data = _make_request(f"matches/{match_id}")

    if isinstance(match_data, dict) and "error" in match_data:
        return f"❌ API 错误: {match_data['error']}"

    players = match_data.get("players") if isinstance(match_data, dict) else None
    if not players:
        return f"❌ 无法获取比赛 {match_id} 的玩家数据"

    hero_map = _build_hero_map()
    item_map = _load_items_map()

    key_map: Dict[str, Dict[str, Any]] = {}
    consumable_keys = set()
    for item_id, info in item_map.items():
        key = info.get("key")
        if not key:
            continue
        entry = {
            "id": item_id,
            "key": key,
            "name": info.get("name"),
            "qual": info.get("qual"),
        }
        key_map[str(key)] = entry
        if str(info.get("qual")) == "consumable":
            consumable_keys.add(str(key))

    def _is_non_consumable(item_key: Any) -> bool:
        return str(item_key) not in consumable_keys

    def _build_purchase_entry(item_key: Any, time_val: Any) -> Dict[str, Any]:
        info = key_map.get(str(item_key))
        try:
            time_int = int(time_val)
        except (TypeError, ValueError):
            time_int = None
        return {
            "time": time_int if time_int is not None else time_val,
            "key": str(item_key),
            "id": info.get("id") if info else None,
            "name": info.get("name") if info else None,
        }

    player_items: List[Dict[str, Any]] = []
    for p in players:
        player_slot = p.get("player_slot", 0)
        is_radiant = p.get("isRadiant")
        if is_radiant is None:
            is_radiant = player_slot < 128 or p.get("is_radiant") == 1

        hero_id = int(p.get("hero_id", 0) or 0)
        hero_en = hero_map.get(hero_id, f"Hero {hero_id}")
        hero_cn = _get_cn_name(hero_en)
        player_name = p.get("name") or p.get("personaname") or p.get("account_id") or "Unknown"

        purchase_log_entries: List[Dict[str, Any]] = []
        for entry in p.get("purchase_log") or []:
            key = entry.get("key")
            if key is None:
                continue
            if not _is_non_consumable(key):
                continue
            purchase_log_entries.append(_build_purchase_entry(key, entry.get("time")))

        purchase_time_entries: List[Dict[str, Any]] = []
        for key, time_val in (p.get("purchase_time") or {}).items():
            if not _is_non_consumable(key):
                continue
            purchase_time_entries.append(_build_purchase_entry(key, time_val))
        purchase_time_entries.sort(
            key=lambda item: item["time"] if isinstance(item.get("time"), int) else 10**9
        )

        player_items.append({
            "player_slot": int(player_slot) if player_slot is not None else None,
            "team": "radiant" if is_radiant else "dire",
            "hero_id": hero_id,
            "hero": hero_cn,
            "hero_en": hero_en,
            "player": str(player_name),
            "purchases": {
                "purchase_log": purchase_log_entries,
                "purchase_time": purchase_time_entries,
            },
        })

    result = {
        "match_id": match_data.get("match_id"),
        "duration": match_data.get("duration"),
        "filter": {"exclude_qual": "consumable"},
        "players": player_items,
    }

    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def get_item_id_map(item_ids: List[int]) -> str:
    """
    查询装备 ID 对照表（基于本地 constants_items_map.json）

    Args:
        item_ids: 物品ID列表

    Returns:
        物品ID到名称/Key的映射(JSON)
    """
    if not item_ids:
        return "❌ item_ids 不能为空"

    by_id = _load_item_id_map_file()
    if not by_id:
        return f"❌ 未找到物品映射文件: {ITEMS_MAP_PATH}"

    items = []
    missing = []
    for item_id in item_ids:
        try:
            item_id_int = int(item_id)
        except (TypeError, ValueError):
            missing.append(item_id)
            continue
        info = by_id.get(str(item_id_int))
        if info:
            items.append({
                "id": item_id_int,
                "key": info.get("key"),
                "name": info.get("name"),
            })
        else:
            missing.append(item_id_int)

    result = {
        "source": ITEMS_MAP_PATH,
        "items": items,
        "missing": missing,
    }

    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def get_heroes() -> str:
    """
    获取所有 Dota 2 英雄的列表
    
    Returns:
        所有英雄的ID、英文名、中文名、主属性、攻击类型等信息
    """
    heroes = _get_heroes_cached()
    
    if not heroes:
        return "❌ 获取英雄列表失败"
    
    lines = [
        "# Dota 2 英雄列表",
        f"共 {len(heroes)} 个英雄",
        "",
        "| ID | 英文名 | 中文名 | 主属性 | 攻击类型 |",
        "|----|--------|--------|--------|----------|",
    ]
    
    for hero in heroes:
        en_name = hero.get("localized_name", "")
        cn_name = _get_cn_name(en_name)
        lines.append(
            f"| {hero.get('id')} | {en_name} | {cn_name} | "
            f"{hero.get('primary_attr', '')} | {hero.get('attack_type', '')} |"
        )
    
    return "\n".join(lines)


@mcp.tool()
def rag_hero_intro(query: str, top_k: int = 1, max_chars: int = 0) -> str:
    """
    使用本地 heroes_txt 知识库检索英雄介绍（RAG）

    Args:
        query: 英雄名称或包含英雄名的提问
        top_k: 返回的候选数量，默认1
        max_chars: 每条返回内容的最大字符数，默认4000
    """
    if not query or not str(query).strip():
        return "❌ query 不能为空"
    if not HAS_FAISS:
        return "❌ 未安装 faiss-cpu，请先安装依赖后重试。"

    try:
        top_k_int = int(top_k)
    except (TypeError, ValueError):
        top_k_int = 1
    top_k_int = max(1, min(5, top_k_int))

    try:
        max_chars_int = int(max_chars)
    except (TypeError, ValueError):
        max_chars_int = 0

    results = _rank_hero_documents(str(query), top_k_int)
    if not results:
        if not os.path.isdir(HEROES_TXT_DIR):
            return f"❌ 未找到 heroes_txt 目录: {HEROES_TXT_DIR}"
        return "❌ 未找到匹配英雄，请提供更准确的英雄名称（中英文均可）。"

    return _format_hero_rag_output(str(query), results, max_chars_int)


@mcp.tool()
def get_hero_matches(hero_id: int, limit: int = 20) -> str:
    """
    获取指定英雄的最近比赛记录

    Args:
        hero_id: 英雄ID
        limit: 返回的比赛数量，默认20

    Returns:
        比赛列表（对阵、阵营、胜负、时长等）
    """
    data = _make_request(f"heroes/{hero_id}/matches")

    if isinstance(data, dict) and "error" in data:
        return f"❌ API 错误: {data['error']}"

    if not isinstance(data, list):
        return "❌ 获取英雄比赛失败"

    hero_map = _build_hero_map()
    hero_en = hero_map.get(hero_id, f"Hero {hero_id}")
    hero_cn = _get_cn_name(hero_en)

    matches = data[:limit]
    lines = [
        f"# 🧭 {hero_cn} 最近比赛",
        "",
        "| Match ID | 对阵 | 阵营 | 结果 | 时长 |",
        "|----------|------|------|------|------|",
    ]

    for m in matches:
        radiant_name = m.get("radiant_name") or "Radiant"
        dire_name = m.get("dire_name") or "Dire"
        side_radiant = bool(m.get("radiant"))
        side = "天辉" if side_radiant else "夜魇"
        radiant_win = m.get("radiant_win")
        if radiant_win is None:
            result = "未知"
        else:
            hero_win = (radiant_win and side_radiant) or (not radiant_win and not side_radiant)
            result = "✅ 胜" if hero_win else "❌ 负"
        duration = int(m.get("duration", 0) or 0)
        minutes, seconds = divmod(duration, 60)

        lines.append(
            f"| {m.get('match_id')} | {radiant_name} vs {dire_name} | "
            f"{side} | {result} | {minutes}:{seconds:02d} |"
        )

    return "\n".join(lines)


@mcp.tool()
def get_hero_matchups(hero_id: int, limit: int = 20) -> str:
    """
    获取指定英雄对阵其他英雄的胜负情况

    Args:
        hero_id: 英雄ID
        limit: 返回的对阵数量，默认20

    Returns:
        对阵英雄列表（场次、胜场、胜率）
    """
    data = _make_request(f"heroes/{hero_id}/matchups")

    if isinstance(data, dict) and "error" in data:
        return f"❌ API 错误: {data['error']}"

    if not isinstance(data, list):
        return "❌ 获取英雄对阵数据失败"

    hero_map = _build_hero_map()
    hero_en = hero_map.get(hero_id, f"Hero {hero_id}")
    hero_cn = _get_cn_name(hero_en)

    matchups = data[:limit]
    lines = [
        f"# ⚔️ {hero_cn} 对阵英雄",
        "",
        "| 对手英雄 | 场次 | 胜场 | 胜率 |",
        "|----------|------|------|------|",
    ]

    for row in matchups:
        opp_id = int(row.get("hero_id", 0) or 0)
        opp_en = hero_map.get(opp_id, f"Hero {opp_id}")
        opp_cn = _get_cn_name(opp_en)
        games = row.get("games_played", 0)
        wins = row.get("wins", 0)
        win_rate = f"{(wins / games * 100):.1f}%" if games > 0 else "N/A"
        lines.append(f"| {opp_cn} | {games} | {wins} | {win_rate} |")

    return "\n".join(lines)


@mcp.tool()
def get_hero_durations(hero_id: int) -> str:
    """
    获取指定英雄在不同时长区间的表现

    Args:
        hero_id: 英雄ID

    Returns:
        不同比赛时长区间的场次与胜率
    """
    data = _make_request(f"heroes/{hero_id}/durations")

    if isinstance(data, dict) and "error" in data:
        return f"❌ API 错误: {data['error']}"

    if not isinstance(data, list):
        return "❌ 获取英雄时长分布失败"

    hero_map = _build_hero_map()
    hero_en = hero_map.get(hero_id, f"Hero {hero_id}")
    hero_cn = _get_cn_name(hero_en)

    lines = [
        f"# ⏱️ {hero_cn} 时长分布",
        "",
        "| 时长起点 | 场次 | 胜场 | 胜率 |",
        "|----------|------|------|------|",
    ]

    for row in data:
        duration_bin = row.get("duration_bin")
        try:
            seconds = int(float(duration_bin))
            minutes, secs = divmod(seconds, 60)
            bin_label = f"{minutes}:{secs:02d}"
        except (TypeError, ValueError):
            bin_label = str(duration_bin)
        games = row.get("games_played", 0)
        wins = row.get("wins", 0)
        win_rate = f"{(wins / games * 100):.1f}%" if games > 0 else "N/A"
        lines.append(f"| {bin_label} | {games} | {wins} | {win_rate} |")

    return "\n".join(lines)


@mcp.tool()
def get_hero_players(hero_id: int, limit: int = 20) -> str:
    """
    获取使用指定英雄的玩家列表

    Args:
        hero_id: 英雄ID
        limit: 返回的玩家数量，默认20

    Returns:
        玩家列表（昵称、战队、场次、胜率）
    """
    data = _make_request(f"heroes/{hero_id}/players")

    if isinstance(data, dict) and "error" in data:
        return f"❌ API 错误: {data['error']}"

    if not isinstance(data, list):
        return "❌ 获取英雄玩家列表失败"

    hero_map = _build_hero_map()
    hero_en = hero_map.get(hero_id, f"Hero {hero_id}")
    hero_cn = _get_cn_name(hero_en)

    players = data[:limit]
    lines = [
        f"# 👥 {hero_cn} 选手列表",
        "",
        "| 玩家 | 战队 | 场次 | 胜场 | 胜率 |",
        "|------|------|------|------|------|",
    ]

    for row in players:
        name = row.get("name") or row.get("personaname") or row.get("account_id") or "Unknown"
        team = row.get("team_name") or row.get("team_tag") or "N/A"
        games = row.get("games", 0)
        wins = row.get("wins", row.get("win", 0))
        win_rate = f"{(wins / games * 100):.1f}%" if games > 0 else "N/A"
        lines.append(f"| {str(name)[:16]} | {str(team)[:12]} | {games} | {wins} | {win_rate} |")

    return "\n".join(lines)


@mcp.tool()
def get_hero_item_popularity(hero_id: int, limit: int = 8) -> str:
    """
    获取指定英雄的出装流行度（按阶段）

    Args:
        hero_id: 英雄ID
        limit: 每个阶段展示的装备数量，默认8

    Returns:
        开局/前期/中期/后期的热门装备列表
    """
    data = _make_request(f"heroes/{hero_id}/itemPopularity")

    if isinstance(data, dict) and "error" in data:
        return f"❌ API 错误: {data['error']}"

    if not isinstance(data, dict):
        return "❌ 获取英雄出装流行度失败"

    hero_map = _build_hero_map()
    hero_en = hero_map.get(hero_id, f"Hero {hero_id}")
    hero_cn = _get_cn_name(hero_en)
    item_map = _load_items_map()

    def _resolve_item_name(item_key: Any) -> str:
        try:
            item_id = int(item_key)
        except (TypeError, ValueError):
            item_id = None
        if item_id and item_id in item_map:
            info = item_map[item_id]
            return info.get("name") or info.get("key") or str(item_key)
        return str(item_key)

    def _format_section(title: str, payload: Any) -> List[str]:
        if not isinstance(payload, dict) or not payload:
            return [f"## {title}", "", "暂无数据", ""]
        pairs = []
        for key, count in payload.items():
            try:
                count_val = int(count)
            except (TypeError, ValueError):
                count_val = 0
            pairs.append((_resolve_item_name(key), count_val))
        pairs.sort(key=lambda x: x[1], reverse=True)
        top_items = pairs[:limit]
        lines = [
            f"## {title}",
            "",
            "| 装备 | 次数 |",
            "|------|------|",
        ]
        for name, count_val in top_items:
            lines.append(f"| {name} | {count_val} |")
        lines.append("")
        return lines

    sections = [
        ("开局装备", data.get("start_game_items")),
        ("前期装备", data.get("early_game_items")),
        ("中期装备", data.get("mid_game_items")),
        ("后期装备", data.get("late_game_items")),
    ]

    lines = [f"# 🧰 {hero_cn} 出装流行度", ""]
    for title, payload in sections:
        lines.extend(_format_section(title, payload))

    return "\n".join(lines)


@mcp.tool()
def get_player_info(account_id: int) -> str:
    """
    获取指定玩家的基本信息
    
    Args:
        account_id: Steam 32位账号ID
    
    Returns:
        玩家的昵称、Steam ID、天梯段位等信息
    """
    data = _make_request(f"players/{account_id}")
    
    if "error" in data:
        return f"❌ API 错误: {data['error']}"
    
    profile = data.get("profile", {})
    
    lines = [
        "# 👤 玩家信息",
        "",
        f"- 昵称: {profile.get('personaname', 'Unknown')}",
        f"- Steam ID: {profile.get('steamid', 'N/A')}",
        f"- 账号 ID: {profile.get('account_id', 'N/A')}",
    ]
    
    if data.get("rank_tier") is not None:
        rank_tier = data.get("rank_tier")
        rank_text = _format_rank_tier(rank_tier)
        if rank_text:
            lines.append(f"- 天梯段位: {rank_text}（{rank_tier}）")
        else:
            lines.append(f"- 天梯段位: {rank_tier}")
    
    if data.get("leaderboard_rank"):
        lines.append(f"- 排行榜排名: {data.get('leaderboard_rank')}")
    
    if profile.get("profileurl"):
        lines.append(f"- Steam 主页: {profile.get('profileurl')}")
    
    return "\n".join(lines)


@mcp.tool()
def get_player_matches(account_id: int, limit: int = 50) -> str:
    """
    获取指定玩家最近的比赛记录
    
    Args:
        account_id: Steam 32位账号ID
        limit: 返回的比赛数量，默认10
    
    Returns:
        玩家最近的比赛列表，包括使用的英雄、KDA、胜负等
    """
    data = _make_request(f"players/{account_id}/recentMatches")
    
    if isinstance(data, dict) and "error" in data:
        return f"❌ API 错误: {data['error']}"
    
    if not isinstance(data, list):
        return "❌ 获取比赛记录失败"
    
    matches = data[:limit]
    hero_map = _build_hero_map()
    
    lines = [
        "# 📋 最近比赛记录",
        "",
        "| Match ID | 英雄 | K/D/A | 结果 | 时长 |",
        "|----------|------|-------|------|------|",
    ]
    
    for m in matches:
        hero_en = hero_map.get(m.get("hero_id"), f"Hero {m.get('hero_id')}")
        hero_cn = _get_cn_name(hero_en)
        kda = f"{m.get('kills', 0)}/{m.get('deaths', 0)}/{m.get('assists', 0)}"
        
        radiant_win = m.get("radiant_win")
        is_radiant = m.get("player_slot", 128) < 128
        won = (radiant_win and is_radiant) or (not radiant_win and not is_radiant)
        result = "✅ 胜" if won else "❌ 负"
        
        duration = m.get("duration", 0)
        minutes, seconds = divmod(duration, 60)
        
        lines.append(f"| {m.get('match_id')} | {hero_cn} | {kda} | {result} | {minutes}:{seconds:02d} |")
    
    return "\n".join(lines)


@mcp.tool()
def request_match_parse(match_id: int) -> str:
    """
    提交比赛录像解析请求

    Args:
        match_id: Dota 2 比赛ID

    Returns:
        解析请求结果（通常包含 jobId）
    """
    data = _make_post_request(f"request/{match_id}")
    if isinstance(data, dict) and "error" in data:
        return f"❌ API 错误: {data['error']}"
    return json.dumps(data, ensure_ascii=False, indent=2)


@mcp.tool()
def request_match_parses(match_ids: List[int]) -> str:
    """
    批量提交比赛录像解析请求

    Args:
        match_ids: 比赛ID列表

    Returns:
        每个比赛ID的解析请求结果列表
    """
    results = []
    for match_id in match_ids:
        data = _make_post_request(f"request/{match_id}")
        if isinstance(data, dict) and "error" in data:
            results.append({"match_id": match_id, "error": data["error"]})
        else:
            results.append({"match_id": match_id, "response": data})
    return json.dumps(results, ensure_ascii=False, indent=2)


@mcp.tool()
def get_parse_request(job_id: str) -> str:
    """
    查询解析请求状态

    Args:
        job_id: 解析请求的 jobId

    Returns:
        解析请求状态信息
    """
    data = _make_request(f"request/{job_id}")
    if isinstance(data, dict) and "error" in data:
        return f"❌ API 错误: {data['error']}"
    return json.dumps(data, ensure_ascii=False, indent=2)


@mcp.tool()
def get_hero_stats() -> str:
    """
    获取所有英雄的统计数据
    
    Returns:
        英雄的胜率、选取率、禁用率等统计数据（按选取率排序，显示前20）
    """
    data = _make_request("heroStats")
    
    if isinstance(data, dict) and "error" in data:
        return f"❌ API 错误: {data['error']}"
    
    if not isinstance(data, list):
        return "❌ 获取英雄统计失败"
    
    lines = [
        "# 📊 英雄统计数据 (职业赛事)",
        "",
        "| 英雄 | 选取数 | 胜率 | 禁用数 |",
        "|------|--------|------|--------|",
    ]
    
    sorted_stats = sorted(data, key=lambda x: x.get("pro_pick", 0), reverse=True)[:20]
    
    for s in sorted_stats:
        en_name = s.get("localized_name", "")
        cn_name = _get_cn_name(en_name)
        picks = s.get("pro_pick", 0)
        wins = s.get("pro_win", 0)
        bans = s.get("pro_ban", 0)
        win_rate = f"{(wins / picks * 100):.1f}%" if picks > 0 else "N/A"
        
        lines.append(f"| {cn_name} | {picks} | {win_rate} | {bans} |")
    
    return "\n".join(lines)


@mcp.tool()
def get_pro_matches(limit: int = 20) -> str:
    """
    获取最近的职业比赛列表
    
    Args:
        limit: 返回的比赛数量，默认20
    
    Returns:
        最近的职业比赛，包括联赛、队伍、获胜方等
    """
    data = _make_request("proMatches")
    
    if isinstance(data, dict) and "error" in data:
        return f"❌ API 错误: {data['error']}"
    
    if not isinstance(data, list):
        return "❌ 获取职业比赛失败"
    
    matches = data[:limit]
    
    lines = [
        "# 🏆 最近职业比赛",
        "",
        "| Match ID | 联赛 | 天辉 | 夜魇 | 获胜方 |",
        "|----------|------|------|------|--------|",
    ]
    
    for m in matches:
        winner = "🟢 天辉" if m.get("radiant_win") else "🔴 夜魇"
        lines.append(
            f"| {m.get('match_id')} | {m.get('league_name', 'N/A')[:20]} | "
            f"{m.get('radiant_name', 'Radiant')[:12]} | {m.get('dire_name', 'Dire')[:12]} | {winner} |"
        )
    
    return "\n".join(lines)


@mcp.tool()
def get_player_win_loss(account_id: int) -> str:
    """
    获取指定玩家的胜负统计
    
    Args:
        account_id: Steam 32位账号ID
    
    Returns:
        玩家的胜场、负场、总场次和胜率
    """
    data = _make_request(f"players/{account_id}/wl")
    
    if "error" in data:
        return f"❌ API 错误: {data['error']}"
    
    wins = data.get("win", 0)
    losses = data.get("lose", 0)
    total = wins + losses
    win_rate = f"{(wins / total * 100):.1f}%" if total > 0 else "N/A"
    
    return f"""# 📈 胜负统计

- ✅ 胜场: {wins}
- ❌ 负场: {losses}
- 📊 总场次: {total}
- 🎯 胜率: {win_rate}
"""


@mcp.tool()
def get_player_heroes(account_id: int, limit: int = 10) -> str:
    """
    获取指定玩家最常用的英雄列表
    
    Args:
        account_id: Steam 32位账号ID
        limit: 返回的英雄数量，默认10
    
    Returns:
        玩家最常用的英雄，包括场次、胜场和胜率
    """
    data = _make_request(f"players/{account_id}/heroes")
    
    if isinstance(data, dict) and "error" in data:
        return f"❌ API 错误: {data['error']}"
    
    if not isinstance(data, list):
        return "❌ 获取玩家英雄数据失败"
    
    heroes = data[:limit]
    hero_map = _build_hero_map()
    
    lines = [
        "# 🎮 常用英雄",
        "",
        "| 英雄 | 场次 | 胜场 | 胜率 |",
        "|------|------|------|------|",
    ]
    
    for h in heroes:
        hero_en = hero_map.get(int(h.get("hero_id", 0)), f"Hero {h.get('hero_id')}")
        hero_cn = _get_cn_name(hero_en)
        games = h.get("games", 0)
        wins = h.get("win", 0)
        win_rate = f"{(wins / games * 100):.1f}%" if games > 0 else "N/A"
        
        lines.append(f"| {hero_cn} | {games} | {wins} | {win_rate} |")
    
    return "\n".join(lines)


@mcp.tool()
def get_player_peers(account_id: int, limit: int = 10) -> str:
    """
    获取指定玩家最常一起游戏的队友
    
    Args:
        account_id: Steam 32位账号ID
        limit: 返回的队友数量，默认10
    
    Returns:
        玩家最常合作的队友列表，包括一起游戏的场次、胜场和胜率
    """
    data = _make_request(f"players/{account_id}/peers")
    
    if isinstance(data, dict) and "error" in data:
        return f"❌ API 错误: {data['error']}"
    
    if not isinstance(data, list):
        return "❌ 获取队友数据失败"
    
    peers = data[:limit]
    
    lines = [
        "# 👥 常合作队友",
        "",
        "| 昵称 | 一起场次 | 一起胜场 | 胜率 |",
        "|------|----------|----------|------|",
    ]
    
    for p in peers:
        name = p.get("personaname", "Unknown")[:15]
        games = p.get("with_games", 0)
        wins = p.get("with_win", 0)
        win_rate = f"{(wins / games * 100):.1f}%" if games > 0 else "N/A"
        
        lines.append(f"| {name} | {games} | {wins} | {win_rate} |")
    
    return "\n".join(lines)


@mcp.tool()
def get_player_totals(account_id: int) -> str:
    """
    获取指定玩家的统计总计数据
    
    Args:
        account_id: Steam 32位账号ID
    
    Returns:
        玩家的各项统计总计，如总击杀、总死亡、总助攻、总GPM等
    """
    data = _make_request(f"players/{account_id}/totals")
    
    if isinstance(data, dict) and "error" in data:
        return f"❌ API 错误: {data['error']}"
    
    if not isinstance(data, list):
        return "❌ 获取统计数据失败"
    
    # 字段中文映射
    field_names = {
        "kills": "击杀", "deaths": "死亡", "assists": "助攻",
        "gold_per_min": "GPM", "xp_per_min": "XPM", "last_hits": "正补",
        "denies": "反补", "hero_damage": "英雄伤害", "tower_damage": "建筑伤害",
        "hero_healing": "治疗量", "stuns": "眩晕时长(秒)",
    }
    
    lines = [
        "# 📊 玩家统计总计",
        "",
        "| 统计项 | 总计 | 场次 | 场均 |",
        "|--------|------|------|------|",
    ]
    
    for item in data:
        field = item.get("field", "")
        if field in field_names:
            cn_name = field_names[field]
            total = item.get("sum", 0)
            n = item.get("n", 0)
            avg = f"{total / n:.1f}" if n > 0 else "N/A"
            lines.append(f"| {cn_name} | {total:.0f} | {n} | {avg} |")
    
    return "\n".join(lines)


@mcp.tool()
def get_pro_players(limit: int = 20) -> str:
    """
    获取职业选手列表
    
    Args:
        limit: 返回的选手数量，默认20
    
    Returns:
        职业选手列表，包括ID、昵称、所属战队等
    """
    data = _make_request("proPlayers")
    
    if isinstance(data, dict) and "error" in data:
        return f"❌ API 错误: {data['error']}"
    
    if not isinstance(data, list):
        return "❌ 获取职业选手列表失败"
    
    players = data[:limit]
    
    lines = [
        "# 🎮 职业选手列表",
        "",
        "| 账号ID | 昵称 | 战队 | 国家 |",
        "|--------|------|------|------|",
    ]
    
    for p in players:
        account_id = p.get("account_id", "N/A")
        name = p.get("name", p.get("personaname", "Unknown"))[:15]
        team = p.get("team_name", "N/A")[:12]
        country = p.get("country_code", "N/A")
        
        lines.append(f"| {account_id} | {name} | {team} | {country} |")
    
    return "\n".join(lines)


@mcp.tool()
def get_public_matches(min_rank: int = 70, limit: int = 20) -> str:
    """
    获取最近的公开比赛列表
    
    Args:
        min_rank: 最低段位等级，默认70（神话），范围10-85
        limit: 返回的比赛数量，默认20
    
    Returns:
        公开比赛列表，包括比赛ID、段位、时长等
    """
    data = _make_request("publicMatches", params={"min_rank": min_rank})
    
    if isinstance(data, dict) and "error" in data:
        return f"❌ API 错误: {data['error']}"
    
    if not isinstance(data, list):
        return "❌ 获取公开比赛失败"
    
    matches = data[:limit]
    
    lines = [
        f"# 🎮 公开比赛 (段位 ≥ {min_rank})",
        "",
        "| Match ID | 段位 | 时长 | 天辉英雄 | 夜魇英雄 |",
        "|----------|------|------|----------|----------|",
    ]
    
    for m in matches:
        match_id = m.get("match_id", "N/A")
        rank = m.get("avg_rank_tier", "N/A")
        rank_text = _format_rank_tier(rank) if rank not in ("N/A", None, "") else None
        rank_display = f"{rank_text}（{rank}）" if rank_text else str(rank)
        duration = m.get("duration", 0)
        minutes = duration // 60
        
        radiant = m.get("radiant_team", "")
        dire = m.get("dire_team", "")
        
        lines.append(f"| {match_id} | {rank_display} | {minutes}分 | {radiant[:20]} | {dire[:20]} |")
    
    return "\n".join(lines)


@mcp.tool()
def get_teams(limit: int = 20) -> str:
    """
    获取战队列表（按评分排序）
    
    Args:
        limit: 返回的战队数量，默认20
    
    Returns:
        战队列表，包括战队名、标签、评分、胜负场次
    """
    data = _make_request("teams")
    
    if isinstance(data, dict) and "error" in data:
        return f"❌ API 错误: {data['error']}"
    
    if not isinstance(data, list):
        return "❌ 获取战队列表失败"
    
    teams = data[:limit]
    
    lines = [
        "# 🏆 战队列表 (按评分排序)",
        "",
        "| 战队ID | 名称 | 标签 | 评分 | 胜/负 |",
        "|--------|------|------|------|-------|",
    ]
    
    for t in teams:
        team_id = t.get("team_id", "N/A")
        name = t.get("name", "Unknown")[:15]
        tag = t.get("tag", "N/A")[:8]
        rating = f"{t.get('rating', 0):.0f}"
        wins = t.get("wins", 0)
        losses = t.get("losses", 0)
        
        lines.append(f"| {team_id} | {name} | {tag} | {rating} | {wins}/{losses} |")
    
    return "\n".join(lines)


@mcp.tool()
def get_team_info(team_id: int) -> str:
    """
    获取指定战队的详细信息
    
    Args:
        team_id: 战队ID
    
    Returns:
        战队详细信息，包括名称、评分、胜负场次等
    """
    data = _make_request(f"teams/{team_id}")
    
    if isinstance(data, dict) and "error" in data:
        return f"❌ API 错误: {data['error']}"
    
    if not data:
        return "❌ 获取战队信息失败"
    
    wins = data.get("wins", 0)
    losses = data.get("losses", 0)
    total = wins + losses
    win_rate = f"{(wins / total * 100):.1f}%" if total > 0 else "N/A"
    
    return f"""# 🏆 战队信息

- 名称: {data.get('name', 'Unknown')}
- 标签: {data.get('tag', 'N/A')}
- 战队ID: {data.get('team_id', 'N/A')}
- 评分: {data.get('rating', 'N/A')}
- 胜场: {wins}
- 负场: {losses}
- 胜率: {win_rate}
"""


@mcp.tool()
def get_team_matches(team_id: int, limit: int = 10) -> str:
    """
    获取指定战队的最近比赛列表
    
    Args:
        team_id: 战队ID
        limit: 返回的比赛数量，默认10
    
    Returns:
        战队最近比赛列表，包括对手、结果、联赛等
    """
    data = _make_request(f"teams/{team_id}/matches")
    
    if isinstance(data, dict) and "error" in data:
        return f"❌ API 错误: {data['error']}"
    
    if not isinstance(data, list):
        return "❌ 获取战队比赛失败"
    
    matches = data[:limit]
    
    hero_map = _build_hero_map()

    def _team_heroes_from_match(match_id: int) -> str:
        for attempt in range(2):
            match_data = _make_request(f"matches/{match_id}")
            if isinstance(match_data, dict) and "error" in match_data:
                if attempt == 0:
                    time.sleep(0.4)
                    continue
                return "未知"
            if not isinstance(match_data, dict):
                if attempt == 0:
                    time.sleep(0.4)
                    continue
                return "未知"
            players = match_data.get("players", [])
            if not players:
                if attempt == 0:
                    time.sleep(0.4)
                    continue
                return "未知"

            side = None
            if match_data.get("radiant_team_id") == team_id:
                side = "radiant"
            elif match_data.get("dire_team_id") == team_id:
                side = "dire"
            if not side:
                return "未知"

            hero_names = []
            for p in players:
                slot = p.get("player_slot", 128)
                is_radiant = slot < 128
                if (side == "radiant" and not is_radiant) or (side == "dire" and is_radiant):
                    continue
                hero_id = p.get("hero_id")
                if hero_id is None:
                    continue
                hero_en = hero_map.get(int(hero_id), f"Hero {hero_id}")
                hero_names.append(_get_cn_name(hero_en))

            if hero_names:
                return ", ".join(hero_names)
        return "未知"

    lines = [
        "# 🎮 战队最近比赛",
        "",
        "| Match ID | 对手 | 结果 | 时长 | 联赛 | 本队英雄 |",
        "|----------|------|------|------|------|----------|",
    ]
    
    for m in matches:
        match_id = m.get("match_id", "N/A")
        opponent = m.get("opposing_team_name", "Unknown")
        
        radiant_win = m.get("radiant_win")
        radiant = m.get("radiant", False)
        if radiant_win is not None:
            team_win = (radiant and radiant_win) or (not radiant and not radiant_win)
            result = "✅ 胜" if team_win else "❌ 负"
        else:
            result = "⏳"
        
        duration = m.get("duration", 0)
        minutes = duration // 60
        league = m.get("league_name", "N/A")
        
        team_heroes = _team_heroes_from_match(int(match_id)) if str(match_id).isdigit() else "未知"
        lines.append(f"| {match_id} | {opponent} | {result} | {minutes}分 | {league} | {team_heroes} |")
    
    return "\n".join(lines)


@mcp.tool()
def get_team_players(team_id: int) -> str:
    """
    获取战队选手列表

    Args:
        team_id: 战队ID

    Returns:
        战队选手列表（account_id、name、games_played、wins、is_current_team_member）
    """
    data = _make_request(f"teams/{team_id}/players")

    if isinstance(data, dict) and "error" in data:
        return f"❌ API 错误: {data['error']}"

    if not isinstance(data, list):
        return "❌ 获取战队选手失败"

    players = data

    lines = [
        "# 👥 战队选手列表",
        "",
        "| account_id | name | games_played | wins | is_current_team_member |",
        "|-----------|------|--------------|------|------------------------|",
    ]

    for p in players:
        account_id = p.get("account_id", "N/A")
        name = p.get("name")
        name_display = name if name is not None else "—"
        games = p.get("games_played", 0)
        wins = p.get("wins", 0)
        is_current = p.get("is_current_team_member")
        if is_current is True:
            is_current_display = "true"
        elif is_current is False:
            is_current_display = "false"
        else:
            is_current_display = "—"
        lines.append(
            f"| {account_id} | {name_display} | {games} | {wins} | {is_current_display} |"
        )

    return "\n".join(lines)


@mcp.tool()
def get_team_heroes(team_id: int, limit: int = 20) -> str:
    """
    获取战队常用英雄

    Args:
        team_id: 战队ID
        limit: 返回的英雄数量，默认20

    Returns:
        战队英雄使用情况列表
    """
    data = _make_request(f"teams/{team_id}/heroes")

    if isinstance(data, dict) and "error" in data:
        return f"❌ API 错误: {data['error']}"

    if not isinstance(data, list):
        return "❌ 获取战队英雄数据失败"

    heroes = data[:limit]
    hero_map = _build_hero_map()

    lines = [
        "# 🎯 战队常用英雄",
        "",
        "| 英雄ID | 英雄名 | 场次 | 胜场 | 胜率 |",
        "|--------|--------|------|------|------|",
    ]

    for h in heroes:
        hero_id = int(h.get("hero_id", 0) or 0)
        hero_name = h.get("name") or hero_map.get(hero_id, f"Hero {hero_id}")
        games = h.get("games_played", 0)
        wins = h.get("wins", 0)
        win_rate = f"{(wins / games * 100):.1f}%" if games > 0 else "N/A"
        lines.append(f"| {hero_id} | {hero_name} | {games} | {wins} | {win_rate} |")

    return "\n".join(lines)


@mcp.tool()
def search_team(team_name: str) -> str:
    """
    通过战队名搜索战队并获取其最近比赛
    
    Args:
        team_name: 战队名称或标签（如 "Team Spirit", "TSpirit", "OG", "LGD"）
    
    Returns:
        匹配的战队信息及其最近比赛
    """
    # 获取战队列表
    teams_data = _make_request("teams")
    
    if isinstance(teams_data, dict) and "error" in teams_data:
        return f"❌ API 错误: {teams_data['error']}"
    
    if not isinstance(teams_data, list):
        return "❌ 获取战队列表失败"
    
    team_name_lower = team_name.lower()
    
    # 精确匹配
    found_team = None
    for team in teams_data:
        if team.get("name", "").lower() == team_name_lower:
            found_team = team
            break
        if team.get("tag", "").lower() == team_name_lower:
            found_team = team
            break
    
    # 模糊匹配
    if not found_team:
        matches = []
        for team in teams_data:
            name = team.get("name", "").lower()
            tag = team.get("tag", "").lower()
            if team_name_lower in name or team_name_lower in tag:
                matches.append(team)
        
        if len(matches) == 1:
            found_team = matches[0]
        elif len(matches) > 1:
            lines = [f"# 🔍 找到 {len(matches)} 个匹配的战队", ""]
            for t in matches[:10]:
                lines.append(f"- {t.get('name')} ({t.get('tag')}) - ID: {t.get('team_id')}")
            lines.append("")
            lines.append("请使用更精确的名称或使用 `get_team_matches(team_id)` 指定战队ID")
            return "\n".join(lines)
    
    if not found_team:
        return f"❌ 未找到战队: {team_name}\n提示: 尝试使用战队标签或完整名称"
    
    # 获取战队比赛
    team_id = found_team.get("team_id")
    wins = found_team.get("wins", 0)
    losses = found_team.get("losses", 0)
    
    lines = [
        f"# 🏆 {found_team.get('name')} ({found_team.get('tag')})",
        "",
        f"- 战队ID: {team_id}",
        f"- 评分: {found_team.get('rating', 'N/A'):.0f}",
        f"- 战绩: {wins} 胜 / {losses} 负",
        "",
    ]
    
    # 获取最近比赛
    matches_data = _make_request(f"teams/{team_id}/matches")
    
    if isinstance(matches_data, list) and matches_data:
        lines.append("## 最近比赛")
        lines.append("")
        lines.append("| Match ID | 对手 | 结果 | 时长 |")
        lines.append("|----------|------|------|------|")
        
        for m in matches_data[:10]:
            match_id = m.get("match_id", "N/A")
            opponent = m.get("opposing_team_name", "Unknown")
            
            radiant_win = m.get("radiant_win")
            radiant = m.get("radiant", False)
            if radiant_win is not None:
                team_win = (radiant and radiant_win) or (not radiant and not radiant_win)
                result = "✅ 胜" if team_win else "❌ 负"
            else:
                result = "⏳"
            
            duration = m.get("duration", 0)
            minutes = duration // 60
            
            lines.append(f"| {match_id} | {opponent} | {result} | {minutes}分 |")
    
    return "\n".join(lines)


@mcp.tool()
def get_leagues(limit: int = 20) -> str:
    """
    获取联赛列表
    
    Args:
        limit: 返回的联赛数量，默认20
    
    Returns:
        联赛列表，包括联赛ID、名称、等级等
    """
    data = _make_request("leagues")
    
    if isinstance(data, dict) and "error" in data:
        return f"❌ API 错误: {data['error']}"
    
    if not isinstance(data, list):
        return "❌ 获取联赛列表失败"
    
    leagues = data[:limit]
    
    lines = [
        "# 🏆 联赛列表",
        "",
        "| 联赛ID | 名称 | 等级 |",
        "|--------|------|------|",
    ]
    
    for l in leagues:
        league_id = l.get("leagueid", "N/A")
        name = l.get("name", "Unknown")[:30]
        tier = l.get("tier", "N/A")
        
        lines.append(f"| {league_id} | {name} | {tier} |")
    
    return "\n".join(lines)


@mcp.tool()
def get_hero_rankings(hero_id: int) -> str:
    """
    获取指定英雄的排行榜
    
    Args:
        hero_id: 英雄ID
    
    Returns:
        该英雄的排行榜，显示顶尖玩家
    """
    data = _make_request("rankings", params={"hero_id": hero_id})
    
    if isinstance(data, dict) and "error" in data:
        return f"❌ API 错误: {data['error']}"
    
    hero_map = _build_hero_map()
    hero_en = hero_map.get(hero_id, f"Hero {hero_id}")
    hero_cn = _get_cn_name(hero_en)
    
    rankings = data.get("rankings", [])[:20]
    
    lines = [
        f"# 🏅 {hero_cn} 排行榜",
        "",
        "| 排名 | 玩家 | 分数 |",
        "|------|------|------|",
    ]
    
    for i, r in enumerate(rankings, 1):
        name = r.get("personaname", "Unknown")[:15]
        score = r.get("score", 0)
        lines.append(f"| {i} | {name} | {score:.2f} |")
    
    return "\n".join(lines)


@mcp.tool()
def get_hero_benchmarks(hero_id: int) -> str:
    """
    获取指定英雄的基准数据（不同百分位的表现标准）
    
    Args:
        hero_id: 英雄ID
    
    Returns:
        英雄的基准数据，如GPM、XPM、击杀等在不同百分位的数值
    """
    data = _make_request("benchmarks", params={"hero_id": hero_id})
    
    if isinstance(data, dict) and "error" in data:
        return f"❌ API 错误: {data['error']}"
    
    hero_map = _build_hero_map()
    hero_en = hero_map.get(hero_id, f"Hero {hero_id}")
    hero_cn = _get_cn_name(hero_en)
    
    result = data.get("result", {})
    
    # 字段中文映射
    field_names = {
        "gold_per_min": "GPM", "xp_per_min": "XPM", "kills_per_min": "每分钟击杀",
        "last_hits_per_min": "每分钟正补", "hero_damage_per_min": "每分钟英雄伤害",
        "hero_healing_per_min": "每分钟治疗", "tower_damage": "建筑伤害",
    }
    
    lines = [
        f"# 📊 {hero_cn} 基准数据",
        "",
        "| 指标 | 50% | 75% | 90% | 99% |",
        "|------|-----|-----|-----|-----|",
    ]
    
    for field, cn_name in field_names.items():
        if field in result:
            values = result[field]
            # 提取不同百分位的值
            p50 = p75 = p90 = p99 = "N/A"
            for v in values:
                pct = v.get("percentile", 0)
                val = v.get("value", 0)
                if pct == 0.5:
                    p50 = f"{val:.1f}"
                elif pct == 0.75:
                    p75 = f"{val:.1f}"
                elif pct == 0.9:
                    p90 = f"{val:.1f}"
                elif pct == 0.99:
                    p99 = f"{val:.1f}"
            
            lines.append(f"| {cn_name} | {p50} | {p75} | {p90} | {p99} |")
    
    return "\n".join(lines)


@mcp.tool()
def get_mmr_distribution() -> str:
    """
    获取全服 MMR 分布数据
    
    Returns:
        MMR 分布统计，包括各段位的玩家数量和百分比
    """
    data = _make_request("distributions")
    
    if isinstance(data, dict) and "error" in data:
        return f"❌ API 错误: {data['error']}"
    
    ranks = data.get("ranks", {})
    rows = ranks.get("rows", [])
    
    lines = [
        "# 📊 段位分布",
        "",
        "| 段位 | 玩家数 | 累计百分比 |",
        "|------|--------|------------|",
    ]
    
    for r in rows[:20]:
        bin_name = r.get("bin_name", "N/A")
        bin_id = r.get("bin")
        bin_display = _format_rank_bin(bin_name, bin_id)
        count = r.get("count", 0)
        cum_sum = r.get("cumulative_sum", 0)
        
        lines.append(f"| {bin_display} | {count:,} | {cum_sum:.2f}% |")
    
    return "\n".join(lines)


@mcp.tool()
def get_records(field: str, limit: int = 20) -> str:
    """
    获取指定字段的最高记录

    Args:
        field: 记录字段名（例如 kills, gpm, xpm, hero_damage 等）
        limit: 返回数量，默认20

    Returns:
        记录列表，包括比赛ID、英雄、分数和时间
    """
    field = str(field or "").strip()
    if not field:
        return "❌ field 不能为空"

    data = _make_request(f"records/{field}")

    if isinstance(data, dict) and "error" in data:
        return f"❌ API 错误: {data['error']}"

    if not isinstance(data, list):
        return "❌ 获取记录数据失败"

    records = data[:limit]
    hero_map = _build_hero_map()

    lines = [
        f"# 🏅 记录排行 - {field}",
        "",
        "| Match ID | 时间 | 英雄 | 记录值 |",
        "|----------|------|------|--------|",
    ]

    for r in records:
        match_id = r.get("match_id", "N/A")
        start_time = r.get("start_time", 0)
        try:
            time_str = time.strftime("%Y-%m-%d", time.gmtime(int(start_time)))
        except (TypeError, ValueError, OSError):
            time_str = "N/A"
        hero_id = int(r.get("hero_id", 0) or 0)
        hero_en = hero_map.get(hero_id, f"Hero {hero_id}")
        hero_cn = _get_cn_name(hero_en)
        score = r.get("score", 0)
        lines.append(f"| {match_id} | {time_str} | {hero_cn} | {score} |")

    return "\n".join(lines)


@mcp.tool()
def get_constants(resource: str) -> str:
    """
    获取 OpenDota constants 静态数据（优先读取本地缓存）

    Args:
        resource: constants 资源名（如 heroes, items, abilities）

    Returns:
        constants 数据（JSON），包含来源与路径
    """
    resource = str(resource or "").strip()
    if not resource:
        return "❌ resource 不能为空"

    resource = resource.replace(".json", "").strip().lower()
    if resource in ("list", "all", "catalog"):
        return json.dumps(
            {"resources": CONSTANT_RESOURCES},
            ensure_ascii=False,
            indent=2,
        )

    if resource not in CONSTANT_RESOURCES:
        return (
            "❌ 未支持的 constants 资源。可用资源："
            + ", ".join(CONSTANT_RESOURCES)
        )

    data, source, output_path, error = _load_constants_resource(resource)
    if error:
        return f"❌ API 错误: {error}"
    result = {"source": source, "path": output_path, "data": data}
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def get_hero_abilities(hero_id: int, include_talents: bool = True) -> str:
    """
    获取英雄的技能列表（优先本地 constants）

    Args:
        hero_id: 英雄ID
        include_talents: 是否包含天赋，默认False

    Returns:
        英雄技能列表
    """
    try:
        hero_id_int = int(hero_id)
    except (TypeError, ValueError):
        return "❌ hero_id 需要是整数"

    heroes_data, _, _, heroes_err = _load_constants_resource("heroes")
    if heroes_err:
        return f"❌ 无法读取 heroes 常量: {heroes_err}"
    if not isinstance(heroes_data, dict):
        return "❌ heroes 常量格式错误"

    hero_entry = heroes_data.get(str(hero_id_int))
    if not isinstance(hero_entry, dict):
        return f"❌ 未找到英雄ID: {hero_id_int}"

    hero_name = str(hero_entry.get("name") or "")
    hero_display = str(hero_entry.get("localized_name") or hero_name or f"Hero {hero_id_int}")
    hero_key_short = hero_name.replace("npc_dota_hero_", "") if hero_name else ""

    ability_map, _, _, ability_err = _load_constants_resource("abilities")
    if ability_err:
        ability_map = None

    abilities_data = None
    for resource in ("hero_abilities", "hero_ability", "heroAbilities"):
        data, _, _, err = _load_constants_resource(resource)
        if err:
            continue
        if isinstance(data, dict):
            abilities_data = data
            break

    if not isinstance(abilities_data, dict):
        return "❌ 未找到 hero_abilities 常量"

    candidates = [hero_name, hero_key_short, hero_key_short.lower()]
    ability_list = None
    for key in candidates:
        if key and key in abilities_data:
            ability_list = abilities_data[key]
            break

    if ability_list is None and hero_key_short:
        for key in abilities_data.keys():
            if key.endswith(hero_key_short):
                ability_list = abilities_data[key]
                break

    if not isinstance(ability_list, list):
        return f"❌ 未找到英雄技能映射: {hero_display}"

    def _is_talent(name: str) -> bool:
        return name.startswith("special_bonus")

    lines = [
        f"# 🧠 {hero_display} 技能列表",
        "",
        "| 技能 | 显示名 | 图标 |",
        "|------|--------|------|",
    ]

    for ability in ability_list:
        ability_key = str(ability)
        if not include_talents and _is_talent(ability_key):
            continue
        display_name = ability_key
        icon = ""
        if isinstance(ability_map, dict) and ability_key in ability_map:
            info = ability_map.get(ability_key, {})
            if isinstance(info, dict):
                display_name = info.get("dname") or ability_key
                icon = info.get("img") or ""
        lines.append(f"| {ability_key} | {display_name} | {icon} |")

    facets_data = None
    for resource in ("facets", "hero_facets", "hero_facet", "heroFacets"):
        data, _, _, err = _load_constants_resource(resource)
        if err:
            continue
        facets_data = data
        break

    facets: List[Dict[str, str]] = []
    hero_key_candidates = [str(hero_id_int), hero_name, hero_key_short, hero_key_short.lower()]
    if isinstance(facets_data, dict):
        for key in hero_key_candidates:
            if key and key in facets_data:
                entries = facets_data.get(key)
                if isinstance(entries, list):
                    facets = entries
                break
        if not facets and hero_key_short:
            for key, entries in facets_data.items():
                if isinstance(key, str) and key.endswith(hero_key_short) and isinstance(entries, list):
                    facets = entries
                    break
    elif isinstance(facets_data, list):
        for entry in facets_data:
            if not isinstance(entry, dict):
                continue
            entry_hero_id = entry.get("hero_id")
            try:
                entry_hero_id = int(entry_hero_id)
            except (TypeError, ValueError):
                entry_hero_id = None
            if entry_hero_id == hero_id_int:
                facets.append(entry)

    if facets:
        lines.extend(["", "## 命石", "", "| 命石 | 描述 |", "|------|------|"])
        for facet in facets:
            if isinstance(facet, dict):
                name = facet.get("name") or facet.get("title") or facet.get("key") or facet.get("id")
                desc = facet.get("desc") or facet.get("description") or facet.get("summary") or ""
            else:
                name = str(facet)
                desc = ""
            lines.append(f"| {name or '未知'} | {desc} |")
    else:
        lines.extend(["", "## 命石", "", "未找到命石数据"])

    return "\n".join(lines)


@mcp.tool()
def get_scenarios_item_timings(
    item: Optional[str] = None,
    hero_id: Optional[int] = None,
    limit: int = 20,
) -> str:
    """
    获取英雄特定装备时机的胜率统计

    Args:
        item: 物品名（如 spirit_vessel）
        hero_id: 英雄ID
        limit: 返回数量，默认20

    Returns:
        装备时机统计列表
    """
    params: Dict[str, Any] = {}
    if item:
        params["item"] = item
    if hero_id is not None:
        params["hero_id"] = hero_id

    data = _make_request("scenarios/itemTimings", params=params or None)

    if isinstance(data, dict) and "error" in data:
        return f"❌ API 错误: {data['error']}"

    if not isinstance(data, list):
        return "❌ 获取装备时机数据失败"

    records = data[:limit]
    hero_map = _build_hero_map()

    lines = [
        "# ⏱️ 装备时机胜率",
        "",
        "| 英雄 | 装备 | 时间 | 场次 | 胜场 | 胜率 |",
        "|------|------|------|------|------|------|",
    ]

    def _to_int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    for row in records:
        hid = int(row.get("hero_id", 0) or 0)
        hero_en = hero_map.get(hid, f"Hero {hid}")
        hero_cn = _get_cn_name(hero_en)
        item_name = row.get("item", "N/A")
        time_val = _to_int(row.get("time", 0))
        minutes, seconds = divmod(time_val, 60)
        time_label = f"{minutes}:{seconds:02d}"
        games = _to_int(row.get("games", 0))
        wins = _to_int(row.get("wins", 0))
        win_rate = f"{(wins / games * 100):.1f}%" if games > 0 else "N/A"
        lines.append(f"| {hero_cn} | {item_name} | {time_label} | {games} | {wins} | {win_rate} |")

    return "\n".join(lines)


@mcp.tool()
def get_scenarios_lane_roles(
    lane_role: Optional[int] = None,
    hero_id: Optional[int] = None,
    limit: int = 20,
) -> str:
    """
    获取英雄在不同分路角色的胜率统计

    Args:
        lane_role: 分路角色 1-4（安全/中/劣/打野）
        hero_id: 英雄ID
        limit: 返回数量，默认20

    Returns:
        分路角色统计列表
    """
    params: Dict[str, Any] = {}
    if lane_role is not None:
        params["lane_role"] = lane_role
    if hero_id is not None:
        params["hero_id"] = hero_id

    data = _make_request("scenarios/laneRoles", params=params or None)

    if isinstance(data, dict) and "error" in data:
        return f"❌ API 错误: {data['error']}"

    if not isinstance(data, list):
        return "❌ 获取分路角色数据失败"

    records = data[:limit]
    hero_map = _build_hero_map()
    lane_map = {1: "安全路", 2: "中路", 3: "劣势路", 4: "打野"}

    lines = [
        "# 🛣️ 分路角色胜率",
        "",
        "| 英雄 | 分路 | 时长上限 | 场次 | 胜场 | 胜率 |",
        "|------|------|----------|------|------|------|",
    ]

    def _to_int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    for row in records:
        hid = int(row.get("hero_id", 0) or 0)
        hero_en = hero_map.get(hid, f"Hero {hid}")
        hero_cn = _get_cn_name(hero_en)
        role_val = _to_int(row.get("lane_role", 0))
        lane_label = lane_map.get(role_val, str(role_val))
        time_val = _to_int(row.get("time", 0))
        minutes, seconds = divmod(time_val, 60)
        time_label = f"{minutes}:{seconds:02d}"
        games = _to_int(row.get("games", 0))
        wins = _to_int(row.get("wins", 0))
        win_rate = f"{(wins / games * 100):.1f}%" if games > 0 else "N/A"
        lines.append(f"| {hero_cn} | {lane_label} | {time_label} | {games} | {wins} | {win_rate} |")

    return "\n".join(lines)


@mcp.tool()
def get_scenarios_misc(scenario: Optional[str] = None, limit: int = 20) -> str:
    """
    获取杂项场景胜率统计

    Args:
        scenario: 场景名称
        limit: 返回数量，默认20

    Returns:
        场景统计列表
    """
    params: Dict[str, Any] = {}
    if scenario:
        params["scenario"] = scenario

    data = _make_request("scenarios/misc", params=params or None)

    if isinstance(data, dict) and "error" in data:
        return f"❌ API 错误: {data['error']}"

    if not isinstance(data, list):
        return "❌ 获取场景统计失败"

    records = data[:limit]

    lines = [
        "# 🧩 场景胜率统计",
        "",
        "| 场景 | 阵营 | 区域 | 场次 | 胜场 | 胜率 |",
        "|------|------|------|------|------|------|",
    ]

    def _to_int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    for row in records:
        scenario_name = row.get("scenario", "N/A")
        is_radiant = row.get("is_radiant")
        side = "天辉" if is_radiant else "夜魇"
        if is_radiant is None:
            side = "未知"
        region = row.get("region", "N/A")
        games = _to_int(row.get("games", 0))
        wins = _to_int(row.get("wins", 0))
        win_rate = f"{(wins / games * 100):.1f}%" if games > 0 else "N/A"
        lines.append(f"| {scenario_name} | {side} | {region} | {games} | {wins} | {win_rate} |")

    return "\n".join(lines)


@mcp.tool()
def search_players(query: str) -> str:
    """
    搜索玩家
    
    Args:
        query: 搜索关键词（玩家昵称）
    
    Returns:
        匹配的玩家列表，包括账号ID、昵称等
    """
    data = _make_request("search", params={"q": query})
    
    if isinstance(data, dict) and "error" in data:
        return f"❌ API 错误: {data['error']}"
    
    if not isinstance(data, list):
        return "❌ 搜索失败"
    
    if not data:
        return f"❌ 未找到匹配 '{query}' 的玩家"
    
    lines = [
        f"# 🔍 搜索结果: {query}",
        "",
        "| 账号ID | 昵称 | 相似度 |",
        "|--------|------|--------|",
    ]
    
    for p in data[:20]:
        account_id = p.get("account_id", "N/A")
        name = p.get("personaname", "Unknown")[:20]
        similarity = p.get("similarity", 0)
        
        lines.append(f"| {account_id} | {name} | {similarity:.2f} |")
    
    return "\n".join(lines)


@mcp.tool()
def search_dota_history(
    query: str,
    num_results: int = 5,
    include_liquipedia: bool = True,
    sites: Optional[List[str]] = None,
    fetch_fulltext: bool = True,
    fulltext_max_chars: int = 8000,
) -> str:
    """
    ?? SerpApi ?? Dota ????????????????

    Args:
        query: ?????
        num_results: ?????1-10???? 5
        include_liquipedia: ?????? Liquipedia??? True?
        sites: ?????????????? ["liquipedia.net/dota2", "gosugamers.net"]?
        fetch_fulltext: ??????????????????????
        fulltext_max_chars: ????????<=0 ?????

    Returns:
        ???????????????????????
    """
    api_key = os.getenv("SERPAPI_API_KEY", "").strip()
    if not api_key:
        return "❌ 未配置 SERPAPI_API_KEY，无法使用搜索工具"

    query = (query or "").strip()
    if not query:
        return "❌ query 不能为空"

    try:
        limit = int(num_results)
    except (TypeError, ValueError):
        limit = 5
    limit = max(1, min(limit, 10))
    try:
        max_chars = int(fulltext_max_chars)
    except (TypeError, ValueError):
        max_chars = 8000
    max_chars = max(0, min(max_chars, 50000))

    default_sites = [
        "liquipedia.net/dota2",
        "gosugamers.net",
        "dotabuff.com",
    ]
    site_filters = []
    if sites:
        site_filters.extend([s.strip() for s in sites if isinstance(s, str) and s.strip()])
    else:
        site_filters.extend(default_sites)
    if include_liquipedia and "liquipedia.net/dota2" not in site_filters:
        site_filters.append("liquipedia.net/dota2")
    site_filters = list(dict.fromkeys(site_filters))

    search_query = query
    if site_filters:
        site_clause = " OR ".join([f"site:{s}" for s in site_filters])
        search_query = f"{query} ({site_clause})"
    used_query = search_query
    used_site_filters = site_filters[:]
    fallback_used = False

    def _serpapi_request(q: str, hl: str) -> Dict[str, Any]:
        response = requests.get(
            SERPAPI_ENDPOINT,
            params={
                "engine": "google",
                "q": q,
                "num": limit,
                "hl": hl,
                "api_key": api_key,
            },
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        return response.json()

    try:
        data = _serpapi_request(search_query, "zh-CN")
    except requests.exceptions.RequestException as exc:
        return f"❌ SerpApi 请求失败: {exc}"
    except ValueError:
        return "❌ SerpApi 返回非 JSON 响应"

    if isinstance(data, dict) and data.get("error"):
        error_message = str(data["error"])
        if "hasn't returned any results" in error_message and site_filters:
            try:
                data = _serpapi_request(query, "en")
                used_query = query
                used_site_filters = []
                fallback_used = True
            except requests.exceptions.RequestException as exc:
                return f"❌ SerpApi 请求失败: {exc}"
            except ValueError:
                return "❌ SerpApi 返回非 JSON 响应"
        else:
            return f"❌ SerpApi 错误: {data['error']}"

    results = data.get("organic_results") if isinstance(data, dict) else None
    if not results:
        if site_filters:
            try:
                data = _serpapi_request(query, "en")
                used_query = query
                used_site_filters = []
                fallback_used = True
            except requests.exceptions.RequestException as exc:
                return f"❌ SerpApi 请求失败: {exc}"
            except ValueError:
                return "❌ SerpApi 返回非 JSON 响应"
            results = data.get("organic_results") if isinstance(data, dict) else None
        if not results:
            return "⚠️ 未找到搜索结果"

    payload = {
        "query": query,
        "search_query": used_query,
        "site_filters": used_site_filters,
        "fallback_used": fallback_used,
        "fulltext_enabled": bool(fetch_fulltext),
        "fulltext_max_chars": max_chars,
        "results": [],
    }

    for item in results[:limit]:
        link = str(item.get("link", ""))
        if not link:
            continue
        result = {
            "title": item.get("title"),
            "snippet": item.get("snippet"),
            "link": link,
            "source": item.get("source"),
            "date": item.get("date") or item.get("published_date"),
        }
        if fetch_fulltext:
            full_text, err, truncated = _fetch_fulltext(link, max_chars=max_chars)
            if err:
                result["full_text_error"] = err
            else:
                result["full_text"] = full_text
                result["full_text_chars"] = len(full_text) if full_text else 0
                if truncated:
                    result["full_text_truncated"] = True
        payload["results"].append(result)

    return json.dumps(payload, ensure_ascii=False, indent=2)


@mcp.tool()
def analyze_match_wards(match_id: int, generate_html: bool = True, generate_image: bool = True) -> str:
    """
    分析指定比赛的眼位数据，生成可视化图表和交互式网页
    
    Args:
        match_id: Dota 2 比赛ID
        generate_html: 是否生成交互式HTML页面，默认True
        generate_image: 已禁用，保留兼容参数
    
    Returns:
        眼位分析结果，包括统计数据、视野分析数据(JSON)和生成的文件路径
    """
    # 获取比赛详情
    match_data = _make_request(f"matches/{match_id}")
    
    if "error" in match_data:
        return f"❌ API 错误: {match_data['error']}"
    
    if not match_data:
        return f"❌ 无法获取比赛 {match_id} 的数据"
    
    # 提取眼位数据
    extractor = WardDataExtractor()
    
    if not extractor.extract_from_match(match_data):
        return f"❌ 比赛 {match_id} 无眼位数据（可能未解析或无观察者数据）"
    
    df_obs, df_sen = extractor.get_dataframes()
    
    if df_obs.empty and df_sen.empty:
        return f"❌ 比赛 {match_id} 无眼位数据"
    
    # 获取队伍名称
    radiant_name = match_data.get("radiant_name") or "天辉 Radiant"
    dire_name = match_data.get("dire_name") or "夜魇 Dire"
    
    # 构建双方阵容信息
    hero_map = _build_hero_map()
    radiant_players: List[Dict[str, Any]] = []
    dire_players: List[Dict[str, Any]] = []
    kill_events: List[Dict[str, Any]] = []
    for p in match_data.get("players", []):
        raw_hero_id = p.get("hero_id")
        hero_id = int(raw_hero_id) if raw_hero_id is not None else 0
        hero_en = hero_map.get(hero_id, f"Hero {hero_id}")
        hero_cn = _get_cn_name(hero_en)
        player_name = p.get("name") or p.get("personaname") or p.get("account_id") or "Unknown"
        entry = {
            "hero_id": hero_id,
            "hero": hero_cn,
            "player": str(player_name)
        }
        is_radiant = p.get("isRadiant")
        if is_radiant is None:
            is_radiant = p.get("player_slot", 128) < 128 or p.get("is_radiant") == 1
        if is_radiant:
            radiant_players.append(entry)
        else:
            dire_players.append(entry)

        for kill in p.get("kills_log", []) or []:
            kill_events.append({
                "time": int(kill.get("time", 0)),
                "killer_team": "radiant" if is_radiant else "dire",
                "killer_hero": hero_cn,
                "killer_player": str(player_name),
                "victim": kill.get("key"),
            })
    
    # 创建分析器
    match_duration = match_data.get("duration")
    analyzer = WardAnalyzer(
        df_obs,
        df_sen,
        radiant_name,
        dire_name,
        match_duration,
        radiant_players=radiant_players,
        dire_players=dire_players
    )
    
    # 获取统计摘要
    stats = analyzer.get_stats_summary()

    # 生成视野分析结构化数据（供 LLM 生成报告）
    tower_status = {
        "radiant": match_data.get("tower_status_radiant"),
        "dire": match_data.get("tower_status_dire"),
    }
    objectives = match_data.get("objectives", [])
    report_data = _build_ward_report_data(
        df_obs,
        df_sen,
        radiant_name,
        dire_name,
        match_duration,
        radiant_players,
        dire_players,
        objectives=objectives,
        tower_status=tower_status,
        kill_events=kill_events,
    )
    report_data_json = json.dumps(report_data, ensure_ascii=False)
    
    # 确保输出目录存在
    if not os.path.exists(WARD_OUTPUT_DIR):
        os.makedirs(WARD_OUTPUT_DIR, exist_ok=True)
    
    # 生成文件
    generated_files = []

    if generate_html:
        html_path = os.path.join(WARD_OUTPUT_DIR, f"ward_timeline_{match_id}.html")
        if analyzer.generate_interactive_html(html_path):
            generated_files.append(f"🌐 交互式网页: {html_path}")
        else:
            generated_files.append("⚠️ 交互式网页生成失败")
    
    # 保存 CSV 文件
    try:
        if not df_obs.empty:
            obs_csv_path = os.path.join(WARD_OUTPUT_DIR, f"df_obs_{match_id}.csv")
            df_obs.to_csv(obs_csv_path, index=False)
            generated_files.append(f"📄 假眼数据: {obs_csv_path}")
        
        if not df_sen.empty:
            sen_csv_path = os.path.join(WARD_OUTPUT_DIR, f"df_sen_{match_id}.csv")
            df_sen.to_csv(sen_csv_path, index=False)
            generated_files.append(f"📄 真眼数据: {sen_csv_path}")
    except Exception as e:
        generated_files.append(f"⚠️ CSV保存失败: {e}")
    
    # 组合结果
    result = [
        f"# 眼位分析 - 比赛 {match_id}",
        "",
        stats,
        "",
        "## 视野分析数据 (JSON)",
        "```json",
        report_data_json,
        "```",
        "",
        "## 生成的文件",
        ""
    ]
    result.extend(generated_files)
    
    return "\n".join(result)


@mcp.tool()
def analyze_multi_match_wards(
    team_id: Optional[int] = None,
    account_id: Optional[int] = None,
    match_ids: Optional[List[int]] = None,
    limit: int = 10,
    sigma: float = 5.0,
    alpha: float = 0.65,
    debug: bool = True,
) -> str:
    """
    获取职业战队或玩家的多场比赛眼位/击杀/防御塔数据，并汇总生成热力图

    Args:
        team_id: 职业战队ID（与 account_id 二选一）
        account_id: 玩家账号ID（与 team_id 二选一）
        match_ids: 指定比赛ID列表（优先级最高）
        limit: 自动获取比赛数量，默认10
        sigma: 热力图高斯 sigma（0-128 坐标系单位）
        alpha: 热力图最大透明度（0-1）
        debug: 是否输出区域映射调试信息（输出到文件）

    Returns:
        汇总结果（包含 HTML/JSON 路径与统计摘要）
    """
    if team_id and account_id:
        return "❌ team_id 与 account_id 不能同时指定"

    if limit <= 0:
        return "❌ limit 需要大于 0"

    # 解析比赛ID列表
    resolved_match_ids: List[int] = []
    source_label = "custom"
    source_display = "custom"
    if match_ids:
        resolved_match_ids = [int(mid) for mid in match_ids if mid is not None]
    elif team_id:
        source_label = f"team_{team_id}"
        source_display = source_label
        team_info = _make_request(f"teams/{team_id}")
        if isinstance(team_info, dict) and "error" not in team_info:
            team_name = team_info.get("name") or team_info.get("tag") or f"Team {team_id}"
            team_tag = team_info.get("tag")
            source_display = f"{team_name} ({team_tag})" if team_tag and team_tag not in team_name else str(team_name)
        matches_data = _make_request(f"teams/{team_id}/matches")
        if isinstance(matches_data, dict) and "error" in matches_data:
            return f"❌ API 错误: {matches_data['error']}"
        if not isinstance(matches_data, list):
            return "❌ 无法获取战队比赛列表"
        matches_sorted = sorted(matches_data, key=lambda x: x.get("start_time", 0), reverse=True)
        resolved_match_ids = [int(m.get("match_id")) for m in matches_sorted[:limit] if m.get("match_id")]
    elif account_id:
        source_label = f"player_{account_id}"
        source_display = source_label
        matches_data = _make_request(f"players/{account_id}/recentMatches")
        if isinstance(matches_data, dict) and "error" in matches_data:
            return f"❌ API 错误: {matches_data['error']}"
        if not isinstance(matches_data, list):
            return "❌ 无法获取玩家比赛列表"
        matches_sorted = sorted(matches_data, key=lambda x: x.get("start_time", 0), reverse=True)
        resolved_match_ids = [int(m.get("match_id")) for m in matches_sorted[:limit] if m.get("match_id")]
    else:
        return "❌ 需要提供 team_id、account_id 或 match_ids"

    if not resolved_match_ids:
        return "❌ 未找到可用的比赛ID"

    obs_rows: List[Dict[str, Any]] = []
    sen_rows: List[Dict[str, Any]] = []
    obs_rows_radiant: List[Dict[str, Any]] = []
    obs_rows_dire: List[Dict[str, Any]] = []
    sen_rows_radiant: List[Dict[str, Any]] = []
    sen_rows_dire: List[Dict[str, Any]] = []
    kill_events: List[Dict[str, Any]] = []
    teamfight_events: List[Dict[str, Any]] = []
    tower_events: List[Dict[str, Any]] = []
    match_summaries: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    max_match_duration = 0
    max_ward_time = 0

    hero_map = _build_hero_map()

    for match_id in resolved_match_ids:
        match_data = _make_request(f"matches/{match_id}")
        if isinstance(match_data, dict) and "error" in match_data:
            skipped.append({"match_id": match_id, "reason": match_data.get("error")})
            continue

        if not match_data or not isinstance(match_data, dict):
            skipped.append({"match_id": match_id, "reason": "match data empty"})
            continue

        players = match_data.get("players", [])
        if not players:
            skipped.append({"match_id": match_id, "reason": "no players"})
            continue

        radiant_label = match_data.get("radiant_name") or "Radiant"
        dire_label = match_data.get("dire_name") or "Dire"

        def _role_label(lane_role: Any) -> str:
            role_map = {
                1: "优势路",
                2: "中路",
                3: "劣势路",
                4: "游走",
                5: "辅助",
            }
            try:
                role_key = int(lane_role)
            except (TypeError, ValueError):
                role_key = None
            return role_map.get(role_key, "未知")

        player_meta: List[Dict[str, str]] = []
        for p in players:
            player_slot = p.get("player_slot", 128)
            is_radiant = player_slot < 128
            player_meta.append({
                "team": radiant_label if is_radiant else dire_label,
                "role": _role_label(p.get("lane_role")),
            })

        # 确定目标阵营或玩家
        target_side: Optional[str] = None
        target_player_slot: Optional[int] = None
        target_player_name: Optional[str] = None
        if team_id:
            radiant_team_id = match_data.get("radiant_team_id")
            dire_team_id = match_data.get("dire_team_id")
            if radiant_team_id == team_id:
                target_side = "radiant"
            elif dire_team_id == team_id:
                target_side = "dire"
            else:
                skipped.append({"match_id": match_id, "reason": "team not in match"})
                continue
        elif account_id:
            for p in players:
                acc = p.get("account_id")
                try:
                    acc_int = int(acc) if acc is not None else None
                except (TypeError, ValueError):
                    acc_int = None
                if acc_int == int(account_id):
                    player_slot = p.get("player_slot", 128)
                    target_side = "radiant" if player_slot < 128 else "dire"
                    target_player_slot = player_slot
                    target_player_name = p.get("name") or p.get("personaname") or str(acc_int)
                    break
            if target_side is None:
                skipped.append({"match_id": match_id, "reason": "player not in match"})
                continue

        if target_side == "radiant":
            own_label = radiant_label
            enemy_label = dire_label
        elif target_side == "dire":
            own_label = dire_label
            enemy_label = radiant_label
        else:
            own_label = radiant_label
            enemy_label = dire_label

        match_duration = int(match_data.get("duration") or 0)
        if match_duration > max_match_duration:
            max_match_duration = match_duration

        def _include_player(p: Dict[str, Any]) -> bool:
            if account_id:
                acc = p.get("account_id")
                try:
                    acc_int = int(acc) if acc is not None else None
                except (TypeError, ValueError):
                    acc_int = None
                return acc_int == int(account_id)
            if team_id:
                player_slot = p.get("player_slot", 128)
                is_radiant = player_slot < 128
                return is_radiant if target_side == "radiant" else not is_radiant
            return True

        obs_count = 0
        sen_count = 0
        obs_count_radiant = 0
        obs_count_dire = 0
        sen_count_radiant = 0
        sen_count_dire = 0
        kill_count = 0
        tower_count = 0

        for p in players:
            if not _include_player(p):
                continue

            raw_hero_id = p.get("hero_id")
            hero_id = int(raw_hero_id) if raw_hero_id is not None else 0
            hero_en = hero_map.get(hero_id, f"Hero {hero_id}")
            hero_cn = _get_cn_name(hero_en)
            player_name = p.get("name") or p.get("personaname") or p.get("account_id") or "Unknown"
            player_slot = p.get("player_slot", 128)
            is_radiant = player_slot < 128

            obs_left_map: Dict[int, int] = {}
            for left_entry in p.get("obs_left_log", []) or []:
                ehandle = left_entry.get("ehandle")
                if ehandle is None:
                    continue
                left_time = int(left_entry.get("time", 0))
                prev_time = obs_left_map.get(ehandle)
                if prev_time is None or left_time < prev_time:
                    obs_left_map[ehandle] = left_time
                if left_time > max_ward_time:
                    max_ward_time = left_time

            sen_left_map: Dict[int, int] = {}
            for left_entry in p.get("sen_left_log", []) or []:
                ehandle = left_entry.get("ehandle")
                if ehandle is None:
                    continue
                left_time = int(left_entry.get("time", 0))
                prev_time = sen_left_map.get(ehandle)
                if prev_time is None or left_time < prev_time:
                    sen_left_map[ehandle] = left_time
                if left_time > max_ward_time:
                    max_ward_time = left_time

            for ward in p.get("obs_log", []) or []:
                time_val = int(ward.get("time", 0))
                if time_val > max_ward_time:
                    max_ward_time = time_val
                ehandle = ward.get("ehandle")
                left_time = obs_left_map.get(ehandle) if ehandle is not None else None
                obs_entry = {
                    "match_id": match_id,
                    "hero_id": hero_id,
                    "player": str(player_name),
                    "is_radiant": 1 if is_radiant else 0,
                    "time": time_val,
                    "x": float(ward.get("x", 0)),
                    "y": float(ward.get("y", 0)),
                    "ehandle": ehandle,
                    "left_time": left_time,
                }
                obs_rows.append(obs_entry)
                if is_radiant:
                    obs_rows_radiant.append(obs_entry)
                    obs_count_radiant += 1
                else:
                    obs_rows_dire.append(obs_entry)
                    obs_count_dire += 1
                obs_count += 1

            for ward in p.get("sen_log", []) or []:
                time_val = int(ward.get("time", 0))
                if time_val > max_ward_time:
                    max_ward_time = time_val
                ehandle = ward.get("ehandle")
                left_time = sen_left_map.get(ehandle) if ehandle is not None else None
                sen_entry = {
                    "match_id": match_id,
                    "hero_id": hero_id,
                    "player": str(player_name),
                    "is_radiant": 1 if is_radiant else 0,
                    "time": time_val,
                    "x": float(ward.get("x", 0)),
                    "y": float(ward.get("y", 0)),
                    "ehandle": ehandle,
                    "left_time": left_time,
                }
                sen_rows.append(sen_entry)
                if is_radiant:
                    sen_rows_radiant.append(sen_entry)
                    sen_count_radiant += 1
                else:
                    sen_rows_dire.append(sen_entry)
                    sen_count_dire += 1
                sen_count += 1

            for kill in p.get("kills_log", []) or []:
                kill_events.append({
                    "match_id": match_id,
                    "time": int(kill.get("time", 0)),
                    "killer_team": "radiant" if is_radiant else "dire",
                    "killer_hero": hero_cn,
                    "killer_player": str(player_name),
                    "victim": kill.get("key"),
                })
                kill_count += 1

        for teamfight in match_data.get("teamfights", []) or []:
            positions: List[Dict[str, Any]] = []
            for player_index, player_data in enumerate(teamfight.get("players", []) or []):
                deaths_pos = player_data.get("deaths_pos", {}) or {}
                if not isinstance(deaths_pos, dict):
                    continue
                meta = player_meta[player_index] if player_index < len(player_meta) else {}
                death_team = meta.get("team", "未知队伍")
                death_role = meta.get("role", "未知角色")
                for x_key, y_map in deaths_pos.items():
                    if not isinstance(y_map, dict):
                        continue
                    try:
                        x_val = int(x_key)
                    except (TypeError, ValueError):
                        continue
                    for y_key, count in y_map.items():
                        try:
                            y_val = int(y_key)
                        except (TypeError, ValueError):
                            continue
                        positions.append({
                            "x": x_val,
                            "y": y_val,
                            "count": int(count) if count is not None else 1,
                            "player_index": player_index,
                            "death_team": death_team,
                            "death_role": death_role,
                        })
            teamfight_events.append({
                "match_id": match_id,
                "start": int(teamfight.get("start", 0)),
                "end": int(teamfight.get("end", 0)),
                "last_death": int(teamfight.get("last_death", 0)),
                "deaths": int(teamfight.get("deaths", 0)),
                "positions": positions,
                "target_side": target_side or "radiant",
                "own_label": own_label,
                "enemy_label": enemy_label,
            })

        # 防御塔事件
        objectives = match_data.get("objectives", [])
        for obj in objectives or []:
            if obj.get("type") != "building_kill":
                continue
            key = str(obj.get("key", ""))
            if "tower" not in key:
                continue
            info = _parse_tower_key(key)
            tower_team = info.get("team")
            destroyed_by_target = True
            if team_id and target_side:
                destroyed_by_target = tower_team != target_side
            elif account_id:
                obj_slot = obj.get("player_slot")
                destroyed_by_target = target_player_slot is not None and obj_slot == target_player_slot

            if not destroyed_by_target:
                continue

            tower_events.append({
                "match_id": match_id,
                "time": int(obj.get("time", 0)),
                "key": key,
                "tower_team": tower_team,
                "lane": info.get("lane"),
                "tier": info.get("tier"),
                "player_slot": obj.get("player_slot"),
            })
            tower_count += 1

        match_summaries.append({
            "match_id": match_id,
            "radiant_name": match_data.get("radiant_name") or "Radiant",
            "dire_name": match_data.get("dire_name") or "Dire",
            "side": target_side or "all",
            "obs_count": obs_count,
            "sen_count": sen_count,
            "obs_count_radiant": obs_count_radiant,
            "sen_count_radiant": sen_count_radiant,
            "obs_count_dire": obs_count_dire,
            "sen_count_dire": sen_count_dire,
            "kill_count": kill_count,
            "tower_count": tower_count,
        })

    if not obs_rows and not sen_rows:
        return "❌ 未获取到目标眼位数据（可能比赛未解析或无观察者数据）"

    if max_match_duration <= 0:
        max_match_duration = 7200

    df_obs = pd.DataFrame(obs_rows) if obs_rows else pd.DataFrame()
    df_sen = pd.DataFrame(sen_rows) if sen_rows else pd.DataFrame()
    df_obs_radiant = pd.DataFrame(obs_rows_radiant) if obs_rows_radiant else pd.DataFrame()
    df_sen_radiant = pd.DataFrame(sen_rows_radiant) if sen_rows_radiant else pd.DataFrame()
    df_obs_dire = pd.DataFrame(obs_rows_dire) if obs_rows_dire else pd.DataFrame()
    df_sen_dire = pd.DataFrame(sen_rows_dire) if sen_rows_dire else pd.DataFrame()

    # 生成热力图
    analyzer_radiant = WardAnalyzer(
        df_obs_radiant,
        df_sen_radiant,
        radiant_name="天辉 Radiant",
        dire_name="夜魇 Dire",
        match_duration=None,
        radiant_players=[],
        dire_players=[],
    )
    analyzer_dire = WardAnalyzer(
        df_obs_dire,
        df_sen_dire,
        radiant_name="天辉 Radiant",
        dire_name="夜魇 Dire",
        match_duration=None,
        radiant_players=[],
        dire_players=[],
    )

    points_base64_radiant = analyzer_radiant._generate_ward_points_base64()
    heatmap_base64_radiant_obs = analyzer_radiant._generate_heatmap_base64(
        sigma=sigma,
        alpha=alpha,
        ward_type="obs",
    )
    heatmap_base64_radiant_sen = analyzer_radiant._generate_heatmap_base64(
        sigma=sigma,
        alpha=alpha,
        ward_type="sen",
    )
    points_base64_dire = analyzer_dire._generate_ward_points_base64()
    heatmap_base64_dire_obs = analyzer_dire._generate_heatmap_base64(
        sigma=sigma,
        alpha=alpha,
        ward_type="obs",
    )
    heatmap_base64_dire_sen = analyzer_dire._generate_heatmap_base64(
        sigma=sigma,
        alpha=alpha,
        ward_type="sen",
    )

    map_base64 = ""
    if analyzer_radiant.map_image:
        buffered = BytesIO()
        analyzer_radiant.map_image.save(buffered, format="JPEG")
        map_base64 = base64.b64encode(buffered.getvalue()).decode()

    total_obs = len(df_obs) if not df_obs.empty else 0
    total_sen = len(df_sen) if not df_sen.empty else 0
    total_obs_radiant = len(df_obs_radiant) if not df_obs_radiant.empty else 0
    total_sen_radiant = len(df_sen_radiant) if not df_sen_radiant.empty else 0
    total_obs_dire = len(df_obs_dire) if not df_obs_dire.empty else 0
    total_sen_dire = len(df_sen_dire) if not df_sen_dire.empty else 0
    total_kills = len(kill_events)
    total_towers = len(tower_events)
    region_summary, region_template = _build_multi_match_region_summary(obs_rows, sen_rows)

    def _ward_lifetime_stats(rows: List[Dict[str, Any]], default_duration: int) -> Dict[str, Any]:
        durations: List[int] = []
        for ward in rows:
            try:
                time_val = int(ward.get("time", 0) or 0)
            except (TypeError, ValueError):
                time_val = 0
            left_raw = ward.get("left_time")
            duration = None
            if left_raw is not None:
                try:
                    left_time = int(left_raw)
                except (TypeError, ValueError):
                    left_time = None
                if left_time is not None and left_time >= time_val:
                    duration = left_time - time_val
            if duration is None:
                duration = default_duration
            if duration >= 0:
                durations.append(int(duration))
        if not durations:
            return {"count": 0, "avg": None, "median": None, "min": None, "max": None}
        return {
            "count": len(durations),
            "avg": sum(durations) / len(durations),
            "median": float(np.median(durations)),
            "min": min(durations),
            "max": max(durations),
        }

    ward_lifetime = {
        "obs": _ward_lifetime_stats(obs_rows, 360),
        "sen": _ward_lifetime_stats(sen_rows, 420),
        "by_side": {
            "radiant": {
                "obs": _ward_lifetime_stats(obs_rows_radiant, 360),
                "sen": _ward_lifetime_stats(sen_rows_radiant, 420),
            },
            "dire": {
                "obs": _ward_lifetime_stats(obs_rows_dire, 360),
                "sen": _ward_lifetime_stats(sen_rows_dire, 420),
            },
        },
    }

    def _region_lifetime_summary(
        obs_rows_src: List[Dict[str, Any]],
        sen_rows_src: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        regions = _load_region_template()
        region_map: Dict[Tuple[str, str], Dict[str, Any]] = {}

        def _duration_seconds(row: Dict[str, Any], default_duration: int) -> Optional[int]:
            try:
                time_val = int(row.get("time", 0) or 0)
            except (TypeError, ValueError):
                time_val = 0
            left_raw = row.get("left_time")
            duration_val = None
            if left_raw is not None:
                try:
                    left_time = int(left_raw)
                except (TypeError, ValueError):
                    left_time = None
                if left_time is not None and left_time >= time_val:
                    duration_val = left_time - time_val
            if duration_val is None:
                duration_val = default_duration
            return duration_val if duration_val >= 0 else None

        def _is_killed(row: Dict[str, Any], max_duration: int) -> bool:
            try:
                time_val = int(row.get("time", 0) or 0)
            except (TypeError, ValueError):
                time_val = 0
            left_raw = row.get("left_time")
            if left_raw is None:
                return False
            try:
                left_time = int(left_raw)
            except (TypeError, ValueError):
                return False
            return left_time < time_val + max_duration

        def _collect(rows: List[Dict[str, Any]], ward_type: str, default_duration: int) -> None:
            for row in rows:
                x = row.get("x")
                y = row.get("y")
                if x is None or y is None:
                    continue
                try:
                    x_val = float(x)
                    y_val = float(y)
                except (TypeError, ValueError):
                    continue
                primary_key, primary_label, _labels = _match_region(x_val, y_val, regions)
                label = primary_label or "未知区域"
                duration_val = _duration_seconds(row, default_duration)
                if duration_val is None:
                    continue
                killed = _is_killed(row, default_duration)
                is_early_kill = killed and duration_val <= 120
                is_full_survival = not killed or duration_val >= default_duration
                ratio_val = min(duration_val, default_duration) / default_duration
                key = (label, ward_type)
                entry = region_map.setdefault(key, {
                    "label": label,
                    "type": ward_type,
                    "key": primary_key,
                    "durations": [],
                    "early_kill_count": 0,
                    "full_survival_count": 0,
                    "ratio_sum": 0.0,
                })
                entry["durations"].append(int(duration_val))
                entry["early_kill_count"] += 1 if is_early_kill else 0
                entry["full_survival_count"] += 1 if is_full_survival else 0
                entry["ratio_sum"] += ratio_val

        _collect(obs_rows_src, "obs", 360)
        _collect(sen_rows_src, "sen", 420)

        summary: List[Dict[str, Any]] = []
        for entry in region_map.values():
            durations = entry.get("durations", [])
            if not durations:
                stats = {
                    "count": 0,
                    "avg": None,
                    "median": None,
                    "min": None,
                    "max": None,
                    "early_kill_rate": None,
                    "full_survival_rate": None,
                    "avg_time_survival_ratio": None,
                }
            else:
                stats = {
                    "count": len(durations),
                    "avg": sum(durations) / len(durations),
                    "median": float(np.median(durations)),
                    "min": min(durations),
                    "max": max(durations),
                    "early_kill_rate": entry.get("early_kill_count", 0) / len(durations),
                    "full_survival_rate": entry.get("full_survival_count", 0) / len(durations),
                    "avg_time_survival_ratio": entry.get("ratio_sum", 0.0) / len(durations),
                }
            summary.append({
                "label": entry.get("label"),
                "type": entry.get("type"),
                "key": entry.get("key"),
                **stats,
            })

        summary.sort(key=lambda x: x.get("count", 0), reverse=True)
        return summary

    ward_lifetime_by_region = _region_lifetime_summary(obs_rows, sen_rows)

    # 输出目录
    if not os.path.exists(WARD_OUTPUT_DIR):
        os.makedirs(WARD_OUTPUT_DIR, exist_ok=True)

    timestamp = int(time.time())
    html_path = os.path.join(WARD_OUTPUT_DIR, f"ward_multi_{source_label}_{timestamp}.html")
    json_path = os.path.join(WARD_OUTPUT_DIR, f"ward_multi_{source_label}_{timestamp}.json")

    # 汇总数据
    output_payload = {
        "source": source_display,
        "source_id": source_label,
        "matches": match_summaries,
        "totals": {
            "obs": total_obs,
            "sen": total_sen,
            "kills": total_kills,
            "tower_kills": total_towers,
            "by_side": {
                "radiant": {"obs": total_obs_radiant, "sen": total_sen_radiant},
                "dire": {"obs": total_obs_dire, "sen": total_sen_dire},
            },
        },
        "wards": {
            "obs": obs_rows,
            "sen": sen_rows,
            "by_side": {
                "radiant": {"obs": obs_rows_radiant, "sen": sen_rows_radiant},
                "dire": {"obs": obs_rows_dire, "sen": sen_rows_dire},
            },
        },
        "region_template": region_template,
        "region_summary": region_summary,
        "ward_lifetime": ward_lifetime,
        "ward_lifetime_by_region": ward_lifetime_by_region,
        "kills": kill_events,
        "teamfights": teamfight_events,
        "tower_events": tower_events,
        "skipped": skipped,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)

    # 生成 HTML
    rows_html = "\n".join(
        f"<tr><td>{m['match_id']}</td><td>{html.escape(m['radiant_name'])} vs {html.escape(m['dire_name'])}</td>"
        f"<td>{m['side']}</td>"
        f"<td>{m.get('obs_count_radiant', 0)}/{m.get('sen_count_radiant', 0)}</td>"
        f"<td>{m.get('obs_count_dire', 0)}/{m.get('sen_count_dire', 0)}</td>"
        f"<td>{m['kill_count']}</td><td>{m['tower_count']}</td></tr>"
        for m in match_summaries
    )
    if not rows_html:
        rows_html = "<tr><td colspan=\"7\">暂无数据</td></tr>"

    total_radiant_side = total_obs_radiant + total_sen_radiant
    total_dire_side = total_obs_dire + total_sen_dire

    def _format_ratio(value: int, total: int) -> str:
        if total <= 0:
            return "-"
        return f"{value / total * 100:.1f}%"

    def _format_time(seconds: int) -> str:
        if seconds is None:
            return "-"
        try:
            sec_val = int(seconds)
        except (TypeError, ValueError):
            return "-"
        sign = "-" if sec_val < 0 else ""
        sec_val = abs(sec_val)
        return f"{sign}{sec_val // 60}:{sec_val % 60:02d}"

    def _format_duration(value: Optional[float]) -> str:
        if value is None:
            return "-"
        try:
            sec_val = int(round(float(value)))
        except (TypeError, ValueError):
            return "-"
        minutes, secs = divmod(max(sec_val, 0), 60)
        return f"{minutes}:{secs:02d}"

    def _format_percent(value: Optional[float]) -> str:
        if value is None:
            return "-"
        try:
            return f"{float(value) * 100:.1f}%"
        except (TypeError, ValueError):
            return "-"

    def _format_ratio_value(value: Optional[float]) -> str:
        if value is None:
            return "-"
        try:
            return f"{float(value) * 100:.1f}%"
        except (TypeError, ValueError):
            return "-"

    debug_lines: List[str] = []

    def _debug_log(message: str) -> None:
        if debug:
            debug_lines.append(message)

    region_radiant = sorted(
        region_summary,
        key=lambda r: int(r.get("obs_radiant", 0)) + int(r.get("sen_radiant", 0)),
        reverse=True,
    )
    region_dire = sorted(
        region_summary,
        key=lambda r: int(r.get("obs_dire", 0)) + int(r.get("sen_dire", 0)),
        reverse=True,
    )

    region_rows_radiant_html = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(r.get('label', '')))}</td>"
        f"<td>{r.get('obs_radiant', 0)}</td>"
        f"<td>{r.get('sen_radiant', 0)}</td>"
        f"<td>{int(r.get('obs_radiant', 0)) + int(r.get('sen_radiant', 0))}</td>"
        f"<td>{_format_ratio(int(r.get('obs_radiant', 0)) + int(r.get('sen_radiant', 0)), total_radiant_side)}</td>"
        "</tr>"
        for r in region_radiant
        if int(r.get("obs_radiant", 0)) + int(r.get("sen_radiant", 0)) > 0
    )
    if not region_rows_radiant_html:
        region_rows_radiant_html = "<tr><td colspan=\"5\">暂无数据</td></tr>"

    region_rows_dire_html = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(r.get('label', '')))}</td>"
        f"<td>{r.get('obs_dire', 0)}</td>"
        f"<td>{r.get('sen_dire', 0)}</td>"
        f"<td>{int(r.get('obs_dire', 0)) + int(r.get('sen_dire', 0))}</td>"
        f"<td>{_format_ratio(int(r.get('obs_dire', 0)) + int(r.get('sen_dire', 0)), total_dire_side)}</td>"
        "</tr>"
        for r in region_dire
        if int(r.get("obs_dire", 0)) + int(r.get("sen_dire", 0)) > 0
    )
    if not region_rows_dire_html:
        region_rows_dire_html = "<tr><td colspan=\"5\">暂无数据</td></tr>"

    def _region_lifetime_row(item: Dict[str, Any]) -> str:
        return (
            "<tr>"
            f"<td>{html.escape(str(item.get('label', '')))}</td>"
            f"<td>{item.get('count', 0)}</td>"
            f"<td>{_format_duration(item.get('avg'))}</td>"
            f"<td>{_format_duration(item.get('median'))}</td>"
            f"<td>{_format_duration(item.get('min'))}</td>"
            f"<td>{_format_duration(item.get('max'))}</td>"
            f"<td>{_format_percent(item.get('early_kill_rate'))}</td>"
            f"<td>{_format_percent(item.get('full_survival_rate'))}</td>"
            f"<td>{_format_ratio_value(item.get('avg_time_survival_ratio'))}</td>"
            "</tr>"
        )

    if ward_lifetime_by_region:
        obs_rows_html = "\n".join(
            _region_lifetime_row(item)
            for item in ward_lifetime_by_region
            if item.get("type") == "obs"
        )
        sen_rows_html = "\n".join(
            _region_lifetime_row(item)
            for item in ward_lifetime_by_region
            if item.get("type") == "sen"
        )
        ward_lifetime_obs_rows_html = obs_rows_html or "<tr><td colspan=\"9\">暂无数据</td></tr>"
        ward_lifetime_sen_rows_html = sen_rows_html or "<tr><td colspan=\"9\">暂无数据</td></tr>"
    else:
        ward_lifetime_obs_rows_html = "<tr><td colspan=\"9\">暂无数据</td></tr>"
        ward_lifetime_sen_rows_html = "<tr><td colspan=\"9\">暂无数据</td></tr>"

    region_note = ""
    if not region_template:
        region_note = "<div class=\"summary\">⚠️ 未加载区域模板，可能全部落入“未知区域”。</div>"

    regions = _load_region_template()
    region_lookup = {
        str(r.get("key") or r.get("label")): r for r in (regions or [])
    }
    obs_by_match_region: Dict[int, List[Dict[str, Any]]] = {}
    sen_by_match_region: Dict[int, List[Dict[str, Any]]] = {}
    if regions:
        for obs in obs_rows:
            try:
                x_val = float(obs.get("x", 0))
                y_val = float(obs.get("y", 0))
            except (TypeError, ValueError):
                continue
            region_key, region_label, _ = _match_region(x_val, y_val, regions)
            if not region_key:
                _debug_log(
                    f"[OBS_MAP] match={obs.get('match_id')} time={obs.get('time')} "
                    f"x={x_val} y={y_val} -> region=未知区域"
                )
                continue
            start_time = int(obs.get("time", 0) or 0)
            left_raw = obs.get("left_time")
            end_time = None
            if left_raw is not None:
                try:
                    end_candidate = int(left_raw)
                except (TypeError, ValueError):
                    end_candidate = None
                if end_candidate is not None and end_candidate >= start_time:
                    end_time = end_candidate
            if end_time is None:
                end_time = start_time + 360
            match_id = int(obs.get("match_id", 0) or 0)
            obs_by_match_region.setdefault(match_id, []).append({
                "region_key": region_key,
                "region_label": region_label,
                "start": start_time,
                "end": end_time,
                "x": x_val,
                "y": y_val,
            })
            _debug_log(
                f"[OBS_MAP] match={match_id} time={start_time} x={x_val} y={y_val} "
                f"-> region={region_label}/{region_key} end={end_time}"
            )
        for sen in sen_rows:
            try:
                x_val = float(sen.get("x", 0))
                y_val = float(sen.get("y", 0))
            except (TypeError, ValueError):
                continue
            region_key, region_label, _ = _match_region(x_val, y_val, regions)
            if not region_key:
                _debug_log(
                    f"[SEN_MAP] match={sen.get('match_id')} time={sen.get('time')} "
                    f"x={x_val} y={y_val} -> region=未知区域"
                )
                continue
            start_time = int(sen.get("time", 0) or 0)
            left_raw = sen.get("left_time")
            end_time = None
            if left_raw is not None:
                try:
                    end_candidate = int(left_raw)
                except (TypeError, ValueError):
                    end_candidate = None
                if end_candidate is not None and end_candidate >= start_time:
                    end_time = end_candidate
            if end_time is None:
                end_time = start_time + 420
            match_id = int(sen.get("match_id", 0) or 0)
            sen_by_match_region.setdefault(match_id, []).append({
                "region_key": region_key,
                "region_label": region_label,
                "start": start_time,
                "end": end_time,
                "x": x_val,
                "y": y_val,
            })
            _debug_log(
                f"[SEN_MAP] match={match_id} time={start_time} x={x_val} y={y_val} "
                f"-> region={region_label}/{region_key} end={end_time}"
            )

    def _resolve_teamfight_region(
        x_val: float,
        y_val: float,
        region_items: List[Dict[str, Any]],
    ) -> Tuple[Optional[str], Optional[str], float, float, str]:
        region_key, region_label, _labels, raw_dist, _ = _match_region_with_distance(
            x_val,
            y_val,
            region_items,
            allow_nearest=True,
        )
        shifted_x = x_val + 64
        shifted_y = y_val + 64
        region_key_shift, region_label_shift, _labels_shift, shift_dist, _ = _match_region_with_distance(
            shifted_x,
            shifted_y,
            region_items,
            allow_nearest=True,
        )
        if raw_dist <= shift_dist:
            return region_key, region_label, x_val, y_val, "raw"
        return region_key_shift, region_label_shift, shifted_x, shifted_y, "shifted"

    def _distance_to_region(x_val: float, y_val: float, region: Optional[Dict[str, Any]]) -> float:
        if not region:
            return float("inf")
        min_dist = float("inf")
        for area in region.get("areas", []):
            area_type = area.get("type")
            if area_type == "bbox":
                dist = _distance_to_bbox(x_val, y_val, area)
            elif area_type == "polygon":
                points = area.get("points") or []
                dist = _distance_to_polygon(x_val, y_val, points)
            else:
                continue
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def _ward_active(ward: Dict[str, Any], window_start: int, window_end: int) -> bool:
        try:
            start_time = int(ward.get("start", 0) or 0)
        except (TypeError, ValueError):
            start_time = 0
        try:
            end_time = int(ward.get("end", 0) or 0)
        except (TypeError, ValueError):
            end_time = 0
        return start_time <= window_end and end_time >= window_start

    def _pick_primary_region(positions: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
        region_weights: Dict[str, int] = {}
        region_labels: Dict[str, str] = {}
        for pos in positions:
            region_key = pos.get("region_key")
            region_label = pos.get("region_label") or "未知区域"
            if not region_key:
                continue
            weight = int(pos.get("count", 1) or 1)
            region_weights[region_key] = region_weights.get(region_key, 0) + weight
            region_labels[region_key] = region_label
        if not region_weights:
            return None, None
        primary_key = max(region_weights.items(), key=lambda item: item[1])[0]
        return primary_key, region_labels.get(primary_key)

    for teamfight in teamfight_events:
        match_id = int(teamfight.get("match_id", 0) or 0)
        tf_start = int(teamfight.get("start", 0) or 0)
        tf_end = int(teamfight.get("end", tf_start) or tf_start)
        last_death = int(teamfight.get("last_death", tf_end) or tf_end)
        if tf_end < tf_start and last_death >= tf_start:
            tf_end = last_death
        obs_candidates = obs_by_match_region.get(match_id, [])
        positions = teamfight.get("positions", []) or []
        enriched_positions: List[Dict[str, Any]] = []
        positions_with_obs = 0
        for pos in positions:
            try:
                pos_x = float(pos.get("x", 0))
                pos_y = float(pos.get("y", 0))
            except (TypeError, ValueError):
                continue
            if regions:
                region_key, region_label, map_x, map_y, map_mode = _resolve_teamfight_region(pos_x, pos_y, regions)
            else:
                region_key, region_label, map_x, map_y, map_mode = None, None, pos_x, pos_y, "none"
            _debug_log(
                f"[TF_MAP] match={match_id} tf={tf_start}-{tf_end} "
                f"raw=({pos_x},{pos_y}) mapped=({map_x},{map_y}) mode={map_mode} "
                f"region={region_label or '未知区域'}/{region_key or 'None'}"
            )
            obs_count = 0
            if region_key:
                for ward in obs_candidates:
                    if ward.get("region_key") != region_key:
                        continue
                    if ward.get("start", 0) <= tf_end and ward.get("end", 0) > tf_start:
                        obs_count += 1
                if obs_count > 0:
                    positions_with_obs += 1
            enriched_positions.append({
                **pos,
                "map_x": map_x,
                "map_y": map_y,
                "region_key": region_key,
                "region_label": region_label or "未知区域",
                "obs_count": obs_count,
                "has_obs": obs_count > 0,
            })
        if not enriched_positions:
            continue

        teamfight["positions"] = enriched_positions
        teamfight["positions_total"] = len(enriched_positions)
        teamfight["positions_with_obs"] = positions_with_obs

        fight_region_key, fight_region_label = _pick_primary_region(enriched_positions)
        if not fight_region_label:
            fight_region_label = "未知区域"

        window_end = tf_start
        window_start = max(0, tf_start - 10)
        wards_in_match = obs_by_match_region.get(match_id, []) + sen_by_match_region.get(match_id, [])
        offensive_vision = "无"
        if fight_region_key:
            has_direct = False
            for ward in wards_in_match:
                if not _ward_active(ward, window_start, window_end):
                    continue
                if ward.get("region_key") == fight_region_key:
                    has_direct = True
                    break
            if has_direct:
                offensive_vision = "有"
            else:
                fight_region = region_lookup.get(fight_region_key)
                if fight_region:
                    for ward in wards_in_match:
                        if not _ward_active(ward, window_start, window_end):
                            continue
                        try:
                            ward_x = float(ward.get("x", 0))
                            ward_y = float(ward.get("y", 0))
                        except (TypeError, ValueError):
                            continue
                        if _distance_to_region(ward_x, ward_y, fight_region) <= 8.0:
                            offensive_vision = "部分"
                            break
        elif enriched_positions:
            total_weight = sum(int(p.get("count", 1) or 1) for p in enriched_positions)
            if total_weight > 0:
                cx = sum(float(p.get("map_x", p.get("x", 0))) * int(p.get("count", 1) or 1) for p in enriched_positions) / total_weight
                cy = sum(float(p.get("map_y", p.get("y", 0))) * int(p.get("count", 1) or 1) for p in enriched_positions) / total_weight
                for ward in wards_in_match:
                    if not _ward_active(ward, window_start, window_end):
                        continue
                    try:
                        ward_x = float(ward.get("x", 0))
                        ward_y = float(ward.get("y", 0))
                    except (TypeError, ValueError):
                        continue
                    if math.hypot(ward_x - cx, ward_y - cy) <= 8.0:
                        offensive_vision = "部分"
                        break

        own_label = teamfight.get("own_label") or ""
        enemy_label = teamfight.get("enemy_label") or ""
        own_deaths = 0
        enemy_deaths = 0
        for pos in enriched_positions:
            count_val = int(pos.get("count", 1) or 1)
            death_team = pos.get("death_team")
            if death_team == own_label:
                own_deaths += count_val
            elif death_team == enemy_label:
                enemy_deaths += count_val

        if not fight_region_key or (enemy_deaths + own_deaths) == 0:
            continue

        kill_diff = enemy_deaths - own_deaths
        if kill_diff > 0:
            fight_result = "✅ 胜"
        elif kill_diff < 0:
            fight_result = "❌ 败"
        else:
            fight_result = "⚖ 平"

        teamfight["region_key"] = fight_region_key
        teamfight["region_label"] = fight_region_label
        teamfight["offensive_vision"] = offensive_vision
        teamfight["enemy_deaths"] = enemy_deaths
        teamfight["own_deaths"] = own_deaths
        teamfight["kill_diff"] = kill_diff
        teamfight["result"] = fight_result

    def _format_positions_html(positions: List[Dict[str, Any]]) -> str:
        if not positions:
            return "暂无"
        return "<br/>".join(
            f"{pos.get('region_label', '未知区域')} {pos.get('map_x', pos.get('x'))},{pos.get('map_y', pos.get('y'))}×{pos.get('count', 1)}"
            f" ({'有假眼' if pos.get('has_obs') else '无假眼'})"
            f" - {pos.get('death_team', '未知队伍')}/{pos.get('death_role', '未知角色')}"
            for pos in positions
        )

    def _format_kill_diff(value: int) -> str:
        if value > 0:
            return f"+{value}"
        return str(value)

    def _format_death_value(value: int) -> str:
        return "0" if value == 0 else str(value)

    def _is_valid_teamfight_row(tf: Dict[str, Any]) -> bool:
        label = str(tf.get("region_label") or "").strip()
        if not label or label == "未知区域":
            return False
        if int(tf.get("enemy_deaths", 0) or 0) + int(tf.get("own_deaths", 0) or 0) == 0:
            return False
        return True

    valid_teamfights = [tf for tf in teamfight_events if _is_valid_teamfight_row(tf)]

    teamfight_rows_html = "\n".join(
        "<tr>"
        f"<td>{tf.get('match_id')}</td>"
        f"<td>{_format_time(tf.get('start'))} - {_format_time(tf.get('end'))}</td>"
        f"<td>{html.escape(str(tf.get('region_label') or '未知区域'))}</td>"
        f"<td>{html.escape(str(tf.get('offensive_vision') or '无'))}</td>"
        f"<td>{_format_death_value(int(tf.get('enemy_deaths', 0) or 0))}</td>"
        f"<td>{_format_death_value(int(tf.get('own_deaths', 0) or 0))}</td>"
        f"<td>{_format_kill_diff(int(tf.get('kill_diff', 0) or 0))}</td>"
        f"<td>{html.escape(str(tf.get('result') or ''))}</td>"
        "</tr>"
        for tf in valid_teamfights
    )
    if not teamfight_rows_html:
        teamfight_rows_html = "<tr><td colspan=\"8\">暂无数据</td></tr>"

    vision_buckets = {
        "有": [],
        "部分": [],
        "无": [],
    }
    for tf in valid_teamfights:
        vision = tf.get("offensive_vision") or "无"
        if vision not in vision_buckets:
            vision = "无"
        vision_buckets[vision].append(tf)

    def _avg_or_dash(values: List[int]) -> str:
        if not values:
            return "-"
        return f"{sum(values) / len(values):.1f}"

    def _win_rate_or_dash(values: List[int]) -> str:
        if not values:
            return "-"
        wins = sum(1 for v in values if v > 0)
        return f"{wins / len(values) * 100:.0f}%"

    teamfight_summary_rows_html = "\n".join(
        "<tr>"
        f"<td>{html.escape(label)}</td>"
        f"<td>{len(items)}</td>"
        f"<td>{_avg_or_dash([int(i.get('enemy_deaths', 0) or 0) for i in items])}</td>"
        f"<td>{_avg_or_dash([int(i.get('own_deaths', 0) or 0) for i in items])}</td>"
        f"<td>{_avg_or_dash([int(i.get('kill_diff', 0) or 0) for i in items])}</td>"
        f"<td>{_win_rate_or_dash([int(i.get('kill_diff', 0) or 0) for i in items])}</td>"
        "</tr>"
        for label, items in [("有", vision_buckets["有"]), ("部分", vision_buckets["部分"]), ("无", vision_buckets["无"])]
    )
    if not teamfight_summary_rows_html:
        teamfight_summary_rows_html = "<tr><td colspan=\"6\">暂无数据</td></tr>"

    icon_base64 = {}
    icon_dir = "figure"
    icon_files = {
        "obs_radiant": "goodguys_observer.png",
        "obs_dire": "badguys_observer.png",
        "sen_radiant": "goodguys_sentry.png",
        "sen_dire": "badguys_sentry.png",
    }
    for key, filename in icon_files.items():
        icon_path = os.path.join(icon_dir, filename)
        if os.path.exists(icon_path):
            try:
                with open(icon_path, "rb") as f:
                    icon_base64[key] = base64.b64encode(f.read()).decode()
            except Exception:
                pass

    hero_cn_cache: Dict[int, str] = {}

    def _hero_cn(hero_id: int) -> str:
        if hero_id not in hero_cn_cache:
            hero_en = hero_map.get(hero_id, f"Hero {hero_id}")
            hero_cn_cache[hero_id] = _get_cn_name(hero_en)
        return hero_cn_cache[hero_id]

    def _build_ward_points(
        rows: List[Dict[str, Any]],
        icon_key: str,
        is_obs: bool,
    ) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for ward in rows:
            try:
                x_val = float(ward.get("x", 0)) - 64
                y_val = float(ward.get("y", 0)) - 64
            except (TypeError, ValueError):
                continue
            if not (np.isfinite(x_val) and np.isfinite(y_val)):
                continue
            x_val = float(np.clip(x_val, 0, 128))
            y_val = float(np.clip(y_val, 0, 128))
            time_val = int(ward.get("time", 0) or 0)
            end_time = None
            left_raw = ward.get("left_time")
            if left_raw is not None:
                try:
                    end_time_val = int(left_raw)
                except (TypeError, ValueError):
                    end_time_val = None
                if end_time_val is not None:
                    if end_time_val >= time_val:
                        end_time = end_time_val
            if end_time is None:
                default_duration = 360 if is_obs else 420
                end_time = time_val + default_duration
            hero_id = int(ward.get("hero_id", 0) or 0)
            items.append({
                "x": x_val,
                "y": y_val,
                "time": time_val,
                "end_time": end_time,
                "type": icon_key,
                "is_obs": is_obs,
                "hero": _hero_cn(hero_id),
                "player": str(ward.get("player", "Unknown")),
                "match_id": ward.get("match_id"),
            })
        return items

    wards_radiant_data: List[Dict[str, Any]] = []
    wards_radiant_data.extend(_build_ward_points(obs_rows_radiant, "obs_radiant", True))
    wards_radiant_data.extend(_build_ward_points(sen_rows_radiant, "sen_radiant", False))
    wards_dire_data: List[Dict[str, Any]] = []
    wards_dire_data.extend(_build_ward_points(obs_rows_dire, "obs_dire", True))
    wards_dire_data.extend(_build_ward_points(sen_rows_dire, "sen_dire", False))
    total_points_radiant = len(wards_radiant_data)
    total_points_dire = len(wards_dire_data)
    player_list = sorted({
        str(w.get("player", "Unknown")) for w in (wards_radiant_data + wards_dire_data)
    })
    if player_list:
        player_filter_html = "\n".join(
            f"<label class=\"filter-item\"><input type=\"checkbox\" name=\"playerFilter\" "
            f"value=\"{html.escape(player)}\" checked> {html.escape(player)}</label>"
            for player in player_list
        )
    else:
        player_filter_html = "<div class=\"placeholder\">暂无选手数据</div>"
    timeline_min_time = -150
    timeline_min_label = "-2:30"

    map_image_html = (
        f"<img src=\"data:image/jpeg;base64,{map_base64}\" class=\"map-image\">"
        if map_base64
        else "<div class=\"placeholder\">地图加载失败</div>"
    )

    points_section_radiant = (
        f"<div class=\"map-body\" id=\"radiantMapBody\">{map_image_html}</div>"
    )
    heatmap_section_radiant_obs = (
        f"<img src=\"data:image/png;base64,{heatmap_base64_radiant_obs}\" class=\"map-image\">"
        if heatmap_base64_radiant_obs
        else (
            f"<img src=\"data:image/jpeg;base64,{map_base64}\" class=\"map-image\">"
            "<div class=\"placeholder\">假眼热力图生成失败</div>"
        )
    )
    heatmap_section_radiant_sen = (
        f"<img src=\"data:image/png;base64,{heatmap_base64_radiant_sen}\" class=\"map-image\">"
        if heatmap_base64_radiant_sen
        else (
            f"<img src=\"data:image/jpeg;base64,{map_base64}\" class=\"map-image\">"
            "<div class=\"placeholder\">真眼热力图生成失败</div>"
        )
    )
    points_section_dire = (
        f"<div class=\"map-body\" id=\"direMapBody\">{map_image_html}</div>"
    )
    heatmap_section_dire_obs = (
        f"<img src=\"data:image/png;base64,{heatmap_base64_dire_obs}\" class=\"map-image\">"
        if heatmap_base64_dire_obs
        else (
            f"<img src=\"data:image/jpeg;base64,{map_base64}\" class=\"map-image\">"
            "<div class=\"placeholder\">假眼热力图生成失败</div>"
        )
    )
    heatmap_section_dire_sen = (
        f"<img src=\"data:image/png;base64,{heatmap_base64_dire_sen}\" class=\"map-image\">"
        if heatmap_base64_dire_sen
        else (
            f"<img src=\"data:image/jpeg;base64,{map_base64}\" class=\"map-image\">"
            "<div class=\"placeholder\">真眼热力图生成失败</div>"
        )
    )

    points_controls_radiant = f"""
        <div class="time-controls" id="radiantControls">
            <div class="time-display" id="radiantTimeDisplay">00:00</div>
            <div class="slider-row">
                <span class="time-label">{timeline_min_label}</span>
                <input type="range" class="time-slider" id="radiantTimeSlider" min="{timeline_min_time}" max="{max_match_duration}" value="{timeline_min_time}">
                <span class="time-label" id="radiantMaxLabel"></span>
            </div>
            <div class="ward-stats">
                <div>当前假眼 <span class="stat-value" id="radiantObsCount">0</span></div>
                <div>当前真眼 <span class="stat-value" id="radiantSenCount">0</span></div>
                <div>总眼位 <span class="stat-value" id="radiantTotalCount">{total_points_radiant}</span></div>
            </div>
            <details class="filter-details">
                <summary>按选手筛选</summary>
                <div class="filter-actions">
                    <button class="filter-button" data-action="select-all-players">全选</button>
                    <button class="filter-button" data-action="clear-all-players">清空</button>
                </div>
                <div class="filter-list">
                    {player_filter_html}
                </div>
            </details>
        </div>
    """
    points_controls_dire = f"""
        <div class="time-controls" id="direControls">
            <div class="time-display" id="direTimeDisplay">00:00</div>
            <div class="slider-row">
                <span class="time-label">{timeline_min_label}</span>
                <input type="range" class="time-slider" id="direTimeSlider" min="{timeline_min_time}" max="{max_match_duration}" value="{timeline_min_time}">
                <span class="time-label" id="direMaxLabel"></span>
            </div>
            <div class="ward-stats">
                <div>当前假眼 <span class="stat-value" id="direObsCount">0</span></div>
                <div>当前真眼 <span class="stat-value" id="direSenCount">0</span></div>
                <div>总眼位 <span class="stat-value" id="direTotalCount">{total_points_dire}</span></div>
            </div>
            <details class="filter-details">
                <summary>按选手筛选</summary>
                <div class="filter-actions">
                    <button class="filter-button" data-action="select-all-players">全选</button>
                    <button class="filter-button" data-action="clear-all-players">清空</button>
                </div>
                <div class="filter-list">
                    {player_filter_html}
                </div>
            </details>
        </div>
    """

    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(source_display)} 的最近比赛的视野分析</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: Arial, sans-serif; background: #0f0f0f; color: #f0f0f0; }}
        .container {{ max-width: 980px; margin: 0 auto; padding: 20px 16px 32px; }}
        h1 {{ text-align: center; margin-bottom: 10px; }}
        .summary {{ text-align: center; color: #aaa; margin-bottom: 14px; }}
        .side-title {{ margin: 18px 0 8px; font-size: 16px; color: #f6f6f6; text-align: center; }}
        .map-container {{ width: 100%; max-width: 800px; margin: 0 auto 16px; border: 2px solid #333; border-radius: 10px; overflow: hidden; background: #111; position: relative; }}
        .map-title {{ padding: 8px 10px; font-size: 13px; color: #f0f0f0; background: rgba(0,0,0,0.5); border-bottom: 1px solid rgba(255,255,255,0.08); }}
        .map-image {{ width: 100%; display: block; }}
        .map-body {{ position: relative; width: 100%; }}
        .ward-dot {{ position: absolute; transform: translate(-50%, -50%); z-index: 5; }}
        .ward-dot img {{ width: 26px; height: 26px; }}
        .ward-dot.hidden {{ opacity: 0; pointer-events: none; }}
        .time-controls {{ width: 100%; max-width: 800px; margin: 6px auto 18px; background: rgba(255,255,255,0.08); padding: 8px 10px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.12); }}
        .time-display {{ text-align: center; font-size: 16px; color: #ffd700; margin-bottom: 6px; }}
        .slider-row {{ display: flex; align-items: center; gap: 8px; }}
        .time-label {{ font-size: 11px; color: #aaa; min-width: 42px; text-align: center; }}
        .time-slider {{ flex: 1; -webkit-appearance: none; height: 6px; border-radius: 6px; background: linear-gradient(to right, #2d5a27 0%, #8b4513 50%, #4a1a1a 100%); outline: none; cursor: pointer; }}
        .time-slider::-webkit-slider-thumb {{ -webkit-appearance: none; width: 14px; height: 14px; border-radius: 50%; background: #ffd700; cursor: pointer; box-shadow: 0 2px 6px rgba(0,0,0,0.4); }}
        .ward-stats {{ display: flex; justify-content: center; gap: 22px; margin-top: 6px; font-size: 12px; color: #ddd; }}
        .ward-stats .stat-value {{ color: #ffd700; font-weight: 600; margin-left: 4px; }}
        .placeholder {{ position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; color: #ccc; }}
        .filter-details {{ margin-top: 8px; border: 1px solid rgba(255,255,255,0.12); border-radius: 8px; padding: 6px 8px; background: rgba(255,255,255,0.06); }}
        .filter-details summary {{ cursor: pointer; font-size: 12px; color: #ddd; }}
        .filter-details[open] summary {{ color: #ffd700; }}
        .filter-actions {{ display: flex; gap: 6px; margin-top: 6px; }}
        .filter-button {{ background: rgba(255,255,255,0.12); color: #f0f0f0; border: 1px solid rgba(255,255,255,0.2); padding: 2px 8px; border-radius: 999px; font-size: 11px; cursor: pointer; }}
        .filter-button:hover {{ background: rgba(255,255,255,0.18); }}
        .filter-list {{ display: flex; flex-wrap: wrap; gap: 6px 10px; margin-top: 6px; max-height: 90px; overflow-y: auto; }}
        .filter-item {{ font-size: 11px; color: #ddd; display: flex; align-items: center; gap: 4px; }}
        .filter-item input {{ accent-color: #ffd700; }}
        .report {{ margin: 12px auto 0; background: rgba(255,255,255,0.06); padding: 12px 14px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.1); }}
        .report-hidden {{ display: none; }}
        .report h2 {{ font-size: 16px; margin-bottom: 8px; color: #f6f6f6; }}
        .report p, .report li {{ color: #e0e0e0; line-height: 1.6; }}
        .report ul, .report ol {{ margin: 6px 0 12px; padding-left: 16px; list-style-position: inside; }}
        .report li {{ margin: 4px 0; }}
        .report .report-section {{ margin-bottom: 12px; }}
        .report .report-section:last-child {{ margin-bottom: 0; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 13px; }}
        th, td {{ padding: 8px 6px; border-bottom: 1px solid rgba(255,255,255,0.1); text-align: left; }}
        th {{ color: #ffd700; }}
        .tag {{ display: inline-block; padding: 2px 8px; border-radius: 999px; background: rgba(255,255,255,0.12); font-size: 12px; color: #ffd700; margin-left: 6px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{html.escape(source_display)} 的最近比赛的视野分析</h1>
        <div class="summary">
            来源: {html.escape(source_display)} |
            比赛数: {len(match_summaries)} |
            假眼: {total_obs} |
            真眼: {total_sen} |
            击杀: {total_kills} |
            推塔: {total_towers}
        </div>
        <div class="summary">
            天辉：假眼 {total_obs_radiant} / 真眼 {total_sen_radiant} |
            夜魇：假眼 {total_obs_dire} / 真眼 {total_sen_dire}
        </div>
        <div class="side-title">天辉 (Radiant) 视野汇总</div>
        <div class="map-container">
            <div class="map-title">眼位点位图</div>
            {points_section_radiant}
        </div>
        {points_controls_radiant}
        <div class="map-container">
            <div class="map-title">假眼热力图</div>
            {heatmap_section_radiant_obs}
        </div>
        <div class="map-container">
            <div class="map-title">真眼热力图</div>
            {heatmap_section_radiant_sen}
        </div>
        <div class="side-title">夜魇 (Dire) 视野汇总</div>
        <div class="map-container">
            <div class="map-title">眼位点位图</div>
            {points_section_dire}
        </div>
        {points_controls_dire}
        <div class="map-container">
            <div class="map-title">假眼热力图</div>
            {heatmap_section_dire_obs}
        </div>
        <div class="map-container">
            <div class="map-title">真眼热力图</div>
            {heatmap_section_dire_sen}
        </div>
        <!--MULTI_MATCH_REPORT-->
        <h2 style="margin-top: 10px; font-size: 16px;">区域假眼存活时长<span class="tag">汇总</span></h2>
        <table>
            <thead>
                <tr>
                    <th>区域</th>
                    <th>样本数</th>
                    <th>平均</th>
                    <th>中位</th>
                    <th>最短</th>
                    <th>最长</th>
                    <th>2分钟内被反率</th>
                    <th>满时存活率</th>
                    <th>时间存活率</th>
                </tr>
            </thead>
            <tbody>
                {ward_lifetime_obs_rows_html}
            </tbody>
        </table>
        <h2 style="margin-top: 10px; font-size: 16px;">区域真眼存活时长<span class="tag">汇总</span></h2>
        <table>
            <thead>
                <tr>
                    <th>区域</th>
                    <th>样本数</th>
                    <th>平均</th>
                    <th>中位</th>
                    <th>最短</th>
                    <th>最长</th>
                    <th>2分钟内被反率</th>
                    <th>满时存活率</th>
                    <th>时间存活率</th>
                </tr>
            </thead>
            <tbody>
                {ward_lifetime_sen_rows_html}
            </tbody>
        </table>
        <h2 style="margin-top: 10px; font-size: 16px;">区域统计<span class="tag">汇总</span></h2>
        {region_note}
        <div class="side-title">天辉区域分布</div>
        <table>
            <thead>
                <tr>
                    <th>区域</th>
                    <th>假眼</th>
                    <th>真眼</th>
                    <th>合计</th>
                    <th>占比</th>
                </tr>
            </thead>
            <tbody>
                {region_rows_radiant_html}
            </tbody>
        </table>
        <div class="side-title" style="margin-top: 12px;">夜魇区域分布</div>
        <table>
            <thead>
                <tr>
                    <th>区域</th>
                    <th>假眼</th>
                    <th>真眼</th>
                    <th>合计</th>
                    <th>占比</th>
                </tr>
            </thead>
            <tbody>
                {region_rows_dire_html}
            </tbody>
        </table>
        <h2 style="margin-top: 10px; font-size: 16px;">比赛列表<span class="tag">汇总</span></h2>
        <table>
            <thead>
                <tr>
                    <th>Match ID</th>
                    <th>对阵</th>
                    <th>阵营</th>
                    <th>天辉假/真</th>
                    <th>夜魇假/真</th>
                    <th>击杀</th>
                    <th>推塔</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
        <h2 style="margin-top: 10px; font-size: 16px;">团战视野分析<span class="tag">Teamfight</span></h2>
        <table>
            <thead>
                <tr>
                    <th>Match</th>
                    <th>时间</th>
                    <th>区域</th>
                    <th>进攻视野</th>
                    <th>敌方死亡</th>
                    <th>己方死亡</th>
                    <th>击杀差</th>
                    <th>团战结果</th>
                </tr>
            </thead>
            <tbody>
                {teamfight_rows_html}
            </tbody>
        </table>
        <h2 style="margin-top: 10px; font-size: 16px;">团战击杀效率 × 进攻视野<span class="tag">汇总</span></h2>
        <table>
            <thead>
                <tr>
                    <th>进攻视野</th>
                    <th>团战数</th>
                    <th>平均敌方死亡</th>
                    <th>平均己方死亡</th>
                    <th>平均击杀差</th>
                    <th>胜率</th>
                </tr>
            </thead>
            <tbody>
                {teamfight_summary_rows_html}
            </tbody>
        </table>
    </div>
    <script>
        const wardIcons = {{
            "obs_radiant": "data:image/png;base64,{icon_base64.get('obs_radiant', '')}",
            "obs_dire": "data:image/png;base64,{icon_base64.get('obs_dire', '')}",
            "sen_radiant": "data:image/png;base64,{icon_base64.get('sen_radiant', '')}",
            "sen_dire": "data:image/png;base64,{icon_base64.get('sen_dire', '')}"
        }};
        const radiantWards = {json.dumps(wards_radiant_data, ensure_ascii=False)};
        const direWards = {json.dumps(wards_dire_data, ensure_ascii=False)};
        const playerFilterInputs = document.querySelectorAll('input[name="playerFilter"]');
        const selectAllButtons = document.querySelectorAll('[data-action="select-all-players"]');
        const clearAllButtons = document.querySelectorAll('[data-action="clear-all-players"]');
        const updateHandlers = [];

        function formatTime(seconds) {{
            const sign = seconds < 0 ? '-' : '';
            const absSeconds = Math.abs(seconds);
            const mins = Math.floor(absSeconds / 60);
            const secs = absSeconds % 60;
            return sign + mins + ':' + secs.toString().padStart(2, '0');
        }}

        function getSelectedPlayers() {{
            const selected = new Set();
            playerFilterInputs.forEach((input) => {{
                if (input.checked) {{
                    selected.add(input.value);
                }}
            }});
            return Array.from(selected);
        }}

        function syncPlayerCheckboxes(value, checked) {{
            playerFilterInputs.forEach((input) => {{
                if (input.value === value) {{
                    input.checked = checked;
                }}
            }});
        }}

        function updateAllMaps() {{
            updateHandlers.forEach((handler) => handler());
        }}

        function initWardMap(options) {{
            const mapBody = document.getElementById(options.mapBodyId);
            const slider = document.getElementById(options.sliderId);
            const display = document.getElementById(options.displayId);
            const obsCount = document.getElementById(options.obsCountId);
            const senCount = document.getElementById(options.senCountId);
            const maxLabel = document.getElementById(options.maxLabelId);
            const totalCount = document.getElementById(options.totalCountId);
            const wards = options.wards || [];

            if (!mapBody || !slider) {{
                return;
            }}

            const wardElements = [];
            wards.forEach((ward) => {{
                const xPercent = (ward.x / 128) * 100;
                const yPercent = (1 - ward.y / 128) * 100;
                const div = document.createElement('div');
                div.className = 'ward-dot hidden';
                div.style.left = xPercent + '%';
                div.style.top = yPercent + '%';
                const img = document.createElement('img');
                img.src = wardIcons[ward.type] || '';
                div.appendChild(img);
                mapBody.appendChild(div);
                wardElements.push(div);
            }});

            if (totalCount) {{
                totalCount.textContent = wards.length;
            }}
            if (maxLabel) {{
                maxLabel.textContent = formatTime(parseInt(slider.max || '0'));
            }}

            const update = (currentTime) => {{
                let obs = 0;
                let sen = 0;
                const minTime = parseInt(slider.min || '0');
                const showAll = currentTime <= minTime;
                const selectedPlayers = getSelectedPlayers();
                const hasPlayerFilter = selectedPlayers.length > 0;
                const playerSet = new Set(selectedPlayers);
                wards.forEach((ward, index) => {{
                    const hasEnd = ward.end_time !== null && ward.end_time !== undefined;
                    const visible = showAll || (ward.time <= currentTime && (!hasEnd || currentTime < ward.end_time));
                    const matchesPlayer = !hasPlayerFilter || playerSet.has(ward.player);
                    if (visible && matchesPlayer) {{
                        wardElements[index].classList.remove('hidden');
                        if (ward.is_obs) {{
                            obs += 1;
                        }} else {{
                            sen += 1;
                        }}
                    }} else {{
                        wardElements[index].classList.add('hidden');
                    }}
                }});
                if (obsCount) {{
                    obsCount.textContent = obs;
                }}
                if (senCount) {{
                    senCount.textContent = sen;
                }}
            }};

            const setDisplay = (value) => {{
                if (display) {{
                    const minTime = parseInt(slider.min || '0');
                    display.textContent = value <= minTime ? '全部' : formatTime(value);
                }}
            }};

            const initValue = parseInt(slider.value || '0');
            setDisplay(initValue);
            update(initValue);
            updateHandlers.push(() => {{
                const value = parseInt(slider.value || '0');
                setDisplay(value);
                update(value);
            }});
            slider.addEventListener('input', () => {{
                const value = parseInt(slider.value || '0');
                setDisplay(value);
                update(value);
            }});
        }}

        initWardMap({{
            mapBodyId: 'radiantMapBody',
            sliderId: 'radiantTimeSlider',
            displayId: 'radiantTimeDisplay',
            obsCountId: 'radiantObsCount',
            senCountId: 'radiantSenCount',
            maxLabelId: 'radiantMaxLabel',
            totalCountId: 'radiantTotalCount',
            wards: radiantWards
        }});
        initWardMap({{
            mapBodyId: 'direMapBody',
            sliderId: 'direTimeSlider',
            displayId: 'direTimeDisplay',
            obsCountId: 'direObsCount',
            senCountId: 'direSenCount',
            maxLabelId: 'direMaxLabel',
            totalCountId: 'direTotalCount',
            wards: direWards
        }});
        if (playerFilterInputs.length) {{
            playerFilterInputs.forEach((input) => {{
                input.addEventListener('change', (event) => {{
                    const target = event.target;
                    if (target && target.value !== undefined) {{
                        syncPlayerCheckboxes(target.value, target.checked);
                    }}
                    updateAllMaps();
                }});
            }});
        }}
        if (selectAllButtons.length) {{
            selectAllButtons.forEach((btn) => {{
                btn.addEventListener('click', () => {{
                    playerFilterInputs.forEach((input) => {{
                        input.checked = true;
                    }});
                    updateAllMaps();
                }});
            }});
        }}
        if (clearAllButtons.length) {{
            clearAllButtons.forEach((btn) => {{
                btn.addEventListener('click', () => {{
                    playerFilterInputs.forEach((input) => {{
                        input.checked = false;
                    }});
                    updateAllMaps();
                }});
            }});
        }}
    </script>
</body>
</html>
"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    debug_log_path = ""
    if debug and debug_lines:
        debug_log_path = os.path.join(
            WARD_OUTPUT_DIR,
            f"ward_mapping_debug_{source_label}_{timestamp}.log",
        )
        with open(debug_log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(debug_lines))

    result_lines = [
        "# 多场比赛视野汇总",
        f"- 来源: {source_display}",
        f"- 比赛数量: {len(match_summaries)}",
        f"- 眼位总计: 假眼 {total_obs} / 真眼 {total_sen}",
        f"- 天辉眼位: 假眼 {total_obs_radiant} / 真眼 {total_sen_radiant}",
        f"- 夜魇眼位: 假眼 {total_obs_dire} / 真眼 {total_sen_dire}",
        f"- 击杀总计: {total_kills}",
        f"- 推塔总计: {total_towers}",
        f"- 交互式网页: {html_path}",
        f"- 汇总数据: {json_path}",
    ]

    def _format_duration(value: Optional[float]) -> str:
        if value is None:
            return "-"
        try:
            seconds = int(round(float(value)))
        except (TypeError, ValueError):
            return "-"
        minutes, secs = divmod(max(seconds, 0), 60)
        return f"{minutes}:{secs:02d}"

    obs_avg = _format_duration(ward_lifetime["obs"].get("avg"))
    obs_median = _format_duration(ward_lifetime["obs"].get("median"))
    sen_avg = _format_duration(ward_lifetime["sen"].get("avg"))
    sen_median = _format_duration(ward_lifetime["sen"].get("median"))
    result_lines.insert(
        6,
        f"- 眼位存活: 假眼 平均 {obs_avg} / 中位 {obs_median} | 真眼 平均 {sen_avg} / 中位 {sen_median}",
    )
    if debug_log_path:
        result_lines.append(f"- 区域映射调试日志: {debug_log_path}")

    if skipped:
        skipped_lines = ", ".join(str(s.get("match_id")) for s in skipped[:10])
        result_lines.append(f"- ⚠️ 跳过 {len(skipped)} 场比赛: {skipped_lines}")

    return "\n".join(result_lines)


@mcp.tool()
def inject_multi_match_ward_report_html(
    summary_path: Optional[str] = None,
    report_html: Optional[str] = None,
    report_path: Optional[str] = None,
    html_path: Optional[str] = None,
) -> str:
    """
    读取多场比赛视野汇总 JSON，并将汇总分析报告写入多场比赛网页

    Args:
        summary_path: 汇总 JSON 文件路径（可选，默认读取 ward_analysis 下最新 ward_multi_*.json）
        report_html: 汇总分析报告 HTML 片段（可选）
        report_path: 汇总分析报告 HTML 文件路径（可选）
        html_path: 多场比赛网页路径（可选，默认尝试与 summary_path 同名）

    Returns:
        JSON 字符串：包含汇总数据与写入结果
    """
    # 读取报告内容
    if report_path:
        resolved_report_path = report_path
        if not os.path.isabs(resolved_report_path):
            resolved_report_path = os.path.join(os.getcwd(), report_path)
        if not os.path.exists(resolved_report_path):
            return f"❌ 未找到报告文件: {resolved_report_path}"
        with open(resolved_report_path, "r", encoding="utf-8") as f:
            report_html = f.read()

    if report_html and report_html.strip().lower().startswith("@file:"):
        path_hint = report_html.strip()[6:].strip()
        resolved_report_path = path_hint
        if not os.path.isabs(resolved_report_path):
            resolved_report_path = os.path.join(os.getcwd(), path_hint)
        if not os.path.exists(resolved_report_path):
            return f"❌ 未找到报告文件: {resolved_report_path}"
        with open(resolved_report_path, "r", encoding="utf-8") as f:
            report_html = f.read()

    # 解析汇总 JSON
    resolved_summary_path = summary_path
    if not resolved_summary_path:
        if not os.path.exists(WARD_OUTPUT_DIR):
            return f"❌ 未找到视野汇总目录: {WARD_OUTPUT_DIR}"
        candidates = []
        for filename in os.listdir(WARD_OUTPUT_DIR):
            if filename.startswith("ward_multi_") and filename.endswith(".json"):
                full_path = os.path.join(WARD_OUTPUT_DIR, filename)
                candidates.append((os.path.getmtime(full_path), full_path))
        if not candidates:
            return "❌ 未找到视野汇总 JSON 文件，请先运行 analyze_multi_match_wards"
        candidates.sort(key=lambda x: x[0], reverse=True)
        resolved_summary_path = candidates[0][1]

    if not os.path.exists(resolved_summary_path):
        return f"❌ 文件不存在: {resolved_summary_path}"

    try:
        with open(resolved_summary_path, "r", encoding="utf-8") as f:
            summary_data = json.load(f)
    except Exception as exc:
        return f"❌ 读取 JSON 失败: {exc}"

    summary_view = dict(summary_data) if isinstance(summary_data, dict) else {"data": summary_data}
    removed_fields: List[str] = []
    if isinstance(summary_view, dict):
        if "wards" in summary_view:
            removed_fields.append("wards")
            totals = summary_view.get("totals", {})
            summary_view["wards_summary"] = {
                "obs": totals.get("obs"),
                "sen": totals.get("sen"),
                "by_side": totals.get("by_side"),
            }
            summary_view.pop("wards", None)
        if "kills" in summary_view:
            removed_fields.append("kills")
            summary_view.pop("kills", None)
        if "tower_events" in summary_view:
            removed_fields.append("tower_events")
            summary_view.pop("tower_events", None)

    # 解析网页路径
    resolved_html_path = html_path
    if not resolved_html_path and resolved_summary_path:
        base_name = os.path.splitext(os.path.basename(resolved_summary_path))[0]
        candidate_html = os.path.join(WARD_OUTPUT_DIR, f"{base_name}.html")
        if os.path.exists(candidate_html):
            resolved_html_path = candidate_html

    if not resolved_html_path:
        candidates = []
        if os.path.exists(WARD_OUTPUT_DIR):
            for filename in os.listdir(WARD_OUTPUT_DIR):
                if filename.startswith("ward_multi_") and filename.endswith(".html"):
                    full_path = os.path.join(WARD_OUTPUT_DIR, filename)
                    candidates.append((os.path.getmtime(full_path), full_path))
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            resolved_html_path = candidates[0][1]

    if not resolved_html_path or not os.path.exists(resolved_html_path):
        return "❌ 未找到多场比赛分析网页，请先运行 analyze_multi_match_wards"

    result_payload: Dict[str, Any] = {
        "summary_path": resolved_summary_path,
        "html_path": resolved_html_path,
        "summary": summary_view,
    }
    if removed_fields:
        result_payload["summary_truncated"] = True
        result_payload["summary_removed_fields"] = removed_fields

    # 多场汇总报告禁用写入（按产品需求）
    result_payload["injected"] = False
    result_payload["message"] = "⏭️ 已禁用多场比赛视野汇总分析报告写入"

    return json.dumps(result_payload, ensure_ascii=False, indent=2)


@mcp.tool()
def inject_ward_report_html(
    match_id: Optional[int] = None,
    report_html: Optional[str] = None,
    report_path: Optional[str] = None,
) -> str:
    """
    将视野分析报告 HTML 插入到已生成的眼位时间线网页中

    Args:
        match_id: Dota 2 比赛ID（可从 report_path 中推断）
        report_html: HTML 片段（建议包含段落/列表/表格）
        report_path: HTML 文件路径（用于长报告，优先读取）

    Returns:
        写入结果与文件路径
    """
    if report_path:
        resolved_path = report_path
        if not os.path.isabs(resolved_path):
            resolved_path = os.path.join(os.getcwd(), report_path)
        if not os.path.exists(resolved_path):
            return f"❌ 未找到报告文件: {resolved_path}"
        with open(resolved_path, "r", encoding="utf-8") as f:
            report_html = f.read()
        if match_id is None:
            name_match = re.search(r"(\\d+)", os.path.basename(resolved_path))
            if name_match:
                match_id = int(name_match.group(1))

    if report_html and report_html.strip().lower().startswith("@file:"):
        path_hint = report_html.strip()[6:].strip()
        resolved_path = path_hint
        if not os.path.isabs(resolved_path):
            resolved_path = os.path.join(os.getcwd(), path_hint)
        if not os.path.exists(resolved_path):
            return f"❌ 未找到报告文件: {resolved_path}"
        with open(resolved_path, "r", encoding="utf-8") as f:
            report_html = f.read()

    if match_id is None:
        candidates = []
        if os.path.exists(WARD_OUTPUT_DIR):
            for filename in os.listdir(WARD_OUTPUT_DIR):
                if filename.startswith("ward_timeline_") and filename.endswith(".html"):
                    full_path = os.path.join(WARD_OUTPUT_DIR, filename)
                    candidates.append((os.path.getmtime(full_path), filename))
        if candidates:
            latest_file = sorted(candidates, key=lambda x: x[0], reverse=True)[0][1]
            match = re.search(r"(\\d+)", latest_file)
            if match:
                match_id = int(match.group(1))

    if match_id is None:
        return "❌ 缺少 match_id，且无法从 report_path 推断比赛ID"

    html_path = os.path.join(WARD_OUTPUT_DIR, f"ward_timeline_{match_id}.html")
    if not os.path.exists(html_path):
        return f"❌ 未找到网页文件: {html_path}，请先生成眼位时间线网页"

    if not report_html or not report_html.strip():
        return "❌ 报告内容为空，未写入"

    try:
        with open(html_path, "r", encoding="utf-8") as f:
            html_text = f.read()

        pattern = re.compile(r"<!-- WARD_REPORT_START -->.*?<!-- WARD_REPORT_END -->", re.DOTALL)
        replacement = f"<!-- WARD_REPORT_START -->\n{report_html}\n<!-- WARD_REPORT_END -->"

        if pattern.search(html_text):
            html_text = pattern.sub(replacement, html_text, count=1)
        else:
            insert_block = f"\n<section class=\"report\" id=\"analysisReport\">\n" \
                          f"  <h2>视野分析报告</h2>\n" \
                          f"  <div class=\"report-content\" id=\"reportContent\">\n{report_html}\n  </div>\n</section>\n"
            if "</body>" in html_text:
                html_text = html_text.replace("</body>", insert_block + "\n</body>")
            else:
                html_text += insert_block

        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_text)

        return f"✅ 已写入视野分析报告: {html_path}"
    except Exception as e:
        return f"❌ 写入失败: {e}"


@mcp.tool()
def save_team_hero_report(
    team_id: int,
    report_title: str,
    report_markdown: str,
    report_html: Optional[str] = None,
    report_path: Optional[str] = None,
) -> str:
    """
    保存战队常用英雄分析报告，并返回可视化路径与内容

    Args:
        team_id: 战队ID
        report_title: 报告标题
        report_markdown: Markdown 报告（用于对话可视化）
        report_html: HTML 报告片段（可选，不含完整 HTML 文档）
        report_path: 指定保存路径（可选，默认保存到 hero_analysis 目录）

    Returns:
        保存结果 + 可视化内容（Markdown） + HTML 路径
    """
    if not team_id:
        return "❌ team_id 不能为空"
    if not report_markdown or not str(report_markdown).strip():
        return "❌ report_markdown 不能为空"

    os.makedirs(HERO_REPORT_DIR, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if report_path:
        resolved_report_path = report_path
        if not os.path.isabs(resolved_report_path):
            resolved_report_path = os.path.join(os.getcwd(), report_path)
    else:
        filename = f"team_hero_report_{team_id}_{timestamp}.html"
        resolved_report_path = os.path.join(HERO_REPORT_DIR, filename)

    markdown_path = os.path.splitext(resolved_report_path)[0] + ".md"

    if report_html:
        report_body = report_html
    else:
        report_body = "<pre>" + html.escape(report_markdown) + "</pre>"

    html_doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <title>{html.escape(report_title or 'Team Hero Report')}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
    h1 {{ font-size: 20px; margin-bottom: 16px; }}
    .report {{ line-height: 1.6; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background: #f5f5f5; }}
    pre {{ background: #f7f7f7; padding: 12px; overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>{html.escape(report_title or 'Team Hero Report')}</h1>
  <div class="report">
    {report_body}
  </div>
</body>
</html>
"""

    try:
        with open(resolved_report_path, "w", encoding="utf-8") as f:
            f.write(html_doc)
        with open(markdown_path, "w", encoding="utf-8") as f:
            f.write(report_markdown)
    except Exception as exc:
        return f"❌ 写入报告失败: {exc}"

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    rel_html = os.path.relpath(resolved_report_path, repo_root).replace("\\", "/")
    rel_md = os.path.relpath(markdown_path, repo_root).replace("\\", "/")

    lines = [
        "# ✅ 战队常用英雄分析报告已保存",
        f"- team_id: {team_id}",
        f"- html_path: {rel_html}",
        f"- markdown_path: {rel_md}",
        "",
        "## 📊 可视化（Markdown）",
        report_markdown.strip(),
    ]
    return "\n".join(lines)


@mcp.tool()
def save_match_details_report(
    report_title: Optional[str] = None,
    report_markdown: Optional[str] = None,
    report_html: Optional[str] = None,
    report_path: Optional[str] = None,
    match_ids: Optional[List[int]] = None,
) -> str:
    """
    保存多场比赛详情分析报告，并返回可视化路径与内容

    Args:
        report_title: 报告标题
        report_markdown: Markdown 报告（用于对话可视化）
        report_html: HTML 报告片段（可选，不含完整 HTML 文档）
        report_path: 指定保存路径（可选，默认保存到 match_analysis 目录）
        match_ids: 本次报告覆盖的比赛ID列表（可选）

    Returns:
        保存结果 + 可视化内容（Markdown） + HTML 路径
    """
    report_title = (report_title or "").strip() or "多场比赛详情分析报告"
    report_markdown = (report_markdown or "").strip()
    if not report_markdown:
        if match_ids:
            lines = [f"# {report_title}", "", "## 📋 比赛列表", ""]
            lines.extend([f"- Match ID: {mid}" for mid in match_ids])
            report_markdown = "\n".join(lines)
        else:
            report_markdown = f"# {report_title}\n\n暂无详细报告内容。"

    os.makedirs(MATCH_REPORT_DIR, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if report_path:
        resolved_report_path = report_path
        if not os.path.isabs(resolved_report_path):
            resolved_report_path = os.path.join(os.getcwd(), report_path)
    else:
        filename = f"match_details_report_{timestamp}.html"
        resolved_report_path = os.path.join(MATCH_REPORT_DIR, filename)

    markdown_path = os.path.splitext(resolved_report_path)[0] + ".md"

    if report_html:
        report_body = report_html
    else:
        report_body = "<pre>" + html.escape(report_markdown) + "</pre>"

    html_doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <title>{html.escape(report_title or 'Match Details Report')}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
    h1 {{ font-size: 20px; margin-bottom: 16px; }}
    h2 {{ font-size: 16px; margin-top: 24px; }}
    .report {{ line-height: 1.6; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background: #f5f5f5; }}
    pre {{ background: #f7f7f7; padding: 12px; overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>{html.escape(report_title or 'Match Details Report')}</h1>
  <div class="report">
    {report_body}
  </div>
</body>
</html>
"""

    try:
        with open(resolved_report_path, "w", encoding="utf-8") as f:
            f.write(html_doc)
        with open(markdown_path, "w", encoding="utf-8") as f:
            f.write(report_markdown)
    except Exception as exc:
        return f"❌ 写入报告失败: {exc}"

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    rel_html = os.path.relpath(resolved_report_path, repo_root).replace("\\", "/")
    rel_md = os.path.relpath(markdown_path, repo_root).replace("\\", "/")

    lines = [
        "# ✅ 多场比赛详情报告已保存",
        f"- html_path: {rel_html}",
        f"- markdown_path: {rel_md}",
    ]
    if match_ids:
        lines.append(f"- match_ids: {', '.join(str(mid) for mid in match_ids)}")
    lines.extend([
        "",
        "## 📊 可视化（Markdown）",
        report_markdown.strip(),
    ])
    return "\n".join(lines)


@mcp.tool()
def get_ward_statistics(match_id: int) -> str:
    """
    获取指定比赛的眼位统计数据（不生成可视化文件）
    
    Args:
        match_id: Dota 2 比赛ID
    
    Returns:
        眼位统计数据，包括假眼、真眼的数量和时间分布
    """
    # 获取比赛详情
    match_data = _make_request(f"matches/{match_id}")
    
    if "error" in match_data:
        return f"❌ API 错误: {match_data['error']}"
    
    if not match_data:
        return f"❌ 无法获取比赛 {match_id} 的数据"
    
    # 提取眼位数据
    extractor = WardDataExtractor()
    
    if not extractor.extract_from_match(match_data):
        return f"❌ 比赛 {match_id} 无眼位数据（可能未解析或无观察者数据）"
    
    df_obs, df_sen = extractor.get_dataframes()
    
    if df_obs.empty and df_sen.empty:
        return f"❌ 比赛 {match_id} 无眼位数据"
    
    # 获取队伍名称
    radiant_name = match_data.get("radiant_name") or "天辉 Radiant"
    dire_name = match_data.get("dire_name") or "夜魇 Dire"
    
    # 创建分析器
    analyzer = WardAnalyzer(df_obs, df_sen, radiant_name, dire_name)
    
    # 返回统计摘要
    return analyzer.get_stats_summary()


@mcp.tool()
def get_live_matches(limit: int = 10) -> str:
    """
    获取正在进行的实时比赛
    
    Args:
        limit: 返回的比赛数量，默认10
    
    Returns:
        正在进行的高分比赛列表，按MMR排序
    """
    data = _make_request("live")
    
    if isinstance(data, dict) and "error" in data:
        return f"❌ API 错误: {data['error']}"
    
    if not isinstance(data, list):
        return "❌ 获取实时比赛失败"
    
    if not data:
        return "当前没有正在进行的比赛"
    
    # 按 MMR 排序
    sorted_matches = sorted(data, key=lambda x: x.get("average_mmr", 0), reverse=True)[:limit]
    
    hero_map = _build_hero_map()
    
    lines = [
        f"# 🔴 正在进行的比赛 ({len(data)} 场)",
        "",
    ]
    
    for i, m in enumerate(sorted_matches, 1):
        match_id = m.get("match_id", "N/A")
        avg_mmr = m.get("average_mmr", 0)
        game_time = m.get("game_time", 0)
        minutes = game_time // 60
        seconds = game_time % 60
        
        radiant_score = m.get("radiant_score", 0)
        dire_score = m.get("dire_score", 0)
        spectators = m.get("spectators", 0)
        
        lines.append(f"## [{i}] Match {match_id}")
        lines.append(f"- MMR: {avg_mmr} | 时间: {minutes}:{seconds:02d} | 观众: {spectators}")
        lines.append(f"- 比分: 天辉 {radiant_score} - {dire_score} 夜魇")
        
        # 显示英雄
        players = m.get("players", [])
        radiant = [p for p in players if p.get("team") == 0]
        dire = [p for p in players if p.get("team") == 1]
        
        radiant_heroes = [_get_cn_name(hero_map.get(p.get("hero_id"), "?")) for p in radiant]
        dire_heroes = [_get_cn_name(hero_map.get(p.get("hero_id"), "?")) for p in dire]
        
        lines.append(f"- 天辉: {', '.join(radiant_heroes)}")
        lines.append(f"- 夜魇: {', '.join(dire_heroes)}")
        lines.append("")
    
    return "\n".join(lines)


# ==================== 主入口 ====================

if __name__ == "__main__":
    mcp.run()

