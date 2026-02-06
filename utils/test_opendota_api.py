# test_opendota_api.py
"""
OpenDota API 测试脚本

用于测试和探索 OpenDota API 的各个端点，了解能获取哪些数据。
API 文档: https://docs.opendota.com/
"""

import requests
import json
from typing import Optional, Dict, Any, List
from datetime import datetime

# ==================== 配置 ====================

BASE_URL = "https://api.opendota.com/api"
TIMEOUT = 30

# 测试用的数据
TEST_MATCH_ID = 8652316527  # 示例比赛 ID
TEST_ACCOUNT_ID = 355962940  # 示例玩家账号 ID
TEST_HERO_ID = 1  # Anti-Mage


# ==================== 字段中文注释 ====================

FIELD_COMMENTS = {
    # 英雄相关
    "id": "唯一标识符",
    "hero_id": "英雄ID",
    "name": "内部名称/代号",
    "localized_name": "本地化显示名称",
    "primary_attr": "主属性 (str=力量/agi=敏捷/int=智力/all=全能)",
    "attack_type": "攻击类型 (Melee=近战/Ranged=远程)",
    "roles": "英雄定位列表",
    "legs": "腿的数量",
    "img": "英雄图片路径",
    "icon": "英雄图标路径",
    "base_health": "基础生命值",
    "base_health_regen": "基础生命恢复",
    "base_mana": "基础魔法值",
    "base_mana_regen": "基础魔法恢复",
    "base_armor": "基础护甲",
    "base_mr": "基础魔法抗性",
    "base_attack_min": "基础攻击力(最小)",
    "base_attack_max": "基础攻击力(最大)",
    "base_str": "基础力量",
    "base_agi": "基础敏捷",
    "base_int": "基础智力",
    "str_gain": "力量成长",
    "agi_gain": "敏捷成长",
    "int_gain": "智力成长",
    "attack_range": "攻击距离",
    "projectile_speed": "弹道速度",
    "attack_rate": "攻击间隔",
    "base_attack_time": "基础攻击时间",
    "attack_point": "攻击前摇",
    "move_speed": "移动速度",
    "turn_rate": "转身速率",
    "cm_enabled": "是否队长模式可用",
    "turbo_picks": "加速模式选用次数",
    "turbo_wins": "加速模式获胜次数",
    "pro_ban": "职业赛禁用次数",
    "pro_win": "职业赛获胜次数",
    "pro_pick": "职业赛选用次数",
    "null_pick": "无效选用",
    "pub_pick": "路人局选用次数",
    "pub_win": "路人局获胜次数",
    
    # 比赛相关
    "match_id": "比赛唯一ID",
    "match_seq_num": "比赛序列号",
    "radiant_win": "天辉是否获胜",
    "duration": "比赛时长(秒)",
    "pre_game_duration": "赛前准备时长(秒)",
    "start_time": "开始时间(Unix时间戳)",
    "game_time": "当前游戏时间(秒)",
    "radiant_score": "天辉击杀数",
    "dire_score": "夜魇击杀数",
    "radiant_lead": "天辉经济领先值",
    "game_mode": "游戏模式ID (1=全选/2=队长模式/22=天梯)",
    "lobby_type": "大厅类型ID (0=普通/7=天梯)",
    "cluster": "服务器集群ID",
    "region": "服务器区域",
    "patch": "游戏版本号",
    "picks_bans": "BP选禁列表",
    "players": "玩家数据列表",
    "first_blood_time": "一血时间(秒)",
    "tower_status_radiant": "天辉塔状态(位掩码)",
    "tower_status_dire": "夜魇塔状态(位掩码)",
    "barracks_status_radiant": "天辉兵营状态",
    "barracks_status_dire": "夜魇兵营状态",
    "radiant_gold_adv": "天辉经济优势时间序列",
    "radiant_xp_adv": "天辉经验优势时间序列",
    "teamfights": "团战数据",
    "objectives": "目标事件(如肉山/塔)",
    "chat": "聊天记录",
    "cosmetics": "饰品数据",
    "series_id": "系列赛ID",
    "series_type": "系列赛类型 (0=非系列/1=BO3/2=BO5)",
    "replay_salt": "录像加密盐值",
    "replay_url": "录像下载URL",
    "human_players": "人类玩家数量",
    "positive_votes": "点赞数",
    "negative_votes": "点踩数",
    "engine": "游戏引擎版本",
    "version": "数据版本",
    "skill": "技能等级 (1=普通/2=高/3=非常高)",
    "avg_rank_tier": "平均段位",
    "num_rank_tier": "有段位的玩家数",
    "radiant_team": "天辉队伍英雄ID列表",
    "dire_team": "夜魇队伍英雄ID列表",
    
    # 玩家相关
    "account_id": "Steam账号ID(32位)",
    "steamid": "Steam完整ID(64位)",
    "personaname": "Steam昵称",
    "avatar": "头像URL(小)",
    "avatarmedium": "头像URL(中)",
    "avatarfull": "头像URL(大)",
    "profileurl": "Steam个人主页URL",
    "last_login": "最后登录时间",
    "loccountrycode": "国家代码",
    "plus": "是否Dota Plus订阅",
    "cheese": "芝士数量(捐赠)",
    "rank_tier": "段位等级 (11-85, 十位=勋章 个位=星数)",
    "leaderboard_rank": "天梯排名",
    "competitive_rank": "竞技天梯分",
    "solo_competitive_rank": "单排天梯分",
    "mmr_estimate": "估算MMR",
    "profile": "玩家资料对象",
    "fh_unavailable": "完整历史是否不可用",
    "is_contributor": "是否贡献者",
    "is_subscriber": "是否订阅者",
    
    # 玩家比赛数据
    "player_slot": "玩家位置 (0-4天辉/128-132夜魇)",
    "team_slot": "队伍内位置(1-5)",
    "team": "所属队伍 (0=天辉/1=夜魇)",
    "kills": "击杀数",
    "deaths": "死亡数",
    "assists": "助攻数",
    "kda": "KDA值",
    "last_hits": "正补数",
    "denies": "反补数",
    "gold_per_min": "每分钟金钱(GPM)",
    "xp_per_min": "每分钟经验(XPM)",
    "level": "等级",
    "net_worth": "身价/总资产",
    "hero_damage": "英雄伤害",
    "tower_damage": "建筑伤害",
    "hero_healing": "治疗量",
    "gold": "当前金钱",
    "gold_spent": "花费金钱",
    "item_0": "物品栏1",
    "item_1": "物品栏2",
    "item_2": "物品栏3",
    "item_3": "物品栏4",
    "item_4": "物品栏5",
    "item_5": "物品栏6",
    "item_neutral": "中立物品",
    "backpack_0": "背包1",
    "backpack_1": "背包2",
    "backpack_2": "背包3",
    "aghanims_scepter": "是否有阿哈利姆神杖",
    "aghanims_shard": "是否有阿哈利姆魔晶",
    "lane": "分路 (1=安全路/2=中路/3=劣势路)",
    "lane_role": "分路角色 (1=安全路/2=中路/3=劣势路/4=打野)",
    "is_roaming": "是否游走",
    "obs_placed": "放置真眼数",
    "sen_placed": "放置假眼数",
    "observer_uses": "使用真眼次数",
    "sentry_uses": "使用假眼次数",
    "camps_stacked": "堆叠野怪次数",
    "rune_pickups": "拾取神符次数",
    "stuns": "眩晕时长(秒)",
    "teamfight_participation": "团战参与率",
    "towers_killed": "推塔数",
    "courier_kills": "击杀信使数",
    "purchase_log": "购买记录",
    "ability_upgrades": "技能加点记录",
    "ability_upgrades_arr": "技能加点ID数组",
    "benchmarks": "表现基准对比",
    "party_id": "组队ID",
    "party_size": "组队人数",
    "permanent_buffs": "永久Buff(如肉山盾)",
    "actions_per_min": "每分钟操作数(APM)",
    "life_state_dead": "死亡状态时间",
    "buyback_log": "买活记录",
    "killed_by": "被击杀记录",
    "purchase": "物品购买统计",
    "damage": "伤害分布",
    "damage_taken": "承受伤害",
    "damage_inflictor": "造成伤害来源",
    "damage_inflictor_received": "受到伤害来源",
    "runes": "神符拾取统计",
    "multi_kills": "多杀统计",
    "kill_streaks": "连杀统计",
    "pings": "信号数量",
    "win": "是否获胜",
    "lose": "是否失败",
    "total_gold": "总金钱",
    "total_xp": "总经验",
    "ancient_kills": "远古野怪击杀",
    "neutral_kills": "中立单位击杀",
    "tower_kills": "防御塔击杀",
    "roshan_kills": "肉山击杀",
    "lane_kills": "线上击杀",
    "hero_kills": "英雄击杀",
    "observer_kills": "真眼排除",
    "sentry_kills": "假眼排除",
    "randomed": "是否随机选英雄",
    "pred_vict": "预测获胜",
    "isRadiant": "是否天辉方",
    "hero_variant": "英雄变体/皮肤",
    
    # 实时比赛
    "average_mmr": "平均MMR",
    "spectators": "观众数",
    "delay": "延迟(秒)",
    "activate_time": "比赛激活时间",
    "deactivate_time": "比赛结束时间",
    "server_steam_id": "服务器Steam ID",
    "lobby_id": "大厅ID",
    "sort_score": "排序分数(用于列表排序)",
    "last_update_time": "最后更新时间",
    "building_state": "建筑状态(位掩码)",
    "is_player_draft": "是否玩家选人模式",
    "is_watch_eligible": "是否可观战",
    "weekend_tourney_tournament_id": "周末联赛ID",
    "weekend_tourney_division": "周末联赛分区",
    "weekend_tourney_skill_level": "周末联赛技能等级",
    "weekend_tourney_bracket_round": "周末联赛淘汰赛轮次",
    "custom_game_difficulty": "自定义游戏难度",
    
    # 联赛/战队
    "league_id": "联赛ID",
    "leagueid": "联赛ID",
    "team_id": "战队ID",
    "team_name": "战队名称",
    "team_name_radiant": "天辉战队名",
    "team_name_dire": "夜魇战队名",
    "team_logo_radiant": "天辉战队Logo",
    "team_logo_dire": "夜魇战队Logo",
    "team_id_radiant": "天辉战队ID",
    "team_id_dire": "夜魇战队ID",
    "tag": "战队标签/简称",
    "logo_url": "Logo URL",
    "rating": "战队/玩家评分",
    "wins": "胜场数",
    "losses": "负场数",
    "last_match_time": "最后比赛时间",
    "tier": "联赛等级",
    "ticket": "门票",
    "banner": "横幅URL",
    "radiant_team_id": "天辉战队ID",
    "dire_team_id": "夜魇战队ID",
    "radiant_team_name": "天辉战队名",
    "dire_team_name": "夜魇战队名",
    "radiant_team_complete": "天辉队伍是否完整",
    "dire_team_complete": "夜魇队伍是否完整",
    "radiant_captain": "天辉队长账号ID",
    "dire_captain": "夜魇队长账号ID",
    
    # 职业选手
    "fantasy_role": "梦幻联赛角色 (1=核心/2=辅助)",
    "team_tag": "战队标签",
    "is_locked": "是否锁定",
    "is_pro": "是否职业选手",
    "country_code": "国家代码",
    "locked_until": "锁定到期时间",
    
    # 搜索相关
    "similarity": "相似度分数",
    
    # 胜负统计
    "games": "比赛场数",
    "n": "样本数量",
    "sum": "总和",
    "field": "统计字段名",
    
    # 队友数据
    "with_games": "一起游戏场数",
    "with_win": "一起获胜场数",
    "against_games": "对抗场数",
    "against_win": "对抗获胜场数",
    "with_gpm_sum": "一起游戏GPM总和",
    "with_xpm_sum": "一起游戏XPM总和",
    "last_played": "最后一起游戏时间",
    
    # 排行/基准
    "score": "分数/评分",
    "percentile": "百分位",
    "raw": "原始值",
    "rankings": "排名列表",
    "result": "结果数据",
    
    # 数据库Schema
    "table_name": "数据库表名",
    "column_name": "字段/列名",
    "data_type": "数据类型",
    
    # 分布数据
    "ranks": "段位分布",
    "mmr": "MMR分布",
    "country_mmr": "国家MMR分布",
    "rows": "数据行",
    "bin": "分组区间",
    "bin_name": "区间名称",
    "count": "计数",
    "cumulative_sum": "累计总和",
}


def get_field_comment(field: str) -> str:
    """获取字段的中文注释"""
    return FIELD_COMMENTS.get(field, "")


def add_comments_to_data(data: Any) -> Any:
    """为数据添加中文注释，返回带注释的新数据结构"""
    if isinstance(data, dict):
        commented_data = {}
        for key, value in data.items():
            comment = get_field_comment(key)
            # 递归处理嵌套结构
            if isinstance(value, (dict, list)):
                processed_value = add_comments_to_data(value)
            else:
                processed_value = value
            
            # 使用带注释的键名格式: "key // 注释"
            if comment:
                commented_key = f"{key}  // {comment}"
            else:
                commented_key = key
            commented_data[commented_key] = processed_value
        return commented_data
    
    elif isinstance(data, list):
        # 对列表中的每个元素递归处理
        return [add_comments_to_data(item) for item in data]
    
    else:
        return data


# ==================== 辅助函数 ====================

def make_request(endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
    """发起 API 请求"""
    url = f"{BASE_URL}/{endpoint}"
    print(f"\n🔗 请求: {url}")
    if params:
        print(f"   参数: {params}")
    
    try:
        response = requests.get(url, params=params, timeout=TIMEOUT)
        response.raise_for_status()
        data = response.json()
        print(f"   状态: ✅ 成功")
        return {"success": True, "data": data}
    except requests.exceptions.RequestException as e:
        print(f"   状态: ❌ 失败 - {e}")
        return {"success": False, "error": str(e)}


def print_data_structure(data: Any, max_depth: int = 2, current_depth: int = 0, prefix: str = ""):
    """打印数据结构"""
    indent = "  " * current_depth
    
    if isinstance(data, dict):
        print(f"{indent}{prefix}Dict with {len(data)} keys:")
        if current_depth < max_depth:
            for key in list(data.keys())[:10]:  # 只显示前10个键
                value = data[key]
                print_data_structure(value, max_depth, current_depth + 1, f"[{key}] ")
            if len(data) > 10:
                print(f"{indent}  ... 还有 {len(data) - 10} 个键")
    elif isinstance(data, list):
        print(f"{indent}{prefix}List with {len(data)} items")
        if current_depth < max_depth and len(data) > 0:
            print_data_structure(data[0], max_depth, current_depth + 1, "[0] ")
            if len(data) > 1:
                print(f"{indent}  ... 还有 {len(data) - 1} 个元素")
    else:
        type_name = type(data).__name__
        value_preview = str(data)[:50] if data is not None else "None"
        if len(str(data)) > 50:
            value_preview += "..."
        print(f"{indent}{prefix}{type_name}: {value_preview}")


def print_section(title: str):
    """打印章节标题"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def save_sample(name: str, data: Any, output_dir: str = "api_samples", with_comments: bool = True):
    """保存示例数据到文件
    
    Args:
        name: 文件名(不含扩展名)
        data: 要保存的数据
        output_dir: 输出目录
        with_comments: 是否添加中文注释
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存带注释的版本
    if with_comments:
        commented_data = add_comments_to_data(data)
        filepath = os.path.join(output_dir, f"{name}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(commented_data, f, ensure_ascii=False, indent=2)
        print(f"   💾 已保存示例(带注释): {filepath}")
    else:
        filepath = os.path.join(output_dir, f"{name}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"   💾 已保存示例: {filepath}")


# ==================== API 测试函数 ====================

def test_heroes():
    """测试英雄列表 API"""
    print_section("1. 英雄列表 API - /heroes")
    
    result = make_request("heroes")
    if result["success"]:
        data = result["data"]
        print(f"\n📊 返回数据:")
        print(f"   类型: List")
        print(f"   数量: {len(data)} 个英雄")
        
        if data:
            print(f"\n📝 英雄数据字段:")
            hero = data[0]
            for key, value in hero.items():
                print(f"   - {key}: {type(value).__name__} = {str(value)[:50]}")
            
            save_sample("heroes", data[:3])
    
    return result


def test_hero_stats():
    """测试英雄统计 API"""
    print_section("2. 英雄统计 API - /heroStats")
    
    result = make_request("heroStats")
    if result["success"]:
        data = result["data"]
        print(f"\n📊 返回数据:")
        print(f"   类型: List")
        print(f"   数量: {len(data)} 个英雄统计")
        
        if data:
            print(f"\n📝 统计数据字段:")
            hero_stat = data[0]
            for key, value in list(hero_stat.items())[:20]:
                print(f"   - {key}: {type(value).__name__}")
            if len(hero_stat) > 20:
                print(f"   ... 还有 {len(hero_stat) - 20} 个字段")
            
            save_sample("hero_stats", data[:2])
    
    return result


def test_match_details(match_id: int = TEST_MATCH_ID):
    """测试比赛详情 API"""
    print_section(f"3. 比赛详情 API - /matches/{match_id}")
    
    result = make_request(f"matches/{match_id}")
    if result["success"]:
        data = result["data"]
        print(f"\n📊 返回数据:")
        print(f"   类型: Dict")
        print(f"   字段数: {len(data)} 个")
        
        print(f"\n📝 主要字段:")
        important_fields = [
            "match_id", "radiant_win", "duration", "start_time",
            "radiant_score", "dire_score", "game_mode", "lobby_type",
            "players", "picks_bans", "patch", "region"
        ]
        for field in important_fields:
            if field in data:
                value = data[field]
                if isinstance(value, list):
                    print(f"   - {field}: List[{len(value)}]")
                elif isinstance(value, dict):
                    print(f"   - {field}: Dict[{len(value)}]")
                else:
                    print(f"   - {field}: {value}")
        
        # 玩家数据字段
        if "players" in data and data["players"]:
            print(f"\n📝 玩家数据字段 (players[0]):")
            player = data["players"][0]
            for key in list(player.keys())[:25]:
                value = player[key]
                print(f"   - {key}: {type(value).__name__}")
            if len(player) > 25:
                print(f"   ... 还有 {len(player) - 25} 个字段")
        
        save_sample("match_details", data)
    
    return result


def test_player_info(account_id: int = TEST_ACCOUNT_ID):
    """测试玩家信息 API"""
    print_section(f"4. 玩家信息 API - /players/{account_id}")
    
    result = make_request(f"players/{account_id}")
    if result["success"]:
        data = result["data"]
        print(f"\n📊 返回数据:")
        print_data_structure(data, max_depth=2)
        
        save_sample("player_info", data)
    
    return result


def test_player_win_loss(account_id: int = TEST_ACCOUNT_ID):
    """测试玩家胜负 API"""
    print_section(f"5. 玩家胜负 API - /players/{account_id}/wl")
    
    result = make_request(f"players/{account_id}/wl")
    if result["success"]:
        data = result["data"]
        print(f"\n📊 返回数据:")
        for key, value in data.items():
            print(f"   - {key}: {value}")
        
        save_sample("player_win_loss", data)
    
    return result


def test_player_recent_matches(account_id: int = TEST_ACCOUNT_ID):
    """测试玩家最近比赛 API"""
    print_section(f"6. 玩家最近比赛 API - /players/{account_id}/recentMatches")
    
    result = make_request(f"players/{account_id}/recentMatches")
    if result["success"]:
        data = result["data"]
        print(f"\n📊 返回数据:")
        print(f"   类型: List")
        print(f"   数量: {len(data)} 场比赛")
        
        if data:
            print(f"\n📝 比赛记录字段:")
            match = data[0]
            for key, value in match.items():
                print(f"   - {key}: {type(value).__name__} = {str(value)[:30]}")
            
            save_sample("player_recent_matches", data[:3])
    
    return result


def test_player_heroes(account_id: int = TEST_ACCOUNT_ID):
    """测试玩家英雄数据 API"""
    print_section(f"7. 玩家英雄数据 API - /players/{account_id}/heroes")
    
    result = make_request(f"players/{account_id}/heroes")
    if result["success"]:
        data = result["data"]
        print(f"\n📊 返回数据:")
        print(f"   类型: List")
        print(f"   数量: {len(data)} 个英雄记录")
        
        if data:
            print(f"\n📝 英雄记录字段:")
            hero_data = data[0]
            for key, value in hero_data.items():
                print(f"   - {key}: {type(value).__name__} = {value}")
            
            save_sample("player_heroes", data[:5])
    
    return result


def test_player_peers(account_id: int = TEST_ACCOUNT_ID):
    """测试玩家队友 API"""
    print_section(f"8. 玩家队友 API - /players/{account_id}/peers")
    
    result = make_request(f"players/{account_id}/peers")
    if result["success"]:
        data = result["data"]
        print(f"\n📊 返回数据:")
        print(f"   类型: List")
        print(f"   数量: {len(data)} 个队友记录")
        
        if data:
            print(f"\n📝 队友记录字段:")
            peer = data[0]
            for key, value in peer.items():
                print(f"   - {key}: {type(value).__name__}")
            
            save_sample("player_peers", data[:5])
    
    return result


def test_player_totals(account_id: int = TEST_ACCOUNT_ID):
    """测试玩家统计总计 API"""
    print_section(f"9. 玩家统计总计 API - /players/{account_id}/totals")
    
    result = make_request(f"players/{account_id}/totals")
    if result["success"]:
        data = result["data"]
        print(f"\n📊 返回数据:")
        print(f"   类型: List")
        print(f"   数量: {len(data)} 个统计项")
        
        if data:
            print(f"\n📝 统计项示例:")
            for item in data[:10]:
                print(f"   - {item.get('field')}: {item.get('sum')} (n={item.get('n')})")
            
            save_sample("player_totals", data)
    
    return result


def test_pro_players():
    """测试职业选手 API"""
    print_section("10. 职业选手 API - /proPlayers")
    
    result = make_request("proPlayers")
    if result["success"]:
        data = result["data"]
        print(f"\n📊 返回数据:")
        print(f"   类型: List")
        print(f"   数量: {len(data)} 个职业选手")
        
        if data:
            print(f"\n📝 职业选手字段:")
            player = data[0]
            for key, value in player.items():
                print(f"   - {key}: {type(value).__name__}")
            
            save_sample("pro_players", data[:5])
    
    return result


def test_pro_matches():
    """测试职业比赛 API"""
    print_section("11. 职业比赛 API - /proMatches")
    
    result = make_request("proMatches")
    if result["success"]:
        data = result["data"]
        print(f"\n📊 返回数据:")
        print(f"   类型: List")
        print(f"   数量: {len(data)} 场职业比赛")
        
        if data:
            print(f"\n📝 职业比赛字段:")
            match = data[0]
            for key, value in match.items():
                print(f"   - {key}: {type(value).__name__} = {str(value)[:30]}")
            
            save_sample("pro_matches", data[:5])
    
    return result


def test_public_matches():
    """测试公开比赛 API"""
    print_section("12. 公开比赛 API - /publicMatches")
    
    result = make_request("publicMatches", params={"min_rank": 70})  # 神话以上
    if result["success"]:
        data = result["data"]
        print(f"\n📊 返回数据:")
        print(f"   类型: List")
        print(f"   数量: {len(data)} 场公开比赛")
        
        if data:
            print(f"\n📝 公开比赛字段:")
            match = data[0]
            for key, value in match.items():
                print(f"   - {key}: {type(value).__name__}")
            
            save_sample("public_matches", data[:5])
    
    return result


def test_teams():
    """测试战队 API"""
    print_section("13. 战队 API - /teams")
    
    result = make_request("teams")
    if result["success"]:
        data = result["data"]
        print(f"\n📊 返回数据:")
        print(f"   类型: List")
        print(f"   数量: {len(data)} 个战队")
        
        if data:
            print(f"\n📝 战队字段:")
            team = data[0]
            for key, value in team.items():
                print(f"   - {key}: {type(value).__name__}")
            
            save_sample("teams", data[:5])
    
    return result


def test_team_info(team_id: int):
    """测试战队信息 API"""
    print_section(f"13a. 战队信息 API - /teams/{team_id}")
    
    result = make_request(f"teams/{team_id}")
    if result["success"]:
        data = result["data"]
        print(f"\n📊 战队信息:")
        if data:
            print(f"   名称: {data.get('name', 'N/A')}")
            print(f"   标签: {data.get('tag', 'N/A')}")
            print(f"   评分: {data.get('rating', 'N/A')}")
            print(f"   胜场: {data.get('wins', 0)}")
            print(f"   负场: {data.get('losses', 0)}")
            
            save_sample("team_info", data)
    
    return result


def test_team_matches(team_id: int):
    """测试战队比赛 API - 获取指定战队的比赛列表"""
    print_section(f"13b. 战队比赛 API - /teams/{team_id}/matches")
    
    result = make_request(f"teams/{team_id}/matches")
    if result["success"]:
        data = result["data"]
        print(f"\n📊 返回数据:")
        print(f"   类型: List")
        print(f"   数量: {len(data)} 场比赛")
        
        if data:
            print(f"\n📝 比赛记录字段:")
            match = data[0]
            for key, value in match.items():
                print(f"   - {key}: {type(value).__name__}")
            
            # 显示最近 5 场比赛
            print(f"\n🎮 最近 5 场比赛:")
            for i, match in enumerate(data[:5]):
                match_id = match.get("match_id", "N/A")
                duration = match.get("duration", 0)
                radiant_win = match.get("radiant_win")
                radiant = match.get("radiant", False)  # 该战队是否为天辉方
                
                # 判断该战队是否获胜
                if radiant_win is not None:
                    team_win = (radiant and radiant_win) or (not radiant and not radiant_win)
                    result_str = "✅ 胜" if team_win else "❌ 负"
                else:
                    result_str = "⏳ 进行中"
                
                # 格式化时长
                minutes = duration // 60
                seconds = duration % 60
                
                # 对手信息
                opposing_team_id = match.get("opposing_team_id", "N/A")
                opposing_team_name = match.get("opposing_team_name", "未知")
                league_name = match.get("league_name", "")
                
                print(f"\n   [{i+1}] 比赛 ID: {match_id}")
                print(f"       结果: {result_str} | 时长: {minutes}:{seconds:02d}")
                print(f"       对手: {opposing_team_name} (ID: {opposing_team_id})")
                if league_name:
                    print(f"       联赛: {league_name}")
            
            save_sample("team_matches", data[:10])
    
    return result


def search_team_by_name(team_name: str) -> Optional[Dict[str, Any]]:
    """通过战队名搜索战队（支持模糊匹配）
    
    Args:
        team_name: 战队名称（支持部分匹配）
    
    Returns:
        匹配的战队信息，或 None
    """
    result = make_request("teams")
    if not result["success"]:
        return None
    
    teams = result["data"]
    team_name_lower = team_name.lower()
    
    # 精确匹配
    for team in teams:
        if team.get("name", "").lower() == team_name_lower:
            return team
        if team.get("tag", "").lower() == team_name_lower:
            return team
    
    # 模糊匹配（名称包含搜索词）
    matches = []
    for team in teams:
        name = team.get("name", "").lower()
        tag = team.get("tag", "").lower()
        if team_name_lower in name or team_name_lower in tag:
            matches.append(team)
    
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        print(f"\n⚠️ 找到 {len(matches)} 个匹配的战队:")
        for i, team in enumerate(matches[:10]):
            print(f"   [{i+1}] {team.get('name')} ({team.get('tag')}) - ID: {team.get('team_id')}")
        return matches[0]  # 返回第一个匹配
    
    return None


def test_team_matches_by_name(team_name: str):
    """通过战队名查询最近比赛
    
    Args:
        team_name: 战队名称（如 "Team Spirit", "OG", "LGD" 等）
    """
    print_section(f"🔍 搜索战队: {team_name}")
    
    # 1. 搜索战队
    team = search_team_by_name(team_name)
    
    if not team:
        print(f"\n❌ 未找到战队: {team_name}")
        print("   提示: 尝试使用战队标签(如 'TSpirit')或完整名称(如 'Team Spirit')")
        return {"success": False, "error": "Team not found"}
    
    team_id = team.get("team_id")
    print(f"\n✅ 找到战队:")
    print(f"   名称: {team.get('name')}")
    print(f"   标签: {team.get('tag')}")
    print(f"   ID: {team_id}")
    print(f"   评分: {team.get('rating', 'N/A')}")
    print(f"   战绩: {team.get('wins', 0)} 胜 / {team.get('losses', 0)} 负")
    
    # 2. 获取战队比赛
    return test_team_matches(team_id)


def test_leagues():
    """测试联赛 API"""
    print_section("14. 联赛 API - /leagues")
    
    result = make_request("leagues")
    if result["success"]:
        data = result["data"]
        print(f"\n📊 返回数据:")
        print(f"   类型: List")
        print(f"   数量: {len(data)} 个联赛")
        
        if data:
            print(f"\n📝 联赛字段:")
            league = data[0]
            for key, value in league.items():
                print(f"   - {key}: {type(value).__name__}")
            
            save_sample("leagues", data[:5])
    
    return result


def test_rankings():
    """测试英雄排行 API"""
    print_section(f"15. 英雄排行 API - /rankings (hero_id={TEST_HERO_ID})")
    
    result = make_request("rankings", params={"hero_id": TEST_HERO_ID})
    if result["success"]:
        data = result["data"]
        print(f"\n📊 返回数据:")
        print_data_structure(data, max_depth=2)
        
        save_sample("rankings", data)
    
    return result


def test_benchmarks():
    """测试英雄基准数据 API"""
    print_section(f"16. 英雄基准数据 API - /benchmarks (hero_id={TEST_HERO_ID})")
    
    result = make_request("benchmarks", params={"hero_id": TEST_HERO_ID})
    if result["success"]:
        data = result["data"]
        print(f"\n📊 返回数据:")
        print_data_structure(data, max_depth=2)
        
        save_sample("benchmarks", data)
    
    return result


def test_distributions():
    """测试分布数据 API"""
    print_section("17. 分布数据 API - /distributions")
    
    result = make_request("distributions")
    if result["success"]:
        data = result["data"]
        print(f"\n📊 返回数据:")
        print(f"   类型: Dict")
        print(f"   包含的分布类型:")
        for key in data.keys():
            print(f"   - {key}")
        
        save_sample("distributions", data)
    
    return result


def test_schema():
    """测试数据库 Schema API - 获取数据库表结构"""
    print_section("18. 数据库 Schema API - /schema")
    
    result = make_request("schema")
    if result["success"]:
        data = result["data"]
        print(f"\n📊 返回数据:")
        print(f"   类型: List")
        print(f"   字段数量: {len(data)} 个")
        
        # 按表名分组统计
        tables = {}
        for item in data:
            table_name = item.get("table_name", "unknown")
            if table_name not in tables:
                tables[table_name] = []
            tables[table_name].append({
                "column": item.get("column_name"),
                "type": item.get("data_type")
            })
        
        print(f"   数据库表数量: {len(tables)} 个")
        
        # 显示主要表及其字段数
        print(f"\n📝 数据库表列表:")
        
        # 按字段数排序，显示最重要的表
        sorted_tables = sorted(tables.items(), key=lambda x: len(x[1]), reverse=True)
        
        # 分类显示
        important_tables = ["matches", "players", "player_matches", "heroes", "items", "teams", "leagues"]
        
        print(f"\n   🎮 核心游戏表:")
        for table in important_tables:
            if table in tables:
                cols = tables[table]
                print(f"      - {table}: {len(cols)} 个字段")
        
        print(f"\n   📊 其他表 (按字段数排序):")
        shown = 0
        for table_name, cols in sorted_tables:
            if table_name not in important_tables and shown < 15:
                print(f"      - {table_name}: {len(cols)} 个字段")
                shown += 1
        
        remaining = len(tables) - len(important_tables) - shown
        if remaining > 0:
            print(f"      ... 还有 {remaining} 个表")
        
        # 显示示例表结构
        print(f"\n📋 示例表结构 (matches):")
        if "matches" in tables:
            for col in tables["matches"][:10]:
                print(f"      - {col['column']}: {col['type']}")
            if len(tables["matches"]) > 10:
                print(f"      ... 还有 {len(tables['matches']) - 10} 个字段")
        
        print(f"\n📋 示例表结构 (players):")
        if "players" in tables:
            for col in tables["players"][:10]:
                print(f"      - {col['column']}: {col['type']}")
            if len(tables["players"]) > 10:
                print(f"      ... 还有 {len(tables['players']) - 10} 个字段")
        
        # 保存完整数据和按表分组的数据
        save_sample("schema_raw", data)
        save_sample("schema_tables", {k: v for k, v in sorted_tables[:20]})
    
    return result


def test_search():
    """测试搜索 API"""
    print_section("19. 搜索 API - /search (q=Miracle)")
    
    result = make_request("search", params={"q": "Miracle"})
    if result["success"]:
        data = result["data"]
        print(f"\n📊 返回数据:")
        print(f"   类型: List")
        print(f"   数量: {len(data)} 个结果")
        
        if data:
            print(f"\n📝 搜索结果字段:")
            item = data[0]
            for key, value in item.items():
                print(f"   - {key}: {type(value).__name__} = {value}")
            
            save_sample("search", data[:5])
    
    return result


def test_live():
    """测试正在进行的比赛 API - 获取实时比赛数据"""
    print_section("20. 实时比赛 API - /live")
    
    result = make_request("live")
    if result["success"]:
        data = result["data"]
        print(f"\n📊 返回数据:")
        print(f"   类型: List")
        print(f"   正在进行的比赛数: {len(data)} 场")
        
        if data:
            # 按平均 MMR 排序，显示高分比赛
            sorted_matches = sorted(data, key=lambda x: x.get("average_mmr", 0), reverse=True)
            
            # 统计信息
            mmr_values = [m.get("average_mmr", 0) for m in data if m.get("average_mmr")]
            if mmr_values:
                print(f"\n📈 MMR 统计:")
                print(f"   最高 MMR: {max(mmr_values)}")
                print(f"   最低 MMR: {min(mmr_values)}")
                print(f"   平均 MMR: {sum(mmr_values) // len(mmr_values)}")
            
            # 显示前5场高分比赛
            print(f"\n🏆 高分比赛 TOP 5:")
            for i, match in enumerate(sorted_matches[:5]):
                match_id = match.get("match_id", "N/A")
                avg_mmr = match.get("average_mmr", 0)
                game_time = match.get("game_time", 0)
                radiant_score = match.get("radiant_score", 0)
                dire_score = match.get("dire_score", 0)
                spectators = match.get("spectators", 0)
                
                # 格式化游戏时间
                minutes = game_time // 60
                seconds = game_time % 60
                time_str = f"{minutes}:{seconds:02d}"
                
                print(f"\n   [{i+1}] 比赛 ID: {match_id}")
                print(f"       MMR: {avg_mmr} | 时长: {time_str} | 比分: {radiant_score}-{dire_score}")
                print(f"       观众: {spectators} | 模式: {match.get('game_mode', 'N/A')}")
                
                # 显示玩家英雄
                players = match.get("players", [])
                radiant = [p for p in players if p.get("team") == 0]
                dire = [p for p in players if p.get("team") == 1]
                
                radiant_heroes = [str(p.get("hero_id", "?")) for p in radiant]
                dire_heroes = [str(p.get("hero_id", "?")) for p in dire]
                
                print(f"       天辉英雄: {', '.join(radiant_heroes)}")
                print(f"       夜魇英雄: {', '.join(dire_heroes)}")
            
            # 显示数据字段
            print(f"\n📝 比赛数据字段:")
            match = data[0]
            important_fields = [
                "match_id", "average_mmr", "game_time", "game_mode", "lobby_type",
                "radiant_score", "dire_score", "radiant_lead", "spectators",
                "team_name_radiant", "team_name_dire", "league_id", "players"
            ]
            for field in important_fields:
                if field in match:
                    value = match[field]
                    if isinstance(value, list):
                        print(f"   - {field}: List[{len(value)}]")
                    else:
                        print(f"   - {field}: {type(value).__name__} = {str(value)[:50]}")
            
            # 玩家数据字段
            if match.get("players"):
                print(f"\n📝 玩家数据字段 (players[0]):")
                player = match["players"][0]
                for key, value in player.items():
                    print(f"   - {key}: {type(value).__name__} = {value}")
            
            save_sample("live_matches", sorted_matches[:5])
    
    return result


# ==================== 主函数 ====================

def run_all_tests():
    """运行所有测试"""
    print("\n" + "🎮 " * 20)
    print("  OpenDota API 测试脚本")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("🎮 " * 20)
    
    tests = [
        ("英雄列表", test_heroes),
        ("英雄统计", test_hero_stats),
        ("比赛详情", test_match_details),
        ("玩家信息", test_player_info),
        ("玩家胜负", test_player_win_loss),
        ("玩家最近比赛", test_player_recent_matches),
        ("玩家英雄数据", test_player_heroes),
        ("玩家队友", test_player_peers),
        ("玩家统计总计", test_player_totals),
        ("职业选手", test_pro_players),
        ("职业比赛", test_pro_matches),
        ("公开比赛", test_public_matches),
        ("战队", test_teams),
        ("联赛", test_leagues),
        ("英雄排行", test_rankings),
        ("英雄基准", test_benchmarks),
        ("分布数据", test_distributions),
        ("数据库Schema", test_schema),
        ("搜索", test_search),
        ("实时比赛", test_live),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result["success"]))
        except Exception as e:
            print(f"\n❌ {name} 测试出错: {e}")
            results.append((name, False))
    
    # 打印测试结果汇总
    print_section("测试结果汇总")
    success_count = sum(1 for _, success in results if success)
    print(f"\n✅ 成功: {success_count}/{len(results)}")
    print(f"❌ 失败: {len(results) - success_count}/{len(results)}")
    
    print(f"\n详细结果:")
    for name, success in results:
        status = "✅" if success else "❌"
        print(f"   {status} {name}")
    
    print(f"\n📁 示例数据已保存到 api_samples/ 目录")


def interactive_test():
    """交互式测试"""
    print("\n🎮 OpenDota API 交互式测试")
    print("=" * 60)
    print("可用命令:")
    print("  1-20  - 运行对应编号的测试")
    print("  all   - 运行所有测试")
    print("  match <id>     - 查询指定比赛")
    print("  player <id>    - 查询指定玩家")
    print("  team <name>    - 🆕 通过战队名搜索最近比赛")
    print("  team_id <id>   - 🆕 通过战队ID查询比赛")
    print("  live           - 查看正在进行的比赛")
    print("  schema         - 查看数据库表结构")
    print("  quit  - 退出")
    print("=" * 60)
    
    while True:
        try:
            cmd = input("\n> ").strip()
            cmd_lower = cmd.lower()
            
            if cmd_lower in ['quit', 'q', 'exit']:
                print("再见！")
                break
            elif cmd_lower == 'all':
                run_all_tests()
            elif cmd_lower.startswith('match '):
                match_id = int(cmd.split()[1])
                test_match_details(match_id)
            elif cmd_lower.startswith('player '):
                account_id = int(cmd.split()[1])
                test_player_info(account_id)
            elif cmd_lower.startswith('team_id '):
                # 通过战队 ID 查询比赛
                team_id = int(cmd.split()[1])
                test_team_matches(team_id)
            elif cmd_lower.startswith('team '):
                # 通过战队名搜索比赛 (保留原始大小写)
                team_name = cmd[5:].strip()
                if team_name:
                    test_team_matches_by_name(team_name)
                else:
                    print("请输入战队名称，如: team Team Spirit")
            elif cmd_lower == 'live':
                test_live()
            elif cmd_lower == 'schema':
                test_schema()
            elif cmd_lower.isdigit():
                num = int(cmd_lower)
                tests = {
                    1: test_heroes, 2: test_hero_stats, 3: test_match_details,
                    4: test_player_info, 5: test_player_win_loss, 6: test_player_recent_matches,
                    7: test_player_heroes, 8: test_player_peers, 9: test_player_totals,
                    10: test_pro_players, 11: test_pro_matches, 12: test_public_matches,
                    13: test_teams, 14: test_leagues, 15: test_rankings,
                    16: test_benchmarks, 17: test_distributions, 18: test_schema,
                    19: test_search, 20: test_live,
                }
                if num in tests:
                    tests[num]()
                else:
                    print(f"未知测试编号: {num} (可用: 1-20)")
            else:
                print("未知命令，输入 'quit' 退出")
                
        except KeyboardInterrupt:
            print("\n再见！")
            break
        except Exception as e:
            print(f"错误: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        run_all_tests()
    else:
        interactive_test()
