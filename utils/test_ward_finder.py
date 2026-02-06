# test_ward_finder.py
"""
Dota 2 单场比赛眼位分析工具 - 集成测试脚本

功能：
1. 支持指定比赛ID进行眼位分析
2. 从 OpenDota API 获取比赛数据
3. 提取假眼/真眼坐标和时间信息
4. 生成眼位分布可视化图表

地图版本：统一使用 maps/740.jpeg (7.40 版本)

使用方法：
    python test_ward_finder.py

    # 或者直接指定比赛ID：
    python -c "from test_ward_finder import fetch_and_analyze; fetch_and_analyze(match_id=8650993496)"
"""

import os
import json
import requests
from copy import deepcopy
from collections import OrderedDict
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

try:
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import silhouette_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("⚠️ scikit-learn 未安装，聚类功能不可用")
    print("   安装命令: pip install scikit-learn")

# ==================== 配置 ====================

BASE_URL = "https://api.opendota.com/api"
TIMEOUT = 30

# 地图目录
MAPS_DIR = "maps"

# 统一使用的地图版本
MAP_VERSION = "740"

def get_map_version_from_patch(patch_id: int) -> str:
    """获取地图版本（统一使用 7.40 地图）"""
    return MAP_VERSION


def get_map_path(version: str) -> Optional[str]:
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


# ==================== OpenDota API 工具 ====================

def make_request(endpoint: str, params: Optional[Dict] = None) -> Any:
    """发起 API 请求"""
    url = f"{BASE_URL}/{endpoint}"
    print(f"🔗 请求: {url}")
    
    try:
        response = requests.get(url, params=params, timeout=TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ 请求失败: {e}")
        return None


def get_pro_matches(limit: int = 10) -> List[Dict]:
    """获取最近的职业比赛"""
    data = make_request("proMatches")
    if data:
        return data[:limit]
    return []


def get_match_details(match_id: int) -> Optional[Dict]:
    """获取比赛详情（包含眼位数据）"""
    return make_request(f"matches/{match_id}")


def get_patch_list() -> List[Dict]:
    """获取版本列表"""
    data = make_request("constants/patch")
    return data if data else []


# ==================== 数据提取类 ====================

class WardDataExtractor:
    """从比赛数据中提取眼位信息"""
    
    def __init__(self):
        self.obs_data = []  # 假眼数据
        self.sen_data = []  # 真眼数据
        self.objectives = []  # 目标数据
        self.patch_info = {}  # patch ID -> 比赛数量
    
    def extract_from_match(self, match_data: Dict) -> bool:
        """从单场比赛数据中提取眼位"""
        if not match_data:
            return False
        
        match_id = match_data.get("match_id")
        start_time = match_data.get("start_time", 0)
        patch = match_data.get("patch", 0)
        
        # 记录 patch 信息
        if patch not in self.patch_info:
            self.patch_info[patch] = 0
        self.patch_info[patch] += 1
        
        # 获取地图版本
        map_version = get_map_version_from_patch(patch)
        
        # 检查是否有解析数据
        if not match_data.get("players"):
            print(f"   ⚠️ 比赛 {match_id} 无玩家数据")
            return False
        
        # 提取目标时间
        objectives = match_data.get("objectives", [])
        obj_times = self._extract_objectives(match_id, objectives)
        self.objectives.append(obj_times)
        
        # 从每个玩家提取眼位
        for player in match_data.get("players", []):
            hero_id = player.get("hero_id")
            player_slot = player.get("player_slot", 0)
            is_radiant = 1 if player_slot < 128 else 0
            
            # 提取假眼
            obs_log = player.get("obs_log", [])
            for ward in obs_log:
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
                    **obj_times  # 添加目标时间
                })
            
            # 提取真眼
            sen_log = player.get("sen_log", [])
            for ward in sen_log:
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
                    **obj_times  # 添加目标时间
                })
        
        obs_count = len([w for w in self.obs_data if w["match_id"] == match_id])
        sen_count = len([w for w in self.sen_data if w["match_id"] == match_id])
        print(f"   ✅ 提取: {obs_count} 假眼, {sen_count} 真眼 (patch={patch}, map={map_version})")
        
        return obs_count > 0 or sen_count > 0
    
    def _extract_objectives(self, match_id: int, objectives: List) -> Dict:
        """提取目标事件时间"""
        result = {"match_id": match_id}
        
        # 最大时间（用于未发生的事件）
        MAX_TIME = 3 * 60 * 60
        
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
            result[tower] = MAX_TIME
        
        # 肉山击杀
        rosh_count = 0
        for i in range(4):
            result[f"ROSHAN_{i}"] = MAX_TIME
        
        # 解析目标事件
        for obj in objectives:
            obj_type = obj.get("type", "")
            key = obj.get("key", "")
            time = obj.get("time", MAX_TIME)
            
            if obj_type == "building_kill":
                # 转换键名
                col_name = key.replace("npc_dota_goodguys", "radiant")
                col_name = col_name.replace("npc_dota_badguys", "dire")
                if col_name in result:
                    result[col_name] = time
            
            elif obj_type == "CHAT_MESSAGE_ROSHAN_KILL":
                if rosh_count < 4:
                    result[f"ROSHAN_{rosh_count}"] = time
                    rosh_count += 1
        
        return result
    
    def get_dataframes(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """返回眼位数据的 DataFrame"""
        df_obs = pd.DataFrame(self.obs_data) if self.obs_data else pd.DataFrame()
        df_sen = pd.DataFrame(self.sen_data) if self.sen_data else pd.DataFrame()
        return df_obs, df_sen
    
    def save_to_csv(self, obs_path: str = "df_obs_test.csv", sen_path: str = "df_sen_test.csv"):
        """保存数据到 CSV"""
        df_obs, df_sen = self.get_dataframes()
        
        if not df_obs.empty:
            df_obs.to_csv(obs_path, index=False)
            print(f"💾 假眼数据已保存: {obs_path} ({len(df_obs)} 条)")
        
        if not df_sen.empty:
            df_sen.to_csv(sen_path, index=False)
            print(f"💾 真眼数据已保存: {sen_path} ({len(df_sen)} 条)")
    
    def print_patch_summary(self):
        """打印版本统计"""
        print("\n📊 版本分布:")
        for patch, count in sorted(self.patch_info.items()):
            map_ver = get_map_version_from_patch(patch)
            print(f"   Patch {patch} (地图 {map_ver}): {count} 场比赛")


# ==================== 眼位分析类 ====================

class WardAnalyzer:
    """眼位分析和可视化（支持多版本地图）"""
    
    def __init__(self, df_obs: pd.DataFrame, df_sen: pd.DataFrame):
        """
        初始化分析器
        
        Args:
            df_obs: 假眼数据（需包含 map_version 列）
            df_sen: 真眼数据（需包含 map_version 列）
        """
        self.df_obs = df_obs.copy()
        self.df_sen = df_sen.copy()
        
        # 坐标转换 (64,64) -> (0,0)
        if not self.df_obs.empty:
            self.df_obs["x"] = self.df_obs["x"] - 64
            self.df_obs["y"] = self.df_obs["y"] - 64
        
        if not self.df_sen.empty:
            self.df_sen["x"] = self.df_sen["x"] - 64
            self.df_sen["y"] = self.df_sen["y"] - 64
        
        # 获取所有地图版本
        self.map_versions = set()
        if not self.df_obs.empty and "map_version" in self.df_obs.columns:
            self.map_versions.update(self.df_obs["map_version"].unique())
        if not self.df_sen.empty and "map_version" in self.df_sen.columns:
            self.map_versions.update(self.df_sen["map_version"].unique())
        
        # 加载地图图片
        self.map_images = {}
        for version in self.map_versions:
            map_path = get_map_path(version)
            if map_path:
                try:
                    self.map_images[version] = Image.open(map_path)
                    print(f"✅ 加载地图 {version}: {map_path}")
                except Exception as e:
                    print(f"⚠️ 无法加载地图 {version}: {e}")
        
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
                    print(f"✅ 加载图标: {filename}")
                except Exception as e:
                    print(f"⚠️ 无法加载图标 {filename}: {e}")
        
        # 图标缩放比例
        self.icon_zoom = 0.55
        
        # 视野半径
        self.radius_obs = 8.96
        self.radius_sen = 5.76
    
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
        
        # 图例项配置: (icon_key, label_template)
        legend_config = [
            ("obs_radiant", "天辉假眼 Observer ({})"),
            ("obs_dire", "夜魇假眼 Observer ({})"),
            ("sen_radiant", "天辉真眼 Sentry ({})"),
            ("sen_dire", "夜魇真眼 Sentry ({})"),
        ]
        
        for icon_key, label_template in legend_config:
            count = counts.get(icon_key, 0)
            if icon_key in self.ward_icons:
                # 创建带图标的图例项
                img = OffsetImage(self.ward_icons[icon_key], zoom=0.25)
                legend_items.append(img)
                labels.append(label_template.format(count))
        
        # 在图的上方创建自定义图例区域
        legend_y = 1.12
        legend_x_start = 0.1
        legend_spacing = 0.22
        
        for i, (item, label) in enumerate(zip(legend_items, labels)):
            x_pos = legend_x_start + i * legend_spacing
            # 添加图标
            ab = AnnotationBbox(item, (x_pos, legend_y), 
                              xycoords='axes fraction', frameon=False)
            ax.add_artist(ab)
            # 添加文字
            ax.text(x_pos + 0.03, legend_y, label, transform=ax.transAxes,
                   fontsize=9, verticalalignment='center')
    
    def plot_scatter_by_version(self, save_dir: str = ".", figsize: Tuple = (12, 12)):
        """为每个地图版本分别绘制眼位散点图（使用图标）"""
        
        for version in self.map_versions:
            print(f"\n📊 生成地图 {version} 的眼位图...")
            
            fig, ax = plt.subplots(figsize=figsize)
            
            # 显示地图
            if version in self.map_images:
                ax.imshow(self.map_images[version], extent=[0, 128, 0, 128])
            else:
                ax.set_facecolor("gray")
            
            # 筛选该版本的数据
            if not self.df_obs.empty and "map_version" in self.df_obs.columns:
                df_obs_ver = self.df_obs[self.df_obs["map_version"] == version]
            else:
                df_obs_ver = self.df_obs
            
            if not self.df_sen.empty and "map_version" in self.df_sen.columns:
                df_sen_ver = self.df_sen[self.df_sen["map_version"] == version]
            else:
                df_sen_ver = self.df_sen
            
            # 统计各类眼位数量
            counts = {"obs_radiant": 0, "obs_dire": 0, "sen_radiant": 0, "sen_dire": 0}
            
            # 绘制假眼 (Observer) - 使用图标
            if not df_obs_ver.empty:
                obs_rad = df_obs_ver[df_obs_ver["is_radiant"] == 1]
                obs_dir = df_obs_ver[df_obs_ver["is_radiant"] == 0]
                counts["obs_radiant"] = len(obs_rad)
                counts["obs_dire"] = len(obs_dir)
                
                for _, row in obs_rad.iterrows():
                    self._add_ward_icon(ax, row["x"], row["y"], "obs_radiant")
                for _, row in obs_dir.iterrows():
                    self._add_ward_icon(ax, row["x"], row["y"], "obs_dire")

            # 绘制真眼 (Sentry) - 使用图标
            if not df_sen_ver.empty:
                sen_rad = df_sen_ver[df_sen_ver["is_radiant"] == 1]
                sen_dir = df_sen_ver[df_sen_ver["is_radiant"] == 0]
                counts["sen_radiant"] = len(sen_rad)
                counts["sen_dire"] = len(sen_dir)
                
                for _, row in sen_rad.iterrows():
                    self._add_ward_icon(ax, row["x"], row["y"], "sen_radiant")
                for _, row in sen_dir.iterrows():
                    self._add_ward_icon(ax, row["x"], row["y"], "sen_dire")

            ax.set_xlim(0, 128)
            ax.set_ylim(0, 128)

            # 获取比赛ID用于标题
            if not self.df_obs.empty and "match_id" in self.df_obs.columns:
                match_ids = self.df_obs["match_id"].unique()
                if len(match_ids) == 1:
                    title = f"Dota 2 眼位分布图 - 比赛 {match_ids[0]}"
                else:
                    title = f"Dota 2 眼位分布图 - {len(match_ids)} 场比赛"
            else:
                title = f"Dota 2 眼位分布图 - 地图版本 7.{version[-2:]}"

            ax.set_title(title, pad=60)  # 增加标题与图的间距，为图例留空间

            # 创建带图标的自定义图例
            self._create_icon_legend(ax, counts)
            
            save_path = os.path.join(save_dir, f"ward_scatter_{version}.png")
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"💾 已保存: {save_path}")
            
            plt.show()
            plt.close()
    
    
    def plot_comparison(self, save_path: str = None, figsize: Tuple = (18, 6)):
        """对比不同版本地图的眼位分布"""
        
        versions = sorted(self.map_versions)
        n_versions = len(versions)
        
        if n_versions == 0:
            print("❌ 无地图版本数据")
            return
        
        fig, axes = plt.subplots(1, n_versions, figsize=figsize)
        if n_versions == 1:
            axes = [axes]
        
        for idx, version in enumerate(versions):
            ax = axes[idx]
            
            # 显示地图
            if version in self.map_images:
                ax.imshow(self.map_images[version], extent=[0, 128, 0, 128], alpha=0.8)
            else:
                ax.set_facecolor("gray")
            
            # 筛选数据
            if not self.df_obs.empty and "map_version" in self.df_obs.columns:
                df_ver = self.df_obs[self.df_obs["map_version"] == version]
                
                if not df_ver.empty:
                    rad = df_ver[df_ver["is_radiant"] == 1]
                    dire = df_ver[df_ver["is_radiant"] == 0]
                    
                    ax.scatter(rad["x"], rad["y"], c="lime", alpha=0.7, s=40, marker="o", edgecolors="black", linewidth=1)
                    ax.scatter(dire["x"], dire["y"], c="red", alpha=0.7, s=40, marker="o", edgecolors="black", linewidth=1)
                    
                    count = len(df_ver)
                else:
                    count = 0
            else:
                count = 0
            
            ax.set_xlim(0, 128)
            ax.set_ylim(0, 128)
            ax.set_title(f"地图 7.{version[-2:]}\n({count} 个眼)")
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.suptitle("不同版本地图眼位对比", fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"💾 已保存: {save_path}")
        
        plt.show()
    
    def print_stats(self):
        """打印统计信息"""
        print("\n" + "=" * 50)
        print("📊 眼位数据统计")
        print("=" * 50)

        # 显示比赛ID
        if not self.df_obs.empty and "match_id" in self.df_obs.columns:
            match_ids = self.df_obs["match_id"].unique()
            if len(match_ids) == 1:
                print(f"🏆 比赛ID: {match_ids[0]}")
            else:
                print(f"🏆 比赛数量: {len(match_ids)} 个")

        # 按队伍统计
        if not self.df_obs.empty:
            obs_rad = len(self.df_obs[self.df_obs["is_radiant"] == 1])
            obs_dir = len(self.df_obs[self.df_obs["is_radiant"] == 0])
            print(f"\n假眼总计: {len(self.df_obs)}")
            print(f"   天辉: {obs_rad} 个")
            print(f"   夜魇: {obs_dir} 个")

        if not self.df_sen.empty:
            sen_rad = len(self.df_sen[self.df_sen["is_radiant"] == 1])
            sen_dir = len(self.df_sen[self.df_sen["is_radiant"] == 0])
            print(f"\n真眼总计: {len(self.df_sen)}")
            print(f"   天辉: {sen_rad} 个")
            print(f"   夜魇: {sen_dir} 个")

        # 时间分布
        if not self.df_obs.empty:
            early_wards = len(self.df_obs[self.df_obs["time"] <= 600])  # 前10分钟
            mid_wards = len(self.df_obs[(self.df_obs["time"] > 600) & (self.df_obs["time"] <= 1800)])  # 10-30分钟
            late_wards = len(self.df_obs[self.df_obs["time"] > 1800])  # 30分钟后

            print(f"\n⏰ 眼位时间分布:")
            print(f"   前10分钟: {early_wards} 个")
            print(f"   10-30分钟: {mid_wards} 个")
            print(f"   30分钟后: {late_wards} 个")

        print("=" * 50)
    
    def generate_interactive_html(self, save_path: str = "ward_timeline.html", 
                                   obs_duration: int = 360, sen_duration: int = 420):
        """
        生成交互式 HTML 页面，带时间滑动条
        
        Args:
            save_path: 保存路径
            obs_duration: 假眼持续时间（秒），默认 360
            sen_duration: 真眼持续时间（秒），默认 420
        """
        import base64
        from io import BytesIO
        
        # 获取地图版本
        version = list(self.map_versions)[0] if self.map_versions else MAP_VERSION
        
        # 将地图图片转为 base64
        map_base64 = ""
        if version in self.map_images:
            buffered = BytesIO()
            self.map_images[version].save(buffered, format="JPEG")
            map_base64 = base64.b64encode(buffered.getvalue()).decode()
        
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
        
        # 准备眼位数据（坐标已经在初始化时转换过了）
        wards_data = []
        
        # 处理假眼数据
        if not self.df_obs.empty:
            for _, row in self.df_obs.iterrows():
                ward_type = "obs_radiant" if row["is_radiant"] == 1 else "obs_dire"
                wards_data.append({
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "time": int(row["time"]),
                    "duration": obs_duration,
                    "type": ward_type,
                    "is_obs": True
                })
        
        # 处理真眼数据
        if not self.df_sen.empty:
            for _, row in self.df_sen.iterrows():
                ward_type = "sen_radiant" if row["is_radiant"] == 1 else "sen_dire"
                wards_data.append({
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "time": int(row["time"]),
                    "duration": sen_duration,
                    "type": ward_type,
                    "is_obs": False
                })
        
        # 计算时间范围（从 -1:30 开始）
        all_times = [w["time"] for w in wards_data]
        min_time = -90  # 固定从 -1:30 开始
        max_time = max(all_times) + max(obs_duration, sen_duration) if all_times else 3600
        
        # 获取比赛ID
        match_id = ""
        if not self.df_obs.empty and "match_id" in self.df_obs.columns:
            match_id = str(self.df_obs["match_id"].iloc[0])
        
        # 生成 HTML
        html_content = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dota 2 眼位时间线 - 比赛 {match_id}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            padding: 20px;
            color: #fff;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
        }}
        h1 {{
            text-align: center;
            margin-bottom: 20px;
            color: #f0f0f0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }}
        .map-container {{
            position: relative;
            width: 100%;
            max-width: 800px;
            margin: 0 auto;
            border: 3px solid #4a4a6a;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        }}
        .map-image {{
            width: 100%;
            display: block;
        }}
        .ward {{
            position: absolute;
            transform: translate(-50%, -50%);
            transition: opacity 0.2s ease;
            pointer-events: none;
            z-index: 10;
        }}
        .ward img {{
            width: 26px;
            height: 26px;
        }}
        .ward.hidden {{
            opacity: 0;
        }}
        .vision-circle {{
            position: absolute;
            border-radius: 50%;
            transform: translate(-50%, -50%);
            transition: opacity 0.2s ease;
            pointer-events: none;
            z-index: 5;
        }}
        .vision-circle.hidden {{
            opacity: 0;
        }}
        .vision-circle.radiant {{
            background: radial-gradient(circle, rgba(0, 255, 128, 0.25) 0%, rgba(0, 255, 128, 0.1) 70%, rgba(0, 255, 128, 0) 100%);
            border: 2px solid rgba(0, 255, 128, 0.4);
        }}
        .vision-circle.dire {{
            background: radial-gradient(circle, rgba(255, 80, 80, 0.25) 0%, rgba(255, 80, 80, 0.1) 70%, rgba(255, 80, 80, 0) 100%);
            border: 2px solid rgba(255, 80, 80, 0.4);
        }}
        .controls {{
            max-width: 800px;
            margin: 20px auto;
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 10px;
            backdrop-filter: blur(10px);
        }}
        .time-display {{
            text-align: center;
            font-size: 28px;
            font-weight: bold;
            margin-bottom: 15px;
            color: #ffd700;
        }}
        .slider-container {{
            display: flex;
            align-items: center;
            gap: 15px;
        }}
        .slider {{
            flex: 1;
            -webkit-appearance: none;
            height: 12px;
            border-radius: 6px;
            background: linear-gradient(to right, #2d5a27 0%, #8b4513 50%, #4a1a1a 100%);
            outline: none;
            cursor: pointer;
        }}
        .slider::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 24px;
            height: 24px;
            border-radius: 50%;
            background: #ffd700;
            cursor: pointer;
            box-shadow: 0 2px 6px rgba(0,0,0,0.4);
        }}
        .slider::-moz-range-thumb {{
            width: 24px;
            height: 24px;
            border-radius: 50%;
            background: #ffd700;
            cursor: pointer;
            border: none;
        }}
        .time-label {{
            font-size: 14px;
            color: #aaa;
            min-width: 60px;
        }}
        .legend {{
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-top: 15px;
            flex-wrap: wrap;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .legend-item img {{
            width: 28px;
            height: 28px;
        }}
        .stats {{
            display: flex;
            justify-content: center;
            gap: 40px;
            margin-top: 15px;
            font-size: 14px;
        }}
        .stat-item {{
            text-align: center;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #ffd700;
        }}
        .play-controls {{
            display: flex;
            justify-content: center;
            gap: 10px;
            margin-top: 15px;
        }}
        .btn {{
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.2s;
        }}
        .btn-play {{
            background: #4CAF50;
            color: white;
        }}
        .btn-play:hover {{
            background: #45a049;
        }}
        .btn-speed {{
            background: #2196F3;
            color: white;
        }}
        .btn-speed:hover {{
            background: #1976D2;
        }}
        .btn-speed.active {{
            background: #ffd700;
            color: #333;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Dota 2 眼位时间线</h1>
        <p style="text-align: center; margin-bottom: 15px; color: #aaa;">比赛 ID: {match_id}</p>
        
        <div class="map-container" id="mapContainer">
            <img src="data:image/jpeg;base64,{map_base64}" class="map-image" id="mapImage">
        </div>
        
        <div class="controls">
            <div class="time-display" id="timeDisplay">00:00</div>
            
            <div class="slider-container">
                <span class="time-label" id="minTimeLabel">{min_time // 60}:{min_time % 60:02d}</span>
                <input type="range" class="slider" id="timeSlider" 
                       min="{min_time}" max="{max_time}" value="{min_time}">
                <span class="time-label" id="maxTimeLabel">{max_time // 60}:{max_time % 60:02d}</span>
            </div>
            
            <div class="play-controls">
                <button class="btn btn-play" id="playBtn">▶ 播放</button>
                <button class="btn btn-speed" data-speed="1">1x</button>
                <button class="btn btn-speed" data-speed="2">2x</button>
                <button class="btn btn-speed active" data-speed="4">4x</button>
                <button class="btn btn-speed" data-speed="8">8x</button>
            </div>
            
            <div class="legend">
                <div class="legend-item">
                    <img src="data:image/png;base64,{icon_base64.get('obs_radiant', '')}" alt="天辉假眼">
                    <span>天辉假眼 ({obs_duration}s)</span>
                </div>
                <div class="legend-item">
                    <img src="data:image/png;base64,{icon_base64.get('obs_dire', '')}" alt="夜魇假眼">
                    <span>夜魇假眼 ({obs_duration}s)</span>
                </div>
                <div class="legend-item">
                    <img src="data:image/png;base64,{icon_base64.get('sen_radiant', '')}" alt="天辉真眼">
                    <span>天辉真眼 ({sen_duration}s)</span>
                </div>
                <div class="legend-item">
                    <img src="data:image/png;base64,{icon_base64.get('sen_dire', '')}" alt="夜魇真眼">
                    <span>夜魇真眼 ({sen_duration}s)</span>
                </div>
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
    </div>

    <script>
        // 眼位数据
        const wardsData = {json.dumps(wards_data)};
        
        // 图标 base64
        const icons = {{
            'obs_radiant': 'data:image/png;base64,{icon_base64.get("obs_radiant", "")}',
            'obs_dire': 'data:image/png;base64,{icon_base64.get("obs_dire", "")}',
            'sen_radiant': 'data:image/png;base64,{icon_base64.get("sen_radiant", "")}',
            'sen_dire': 'data:image/png;base64,{icon_base64.get("sen_dire", "")}'
        }};
        
        const mapContainer = document.getElementById('mapContainer');
        const mapImage = document.getElementById('mapImage');
        const timeSlider = document.getElementById('timeSlider');
        const timeDisplay = document.getElementById('timeDisplay');
        const playBtn = document.getElementById('playBtn');
        const activeObs = document.getElementById('activeObs');
        const activeSen = document.getElementById('activeSen');
        
        let wardElements = [];
        let visionElements = [];
        let isPlaying = false;
        let playInterval = null;
        let playSpeed = 4;
        
        // 视野半径（游戏单位转百分比）
        const OBS_VISION_RADIUS = 8.96 / 128 * 100;  // 假眼视野
        const SEN_VISION_RADIUS = 5.76 / 128 * 100;  // 真眼视野
        
        // 创建眼位元素
        function createWardElements() {{
            wardsData.forEach((ward, index) => {{
                // 坐标转换：游戏坐标 (0-128) -> 百分比
                // 注意：y 坐标需要翻转（游戏中 y 向上增加，但 CSS 中 top 向下增加）
                const xPercent = (ward.x / 128) * 100;
                const yPercent = (1 - ward.y / 128) * 100;
                
                // 创建视野圈
                const visionDiv = document.createElement('div');
                const isRadiant = ward.type.includes('radiant');
                const visionRadius = ward.is_obs ? OBS_VISION_RADIUS : SEN_VISION_RADIUS;
                
                visionDiv.className = 'vision-circle hidden ' + (isRadiant ? 'radiant' : 'dire');
                visionDiv.style.left = xPercent + '%';
                visionDiv.style.top = yPercent + '%';
                visionDiv.style.width = (visionRadius * 2) + '%';
                visionDiv.style.height = (visionRadius * 2) + '%';
                
                mapContainer.appendChild(visionDiv);
                visionElements.push(visionDiv);
                
                // 创建眼位图标
                const div = document.createElement('div');
                div.className = 'ward hidden';
                div.dataset.index = index;
                
                const img = document.createElement('img');
                img.src = icons[ward.type];
                div.appendChild(img);
                
                div.style.left = xPercent + '%';
                div.style.top = yPercent + '%';
                
                mapContainer.appendChild(div);
                wardElements.push(div);
            }});
        }}
        
        // 更新眼位显示
        function updateWards(currentTime) {{
            let obsCount = 0;
            let senCount = 0;
            
            wardsData.forEach((ward, index) => {{
                const isActive = currentTime >= ward.time && currentTime < ward.time + ward.duration;
                
                if (isActive) {{
                    wardElements[index].classList.remove('hidden');
                    visionElements[index].classList.remove('hidden');
                    if (ward.is_obs) obsCount++;
                    else senCount++;
                }} else {{
                    wardElements[index].classList.add('hidden');
                    visionElements[index].classList.add('hidden');
                }}
            }});
            
            activeObs.textContent = obsCount;
            activeSen.textContent = senCount;
        }}
        
        // 格式化时间
        function formatTime(seconds) {{
            const sign = seconds < 0 ? '-' : '';
            const absSeconds = Math.abs(seconds);
            const mins = Math.floor(absSeconds / 60);
            const secs = absSeconds % 60;
            return sign + mins + ':' + secs.toString().padStart(2, '0');
        }}
        
        // 滑动条事件
        timeSlider.addEventListener('input', function() {{
            const currentTime = parseInt(this.value);
            timeDisplay.textContent = formatTime(currentTime);
            updateWards(currentTime);
        }});
        
        // 播放/暂停
        playBtn.addEventListener('click', function() {{
            if (isPlaying) {{
                stopPlay();
            }} else {{
                startPlay();
            }}
        }});
        
        function startPlay() {{
            isPlaying = true;
            playBtn.textContent = '⏸ 暂停';
            playInterval = setInterval(() => {{
                let currentTime = parseInt(timeSlider.value);
                currentTime += playSpeed;
                
                if (currentTime > parseInt(timeSlider.max)) {{
                    currentTime = parseInt(timeSlider.min);
                }}
                
                timeSlider.value = currentTime;
                timeDisplay.textContent = formatTime(currentTime);
                updateWards(currentTime);
            }}, 100);
        }}
        
        function stopPlay() {{
            isPlaying = false;
            playBtn.textContent = '▶ 播放';
            if (playInterval) {{
                clearInterval(playInterval);
                playInterval = null;
            }}
        }}
        
        // 速度按钮
        document.querySelectorAll('.btn-speed').forEach(btn => {{
            btn.addEventListener('click', function() {{
                document.querySelectorAll('.btn-speed').forEach(b => b.classList.remove('active'));
                this.classList.add('active');
                playSpeed = parseInt(this.dataset.speed);
            }});
        }});
        
        // 初始化
        createWardElements();
        updateWards(parseInt(timeSlider.value));
        timeDisplay.textContent = formatTime(parseInt(timeSlider.value));
    </script>
</body>
</html>'''
        
        # 保存 HTML 文件
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"💾 已生成交互式页面: {save_path}")
        print(f"   假眼持续时间: {obs_duration} 秒")
        print(f"   真眼持续时间: {sen_duration} 秒")
        print(f"   时间范围: {min_time // 60}:{min_time % 60:02d} ~ {max_time // 60}:{max_time % 60:02d}")
        
        return save_path


# ==================== 主函数 ====================

def fetch_and_analyze(match_id: int = None, use_cache: bool = False):
    """
    获取比赛数据并分析眼位

    Args:
        match_id: 指定的比赛ID，如果为None则获取最近的职业比赛
        use_cache: 是否使用缓存的 CSV 文件
    """
    print("=" * 60)
    print("  🎮 Dota 2 单场比赛眼位分析工具")
    print("=" * 60)

    # 检查地图文件
    print("\n📁 检查地图文件...")
    available_maps = []
    for f in os.listdir(MAPS_DIR) if os.path.exists(MAPS_DIR) else []:
        if f.endswith((".jpeg", ".jpg", ".png")):
            version = f.split(".")[0]
            available_maps.append(version)
            print(f"   ✅ 找到地图: {f}")

    if not available_maps:
        print(f"   ⚠️ maps/ 目录下无地图文件")

    cache_obs = f"df_obs_{match_id}.csv" if match_id else "df_obs_test.csv"
    cache_sen = f"df_sen_{match_id}.csv" if match_id else "df_sen_test.csv"

    # 检查缓存
    if use_cache and os.path.exists(cache_obs) and os.path.exists(cache_sen):
        print(f"\n📂 使用缓存数据...")
        df_obs = pd.read_csv(cache_obs)
        df_sen = pd.read_csv(cache_sen)
        print(f"   假眼: {len(df_obs)} 条")
        print(f"   真眼: {len(df_sen)} 条")
    else:
        if match_id:
            # 使用指定的比赛ID
            print(f"\n🔍 获取指定比赛 {match_id}...")
            match_data = get_match_details(match_id)

            if not match_data:
                print(f"❌ 无法获取比赛 {match_id} 的数据")
                return

            # 提取眼位数据
            extractor = WardDataExtractor()

            print(f"\n[1] 提取比赛 {match_id}...")
            if extractor.extract_from_match(match_data):
                print("✅ 成功提取比赛数据")
                extractor.print_patch_summary()
            else:
                print("❌ 该比赛无眼位数据")
                return

            # 获取数据并保存
            df_obs, df_sen = extractor.get_dataframes()
            extractor.save_to_csv(cache_obs, cache_sen)
        else:
            # 获取最近比赛
            print(f"\n🔍 获取最近职业比赛...")
            match_data = get_match_details(8650993496)  # 示例比赛ID

            if not match_data:
                print("❌ 无法获取比赛数据")
                return

            # 提取眼位数据
            extractor = WardDataExtractor()

            print(f"\n[1] 提取比赛 {match_data.get('match_id')}...")
            if extractor.extract_from_match(match_data):
                print("✅ 成功提取比赛数据")
                extractor.print_patch_summary()
            else:
                print("❌ 该比赛无眼位数据")
                return

            # 显示比赛基本信息
            print(f"\n🏆 比赛基本信息:")
            print(f"   比赛ID: {match_data.get('match_id')}")
            print(f"   时长: {match_data.get('duration', 0) // 60}分{match_data.get('duration', 0) % 60}秒")
            print(f"   结果: {'天辉获胜' if match_data.get('radiant_win') else '夜魇获胜'}")
            print(f"   比分: 天辉 {match_data.get('radiant_score', 0)} - {match_data.get('dire_score', 0)} 夜魇")
            print(f"   版本: {match_data.get('patch', '未知')}")

            # 获取数据并保存
            df_obs, df_sen = extractor.get_dataframes()
            extractor.save_to_csv(cache_obs, cache_sen)

    # 分析
    if df_obs.empty and df_sen.empty:
        print("❌ 无眼位数据可分析")
        return

    # 创建分析器
    analyzer = WardAnalyzer(df_obs, df_sen)

    # 打印统计
    analyzer.print_stats()

    # 绘制图表
    print("\n📊 生成可视化...")

    # 按版本生成散点图
    analyzer.plot_scatter_by_version(save_dir=".")
    
    # 生成交互式 HTML 页面
    print("\n🌐 生成交互式网页...")
    html_path = f"ward_timeline_{match_id}.html" if match_id else "ward_timeline.html"
    analyzer.generate_interactive_html(save_path=html_path)

    print("\n✅ 分析完成!")


def quick_demo():
    """快速演示 - 使用模拟数据"""
    print("=" * 60)
    print("  🎮 Dota 2 眼位分析 - 快速演示")
    print("=" * 60)

    # 检查可用地图
    available_versions = []
    if os.path.exists(MAPS_DIR):
        for f in os.listdir(MAPS_DIR):
            if f.endswith((".jpeg", ".jpg", ".png")):
                version = f.split(".")[0]
                if version.isdigit():
                    available_versions.append(version)

    if not available_versions:
        available_versions = ["740"]  # 默认版本

    print(f"\n📁 可用地图版本: {available_versions}")

    # 生成示例数据 (模拟一场比赛)
    np.random.seed(42)
    n_samples = 120  # 模拟一场比赛的眼位数量

    obs_data = []
    sample_match_id = 8650993496  # 示例比赛ID

    # 模拟一场比赛的眼位分布
    for _ in range(n_samples // 4):
        # 天辉三角区域 (热门眼位)
        obs_data.append({
            "match_id": sample_match_id,
            "x": np.random.normal(100, 8) + 64,
            "y": np.random.normal(40, 8) + 64,
            "is_radiant": 0,  # 天辉放置的眼
            "time": np.random.randint(-60, 2400),
            "map_version": available_versions[0],
        })
        # 夜魇三角区域
        obs_data.append({
            "match_id": sample_match_id,
            "x": np.random.normal(30, 8) + 64,
            "y": np.random.normal(90, 8) + 64,
            "is_radiant": 1,  # 夜魇放置的眼
            "time": np.random.randint(-60, 2400),
            "map_version": available_versions[0],
        })
        # 中路河道
        obs_data.append({
            "match_id": sample_match_id,
            "x": np.random.normal(64, 10) + 64,
            "y": np.random.normal(64, 10) + 64,
            "is_radiant": np.random.randint(0, 2),
            "time": np.random.randint(-60, 2400),
            "map_version": available_versions[0],
        })
        # 肉山坑
        obs_data.append({
            "match_id": sample_match_id,
            "x": np.random.normal(35, 5) + 64,
            "y": np.random.normal(100, 5) + 64,
            "is_radiant": np.random.randint(0, 2),
            "time": np.random.randint(600, 2400),  # 晚期游戏
            "map_version": available_versions[0],
        })

    df_obs = pd.DataFrame(obs_data)
    df_sen = pd.DataFrame(obs_data[:len(obs_data) // 3])  # 真眼少一些

    print(f"\n📊 示例数据 (模拟比赛 {sample_match_id}):")
    print(f"   假眼: {len(df_obs)} 个")
    print(f"   真眼: {len(df_sen)} 个")

    # 分析
    analyzer = WardAnalyzer(df_obs, df_sen)

    analyzer.print_stats()
    analyzer.plot_scatter_by_version(save_dir=".")

    print("\n✅ 演示完成!")


# ==================== 入口 ====================

if __name__ == "__main__":
    import sys

    print("\n选择模式:")
    print("  1. 快速演示 (使用模拟数据)")
    print("  2. 指定比赛ID分析 (输入比赛ID)")
    print("  3. 使用缓存数据分析")

    try:
        choice = input("\n请选择 (1/2/3): ").strip()

        if choice == "1":
            quick_demo()
        elif choice == "2":
            match_id_str = input("请输入比赛ID (例如: 8650993496): ").strip()
            try:
                match_id = int(match_id_str)
                fetch_and_analyze(match_id=match_id, use_cache=False)
            except ValueError:
                print("❌ 无效的比赛ID，请输入数字")
                sys.exit(1)
        elif choice == "3":
            match_id_str = input("请输入比赛ID (留空使用默认缓存): ").strip()
            if match_id_str:
                try:
                    match_id = int(match_id_str)
                    cache_obs = f"df_obs_{match_id}.csv"
                    cache_sen = f"df_sen_{match_id}.csv"
                    if os.path.exists(cache_obs) and os.path.exists(cache_sen):
                        fetch_and_analyze(match_id=match_id, use_cache=True)
                    else:
                        print(f"❌ 找不到缓存文件: {cache_obs} 或 {cache_sen}")
                        sys.exit(1)
                except ValueError:
                    print("❌ 无效的比赛ID，请输入数字")
                    sys.exit(1)
            else:
                fetch_and_analyze(use_cache=True)
        else:
            print("默认选择快速演示...")
            quick_demo()

    except KeyboardInterrupt:
        print("\n\n已取消")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
