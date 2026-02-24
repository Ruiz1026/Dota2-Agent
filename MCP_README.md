# Dota 2 ReAct Agent

**ReAct 范式 + MCP 工具调用**

## 🎯 设计理念

```
用户输入
    ↓
┌─────────────────────────────────────┐
│           ReAct 循环                 │
│  ┌─────────────────────────────┐   │
│  │ Thought: 分析用户需求        │   │
│  │ Action: 选择 MCP 工具        │   │
│  │ Action Input: 构造参数       │   │
│  └─────────────────────────────┘   │
│              ↓                      │
│  ┌─────────────────────────────┐   │
│  │ 调用 MCP Server 工具         │   │
│  └─────────────────────────────┘   │
│              ↓                      │
│  ┌─────────────────────────────┐   │
│  │ Observation: 获取工具返回    │   │
│  │ Thought: 分析结果            │   │
│  │ (继续循环或给出 Final Answer)│   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
    ↓
最终回答
```

## 📁 项目结构

```
hello_agents/
├── mcp_server/
│   └── dota2_fastmcp.py       # MCP Server（提供工具）
├── prompts/
│   └── dota2_agent_prompt.py  # ReAct 系统提示词
├── dota2_agent.py             # ReAct Agent 主程序
└── requirements.txt
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install mcp requests openai python-dotenv
```

### 2. 配置环境变量（可选，用于完整 ReAct 模式）

```bash
# .env 文件
LLM_API_KEY=your_api_key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4
SERPAPI_API_KEY=your_serpapi_key
```

### 3. 运行

```bash
python dota2_agent.py
```

## 💡 两种模式

### 简化模式（无需 LLM）
直接根据关键词调用 MCP 工具：
```
你: 查询比赛 8650430843
助手: [直接调用 get_match_details] ...
```

### ReAct 模式（需要 LLM）
使用 ReAct 循环进行推理：
```
你: react 分析玩家 139939567 最近的表现
助手: 
Thought: 需要先获取玩家信息和最近比赛...
Action: get_player_info
Action Input: {"account_id": 139939567}

Observation: [玩家信息]

Thought: 现在获取最近比赛记录...
Action: get_player_matches
Action Input: {"account_id": 139939567, "limit": 10}

Observation: [比赛记录]

Thought: 我已经有足够信息进行分析
Final Answer: 
## 玩家表现分析
...
```

## 🛠️ MCP 工具列表

| 工具 | 描述 | 参数 |
|------|------|------|
| `get_match_details` | 获取比赛摘要 + 原始数据（matches/{match_id}） | `match_id` |
| `get_match_items` | 获取比赛购买记录（非消耗品） | `match_id` |
| `get_heroes` | 获取英雄列表 | - |
| `get_player_info` | 获取玩家信息 | `account_id` |
| `get_player_matches` | 获取玩家比赛 | `account_id`, `limit?` |
| `get_hero_stats` | 获取英雄统计 | - |
| `get_pro_matches` | 获取职业比赛 | `limit?` |
| `get_player_win_loss` | 获取胜负统计 | `account_id` |
| `get_player_heroes` | 获取常用英雄 | `account_id`, `limit?` |
| `search_dota_history` | SerpApi 搜索 Dota 历史信息 | `query`, `num_results?`, `include_liquipedia?`, `sites?`（默认多站点） |
| `analyze_multi_match_wards` | 多场比赛眼位/击杀/推塔汇总与热力图 | `team_id?`, `account_id?`, `match_ids?`, `limit?(默认10)` |
| `inject_multi_match_ward_report_html` | 写入多场比赛汇总分析报告并读取汇总JSON | `summary_path?`, `report_html?`, `report_path?`, `html_path?` |

## 🔧 自定义开发

### 添加新的 MCP 工具

在 `mcp_server/dota2_fastmcp.py` 中：

```python
@mcp.tool()
def your_tool(param: int) -> str:
    """工具描述"""
    # 实现
    return "结果"
```

### 修改 ReAct 提示词

编辑 `prompts/dota2_agent_prompt.py` 中的 `DOTA2_SYSTEM_PROMPT`。

## 📊 数据来源

[OpenDota API](https://docs.opendota.com/)
