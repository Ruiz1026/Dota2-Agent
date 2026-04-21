# Dota 2 ReAct Agent ⚔️

基于 ReAct 推理框架的 Dota 2 数据分析系统，内置 MCP 工具用于查询比赛、战队、选手、英雄与视野相关数据，并提供 Web UI 进行交互式分析与报告展示。

**系统介绍 🧭**

本系统由 ReAct Agent、MCP 工具服务与 Web UI 组成：Agent 负责“思考-行动-观察”的多步推理流程，MCP 以工具化方式封装 OpenDota 数据查询与分析能力，Web UI 提供实时对话、历史回放与报告展示。系统支持视野热力图、眼位分析、战队与英雄画像等结果输出，并可选接入 OpenViking 记忆模块以提升多轮对话的上下文理解。

**系统展示 🖼️**

![首页](introduce/首页.png)
![玩家单场比赛](introduce/玩家单场比赛.png)
![单场比赛视野分析](introduce/单场比赛视野分析.png)
![眼位存活统计](introduce/眼位存活统计.png)
![视野热力图](introduce/视野热力图.png)
![事业点位图](introduce/事业点位图.png)
![战队英雄解析](introduce/战队英雄解析.png)
![英雄介绍](introduce/英雄介绍.png)

**功能概览 ✨**

- ReAct 方式调用 MCP 工具完成多步分析
- 战队/比赛/选手/英雄数据查询与统计
- 视野热力图与分析报告生成
- Web UI 实时对话与历史会话回看

**最近更新 🆕**

1. 大幅缩减系统 Prompt，减少无效上下文占用，整体问答效率更高。
2. 优化上下文机制与记忆注入策略，提升 Agent 在多轮任务中的智能性与回复质量。

**快速开始 🚀**

1. 创建 Conda 环境

```bash
conda create -n dota2-agent python=3.10 -y
conda activate dota2-agent
```

2. 安装依赖

```bash
pip install -r requirements.txt
```

3. 配置环境变量
   在项目根目录创建 `.env`：

```bash
LLM_API_KEY="your_api_key"
LLM_BASE_URL="https://api.deepseek.com/v1"
LLM_MODEL_ID="deepseek-chat"
SERPAPI_API_KEY=your_serpapi_key
```

也可以申请有一点免费token额度的豆包 API 来替代 DeepSeek（示例）：

```bash
LLM_API_KEY="your_api_key"
LLM_MODEL_ID="doubao-seed-1-8-251228"
LLM_BASE_URL="https://ark.cn-beijing.volces.com/api/v3"
```

4. 配置 OpenViking（可选，默认启用记忆）

   - 编辑 ov.conf，填入你的 embedding / VLM 供应商与 API Key
   - 如需自定义路径，可设置 OPENVIKING_CONFIG_FILE 指向配置文件
   - 运行后自动生成 ov_data/

   ov.conf example (replace api_key):

```json
{
  "embedding": {
    "dense": {
      "api_base": "https://ark.cn-beijing.volces.com/api/v3",
      "api_key": "your_api_key",
      "provider": "volcengine",
      "dimension": 1024,
      "model": "doubao-embedding-vision-250615"
    }
  },
  "vlm": {
    "api_base": "https://ark.cn-beijing.volces.com/api/v3",
    "api_key": "your_api_key",
    "provider": "volcengine",
    "model": "doubao-seed-1-8-251228"
  }
}
```

5. 启动 Web 服务

```bash
python web_app.py
```

浏览器访问 `http://127.0.0.1:8000`。

**目录结构 📁**

```
hello_agents/
├─ mcp_server/            # MCP 工具集合
├─ prompts/               # ReAct 系统提示词
├─ web/                   # Web UI
├─ dota2_agent.py         # ReAct Agent 主程序
├─ web_app.py             # Web 服务入口
├─ requirements.txt
└─ logs/                  # 会话日志
```

**备注 📌**

- 数据来源：OpenDota API
- MCP 工具说明详见 `MCP_README.md`

## 2026-04 记忆与工具链更新

- OpenViking 记忆检索延长了默认超时时间，并可通过 `memory_status` 查看更清晰的检索状态。
- `memory_search` 现在在 OpenViking 检索超时或无结果时支持本地文件回退，便于排查记忆检索问题。
- 运行时工具调用加入了预检校验，会检查工具名和参数名，并在调用格式错误时返回修复提示。
- 新增内置 workflow skill `team_hero_analysis`，用于战队英雄池、近期英雄使用情况和近期出装路线分析。
- 优化了最终答案归一化逻辑，降低模型把完整回答写在 `Final Answer:` 之外的概率。
