# Dota 2 ReAct Agent

基于 ReAct 推理框架的 Dota 2 数据分析系统，内置 MCP 工具用于查询比赛、战队、选手、英雄与视野相关数据，并提供 Web UI 进行交互式分析与报告展示。

**功能概览**
- ReAct 方式调用 MCP 工具完成多步分析
- 战队/比赛/选手/英雄数据查询与统计
- 视野热力图与分析报告生成
- Web UI 实时对话与历史会话回看

**快速开始**
1. 创建 Conda 环境
```bash
conda create -n dota2-agent python=3.10 -y
conda activate dota2-agent
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 配置环境变量（可选）
在项目根目录创建 `.env`：
```bash
LLM_API_KEY=your_api_key
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL_ID=deepseek-v3.2
SERPAPI_API_KEY=your_serpapi_key
```

4. 启动 Web 服务
```bash
python web_app.py
```
浏览器访问 `http://127.0.0.1:8000`。

**目录结构**
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

**备注**
- 数据来源：OpenDota API
- MCP 工具说明详见 `MCP_README.md`
