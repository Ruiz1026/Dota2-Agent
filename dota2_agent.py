# dota2_agent.py
"""
Dota 2 ReAct Agent
使用 ReAct 范式 + MCP 工具调用
"""

import asyncio
import difflib
import json
import os
import re
from pathlib import Path
from typing import Optional, Dict, Any, List, AsyncGenerator, Tuple
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 导入提示词
from prompts.dota2_agent_prompt import DOTA2_SYSTEM_PROMPT

# 导入日志模块
from utils.logger import ConversationLogger
from utils.openviking_memory import OpenVikingMemoryManager
from utils.skill_loader import SkillLoader

# MCP 客户端
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    HAS_MCP = True
except ImportError:
    HAS_MCP = False
    print("❌ MCP 未安装，请运行: pip install mcp")

# LLM 客户端
try:
    from openai import OpenAI, AsyncOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("❌ OpenAI 未安装，请运行: pip install openai")

# OpenViking 客户端
try:
    import openviking as ov
    from openviking.message.part import TextPart
    HAS_OPENVIKING = True
except ImportError:
    HAS_OPENVIKING = False
    print("❌ OpenViking 未安装，请运行: pip install openviking")


class Dota2ReActAgent:
    """
    Dota 2 ReAct Agent
    
    使用 ReAct (Reasoning + Acting) 范式：
    1. Thought - 思考分析
    2. Action - 调用 MCP 工具
    3. Observation - 观察结果
    4. 循环或给出 Final Answer
    """
    
    def __init__(
        self,
        mcp_server_path: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        llm_model: Optional[str] = None,
        llm_timeout: float = 150.0,
        max_observation_chars: int = 12000,
        max_iterations: int = 20,
        log_dir: str = "logs",
        enable_logging: bool = True,
        enable_memory: bool = True,
        ov_config_path: Optional[str] = None,
        ov_data_path: Optional[str] = None,
        memory_top_k: int = 3,
        memory_commit_every_n: int = 5,
        memory_commit_min_chars: int = 200,
        memory_commit_only_success: bool = True,
        memory_retrieve_every_n: int = 1,
        memory_retrieve_min_chars: int = 12,
        memory_retrieve_timeout: float = 4.0,
        memory_commit_timeout: float = 1.2,
        memory_record_user_min_chars: int = 8,
        memory_record_assistant_min_chars: int = 80,
    ):
        """
        初始化 ReAct Agent
        
        Args:
            mcp_server_path: MCP Server 脚本路径
            llm_api_key: LLM API Key
            llm_base_url: LLM API Base URL
            llm_model: LLM 模型名称
            max_iterations: 最大迭代次数
            log_dir: 日志保存目录
            enable_logging: 是否启用日志
            memory_commit_every_n: 记忆提交间隔（对话数）
            memory_commit_min_chars: 记忆提交最小字数
            memory_commit_only_success: 仅成功时提交记忆
            memory_retrieve_every_n: 记忆检索间隔（对话数）
            memory_retrieve_min_chars: 触发记忆检索最小字数
            memory_retrieve_timeout: 记忆检索超时（秒）
            memory_commit_timeout: 记忆提交超时（秒）
            memory_record_user_min_chars: 记录用户最小字数
            memory_record_assistant_min_chars: 记录助手最小字数
        """
        # MCP 配置
        self.mcp_server_path = mcp_server_path or os.path.join(
            os.path.dirname(__file__),
            "mcp_server",
            "dota2_fastmcp.py"
        )
        self.session: Optional[ClientSession] = None
        self.mcp_tools: List[Dict] = []
        self.skills_dir = Path(__file__).resolve().parent / ".skills"
        self.skill_loader = SkillLoader(self.skills_dir)
        
        # LLM 配置
        self.llm_api_key = llm_api_key or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
        self.llm_base_url = llm_base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL")
        self.llm_model = llm_model or os.getenv("LLM_MODEL_ID") or "deepseek-chat"
        self.llm_max_tokens = max(1, min(int(os.getenv("LLM_MAX_TOKENS", "8000")), 8000))
        self.llm_timeout = float(os.getenv("LLM_TIMEOUT", llm_timeout))
        self.max_observation_chars = int(os.getenv("MAX_OBSERVATION_CHARS", max_observation_chars))
        
        self.max_iterations = max_iterations
        self.system_prompt = DOTA2_SYSTEM_PROMPT
        self._register_local_tools()

        # 上下文预算配置（用于控制每轮注入的 system/context 长度）
        self.context_budget_chars = int(os.getenv("CONTEXT_BUDGET_CHARS", "3200"))
        self.runtime_tools_context_chars = int(os.getenv("RUNTIME_TOOLS_CONTEXT_CHARS", "1500"))
        self.skills_overview_context_chars = int(os.getenv("SKILLS_OVERVIEW_CONTEXT_CHARS", "900"))
        self.memory_context_max_chars = int(os.getenv("MEMORY_CONTEXT_MAX_CHARS", "1200"))
        self.recent_context_max_chars = int(os.getenv("RECENT_CONTEXT_MAX_CHARS", "900"))
        self.recent_turn_window = max(1, int(os.getenv("RECENT_CONTEXT_TURN_WINDOW", "5")))
        self.context_chat_turns_only = str(os.getenv("CONTEXT_CHAT_TURNS_ONLY", "0")).strip().lower() not in {"0", "false", "no"}
        self.force_full_tool_context = str(os.getenv("FORCE_FULL_TOOL_CONTEXT", "1")).strip().lower() not in {"0", "false", "no"}
        self.force_full_skill_overview = str(os.getenv("FORCE_FULL_SKILL_OVERVIEW", "1")).strip().lower() not in {"0", "false", "no"}
        
        # 日志
        self.enable_logging = enable_logging
        self.log_dir = log_dir
        self.logger = ConversationLogger(log_dir) if enable_logging else None

        # OpenViking 记忆配置
        self.enable_memory = enable_memory
        self.ov_config_path = ov_config_path or os.path.join(os.path.dirname(__file__), "ov.conf")
        self.ov_data_path = ov_data_path or os.path.join(os.path.dirname(__file__), "ov_data")
        self.memory_top_k = memory_top_k
        self.memory_commit_every_n = max(1, int(memory_commit_every_n))
        self.memory_commit_min_chars = max(0, int(memory_commit_min_chars))
        self.memory_commit_only_success = bool(memory_commit_only_success)
        self.memory_retrieve_every_n = max(1, int(memory_retrieve_every_n))
        self.memory_retrieve_min_chars = max(0, int(memory_retrieve_min_chars))
        self.memory_retrieve_timeout = max(0.2, float(memory_retrieve_timeout))
        self.memory_commit_timeout = max(0.2, float(memory_commit_timeout))
        self.memory_record_user_min_chars = max(0, int(memory_record_user_min_chars))
        self.memory_record_assistant_min_chars = max(0, int(memory_record_assistant_min_chars))
        self.memory_tool_trace_max_chars = int(os.getenv("MEMORY_TOOL_TRACE_MAX_CHARS", "0"))
        self.ov_session_id = self.logger.session_id if self.logger else None
        self.memory_manager = OpenVikingMemoryManager(
            enable_memory=self.enable_memory,
            has_openviking=HAS_OPENVIKING,
            ov_module=ov if HAS_OPENVIKING else None,
            text_part_cls=TextPart if HAS_OPENVIKING else None,
            ov_config_path=self.ov_config_path,
            ov_data_path=self.ov_data_path,
            session_id=self.ov_session_id,
            memory_top_k=self.memory_top_k,
            memory_commit_every_n=self.memory_commit_every_n,
            memory_commit_min_chars=self.memory_commit_min_chars,
            memory_commit_only_success=self.memory_commit_only_success,
            memory_retrieve_every_n=self.memory_retrieve_every_n,
            memory_retrieve_min_chars=self.memory_retrieve_min_chars,
            memory_retrieve_timeout=self.memory_retrieve_timeout,
            memory_commit_timeout=self.memory_commit_timeout,
            memory_record_user_min_chars=self.memory_record_user_min_chars,
            memory_record_assistant_min_chars=self.memory_record_assistant_min_chars,
            tool_trace_max_chars=self.memory_tool_trace_max_chars,
        )
        self._last_user_input = ""
        self._pending_visual_markdown: List[str] = []
        self._last_assistant_answer = ""
        self._recent_turns: List[Dict[str, str]] = []
        self._last_llm_finish_reason: Optional[str] = None
        self._loaded_skills_current_turn: List[str] = []
        self._tool_history_current_turn: List[str] = []
        
        # LLM 客户端
        if HAS_OPENAI and self.llm_api_key:
            self.llm_client = OpenAI(
                api_key=self.llm_api_key,
                base_url=self.llm_base_url,
                max_retries=0,
            )
            try:
                self.llm_async_client = AsyncOpenAI(
                    api_key=self.llm_api_key,
                    base_url=self.llm_base_url,
                    max_retries=0,
                )
            except Exception:
                self.llm_async_client = None
        else:
            self.llm_client = None
            self.llm_async_client = None

    async def _ensure_memory_ready(self) -> bool:
        """初始化 OpenViking 记忆存储（只做一次）"""
        if not self.ov_session_id and self.logger:
            self.ov_session_id = self.logger.session_id
        ready = await self.memory_manager.ensure_ready(self.ov_session_id)
        self.enable_memory = self.memory_manager.enable_memory
        return ready

    async def _record_memory_message(self, role: str, content: str, session: Optional[Any] = None) -> None:
        await self.memory_manager.record_message(role, content, session=session)

    async def _commit_memory_session(self, session: Optional[Any] = None) -> None:
        await self.memory_manager.commit_session(session=session)

    def _build_memory_query(self, user_input: str) -> tuple[str, bool]:
        return self.memory_manager.build_memory_query(
            user_input,
            self._last_user_input,
            self._last_assistant_answer,
        )

    def _load_skill_tool_spec(self) -> Dict[str, Any]:
        return {
            "name": "load_skill",
            "description": "按技能名加载完整技能说明（第二层注入）",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "技能名称，例如 team_recent_analysis/team_hero_analysis 或 tool:get_match_details",
                    }
                },
                "required": ["name"],
            },
        }

    def _memory_tool_specs(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "memory_search",
                "description": "在 OpenViking 记忆中检索相关历史片段",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "检索关键词"},
                        "limit": {"type": "integer", "description": "返回数量，默认 5，范围 1-20"},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "memory_commit",
                "description": "立即提交当前待提交的 OpenViking 记忆",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "name": "memory_status",
                "description": "查看 OpenViking 记忆运行状态与阈值配置",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        ]

    def _register_local_tools(self) -> None:
        names = {
            str(tool.get("name") or "").strip()
            for tool in self.mcp_tools
            if isinstance(tool, dict)
        }
        for spec in [self._load_skill_tool_spec(), *self._memory_tool_specs()]:
            name = str(spec.get("name") or "").strip()
            if not name or name in names:
                continue
            self.mcp_tools.append(spec)
            names.add(name)

    @staticmethod
    def _clip_context_text(text: str, max_chars: int, suffix: str = "\n...[truncated]") -> str:
        value = str(text or "")
        if max_chars <= 0 or len(value) <= max_chars:
            return value
        keep = max(0, max_chars - len(suffix))
        return value[:keep] + suffix

    @staticmethod
    def _extract_query_tokens(user_input: str, memory_context: str = "") -> Tuple[str, set]:
        text = str(user_input or "").strip().lower()
        memory_hint = str(memory_context or "").strip().lower()
        # 记忆内容仅取前段参与打分，避免噪声过大
        if memory_hint:
            memory_hint = memory_hint[:600]
            text_for_tokens = f"{text}\n{memory_hint}"
        else:
            text_for_tokens = text

        tokens = set(re.findall(r"[a-z][a-z0-9_]{2,}", text))
        tokens.update(re.findall(r"[a-z][a-z0-9_]{2,}", text_for_tokens))
        zh_keywords = (
            "视野", "眼位", "插眼", "多场", "单场", "战队", "比赛", "玩家", "英雄",
            "出装", "装备", "技能", "参数", "怎么用", "说明", "规则", "记忆",
            "天辉", "夜魇", "近期", "样本", "胜率", "分路", "对线",
        )
        for kw in zh_keywords:
            if kw in text_for_tokens:
                tokens.add(kw)
        return text_for_tokens, tokens

    def _tool_relevance_score(self, tool: Dict[str, Any], query_text: str, query_tokens: set) -> int:
        name = str(tool.get("name") or "").strip().lower()
        desc = str(tool.get("description") or "").strip().lower()
        if not name:
            return -1
        score = 0
        if name in query_text:
            score += 8

        haystack = f"{name} {desc}"
        for token in query_tokens:
            if token and token in haystack:
                score += 2

        keyword_rules = [
            (("视野", "眼位", "插眼"), ("ward", "analyze_multi_match_wards", "analyze_match_wards", "inject_ward")),
            (("战队", "team", "近期"), ("team_", "search_team", "analyze_multi_match_wards")),
            (("玩家", "player"), ("player_", "search_players")),
            (("英雄", "hero"), ("hero_", "get_heroes", "rag_hero_intro")),
            (("装备", "出装", "item"), ("item", "get_match_items", "get_item_id_map")),
            (("记忆", "memory"), ("memory_",)),
            (("技能", "说明", "参数", "怎么用", "规则"), ("load_skill",)),
        ]
        for q_keys, n_keys in keyword_rules:
            if any(k in query_text for k in q_keys) and any(k in name for k in n_keys):
                score += 6

        return score

    def _select_context_tools(self, user_input: str, memory_context: str = "", max_tools: int = 16) -> List[Dict[str, Any]]:
        if not self.mcp_tools:
            return []
        if self.force_full_tool_context:
            all_tools = sorted(
                [t for t in self.mcp_tools if str(t.get("name") or "").strip()],
                key=lambda t: str(t.get("name") or "").strip(),
            )
            if max_tools > 0:
                return all_tools[:max_tools]
            return all_tools
        query_text, query_tokens = self._extract_query_tokens(user_input, memory_context=memory_context)
        core_names = {"load_skill", "memory_search", "memory_commit", "memory_status"}
        core_tools: List[Dict[str, Any]] = []
        scored: List[Tuple[int, str, Dict[str, Any]]] = []

        for tool in self.mcp_tools:
            name = str(tool.get("name") or "").strip()
            if not name:
                continue
            if name in core_names:
                core_tools.append(tool)
                continue
            score = self._tool_relevance_score(tool, query_text, query_tokens)
            scored.append((score, name, tool))

        selected: List[Dict[str, Any]] = []
        selected.extend(sorted(core_tools, key=lambda t: str(t.get("name") or "")))

        scored.sort(key=lambda item: (-item[0], item[1]))
        for score, _, tool in scored:
            if len(selected) >= max_tools:
                break
            # 若完全无关，仅填充少量工具避免上下文膨胀
            if score <= 0 and len(selected) >= min(8, max_tools):
                break
            selected.append(tool)

        if not selected:
            selected = sorted(
                [t for t in self.mcp_tools if str(t.get("name") or "").strip()],
                key=lambda t: str(t.get("name") or "").strip(),
            )[:max_tools]
        return selected

    def _build_runtime_tools_context(
        self,
        user_input: str,
        memory_context: str = "",
        max_chars: Optional[int] = None,
        selected_tools: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        if not self.mcp_tools:
            return ""
        budget = self.runtime_tools_context_chars if max_chars is None else int(max_chars)
        tools = selected_tools or self._select_context_tools(user_input=user_input, memory_context=memory_context)
        all_tool_names = sorted(
            [
                str(tool.get("name") or "").strip()
                for tool in self.mcp_tools
                if str(tool.get("name") or "").strip()
            ]
        )
        lines = [
            "运行时工具约束：只能使用已注册工具名，必须精确拼写，禁止自造工具名或参数名。",
            "全部已注册工具名：",
            ", ".join(all_tool_names),
            "",
            f"高相关工具速查（展示 {len(tools)}/{len(self.mcp_tools)}）：",
        ]
        for tool in tools[:10]:
            name = str(tool.get("name") or "").strip()
            if not name:
                continue
            signature = self._tool_signature(tool, max_params=6)
            desc = re.sub(r"\s+", " ", str(tool.get("description") or "").strip())
            if len(desc) > 64:
                desc = desc[:61] + "..."
            param_hint = self._tool_param_hint(tool, max_optional=4)
            if desc and param_hint:
                lines.append(f"- {signature}: {desc} | {param_hint}")
            elif desc:
                lines.append(f"- {signature}: {desc}")
            else:
                lines.append(f"- {signature}")
        lines.append('工具名或参数不确定时：先调用 load_skill，示例 {"name":"tool:get_match_details"}，按返回的参数名重试。')
        text = "\n".join(lines)
        return self._clip_context_text(text, budget)

    @staticmethod
    def _schema_type_text(schema: Dict[str, Any]) -> str:
        if not isinstance(schema, dict):
            return "any"
        schema_type = schema.get("type")
        if isinstance(schema_type, list):
            return "|".join(str(t) for t in schema_type)
        if isinstance(schema_type, str):
            if schema_type == "array":
                item_type = "any"
                items = schema.get("items")
                if isinstance(items, dict):
                    item_type = str(items.get("type") or "any")
                return f"array[{item_type}]"
            return schema_type
        if "enum" in schema and isinstance(schema["enum"], list):
            return "enum"
        return "any"

    def _find_tool_spec(self, tool_name: str) -> Optional[Dict[str, Any]]:
        key = (tool_name or "").strip()
        if not key:
            return None
        for tool in self.mcp_tools:
            if str(tool.get("name") or "").strip() == key:
                return tool
        return None

    @staticmethod
    def _tool_schema_parts(tool: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], set]:
        schema = tool.get("inputSchema") if isinstance(tool.get("inputSchema"), dict) else {}
        props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        required = schema.get("required") if isinstance(schema.get("required"), list) else []
        required_set = {str(x) for x in required}
        return schema, props, required_set

    def _tool_signature(self, tool: Dict[str, Any], max_params: int = 8) -> str:
        name = str(tool.get("name") or "").strip()
        if not name:
            return ""
        _, props, required_set = self._tool_schema_parts(tool)
        if not props:
            return f"{name}()"

        ordered_params = [str(p) for p in props.keys() if str(p) in required_set]
        ordered_params.extend([str(p) for p in props.keys() if str(p) not in required_set])
        labels: List[str] = []
        for idx, param_name in enumerate(ordered_params):
            if idx >= max_params:
                labels.append("...")
                break
            label = param_name if param_name in required_set else f"{param_name}?"
            labels.append(label)
        return f"{name}({', '.join(labels)})"

    def _tool_param_hint(self, tool: Dict[str, Any], max_optional: int = 4) -> str:
        _, props, required_set = self._tool_schema_parts(tool)
        if not props:
            return "无参数"
        required_names = [str(p) for p in props.keys() if str(p) in required_set]
        optional_names = [str(p) for p in props.keys() if str(p) not in required_set]
        parts: List[str] = []
        if required_names:
            parts.append("必填: " + ", ".join(required_names))
        if optional_names:
            shown_optional = optional_names[:max_optional]
            optional_text = ", ".join(shown_optional)
            if len(optional_names) > max_optional:
                optional_text += ", ..."
            parts.append("可选: " + optional_text)
        return "；".join(parts)

    @staticmethod
    def _schema_type_matches(value: Any, schema: Dict[str, Any]) -> bool:
        if not isinstance(schema, dict):
            return True
        enum_values = schema.get("enum")
        if isinstance(enum_values, list) and enum_values and value not in enum_values:
            return False
        schema_type = schema.get("type")
        if isinstance(schema_type, list):
            return any(Dota2ReActAgent._schema_type_matches(value, {"type": t, "enum": enum_values}) for t in schema_type)
        if not isinstance(schema_type, str):
            return True
        if schema_type == "string":
            return isinstance(value, str)
        if schema_type == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if schema_type == "number":
            return (isinstance(value, int) or isinstance(value, float)) and not isinstance(value, bool)
        if schema_type == "boolean":
            return isinstance(value, bool)
        if schema_type == "array":
            return isinstance(value, list)
        if schema_type == "object":
            return isinstance(value, dict)
        return True

    def _tool_repair_observation(
        self,
        tool_name: str,
        arguments: Any,
        issues: List[str],
        include_schema: bool = True,
    ) -> str:
        lines = [f"工具调用校验失败：{tool_name or '<empty>'}"]
        for issue in issues:
            lines.append(f"- {issue}")

        tool = self._find_tool_spec(tool_name)
        if tool:
            lines.append(f"- 正确签名: {self._tool_signature(tool, max_params=10)}")
            param_hint = self._tool_param_hint(tool, max_optional=6)
            if param_hint:
                lines.append(f"- 参数提示: {param_hint}")
            if include_schema:
                lines.append("")
                lines.append(self._build_tool_skill_content(tool_name))
        else:
            valid_names = sorted(
                [
                    str(item.get("name") or "").strip()
                    for item in self.mcp_tools
                    if str(item.get("name") or "").strip()
                ]
            )
            suggestions = difflib.get_close_matches(str(tool_name or "").strip(), valid_names, n=5, cutoff=0.45)
            if suggestions:
                lines.append("- 最接近的已注册工具: " + ", ".join(suggestions))
            lines.append('- 不确定时先调用 load_skill，示例 {"name":"tool:<工具名>"}')

        raw_args = "{}"
        if isinstance(arguments, dict):
            try:
                raw_args = json.dumps(arguments, ensure_ascii=False)
            except Exception:
                raw_args = str(arguments)
        elif arguments is not None:
            raw_args = str(arguments)
        lines.append(f"- 本次 Action Input: {raw_args}")
        lines.append("请改用上面给出的精确工具名和参数名继续，不要重复同一错误。")
        return "\n".join(lines)

    def _preflight_tool_call(self, tool_name: str, arguments: Any) -> Optional[str]:
        key = str(tool_name or "").strip()
        if not key:
            return self._tool_repair_observation(tool_name, arguments, ["缺少工具名。"])

        tool = self._find_tool_spec(key)
        if not tool:
            return self._tool_repair_observation(key, arguments, [f"未知工具 `{key}`。"])

        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            return self._tool_repair_observation(
                key,
                arguments,
                ["Action Input 必须是 JSON 对象。"],
            )

        _, props, required_set = self._tool_schema_parts(tool)
        issues: List[str] = []

        missing = [name for name in required_set if name not in arguments]
        if missing:
            issues.append("缺少必填字段: " + ", ".join(sorted(missing)))

        unknown = [name for name in arguments.keys() if name not in props]
        if unknown:
            issues.append("存在未注册字段: " + ", ".join(sorted(unknown)))

        type_errors: List[str] = []
        for arg_name, arg_value in arguments.items():
            spec = props.get(arg_name)
            if isinstance(spec, dict) and not self._schema_type_matches(arg_value, spec):
                expected = self._schema_type_text(spec)
                actual = type(arg_value).__name__
                type_errors.append(f"{arg_name} 期望类型 {expected}，实际为 {actual}")
        if type_errors:
            issues.extend(type_errors)

        if issues:
            return self._tool_repair_observation(key, arguments, issues)
        return None

    def _tool_runtime_error_observation(self, tool_name: str, arguments: Any, error_text: str) -> str:
        return self._tool_repair_observation(
            tool_name,
            arguments,
            [f"运行时错误: {str(error_text or '').strip() or '未知错误'}"],
            include_schema=False,
        )

    def _build_tool_skill_content(self, tool_name: str) -> str:
        tool = self._find_tool_spec(tool_name)
        if not tool:
            return f"Error: Unknown tool '{tool_name}'."

        name = str(tool.get("name") or "").strip()
        desc = str(tool.get("description") or "").strip() or "无描述"
        schema = tool.get("inputSchema") if isinstance(tool.get("inputSchema"), dict) else {}
        required = schema.get("required") if isinstance(schema.get("required"), list) else []
        required_set = {str(x) for x in required}
        props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}

        lines = [
            f"<skill name=\"tool:{name}\">",
            f"工具名称: {name}",
            f"用途: {desc}",
            "参数说明:",
        ]
        if props:
            for param_name, spec in props.items():
                if not isinstance(spec, dict):
                    spec = {}
                p_type = self._schema_type_text(spec)
                req = "必填" if str(param_name) in required_set else "可选"
                p_desc = re.sub(r"\s+", " ", str(spec.get("description") or "").strip())
                if not p_desc:
                    p_desc = "无描述"
                lines.append(f"- {param_name} ({p_type}, {req}): {p_desc}")
        else:
            lines.append("- 无参数")

        lines.append("调用格式示例:")
        if props:
            example_parts = []
            for param_name in props.keys():
                if str(param_name) in required_set:
                    example_parts.append(f"\"{param_name}\": <value>")
            example_json = "{ " + ", ".join(example_parts) + " }" if example_parts else "{}"
        else:
            example_json = "{}"
        lines.append(f"Action: {name}")
        lines.append(f"Action Input: {example_json}")
        lines.append("</skill>")
        return "\n".join(lines)

    def _build_skills_overview_context(
        self,
        user_input: str,
        memory_context: str = "",
        max_chars: Optional[int] = None,
        selected_tool_names: Optional[List[str]] = None,
    ) -> str:
        budget = self.skills_overview_context_chars if max_chars is None else int(max_chars)
        tool_name_set = set()
        if not self.force_full_skill_overview:
            tool_name_set = {
                str(tool.get("name") or "").strip()
                for tool in self.mcp_tools
                if str(tool.get("name") or "").strip() and str(tool.get("name") or "").strip() != "load_skill"
            }
        domain_desc_lines = []
        for name, skill in self.skill_loader.skills.items():
            if name in tool_name_set:
                continue
            meta = skill.get("meta") if isinstance(skill, dict) else {}
            desc = ""
            if isinstance(meta, dict):
                desc = str(meta.get("description") or "").strip()
            if not desc:
                desc = "No description"
            domain_desc_lines.append(f"- {name}: {desc}")
        domain_desc_lines = sorted(domain_desc_lines)

        shown_domain = domain_desc_lines

        lines = ["可用技能（第一层摘要，完整内容请用 load_skill 加载）："]
        if shown_domain:
            lines.append("领域技能:")
            lines.extend(shown_domain)

        if selected_tool_names:
            preview = ", ".join(selected_tool_names[:12])
            if preview:
                lines.append(f"本轮高相关工具: {preview}")

        lines.append('工具详解按需加载：Action: load_skill / Action Input: {"name": "tool:<工具名>"}')
        lines.append('领域流程按需加载：如 {"name":"team_recent_analysis"} / {"name":"team_hero_analysis"}')
        text = "\n".join(lines)
        if self.force_full_skill_overview and budget <= 0:
            return text
        return self._clip_context_text(text, budget)

    def _build_dynamic_prompt_contexts(
        self,
        user_input: str,
        memory_context: str = "",
        budget_chars: Optional[int] = None,
    ) -> List[str]:
        contexts: List[str] = []
        total_budget = self.context_budget_chars if budget_chars is None else int(budget_chars)
        budget_left = max(220, total_budget)

        selected_tools = self._select_context_tools(
            user_input=user_input,
            memory_context=memory_context,
            max_tools=0 if self.force_full_tool_context else 16,
        )
        selected_tool_names = [str(t.get("name") or "").strip() for t in selected_tools if str(t.get("name") or "").strip()]

        runtime_budget = budget_left if self.force_full_tool_context else min(self.runtime_tools_context_chars, budget_left)
        runtime_tools = self._build_runtime_tools_context(
            user_input=user_input,
            memory_context=memory_context,
            max_chars=runtime_budget,
            selected_tools=selected_tools,
        )
        if runtime_tools:
            contexts.append(runtime_tools)
            budget_left -= len(runtime_tools)

        if budget_left <= 120:
            return contexts

        skills_budget = budget_left if self.force_full_skill_overview else min(self.skills_overview_context_chars, budget_left)
        skills_overview = self._build_skills_overview_context(
            user_input=user_input,
            memory_context=memory_context,
            max_chars=skills_budget,
            selected_tool_names=selected_tool_names,
        )
        if skills_overview:
            contexts.append(skills_overview)
        return contexts

    def _needs_recent_context(self, user_input: str) -> bool:
        if not user_input:
            return False
        cleaned = user_input.strip()
        if len(cleaned) <= 14:
            return True
        pronouns = ("他们", "她们", "他们的", "她们的", "他", "她", "它", "这些", "那些", "这个", "那个", "上述", "上面", "之前", "刚才", "那次", "那他们", "他们最近")
        return any(p in cleaned for p in pronouns)

    def _build_recent_context(self) -> str:
        if not self._recent_turns:
            return ""
        def _clip(text: str, limit: int = 240) -> str:
            if not text:
                return ""
            return text if len(text) <= limit else text[:limit] + "…"
        lines = ["最近对话摘要（用于指代消解）："]
        for idx, turn in enumerate(self._recent_turns[-self.recent_turn_window :], start=1):
            user_text = _clip(turn.get("user", ""))
            assistant_text = _clip(turn.get("assistant", ""), limit=320)
            if user_text:
                lines.append(f"- 第{idx}轮用户：{user_text}")
            if assistant_text:
                lines.append(f"- 第{idx}轮助手：{assistant_text}")
        return self._clip_context_text("\n".join(lines), self.recent_context_max_chars)

    def _record_recent_turn(self, user_input: str, assistant_answer: str) -> None:
        if not (user_input or assistant_answer):
            return
        self._recent_turns.append({
            "user": user_input,
            "assistant": assistant_answer,
        })
        if len(self._recent_turns) > 20:
            self._recent_turns = self._recent_turns[-20:]

    def _reset_turn_skill_state(self) -> None:
        self._loaded_skills_current_turn = []
        self._tool_history_current_turn = []

    def _record_turn_tool_result(self, tool_name: str, tool_input: Dict[str, Any], observation: str) -> None:
        name = str(tool_name or "").strip()
        if not name:
            return
        self._tool_history_current_turn.append(name)
        if name == "load_skill":
            skill_name = str((tool_input or {}).get("name") or "").strip()
            if skill_name and not str(observation or "").startswith("Error:") and skill_name not in self._loaded_skills_current_turn:
                self._loaded_skills_current_turn.append(skill_name)

    @staticmethod
    def _contains_any(text: str, keywords: Tuple[str, ...]) -> bool:
        hay = str(text or "")
        return any(keyword in hay for keyword in keywords)

    def _skill_completion_followup(self, user_input: str, final_answer: str) -> Optional[str]:
        if not self._loaded_skills_current_turn:
            return None

        query = str(user_input or "").strip()
        answer = str(final_answer or "").strip()
        used_tools = set(self._tool_history_current_turn)

        if "team_hero_analysis" in self._loaded_skills_current_turn:
            missing_tools: List[str] = []
            missing_sections: List[str] = []

            if "get_team_heroes" not in used_tools:
                missing_tools.append("先补充 `get_team_heroes(team_id, limit>=20)`，完成“历史常用英雄”层。")

            if self._contains_any(query, ("最近", "近期", "选手", "玩什么英雄")):
                if "get_team_players" not in used_tools:
                    missing_tools.append("先补充 `get_team_players(team_id)`，确认当前成员。")
                if "选手" in query or "们" in query:
                    if "get_player_matches" not in used_tools:
                        missing_tools.append("先逐个调用 `get_player_matches(account_id, limit>=5)`，完成选手近期英雄统计。")
                elif "get_team_matches" not in used_tools and "get_player_matches" not in used_tools:
                    missing_tools.append("先补充 `get_team_matches(team_id, limit>=10)` 或 `get_player_matches(...)`，完成近期英雄统计。")

            if self._contains_any(query, ("出装", "装备", "路线")) and "get_match_items" not in used_tools:
                missing_tools.append("先补充 `get_match_items(match_id)`，完成近期英雄出装路线层。")

            if "get_team_heroes" in used_tools and not self._contains_any(answer, ("历史常用英雄", "历史英雄池")):
                missing_sections.append("在最终答案中增加“历史常用英雄”小节。")

            if self._contains_any(query, ("最近", "近期", "选手", "玩什么英雄")) and (
                ("get_player_matches" in used_tools or "get_team_matches" in used_tools)
                and not self._contains_any(answer, ("当前选手近期英雄", "近期英雄统计", "最近比赛英雄统计"))
            ):
                missing_sections.append("在最终答案中增加“当前选手近期英雄”或“最近比赛英雄统计”小节。")

            if self._contains_any(query, ("出装", "装备", "路线")) and "get_match_items" in used_tools and not self._contains_any(answer, ("出装路线", "核心成型件", "开局常见件")):
                missing_sections.append("在最终答案中增加“最近比赛出装路线”小节。")

            if missing_tools or missing_sections:
                lines = [
                    "你已加载 `team_hero_analysis`，但当前还没有严格完成该 skill 要求，暂时不能直接给出 Final Answer。",
                ]
                if missing_tools:
                    lines.append("请先补齐以下工具步骤：")
                    lines.extend([f"- {item}" for item in missing_tools])
                if missing_sections:
                    lines.append("补齐数据后，最终答案还需包含以下内容：")
                    lines.extend([f"- {item}" for item in missing_sections])
                lines.append("请继续使用 Thought/Action/Action Input 获取缺失数据，补齐后再输出完整 Final Answer。")
                return "\n".join(lines)

        return None

    def load_recent_context_from_session(self, conversations: List[Dict[str, Any]]) -> None:
        """从历史会话加载最近2-3轮上下文。"""
        turns: List[Dict[str, str]] = []
        for conv in conversations or []:
            user_text = str(conv.get("user_input") or "").strip()
            assistant_text = str(conv.get("final_answer") or "").strip()
            if not (user_text or assistant_text):
                continue
            turns.append({"user": user_text, "assistant": assistant_text})
        if not turns:
            return
        self._recent_turns = turns
        last = self._recent_turns[-1]
        self._last_user_input = last.get("user", "")
        self._last_assistant_answer = last.get("assistant", "")

    def _build_messages_for_turn(self, user_input: str, memory_context: str = "") -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = [{"role": "system", "content": self.system_prompt}]

        # 统一上下文总预算：动态工具/技能 + 记忆 + 最近对话
        budget_left = max(240, int(self.context_budget_chars))
        if self.context_chat_turns_only:
            memory_context = ""
        need_recent = bool(self._recent_turns) if self.context_chat_turns_only else self._needs_recent_context(user_input)
        reserve_memory = self._estimate_memory_reserve(user_input) if (memory_context and not self.context_chat_turns_only) else 0
        reserve_recent = min(self.recent_context_max_chars, max(0, budget_left // 4)) if need_recent else 0
        dynamic_budget = max(220, budget_left - reserve_memory - reserve_recent)

        # 记忆优先：先注入检索记忆，避免被后续工具上下文挤占。
        if memory_context and (not self.context_chat_turns_only) and budget_left > 120:
            mem_limit = min(max(120, reserve_memory or self.memory_context_max_chars), budget_left)
            mem_ctx = str(memory_context or "").strip()
            if mem_ctx and len(mem_ctx) <= mem_limit:
                messages.append({"role": "system", "content": mem_ctx})
                budget_left = max(0, budget_left - len(mem_ctx))
            elif mem_ctx:
                # retrieve_context(max_chars=预算) 已优先返回摘要；这里仅做兜底提示，避免硬截断。
                fallback = "相关记忆摘要（OpenViking）：当前上下文预算不足，已返回压缩摘要；需要细节请调用 memory_search。"
                messages.append({"role": "system", "content": fallback})
                budget_left = max(0, budget_left - len(fallback))

        dynamic_contexts = self._build_dynamic_prompt_contexts(
            user_input=user_input,
            memory_context=memory_context,
            budget_chars=min(dynamic_budget, budget_left),
        )
        for context in dynamic_contexts:
            if not context:
                continue
            messages.append({"role": "system", "content": context})
            budget_left = max(0, budget_left - len(context))

        if need_recent and budget_left > 120:
            recent_context = self._build_recent_context()
            recent_context = self._clip_context_text(
                recent_context,
                min(self.recent_context_max_chars, budget_left),
                suffix="\n...[最近对话已截断]",
            )
            if recent_context:
                messages.append({"role": "system", "content": recent_context})

        messages.append({"role": "user", "content": user_input})
        return messages

    def _should_retrieve_memory(self, query: str, force: bool) -> bool:
        return self.memory_manager.should_retrieve(query, force)

    async def _maybe_commit_recent_memory(self, force: bool) -> None:
        await self.memory_manager.maybe_commit_recent(force)

    async def _retrieve_memory_context(self, query: str) -> str:
        raw = await self.memory_manager.retrieve_context(
            query,
            max_chars=self.memory_context_max_chars,
            prefer_summary=True,
        )
        if not raw:
            return ""
        return raw

    def _estimate_memory_reserve(self, user_input: str) -> int:
        """估算本轮可用于 OpenViking 记忆注入的预算（字符）。"""
        if self.context_chat_turns_only:
            return 0
        budget = max(240, int(self.context_budget_chars))
        need_recent = self._needs_recent_context(user_input)
        reserve_recent = min(self.recent_context_max_chars, max(0, budget // 4)) if need_recent else 0
        budget_after_recent = max(0, budget - reserve_recent)
        reserve_memory = min(self.memory_context_max_chars, max(0, budget_after_recent // 3))
        return int(max(0, reserve_memory))

    async def _retrieve_memory_context_with_budget(self, query: str, max_chars: int) -> str:
        budget = int(max_chars or 0)
        if budget <= 0:
            return ""
        raw = await self.memory_manager.retrieve_context(
            query,
            max_chars=budget,
            prefer_summary=True,
        )
        if not raw:
            return ""
        return raw

    async def _finalize_memory(
        self,
        assistant_text: str,
        status: str = "success",
        force_commit: bool = False,
        session: Optional[Any] = None,
    ) -> None:
        await self.memory_manager.finalize_assistant(
            assistant_text,
            status=status,
            force_commit=force_commit,
            session=session,
        )

    def _finalize_memory_background(
        self,
        assistant_text: str,
        status: str = "success",
        force_commit: bool = False,
        session: Optional[Any] = None,
    ) -> None:
        """后台提交记忆，避免阻塞前端响应。"""
        self.memory_manager.finalize_assistant_background(
            assistant_text,
            status=status,
            force_commit=force_commit,
            session=session,
        )
    
    async def connect_mcp(self):
        """连接到 MCP Server"""
        if not HAS_MCP:
            raise RuntimeError("MCP 未安装")
        
        server_params = StdioServerParameters(
            command="python",
            args=[self.mcp_server_path],
        )
        
        self._stdio_client = stdio_client(server_params)
        self._read, self._write = await self._stdio_client.__aenter__()
        
        self.session = ClientSession(self._read, self._write)
        await self.session.__aenter__()
        await self.session.initialize()
        
        # 获取可用工具
        tools_result = await self.session.list_tools()
        self.mcp_tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.inputSchema
            }
            for tool in tools_result.tools
        ]
        self._register_local_tools()
        
        print(f"✅ MCP 连接成功，可用工具: {[t['name'] for t in self.mcp_tools]}")
    
    async def disconnect_mcp(self):
        """断开 MCP 连接"""
        await self.memory_manager.flush_and_close()

        if self.session:
            try:
                await asyncio.wait_for(self.session.__aexit__(None, None, None), timeout=1.5)
            except asyncio.TimeoutError:
                pass
        if hasattr(self, '_stdio_client'):
            try:
                await asyncio.wait_for(self._stdio_client.__aexit__(None, None, None), timeout=1.5)
            except asyncio.TimeoutError:
                pass
        
        # 保存会话日志
        if self.logger:
            self.logger.save_session()

    async def start_new_session(self) -> None:
        """创建一个新的日志/记忆会话"""
        if self.logger:
            self.logger.save_session()
            self.logger = ConversationLogger(self.log_dir)

        self.ov_session_id = self.logger.session_id if self.logger else None
        self._last_user_input = ""
        self._last_assistant_answer = ""
        self._recent_turns = []
        await self.memory_manager.start_new_session(self.ov_session_id)
    
    async def call_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """调用 MCP 工具"""
        if tool_name == "memory_search":
            query = str((arguments or {}).get("query") or "").strip()
            limit = (arguments or {}).get("limit", 5)
            return await self.memory_manager.memory_search_tool(query, limit=limit)

        if tool_name == "memory_commit":
            return await self.memory_manager.memory_commit_tool()

        if tool_name == "memory_status":
            return self.memory_manager.memory_status_tool()

        if tool_name == "load_skill":
            self.skill_loader.reload()
            name = str((arguments or {}).get("name") or "").strip()
            if not name:
                return "Error: Missing required field 'name' for load_skill."

            # 1) 先匹配本地 .skills 目录
            if name in self.skill_loader.skills:
                content = self.skill_loader.get_content(name)
                await self.memory_manager.record_tool_trace(
                    "load_skill",
                    arguments or {},
                    content,
                )
                return content

            # 2) 再匹配 tool:<tool_name> 或直接 tool_name
            target_tool = ""
            if name.startswith("tool:"):
                target_tool = name.split(":", 1)[1].strip()
            else:
                target_tool = name
            if target_tool:
                detail = self._build_tool_skill_content(target_tool)
                if not detail.startswith("Error: Unknown tool"):
                    await self.memory_manager.record_tool_trace(
                        "load_skill",
                        arguments or {},
                        detail,
                    )
                    return detail

            return f"Error: Unknown skill '{name}'."

        if not self.session:
            raise RuntimeError("MCP 未连接")
        
        result = await self.session.call_tool(tool_name, arguments)
        
        if result.content:
            text = result.content[0].text
            await self.memory_manager.record_tool_trace(
                tool_name,
                arguments or {},
                text,
            )
            return text
        return "无结果"
    
    def _call_llm(self, messages: List[Dict]) -> str:
        """调用 LLM"""
        if not self.llm_client:
            raise RuntimeError("LLM 客户端未初始化，请设置 API Key")

        request_kwargs = {
            "model": self.llm_model,
            "messages": messages,
            "temperature": 0.7,
            "timeout": self.llm_timeout,
        }
        if self.llm_max_tokens > 0:
            request_kwargs["max_tokens"] = self.llm_max_tokens

        response = self.llm_client.chat.completions.create(**request_kwargs)
        try:
            self._last_llm_finish_reason = str(response.choices[0].finish_reason or "")
        except Exception:
            self._last_llm_finish_reason = None
        return response.choices[0].message.content

    async def _call_llm_async(self, messages: List[Dict]) -> str:
        """异步调用 LLM（避免阻塞事件循环）"""
        if self.llm_async_client:
            request_kwargs = {
                "model": self.llm_model,
                "messages": messages,
                "temperature": 0.7,
                "timeout": self.llm_timeout,
            }
            if self.llm_max_tokens > 0:
                request_kwargs["max_tokens"] = self.llm_max_tokens
            response = await self.llm_async_client.chat.completions.create(**request_kwargs)
            try:
                self._last_llm_finish_reason = str(response.choices[0].finish_reason or "")
            except Exception:
                self._last_llm_finish_reason = None
            return response.choices[0].message.content
        return await asyncio.wait_for(
            asyncio.to_thread(self._call_llm, messages),
            timeout=self.llm_timeout,
        )

    async def _call_llm_stream(self, messages: List[Dict]) -> AsyncGenerator[str, None]:
        """流式调用 LLM（逐段产出内容）"""
        if self.llm_async_client:
            request_kwargs = {
                "model": self.llm_model,
                "messages": messages,
                "temperature": 0.7,
                "timeout": self.llm_timeout,
                "stream": True,
            }
            if self.llm_max_tokens > 0:
                request_kwargs["max_tokens"] = self.llm_max_tokens
            stream = await self.llm_async_client.chat.completions.create(**request_kwargs)
            finish_reason = None
            async for chunk in stream:
                choice = chunk.choices[0]
                if getattr(choice, "finish_reason", None):
                    finish_reason = choice.finish_reason
                delta = choice.delta
                if delta and delta.content:
                    yield delta.content
            self._last_llm_finish_reason = str(finish_reason or "")
            return
        response = await self._call_llm_async(messages)
        if response:
            yield response
    
    def _parse_action(self, response: str) -> Optional[Dict]:
        """
        解析 LLM 响应中的 Action
        
        Returns:
            {"action": "tool_name", "action_input": {...}} 或 None
        """
        # 匹配 Action 和 Action Input
        action_match = re.search(r'Action:\s*(\w+)', response)
        input_match = re.search(r'Action Input:\s*(\{.*?\})', response, re.DOTALL)
        
        if action_match:
            action = action_match.group(1)
            action_input = {}
            
            if input_match:
                try:
                    action_input = json.loads(input_match.group(1))
                except json.JSONDecodeError:
                    pass
            
            return {"action": action, "action_input": action_input}
        
        return None
    
    def _clean_llm_response(self, response: str) -> str:
        """
        清理 LLM 响应，移除 LLM 自己生成的假 Observation
        
        有些 LLM 会自己幻觉生成 Observation，需要将其移除
        """
        # 检测并移除 LLM 自己生成的 Observation 及其后面的内容
        # 只保留 Action Input 之前（包括）的内容
        
        # 查找 Action Input 的位置
        input_match = re.search(r'Action Input:\s*\{[^}]*\}', response, re.DOTALL)
        if input_match:
            # 规范化：Action Input 后不允许继续输出
            after_input = response[input_match.end():]
            if after_input.strip():
                print("⚠️ 检测到 Action Input 后继续输出，已截断")
                return response[:input_match.end()]

        return self._normalize_final_answer_response(response)

    @staticmethod
    def _is_generic_final_answer_suffix(suffix: str) -> bool:
        text = str(suffix or "").strip()
        if not text:
            return True
        if len(text) <= 40:
            return True
        return any(token in text for token in ("如上", "以上", "详见上文", "详见上述", "已整理"))

    @staticmethod
    def _split_leading_thought_block(text: str) -> Tuple[str, str]:
        raw = str(text or "").strip()
        if not raw:
            return "", ""
        match = re.match(r'^\s*Thought:\s*(.*?)(?=\n(?:Action:|Final Answer:)|\Z)', raw, re.DOTALL)
        if not match:
            return "", raw
        thought = match.group(1).strip()
        rest = raw[match.end():].lstrip()
        thought_block = f"Thought: {thought}" if thought else "Thought:"
        return thought_block, rest

    def _normalize_final_answer_response(self, response: str) -> str:
        text = str(response or "").strip()
        if not text or "Final Answer:" not in text:
            return text

        thought_block, rest = self._split_leading_thought_block(text)
        if not rest:
            return text
        if re.match(r'^\s*Final Answer:\s*', rest, re.DOTALL):
            return text
        if re.search(r'^\s*Action:\s*', rest, re.DOTALL):
            return text

        inline_match = re.search(r'\n\s*Final Answer:\s*(.*)$', rest, re.DOTALL)
        if not inline_match:
            return text

        prefix_body = rest[:inline_match.start()].strip()
        suffix = inline_match.group(1).strip()
        if not prefix_body:
            normalized_answer = suffix
        elif self._is_generic_final_answer_suffix(suffix):
            normalized_answer = prefix_body
        else:
            normalized_answer = f"{prefix_body}\n\n{suffix}"

        normalized_answer = normalized_answer.strip()
        if not normalized_answer:
            return text
        if thought_block:
            return f"{thought_block}\nFinal Answer: {normalized_answer}"
        return f"Final Answer: {normalized_answer}"

    def _sanitize_final_answer(self, answer: str) -> str:
        if not answer:
            return answer
        # Replace HTML line breaks to keep markdown clean and avoid tags in replies
        cleaned = re.sub(r"\s*<br\s*/?>\s*", " / ", answer, flags=re.IGNORECASE)
        # Remove markdown links; keep visible label as plain text
        cleaned = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", cleaned)
        # Remove raw URLs in final reply (file/http/https)
        cleaned = re.sub(r"\b(?:file|https?)://\S+", "", cleaned)
        # Remove horizontal rules (but keep markdown table separators that include pipes)
        cleaned = re.sub(r"(?m)^[ \t]*-{3,}[ \t]*$", "", cleaned)
        cleaned = re.sub(r"(?m)^[ \t:]*-{3,}[ \t:]*$", "", cleaned)
        # Remove blockquote markers to avoid styled highlight
        cleaned = re.sub(r"(?m)^[ \t]*>\s?", "", cleaned)
        # Remove inline code markers to avoid unintended highlighting
        cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
        cleaned = cleaned.replace("`", "")
        # Remove emphasis markers that may render as highlighted text
        cleaned = cleaned.replace("**", "")
        cleaned = cleaned.replace("__", "")
        cleaned = re.sub(r"(?<!\S)\*([^*\n]+)\*", r"\1", cleaned)
        cleaned = re.sub(r"(?<!\S)_([^_\n]+)_", r"\1", cleaned)
        # Collapse extra blank lines introduced by removals
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def _trim_observation_for_llm(self, observation: str) -> str:
        if not observation:
            return observation
        # 减少无意义空白，降低上下文噪声
        observation = observation.replace("\n", " ").replace("\"", "")
        observation = re.sub(r"[ \t]+", " ", observation)
        if self.max_observation_chars <= 0:
            return observation
        if len(observation) <= self.max_observation_chars:
            return observation
        truncated = observation[: self.max_observation_chars]
        return truncated + "\n...[truncated for context limit]"
    
    def _parse_final_answer(self, response: str) -> Optional[str]:
        """解析最终答案"""
        text = str(response or "").strip()
        if not text:
            return None

        leading_match = re.match(r'^\s*Final Answer:\s*(.*)$', text, re.DOTALL)
        if leading_match:
            answer = leading_match.group(1).strip()
            return self._sanitize_final_answer(answer)

        inline_match = re.search(r'\n\s*Final Answer:\s*(.*)$', text, re.DOTALL)
        if inline_match:
            prefix = text[:inline_match.start()].strip()
            suffix = inline_match.group(1).strip()

            # 有些模型会先输出完整总结，最后再补一句 "Final Answer: 详见上文"。
            # 这时应保留前面的正文，而不是只截取尾部摘要句。
            if prefix and not re.match(r'^\s*Thought\s*:', prefix, re.IGNORECASE):
                prefix_clean = self._sanitize_final_answer(prefix)
                generic_suffix = self._is_generic_final_answer_suffix(suffix)
                if generic_suffix:
                    return prefix_clean
                combined = f"{prefix}\n\n{suffix}"
                return self._sanitize_final_answer(combined)

            return self._sanitize_final_answer(suffix)
        return None

    def _extract_answer_text(self, response: str) -> str:
        """从续写响应中提取正文（兼容带/不带 Final Answer 前缀）"""
        parsed = self._parse_final_answer(response)
        if parsed is not None:
            return parsed
        text = str(response or "").strip()
        text = re.sub(r"^\s*(?:Thought|Action|Action Input|Observation)\s*:[\s\S]*$", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^\s*Final Answer\s*:\s*", "", text, flags=re.IGNORECASE)
        return self._sanitize_final_answer(text)

    def _looks_truncated_answer(self, answer: str, finish_reason: Optional[str]) -> bool:
        tail = str(answer or "").rstrip()
        if not tail:
            return False
        if str(finish_reason or "").lower() == "length":
            return True
        if tail.endswith(("...", "…", "，", ",", "：", ":", "|", "、", "和", "及")):
            return True
        if tail.count("```") % 2 != 0:
            return True
        if "|" in tail:
            last = tail.splitlines()[-1].strip()
            if last.startswith("|") and not last.endswith("|"):
                return True
        return False

    @staticmethod
    def _merge_answer_with_continuation(base: str, continuation: str) -> str:
        b = str(base or "").rstrip()
        c = str(continuation or "").strip()
        if not c:
            return b
        if not b:
            return c
        # 去重拼接：若续写开头与 base 末尾重叠，裁掉重复前缀
        max_overlap = min(len(b), len(c), 240)
        overlap = 0
        for k in range(max_overlap, 0, -1):
            if b.endswith(c[:k]):
                overlap = k
                break
        c2 = c[overlap:].lstrip()
        if not c2:
            return b
        joiner = "" if b.endswith(("\n", " ", "\t")) else "\n"
        return f"{b}{joiner}{c2}".strip()

    async def _expand_truncated_final_answer(
        self,
        messages: List[Dict[str, str]],
        llm_response: str,
        final_answer: str,
        max_rounds: int = 2,
    ) -> str:
        """
        当 Final Answer 因长度截断时，自动补充续写并拼接。
        """
        combined = self._sanitize_final_answer(final_answer)
        for _ in range(max_rounds):
            if not self._looks_truncated_answer(combined, self._last_llm_finish_reason):
                break
            continue_prompt = (
                "你上一条 Final Answer 可能因长度被截断。"
                "请只输出后续内容，不要重复前文，不要输出 Thought/Action/Observation，"
                "不要再次输出“Final Answer:”前缀。"
            )
            continuation_messages = [
                *messages,
                {"role": "assistant", "content": llm_response},
                {"role": "user", "content": continue_prompt},
            ]
            try:
                extra = await self._call_llm_async(continuation_messages)
            except Exception:
                break
            addition = self._extract_answer_text(extra)
            if not addition:
                break
            prev = combined
            combined = self._merge_answer_with_continuation(combined, addition)
            llm_response = f"Final Answer: {combined}"
            if combined == prev:
                break
        return self._sanitize_final_answer(combined)

    def _extract_thought(self, response: str) -> Optional[str]:
        match = re.search(r'Thought:\s*(.*?)(?:\nAction:|\nFinal Answer:|\Z)', response, re.DOTALL)
        if match:
            thought = match.group(1).strip()
            return thought or None
        return None

    def _extract_ward_html_path(self, text: str) -> Optional[str]:
        match = re.search(r'(ward_analysis[\\/](ward_(?:timeline|multi)_[^\\/]+\.html))', text)
        if match:
            return "/" + match.group(1).replace("\\", "/")
        return None

    def _reset_visual_reports(self) -> None:
        self._pending_visual_markdown = []

    def _maybe_capture_visual_report(self, tool_name: str, observation: str) -> None:
        return

    def _append_visual_reports(self, final_answer: str) -> Tuple[str, str]:
        if not self._pending_visual_markdown:
            return self._sanitize_final_answer(final_answer), ""
        appended: List[str] = []
        for report in self._pending_visual_markdown:
            report_text = (report or "").strip()
            if not report_text:
                continue
            if report_text in final_answer:
                continue
            first_line = next((line for line in report_text.splitlines() if line.strip()), "")
            if first_line and first_line in final_answer:
                continue
            appended.append(report_text)
        if not appended:
            self._pending_visual_markdown = []
            return self._sanitize_final_answer(final_answer), ""
        separator = "\n\n" if final_answer.strip() else ""
        appended_text = separator + "\n\n".join(appended)
        appended_text = self._sanitize_final_answer(appended_text)
        self._pending_visual_markdown = []
        combined = f"{self._sanitize_final_answer(final_answer)}{appended_text}"
        return combined, appended_text
    
    async def run(self, user_input: str) -> str:
        """
        执行 ReAct 循环
        
        Args:
            user_input: 用户输入
            
        Returns:
            最终回答
        """
        if not self.llm_client:
            raise RuntimeError("LLM 未配置，无法运行 ReAct 模式")

        memory_context = ""
        memory_ready = await self._ensure_memory_ready() if self.enable_memory else False
        memory_query, force_retrieve = self._build_memory_query(user_input)
        memory_budget = self._estimate_memory_reserve(user_input)
        if (not self.context_chat_turns_only) and self._should_retrieve_memory(memory_query, force_retrieve) and memory_ready:
            await self._maybe_commit_recent_memory(force_retrieve)
            memory_context = await self._retrieve_memory_context_with_budget(memory_query, memory_budget)
        if memory_ready and len(user_input.strip()) >= self.memory_record_user_min_chars:
            await self._record_memory_message("user", user_input)

        self._reset_visual_reports()
        self._reset_turn_skill_state()

        # 开始记录对话
        if self.logger:
            self.logger.start_conversation(user_input, self.llm_model)
        
        # 构建消息（system + 动态上下文 + memory + recent + user）
        messages = self._build_messages_for_turn(user_input=user_input, memory_context=memory_context)
        ward_html = None
        
        try:
            for i in range(self.max_iterations):
                # 调用 LLM（异步，避免阻塞）
                print("\n⏳ 正在请求 LLM...")
                try:
                    response = await asyncio.wait_for(
                        self._call_llm_async(messages),
                        timeout=self.llm_timeout,
                    )
                except asyncio.TimeoutError:
                    result = f"LLM 请求超时（>{self.llm_timeout:.0f}s），请稍后重试。"
                    if self.logger:
                        self.logger.end_conversation(result, "timeout", ward_html=ward_html)
                    self._last_user_input = user_input
                    self._last_assistant_answer = result
                    self._record_recent_turn(user_input, result)
                    self._finalize_memory_background(result, status="timeout")
                    return result
                except Exception as e:
                    result = f"LLM 调用失败: {e}"
                    if self.logger:
                        self.logger.end_conversation(result, "error", ward_html=ward_html)
                    self._last_user_input = user_input
                    self._last_assistant_answer = result
                    self._record_recent_turn(user_input, result)
                    self._finalize_memory_background(result, status="error")
                    return result
                
                # 清理 LLM 响应，移除自己生成的假 Observation
                response = self._clean_llm_response(response)
                
                print(f"\n--- 迭代 {i+1} ---")
                print(response)
                
                # 检查是否有最终答案
                final_answer = self._parse_final_answer(response)
                if final_answer:
                    followup = self._skill_completion_followup(user_input, final_answer)
                    if followup:
                        if self.logger:
                            self.logger.log_iteration(i + 1, response)
                        messages.append({"role": "assistant", "content": response})
                        messages.append({"role": "user", "content": followup})
                        continue
                    final_answer = await self._expand_truncated_final_answer(
                        messages=messages,
                        llm_response=response,
                        final_answer=final_answer,
                    )
                    final_answer, _ = self._append_visual_reports(final_answer)
                    # 记录最后一次迭代
                    if self.logger:
                        self.logger.log_iteration(i + 1, response)
                        self.logger.end_conversation(final_answer, "success", ward_html=ward_html)
                    self._last_user_input = user_input
                    self._last_assistant_answer = final_answer
                    self._record_recent_turn(user_input, final_answer)
                    self._finalize_memory_background(final_answer, status="success")
                    return final_answer
                
                # 解析 Action
                action_data = self._parse_action(response)
                
                if action_data:
                    tool_name = action_data["action"]
                    tool_input = action_data["action_input"]
                    
                    print(f"\n🔧 调用工具: {tool_name}")
                    print(f"   参数: {tool_input}")
                    
                    preflight_error = self._preflight_tool_call(tool_name, tool_input)
                    if preflight_error is not None:
                        observation = preflight_error
                    else:
                        # 调用 MCP 工具
                        try:
                            observation = await self.call_mcp_tool(tool_name, tool_input)
                        except Exception as e:
                            observation = self._tool_runtime_error_observation(tool_name, tool_input, str(e))

                    if tool_name in (
                        "analyze_match_wards",
                        "analyze_multi_match_wards",
                        "inject_ward_report_html",
                        "inject_multi_match_ward_report_html",
                    ):
                        ward_html = self._extract_ward_html_path(observation) or ward_html
                    
                    print(f"\n📋 Observation:\n{observation[:500]}...")
                    self._maybe_capture_visual_report(tool_name, observation)
                    self._record_turn_tool_result(tool_name, tool_input, observation)
                    
                    # 记录迭代
                    if self.logger:
                        self.logger.log_iteration(
                            i + 1, response,
                            action=tool_name,
                            action_input=tool_input,
                            observation=observation
                        )
                    
                    # 将结果加入消息历史（避免上下文过长）
                    observation_for_llm = self._trim_observation_for_llm(observation)
                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content": f"Observation: {observation_for_llm}"})
                else:
                    # 没有 Action 也没有 Final Answer，可能是格式问题
                    if self.logger:
                        self.logger.log_iteration(i + 1, response)
                    
                    messages.append({"role": "assistant", "content": response})
                    messages.append({
                        "role": "user",
                        "content": "请按照 ReAct 格式回复：使用 Action/Action Input 调用工具，或使用 Final Answer 给出最终答案。"
                    })
            
            # 达到最大迭代次数
            result = "达到最大迭代次数，无法完成任务。"
            if self.logger:
                self.logger.end_conversation(result, "max_iterations", ward_html=ward_html)
            self._last_user_input = user_input
            self._last_assistant_answer = result
            self._record_recent_turn(user_input, result)
            self._finalize_memory_background(result, status="max_iterations")
            return result
            
        except Exception as e:
            # 记录错误
            if self.logger:
                self.logger.end_conversation(str(e), "error", ward_html=ward_html)
            self._last_user_input = user_input
            self._last_assistant_answer = str(e)
            self._record_recent_turn(user_input, str(e))
            self._finalize_memory_background(str(e), status="error")
            raise

    async def run_stream(self, user_input: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        执行 ReAct 循环（流式输出 Thought/Action/Observation 与最终答案）
        """
        if not self.llm_client:
            raise RuntimeError("LLM 未配置，无法运行 ReAct 模式")

        memory_context = ""
        memory_ready = await self._ensure_memory_ready() if self.enable_memory else False
        memory_query, force_retrieve = self._build_memory_query(user_input)
        memory_budget = self._estimate_memory_reserve(user_input)
        if (not self.context_chat_turns_only) and self._should_retrieve_memory(memory_query, force_retrieve) and memory_ready:
            await self._maybe_commit_recent_memory(force_retrieve)
            memory_context = await self._retrieve_memory_context_with_budget(memory_query, memory_budget)
        if memory_ready and len(user_input.strip()) >= self.memory_record_user_min_chars:
            await self._record_memory_message("user", user_input)

        self._reset_visual_reports()
        self._reset_turn_skill_state()

        if self.logger:
            self.logger.start_conversation(user_input, self.llm_model)
            if self.logger.current_conversation:
                yield {
                    "type": "session",
                    "session_id": self.logger.session_id,
                    "conversation_id": self.logger.current_conversation.get("id"),
                    "timestamp": self.logger.current_conversation.get("timestamp"),
                    "status": self.logger.current_conversation.get("status", "running"),
                }

        messages = self._build_messages_for_turn(user_input=user_input, memory_context=memory_context)

        ward_html = None

        try:
            for i in range(self.max_iterations):
                print("\n⏳ 正在请求 LLM...")
                try:
                    response = ""

                    async for chunk in self._call_llm_stream(messages):
                        response += chunk
                except asyncio.TimeoutError:
                    result = f"LLM 请求超时（>{self.llm_timeout:.0f}s），请稍后重试。"
                    if self.logger:
                        self.logger.end_conversation(result, "timeout", ward_html=ward_html)
                    yield {"type": "final", "content": result, "ward_html": ward_html}
                    self._last_user_input = user_input
                    self._last_assistant_answer = result
                    self._record_recent_turn(user_input, result)
                    self._finalize_memory_background(result, status="timeout")
                    return
                except Exception as e:
                    result = f"LLM 调用失败: {e}"
                    if self.logger:
                        self.logger.end_conversation(result, "error", ward_html=ward_html)
                    yield {"type": "final", "content": result, "ward_html": ward_html}
                    self._last_user_input = user_input
                    self._last_assistant_answer = result
                    self._record_recent_turn(user_input, result)
                    self._finalize_memory_background(result, status="error")
                    return
                if not response:
                    result = "LLM 返回为空，请稍后重试。"
                    if self.logger:
                        self.logger.end_conversation(result, "error", ward_html=ward_html)
                    yield {"type": "final", "content": result, "ward_html": ward_html}
                    self._last_user_input = user_input
                    self._last_assistant_answer = result
                    self._record_recent_turn(user_input, result)
                    self._finalize_memory_background(result, status="error")
                    return
                response = self._clean_llm_response(response)

                print(f"\n--- 迭代 {i+1} ---")
                print(response)

                thought = self._extract_thought(response)
                if thought:
                    yield {"type": "thought", "content": thought}

                final_answer = self._parse_final_answer(response)
                if final_answer:
                    followup = self._skill_completion_followup(user_input, final_answer)
                    if followup:
                        if self.logger:
                            self.logger.log_iteration(i + 1, response)
                        messages.append({"role": "assistant", "content": response})
                        messages.append({"role": "user", "content": followup})
                        yield {"type": "observation", "content": followup}
                        continue
                    final_answer = await self._expand_truncated_final_answer(
                        messages=messages,
                        llm_response=response,
                        final_answer=final_answer,
                    )
                    final_answer, appended_text = self._append_visual_reports(final_answer)
                    if appended_text:
                        yield {"type": "final_delta", "content": appended_text}
                    if self.logger:
                        self.logger.log_iteration(i + 1, response)
                        self.logger.end_conversation(final_answer, "success", ward_html=ward_html)
                    yield {"type": "final", "content": final_answer, "ward_html": ward_html}
                    self._last_user_input = user_input
                    self._last_assistant_answer = final_answer
                    self._record_recent_turn(user_input, final_answer)
                    self._finalize_memory_background(final_answer, status="success")
                    return

                action_data = self._parse_action(response)

                if action_data:
                    tool_name = action_data["action"]
                    tool_input = action_data["action_input"]

                    yield {"type": "action", "content": tool_name, "input": tool_input}

                    print(f"\n🔧 调用工具: {tool_name}")
                    print(f"   参数: {tool_input}")

                    preflight_error = self._preflight_tool_call(tool_name, tool_input)
                    if preflight_error is not None:
                        observation = preflight_error
                    else:
                        try:
                            observation = await self.call_mcp_tool(tool_name, tool_input)
                        except Exception as e:
                            observation = self._tool_runtime_error_observation(tool_name, tool_input, str(e))

                    if tool_name in (
                        "analyze_match_wards",
                        "analyze_multi_match_wards",
                        "inject_ward_report_html",
                        "inject_multi_match_ward_report_html",
                    ):
                        ward_html = self._extract_ward_html_path(observation) or ward_html

                    print(f"\n📋 Observation:\n{observation[:500]}...")
                    self._maybe_capture_visual_report(tool_name, observation)
                    self._record_turn_tool_result(tool_name, tool_input, observation)

                    yield {"type": "observation", "content": observation}

                    if self.logger:
                        self.logger.log_iteration(
                            i + 1, response,
                            action=tool_name,
                            action_input=tool_input,
                            observation=observation
                        )

                    observation_for_llm = self._trim_observation_for_llm(observation)
                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content": f"Observation: {observation_for_llm}"})
                else:
                    if self.logger:
                        self.logger.log_iteration(i + 1, response)

                    messages.append({"role": "assistant", "content": response})
                    messages.append({
                        "role": "user",
                        "content": "请按照 ReAct 格式回复：使用 Action/Action Input 调用工具，或使用 Final Answer 给出最终答案。"
                    })

            result = "达到最大迭代次数，无法完成任务。"
            if self.logger:
                self.logger.end_conversation(result, "max_iterations", ward_html=ward_html)
            yield {"type": "final", "content": result, "ward_html": ward_html}
            self._last_user_input = user_input
            self._last_assistant_answer = result
            self._record_recent_turn(user_input, result)
            self._finalize_memory_background(result, status="max_iterations")

        except Exception as e:
            if self.logger:
                self.logger.end_conversation(str(e), "error", ward_html=ward_html)
            self._last_user_input = user_input
            self._last_assistant_answer = str(e)
            self._record_recent_turn(user_input, str(e))
            self._finalize_memory_background(str(e), status="error")
            raise


# ==================== 主函数 ====================

async def main():
    """主入口"""
    print("=" * 60)
    print("  🎮 Dota 2 ReAct Agent")
    print("  (ReAct 范式 + MCP 工具调用)")
    print("=" * 60)
    print()
    
    agent = Dota2ReActAgent(enable_logging=True)
    
    # 显示 LLM 配置信息
    print("📌 LLM 配置:")
    if agent.llm_client:
        print(f"   模型: {agent.llm_model}")
        print(f"   API: {agent.llm_base_url or 'OpenAI 默认'}")
        print(f"   状态: ✅ 已连接")
    else:
        print(f"   状态: ❌ 未配置")
        print(f"   提示: 设置 LLM_API_KEY 和 LLM_BASE_URL 环境变量")
        print("\n❌ LLM 未配置，无法启动 ReAct Agent")
        return
    
    # 显示日志配置
    if agent.logger:
        print(f"\n📌 日志配置:")
        print(f"   目录: {agent.logger.log_dir}/")
        print(f"   会话: {agent.logger.session_id}")
    print()
    
    try:
        await agent.connect_mcp()
        
        # 显示 MCP 工具信息
        print(f"📌 MCP 工具: {len(agent.mcp_tools)} 个可用")
        print("\n输入 'quit' 或 'exit' 退出\n")
        
        while True:
            try:
                user_input = input("你: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("再见！")
                    break
                
                if not user_input:
                    continue
                
                # 执行 ReAct 循环
                response = await agent.run(user_input)
                print(f"\n{'='*60}")
                print(f"✅ 最终回答:\n{response}")
                print(f"{'='*60}\n")
                
            except KeyboardInterrupt:
                print("\n再见！")
                break
            except Exception as e:
                print(f"错误: {e}\n")
    
    finally:
        await agent.disconnect_mcp()


if __name__ == "__main__":
    asyncio.run(main())
