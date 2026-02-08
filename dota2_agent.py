# dota2_agent.py
"""
Dota 2 ReAct Agent
使用 ReAct 范式 + MCP 工具调用
"""

import os
import re
import json
import asyncio
from typing import Optional, Dict, Any, List, AsyncGenerator, Tuple
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 导入提示词和工具定义
from prompts.dota2_agent_prompt import DOTA2_SYSTEM_PROMPT, DOTA2_MCP_TOOLS

# 导入日志模块
from utils.logger import ConversationLogger

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
        max_observation_chars: int = 40000,
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
        memory_retrieve_every_n: int = 2,
        memory_retrieve_min_chars: int = 12,
        memory_retrieve_timeout: float = 2.0,
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
        
        # LLM 配置
        self.llm_api_key = llm_api_key or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
        self.llm_base_url = llm_base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL")
        self.llm_model = llm_model or os.getenv("LLM_MODEL_ID") or "deepseek-v3.2"
        self.llm_timeout = float(os.getenv("LLM_TIMEOUT", llm_timeout))
        self.max_observation_chars = int(os.getenv("MAX_OBSERVATION_CHARS", max_observation_chars))
        
        self.max_iterations = max_iterations
        self.system_prompt = DOTA2_SYSTEM_PROMPT
        
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
        self.ov_client = None
        self.ov_session = None
        self.ov_session_id = self.logger.session_id if self.logger else None
        self._memory_pending_count = 0
        self._memory_commit_lock = asyncio.Lock()
        self._memory_turn = 0
        self._memory_last_query = ""
        self._memory_last_context = ""
        self._last_user_input = ""
        self._pending_visual_markdown: List[str] = []
        self._last_assistant_answer = ""
        self._recent_turns: List[Dict[str, str]] = []
        self._background_tasks: set = set()
        
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
        if not self.enable_memory or not HAS_OPENVIKING:
            return False
        if self.ov_client and self.ov_session:
            return True
        try:
            if not (self.ov_config_path and os.path.exists(self.ov_config_path)):
                print("⚠️ OpenViking 配置文件不存在，已关闭记忆功能")
                self.enable_memory = False
                return False

            os.environ["OPENVIKING_CONFIG_FILE"] = os.path.abspath(self.ov_config_path)
            os.makedirs(self.ov_data_path, exist_ok=True)

            self.ov_client = ov.AsyncOpenViking(path=self.ov_data_path)
            await self.ov_client.initialize()

            if not self.ov_session_id and self.logger:
                self.ov_session_id = self.logger.session_id
            self.ov_session = self.ov_client.session(session_id=self.ov_session_id)
            await asyncio.to_thread(self.ov_session.load)
            return True
        except Exception as e:
            print(f"⚠️ OpenViking 初始化失败，已关闭记忆功能: {e}")
            self.enable_memory = False
            return False

    async def _record_memory_message(self, role: str, content: str, session: Optional[Any] = None) -> None:
        target_session = session or self.ov_session
        if not target_session or not content:
            return

        def _add():
            target_session.add_message(role, [TextPart(text=content)])

        await asyncio.to_thread(_add)

    async def _commit_memory_session(self, session: Optional[Any] = None) -> None:
        target_session = session or self.ov_session
        if not target_session:
            return
        async with self._memory_commit_lock:
            await asyncio.to_thread(target_session.commit)

    def _build_memory_query(self, user_input: str) -> tuple[str, bool]:
        cleaned = user_input.strip()
        if len(cleaned) >= self.memory_retrieve_min_chars:
            return cleaned, False
        parts = [p for p in (self._last_user_input, self._last_assistant_answer, cleaned) if p]
        if parts:
            return "\n".join(parts), True
        return cleaned, False

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
        def _clip(text: str, limit: int = 800) -> str:
            if not text:
                return ""
            return text if len(text) <= limit else text[:limit] + "…"
        lines = ["最近对话摘要（用于指代消解）："]
        for idx, turn in enumerate(self._recent_turns[-3:], start=1):
            user_text = _clip(turn.get("user", ""))
            assistant_text = _clip(turn.get("assistant", ""))
            if user_text:
                lines.append(f"- 第{idx}轮用户：{user_text}")
            if assistant_text:
                lines.append(f"- 第{idx}轮助手：{assistant_text}")
        return "\n".join(lines)

    def _record_recent_turn(self, user_input: str, assistant_answer: str) -> None:
        if not (user_input or assistant_answer):
            return
        self._recent_turns.append({
            "user": user_input,
            "assistant": assistant_answer,
        })
        if len(self._recent_turns) > 20:
            self._recent_turns = self._recent_turns[-20:]

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

    def _should_retrieve_memory(self, query: str, force: bool) -> bool:
        if not self.enable_memory or not HAS_OPENVIKING:
            return False
        if not query or len(query.strip()) < self.memory_retrieve_min_chars:
            return False
        if force:
            return True
        self._memory_turn += 1
        if self._memory_turn % self.memory_retrieve_every_n != 1:
            return False
        return True

    async def _maybe_commit_recent_memory(self, force: bool) -> None:
        if not force or not self.ov_session or self._memory_pending_count <= 0:
            return
        try:
            await asyncio.wait_for(
                self._commit_memory_session(session=self.ov_session),
                timeout=self.memory_commit_timeout,
            )
            self._memory_pending_count = 0
        except asyncio.TimeoutError:
            return

    async def _retrieve_memory_context(self, query: str) -> str:
        if not self.ov_client or not query:
            return ""
        if query == self._memory_last_query and self._memory_last_context:
            return self._memory_last_context
        try:
            result = await asyncio.wait_for(
                self.ov_client.search(
                    query=query,
                    target_uri="viking://user/memories",
                    session=self.ov_session,
                    limit=self.memory_top_k,
                ),
                timeout=self.memory_retrieve_timeout,
            )
            memories = result.memories if result else []
            if not memories:
                self._memory_last_query = query
                self._memory_last_context = ""
                return ""

            try:
                self.ov_session.used(contexts=[m.uri for m in memories])
            except Exception:
                pass

            lines = []
            for mem in memories[: self.memory_top_k]:
                summary = mem.abstract or ""
                if mem.overview and mem.overview != summary:
                    summary = f"{summary}（详情：{mem.overview}）" if summary else mem.overview
                if mem.category:
                    summary = f"[{mem.category}] {summary}" if summary else f"[{mem.category}]"
                if mem.match_reason:
                    summary = f"{summary}（匹配原因：{mem.match_reason}）"
                if summary:
                    lines.append(f"- {summary}")

            if not lines:
                self._memory_last_query = query
                self._memory_last_context = ""
                return ""

            context = "相关记忆（供参考，可能不完整）：\n" + "\n".join(lines)
            self._memory_last_query = query
            self._memory_last_context = context
            return context
        except asyncio.TimeoutError:
            return ""
        except Exception as e:
            print(f"⚠️ 记忆检索失败: {e}")
            return ""

    async def _finalize_memory(
        self,
        assistant_text: str,
        status: str = "success",
        force_commit: bool = False,
        session: Optional[Any] = None,
    ) -> None:
        target_session = session or self.ov_session
        if not target_session:
            return

        if assistant_text and len(assistant_text.strip()) >= self.memory_record_assistant_min_chars:
            await self._record_memory_message("assistant", assistant_text, session=target_session)
            if target_session is self.ov_session:
                self._memory_pending_count += 1

        if not force_commit:
            if self.memory_commit_only_success and status != "success":
                return
            if assistant_text and len(assistant_text.strip()) < self.memory_commit_min_chars:
                return
            if target_session is self.ov_session and self._memory_pending_count < self.memory_commit_every_n:
                return

        await self._commit_memory_session(session=target_session)
        if target_session is self.ov_session:
            self._memory_pending_count = 0

    def _finalize_memory_background(
        self,
        assistant_text: str,
        status: str = "success",
        force_commit: bool = False,
        session: Optional[Any] = None,
    ) -> None:
        """后台提交记忆，避免阻塞前端响应。"""
        target_session = session or self.ov_session
        if not target_session:
            return
        task = asyncio.create_task(
            self._finalize_memory(
                assistant_text,
                status=status,
                force_commit=force_commit,
                session=target_session,
            )
        )
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
    
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
        
        print(f"✅ MCP 连接成功，可用工具: {[t['name'] for t in self.mcp_tools]}")
    
    async def disconnect_mcp(self):
        """断开 MCP 连接"""
        # 尽量快速取消后台任务，避免阻塞退出
        if self._background_tasks:
            for task in list(self._background_tasks):
                task.cancel()
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()

        if self.ov_client:
            if self.ov_session and self._memory_pending_count > 0:
                try:
                    await asyncio.wait_for(
                        self._finalize_memory("", force_commit=True),
                        timeout=self.memory_commit_timeout,
                    )
                except asyncio.TimeoutError:
                    pass
            try:
                await asyncio.wait_for(self.ov_client.close(), timeout=1.5)
            except asyncio.TimeoutError:
                pass
            self.ov_client = None
            self.ov_session = None

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
        old_session = self.ov_session
        had_pending = self._memory_pending_count > 0
        if self.logger:
            self.logger.save_session()
            self.logger = ConversationLogger(self.log_dir)

        self.ov_session_id = self.logger.session_id if self.logger else None
        self.ov_session = None
        self._memory_pending_count = 0
        self._memory_turn = 0
        self._memory_last_query = ""
        self._memory_last_context = ""
        self._last_user_input = ""
        self._last_assistant_answer = ""
        self._recent_turns = []

        if self.enable_memory and self.ov_client and self.ov_session_id:
            self.ov_session = self.ov_client.session(session_id=self.ov_session_id)
            await asyncio.to_thread(self.ov_session.load)

        if old_session and had_pending:
            self._finalize_memory_background("", force_commit=True, session=old_session)
    
    async def call_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """调用 MCP 工具"""
        if not self.session:
            raise RuntimeError("MCP 未连接")
        
        result = await self.session.call_tool(tool_name, arguments)
        
        if result.content:
            return result.content[0].text
        return "无结果"
    
    def _call_llm(self, messages: List[Dict]) -> str:
        """调用 LLM"""
        if not self.llm_client:
            raise RuntimeError("LLM 客户端未初始化，请设置 API Key")
        
        response = self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            temperature=0.7,
            timeout=self.llm_timeout,
        )
        
        return response.choices[0].message.content

    async def _call_llm_async(self, messages: List[Dict]) -> str:
        """异步调用 LLM（避免阻塞事件循环）"""
        if self.llm_async_client:
            response = await self.llm_async_client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=0.7,
                timeout=self.llm_timeout,
            )
            return response.choices[0].message.content
        return await asyncio.wait_for(
            asyncio.to_thread(self._call_llm, messages),
            timeout=self.llm_timeout,
        )

    async def _call_llm_stream(self, messages: List[Dict]) -> AsyncGenerator[str, None]:
        """流式调用 LLM（逐段产出内容）"""
        if self.llm_async_client:
            stream = await self.llm_async_client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=0.7,
                timeout=self.llm_timeout,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content
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
        
        return response

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
        match = re.search(r'Final Answer:\s*(.*)', response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

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
        if tool_name != "save_match_details_report":
            return
        marker_line = ""
        lines = observation.splitlines()
        marker_index = None
        for idx, line in enumerate(lines):
            if "Markdown" in line and line.strip().startswith("##"):
                marker_line = line.strip()
                marker_index = idx
                break
        if marker_index is None:
            return
        report_body = "\n".join(lines[marker_index + 1:]).strip()
        if not report_body:
            return
        report_markdown = f"{marker_line}\n{report_body}".strip()
        if report_markdown not in self._pending_visual_markdown:
            self._pending_visual_markdown.append(report_markdown)

    def _append_visual_reports(self, final_answer: str) -> Tuple[str, str]:
        if not self._pending_visual_markdown:
            return final_answer, ""
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
            return final_answer, ""
        separator = "\n\n" if final_answer.strip() else ""
        appended_text = separator + "\n\n".join(appended)
        self._pending_visual_markdown = []
        return f"{final_answer}{appended_text}", appended_text
    
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
        if self._should_retrieve_memory(memory_query, force_retrieve) and memory_ready:
            await self._maybe_commit_recent_memory(force_retrieve)
            memory_context = await self._retrieve_memory_context(memory_query)
        if memory_ready and len(user_input.strip()) >= self.memory_record_user_min_chars:
            await self._record_memory_message("user", user_input)

        self._reset_visual_reports()

        # 开始记录对话
        if self.logger:
            self.logger.start_conversation(user_input, self.llm_model)
        
        # 构建消息
        messages = [{"role": "system", "content": self.system_prompt}]
        if memory_context:
            messages.append({"role": "system", "content": memory_context})
        if self._needs_recent_context(user_input):
            recent_context = self._build_recent_context()
            if recent_context:
                messages.append({"role": "system", "content": recent_context})
        messages.append({"role": "user", "content": user_input})
        
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
                        self.logger.end_conversation(result, "timeout")
                    self._last_user_input = user_input
                    self._last_assistant_answer = result
                    self._record_recent_turn(user_input, result)
                    self._finalize_memory_background(result, status="timeout")
                    return result
                except Exception as e:
                    result = f"LLM 调用失败: {e}"
                    if self.logger:
                        self.logger.end_conversation(result, "error")
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
                    final_answer, _ = self._append_visual_reports(final_answer)
                    # 记录最后一次迭代
                    if self.logger:
                        self.logger.log_iteration(i + 1, response)
                        self.logger.end_conversation(final_answer, "success")
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
                    
                    # 调用 MCP 工具
                    try:
                        observation = await self.call_mcp_tool(tool_name, tool_input)
                    except Exception as e:
                        observation = f"工具调用错误: {str(e)}"
                    
                    print(f"\n📋 Observation:\n{observation[:500]}...")
                    self._maybe_capture_visual_report(tool_name, observation)
                    
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
                self.logger.end_conversation(result, "max_iterations")
            self._last_user_input = user_input
            self._last_assistant_answer = result
            self._record_recent_turn(user_input, result)
            self._finalize_memory_background(result, status="max_iterations")
            return result
            
        except Exception as e:
            # 记录错误
            if self.logger:
                self.logger.end_conversation(str(e), "error")
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
        if self._should_retrieve_memory(memory_query, force_retrieve) and memory_ready:
            await self._maybe_commit_recent_memory(force_retrieve)
            memory_context = await self._retrieve_memory_context(memory_query)
        if memory_ready and len(user_input.strip()) >= self.memory_record_user_min_chars:
            await self._record_memory_message("user", user_input)

        self._reset_visual_reports()

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

        messages = [
            {"role": "system", "content": self.system_prompt},
        ]
        if memory_context:
            messages.append({"role": "system", "content": memory_context})
        if self._needs_recent_context(user_input):
            recent_context = self._build_recent_context()
            if recent_context:
                messages.append({"role": "system", "content": recent_context})
        messages.append({"role": "user", "content": user_input})

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
                        self.logger.end_conversation(result, "timeout")
                    yield {"type": "final", "content": result, "ward_html": ward_html}
                    self._last_user_input = user_input
                    self._last_assistant_answer = result
                    self._record_recent_turn(user_input, result)
                    self._finalize_memory_background(result, status="timeout")
                    return
                except Exception as e:
                    result = f"LLM 调用失败: {e}"
                    if self.logger:
                        self.logger.end_conversation(result, "error")
                    yield {"type": "final", "content": result, "ward_html": ward_html}
                    self._last_user_input = user_input
                    self._last_assistant_answer = result
                    self._record_recent_turn(user_input, result)
                    self._finalize_memory_background(result, status="error")
                    return
                if not response:
                    result = "LLM 返回为空，请稍后重试。"
                    if self.logger:
                        self.logger.end_conversation(result, "error")
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
                    final_answer, appended_text = self._append_visual_reports(final_answer)
                    if appended_text:
                        yield {"type": "final_delta", "content": appended_text}
                    if self.logger:
                        self.logger.log_iteration(i + 1, response)
                        self.logger.end_conversation(final_answer, "success")
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

                    try:
                        observation = await self.call_mcp_tool(tool_name, tool_input)
                    except Exception as e:
                        observation = f"工具调用错误: {str(e)}"

                    if tool_name in ("analyze_match_wards", "analyze_multi_match_wards", "inject_multi_match_ward_report_html"):
                        ward_html = self._extract_ward_html_path(observation) or ward_html

                    print(f"\n📋 Observation:\n{observation[:500]}...")
                    self._maybe_capture_visual_report(tool_name, observation)

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
                self.logger.end_conversation(result, "max_iterations")
            yield {"type": "final", "content": result, "ward_html": ward_html}
            self._last_user_input = user_input
            self._last_assistant_answer = result
            self._record_recent_turn(user_input, result)
            self._finalize_memory_background(result, status="max_iterations")

        except Exception as e:
            if self.logger:
                self.logger.end_conversation(str(e), "error")
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
