"""
OpenViking memory manager.

This module centralizes OpenViking lifecycle, retrieval, write/commit strategy,
and session switching so agent code only orchestrates high-level flow.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class OpenVikingMemoryManager:
    def __init__(
        self,
        *,
        enable_memory: bool,
        has_openviking: bool,
        ov_module: Optional[Any],
        text_part_cls: Optional[Any],
        ov_config_path: str,
        ov_data_path: str,
        session_id: Optional[str] = None,
        memory_top_k: int = 3,
        memory_commit_every_n: int = 5,
        memory_commit_min_chars: int = 200,
        memory_commit_only_success: bool = True,
        memory_retrieve_every_n: int = 2,
        memory_retrieve_min_chars: int = 12,
        memory_retrieve_timeout: float = 4.0,
        memory_commit_timeout: float = 1.2,
        memory_record_user_min_chars: int = 8,
        memory_record_assistant_min_chars: int = 80,
        tool_trace_max_chars: int = 0,
    ) -> None:
        self.enable_memory = bool(enable_memory)
        self.has_openviking = bool(has_openviking)
        self.ov_module = ov_module
        self.text_part_cls = text_part_cls

        self.ov_config_path = ov_config_path
        self.ov_data_path = ov_data_path
        self.ov_session_id = session_id

        self.memory_top_k = max(1, int(memory_top_k))
        self.memory_commit_every_n = max(1, int(memory_commit_every_n))
        self.memory_commit_min_chars = max(0, int(memory_commit_min_chars))
        self.memory_commit_only_success = bool(memory_commit_only_success)
        self.memory_retrieve_every_n = max(1, int(memory_retrieve_every_n))
        self.memory_retrieve_min_chars = max(0, int(memory_retrieve_min_chars))
        self.memory_retrieve_timeout = max(0.2, float(memory_retrieve_timeout))
        self.memory_commit_timeout = max(0.2, float(memory_commit_timeout))
        self.memory_record_user_min_chars = max(0, int(memory_record_user_min_chars))
        self.memory_record_assistant_min_chars = max(0, int(memory_record_assistant_min_chars))
        self.tool_trace_max_chars = int(tool_trace_max_chars)

        self.ov_client: Optional[Any] = None
        self.ov_session: Optional[Any] = None

        self._memory_pending_count = 0
        self._memory_turn = 0
        self._memory_last_query = ""
        self._memory_last_context = ""
        self._last_search_error = ""
        self._last_search_mode = ""
        self._memory_commit_lock = asyncio.Lock()
        self._background_tasks: set = set()

    async def ensure_ready(self, session_id: Optional[str] = None) -> bool:
        if session_id:
            self.ov_session_id = session_id
        if not self.enable_memory or not self.has_openviking:
            return False
        if self.ov_client and self.ov_session:
            return True
        try:
            if not (self.ov_config_path and os.path.exists(self.ov_config_path)):
                self.enable_memory = False
                return False

            os.environ["OPENVIKING_CONFIG_FILE"] = os.path.abspath(self.ov_config_path)
            os.makedirs(self.ov_data_path, exist_ok=True)

            if not self.ov_module:
                self.enable_memory = False
                return False

            self.ov_client = self.ov_module.AsyncOpenViking(path=self.ov_data_path)
            await self.ov_client.initialize()

            if not self.ov_session_id:
                self.ov_session_id = "default"
            self.ov_session = self.ov_client.session(session_id=self.ov_session_id)
            await asyncio.to_thread(self.ov_session.load)
            return True
        except Exception:
            self.enable_memory = False
            self.ov_client = None
            self.ov_session = None
            return False

    def build_memory_query(self, user_input: str, last_user: str = "", last_assistant: str = "") -> Tuple[str, bool]:
        cleaned = (user_input or "").strip()
        if len(cleaned) >= self.memory_retrieve_min_chars:
            return cleaned, False
        parts = [p for p in (last_user, last_assistant, cleaned) if p]
        if parts:
            return "\n".join(parts), True
        return cleaned, False

    def should_retrieve(self, query: str, force: bool) -> bool:
        if not self.enable_memory or not self.has_openviking:
            return False
        if not query or len(query.strip()) < self.memory_retrieve_min_chars:
            return False
        if force:
            return True
        self._memory_turn += 1
        return self._memory_turn % self.memory_retrieve_every_n == 1

    async def record_message(self, role: str, content: str, session: Optional[Any] = None) -> bool:
        target_session = session or self.ov_session
        if not target_session or not content or not self.text_part_cls:
            return False

        def _add() -> None:
            target_session.add_message(role, [self.text_part_cls(text=content)])

        await asyncio.to_thread(_add)
        return True

    async def commit_session(self, session: Optional[Any] = None) -> bool:
        target_session = session or self.ov_session
        if not target_session:
            return False
        async with self._memory_commit_lock:
            await asyncio.to_thread(target_session.commit)
        return True

    async def maybe_commit_recent(self, force: bool) -> None:
        if not force or not self.ov_session or self._memory_pending_count <= 0:
            return
        try:
            await asyncio.wait_for(
                self.commit_session(session=self.ov_session),
                timeout=self.memory_commit_timeout,
            )
            self._memory_pending_count = 0
        except asyncio.TimeoutError:
            return

    @staticmethod
    def _compact_text(text: str) -> str:
        return " ".join(str(text or "").split())

    @classmethod
    def _memory_line(
        cls,
        mem: Any,
        *,
        include_overview: bool,
        include_reason: bool,
        line_max_chars: int = 0,
    ) -> str:
        abstract = cls._compact_text(getattr(mem, "abstract", "") or "")
        overview = cls._compact_text(getattr(mem, "overview", "") or "")
        category = cls._compact_text(getattr(mem, "category", "") or "")
        reason = cls._compact_text(getattr(mem, "match_reason", "") or "")

        summary = abstract
        if include_overview and overview and overview != abstract:
            summary = f"{summary} (details: {overview})" if summary else overview
        if category:
            summary = f"[{category}] {summary}" if summary else f"[{category}]"
        if include_reason and reason:
            summary = f"{summary} (match: {reason})" if summary else reason

        if line_max_chars > 0 and len(summary) > line_max_chars:
            summary = summary[: max(0, line_max_chars - 3)] + "..."
        return f"- {summary}" if summary else ""

    @classmethod
    def _compose_context(
        cls,
        memories: List[Any],
        *,
        include_overview: bool,
        include_reason: bool,
        line_max_chars: int = 0,
        header: str = "Relevant memories (for reference, may be incomplete):",
    ) -> str:
        lines: List[str] = []
        for mem in memories:
            line = cls._memory_line(
                mem,
                include_overview=include_overview,
                include_reason=include_reason,
                line_max_chars=line_max_chars,
            )
            if line:
                lines.append(line)
        if not lines:
            return ""
        return header + "\n" + "\n".join(lines)

    async def retrieve_context(
        self,
        query: str,
        max_chars: int = 0,
        prefer_summary: bool = True,
    ) -> str:
        if not self.ov_client or not query:
            return ""
        cache_key = f"{query}::{int(max_chars)}::{1 if prefer_summary else 0}"
        if cache_key == self._memory_last_query and self._memory_last_context:
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
                self._memory_last_query = cache_key
                self._memory_last_context = ""
                return ""

            try:
                self.ov_session.used(contexts=[m.uri for m in memories])
            except Exception:
                pass

            context = self._compose_context(
                memories[: self.memory_top_k],
                include_overview=True,
                include_reason=True,
            )
            if not context:
                self._memory_last_query = cache_key
                self._memory_last_context = ""
                return ""

            if max_chars > 0 and len(context) > max_chars and prefer_summary:
                compact = self._compose_context(
                    memories[: self.memory_top_k],
                    include_overview=True,
                    include_reason=False,
                    header="Relevant memory summary (OpenViking):",
                )
                if compact and len(compact) <= max_chars:
                    context = compact
                else:
                    compact = self._compose_context(
                        memories[: self.memory_top_k],
                        include_overview=False,
                        include_reason=False,
                        header="Relevant memory summary (OpenViking):",
                    )
                    if compact and len(compact) <= max_chars:
                        context = compact
                    else:
                        used = False
                        for keep in range(min(self.memory_top_k, len(memories)), 0, -1):
                            compact = self._compose_context(
                                memories[:keep],
                                include_overview=False,
                                include_reason=False,
                                header=f"Relevant memory summary (OpenViking, top {keep}):",
                            )
                            if compact and len(compact) <= max_chars:
                                context = compact
                                used = True
                                break
                        if not used:
                            per_line_max = max(
                                24,
                                (max_chars // max(1, min(self.memory_top_k, len(memories)))) - 10,
                            )
                            compact = self._compose_context(
                                memories[: min(self.memory_top_k, len(memories))],
                                include_overview=False,
                                include_reason=False,
                                line_max_chars=per_line_max,
                                header="Relevant memory summary (OpenViking, compressed):",
                            )
                            if compact and len(compact) <= max_chars:
                                context = compact
                            else:
                                context = (
                                    f"Relevant memory summary (OpenViking): found {len(memories)} matches, "
                                    "but the current context budget is too small. Use memory_search for details."
                                )

            if max_chars > 0 and len(context) > max_chars:
                context = context[:max_chars]

            self._memory_last_query = cache_key
            self._memory_last_context = context
            return context
        except asyncio.TimeoutError:
            return ""
        except Exception:
            return ""

    async def record_user(self, user_input: str) -> None:
        text = (user_input or "").strip()
        if not text or len(text) < self.memory_record_user_min_chars:
            return
        await self.record_message("user", text)

    async def record_tool_trace(self, tool_name: str, tool_input: Any, observation: str) -> None:
        if not self.ov_session:
            return
        tool_name = (tool_name or "").strip()
        if not tool_name:
            return
        try:
            input_text = json.dumps(tool_input, ensure_ascii=False)
        except Exception:
            input_text = str(tool_input)
        obs = str(observation or "")
        if self.tool_trace_max_chars > 0 and len(obs) > self.tool_trace_max_chars:
            obs = obs[: self.tool_trace_max_chars] + "...[truncated]"
        content = (
            "[tool_trace]\n"
            f"tool={tool_name}\n"
            f"input={input_text}\n"
            f"observation={obs}"
        )
        ok = await self.record_message("assistant", content)
        if ok:
            self._memory_pending_count += 1

    async def finalize_assistant(
        self,
        assistant_text: str,
        status: str = "success",
        force_commit: bool = False,
        session: Optional[Any] = None,
    ) -> None:
        target_session = session or self.ov_session
        if not target_session:
            return

        text = (assistant_text or "").strip()
        if text and len(text) >= self.memory_record_assistant_min_chars:
            ok = await self.record_message("assistant", text, session=target_session)
            if ok and target_session is self.ov_session:
                self._memory_pending_count += 1

        if not force_commit:
            if self.memory_commit_only_success and status != "success":
                return
            if text and len(text) < self.memory_commit_min_chars:
                return
            if target_session is self.ov_session and self._memory_pending_count < self.memory_commit_every_n:
                return

        await self.commit_session(session=target_session)
        if target_session is self.ov_session:
            self._memory_pending_count = 0

    def finalize_assistant_background(
        self,
        assistant_text: str,
        status: str = "success",
        force_commit: bool = False,
        session: Optional[Any] = None,
    ) -> None:
        target_session = session or self.ov_session
        if not target_session:
            return
        task = asyncio.create_task(
            self.finalize_assistant(
                assistant_text,
                status=status,
                force_commit=force_commit,
                session=target_session,
            )
        )
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def start_new_session(self, new_session_id: Optional[str]) -> None:
        old_session = self.ov_session
        had_pending = self._memory_pending_count > 0

        self.ov_session_id = new_session_id
        self.ov_session = None
        self._memory_pending_count = 0
        self._memory_turn = 0
        self._memory_last_query = ""
        self._memory_last_context = ""
        self._last_search_error = ""
        self._last_search_mode = ""

        if self.enable_memory and self.ov_client and self.ov_session_id:
            self.ov_session = self.ov_client.session(session_id=self.ov_session_id)
            await asyncio.to_thread(self.ov_session.load)

        if old_session and had_pending:
            self.finalize_assistant_background("", force_commit=True, session=old_session)

    async def flush_and_close(self) -> None:
        if self._background_tasks:
            for task in list(self._background_tasks):
                task.cancel()
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()

        if not self.ov_client:
            return
        if self.ov_session and self._memory_pending_count > 0:
            try:
                await asyncio.wait_for(
                    self.finalize_assistant("", force_commit=True),
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

    @staticmethod
    def _query_keywords(query: str) -> List[str]:
        keywords: List[str] = []
        seen: set = set()
        for token in re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9_\-]{3,}", str(query or "")):
            norm = token.strip().lower()
            if not norm:
                continue
            variants = [norm]
            if re.fullmatch(r"[\u4e00-\u9fff]{3,}", norm):
                variants.extend(norm[i : i + 2] for i in range(len(norm) - 1))
            for item in variants:
                if not item or item in seen:
                    continue
                seen.add(item)
                keywords.append(item)
        return keywords

    def _local_memory_fallback_search(self, query: str, limit: int = 5) -> List[Dict[str, str]]:
        root = Path(self.ov_data_path) / "viking" / "user" / "memories"
        if not root.exists():
            return []

        query_lower = str(query or "").strip().lower()
        keywords = self._query_keywords(query)
        matches: List[Tuple[int, float, Dict[str, str]]] = []

        for path in root.rglob("*.md"):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore").strip()
            except Exception:
                continue
            if not text:
                continue

            text_lower = text.lower()
            score = 0
            if query_lower and query_lower in text_lower:
                score += 8
            hits = 0
            for keyword in keywords:
                if keyword in text_lower:
                    hits += 1
            score += hits * 3
            if score <= 0:
                continue

            first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
            if not first_line:
                continue
            snippet = first_line if len(first_line) <= 220 else first_line[:217] + "..."
            item = {
                "category": path.parent.name,
                "abstract": snippet,
                "path": str(path),
            }
            try:
                mtime = path.stat().st_mtime
            except Exception:
                mtime = 0.0
            matches.append((score, mtime, item))

        matches.sort(key=lambda x: (x[0], x[1]), reverse=True)
        lim = max(1, min(int(limit or 5), 20))
        return [item for _, _, item in matches[:lim]]

    @staticmethod
    def _format_local_memory_results(query: str, matches: List[Dict[str, str]], reason: str = "") -> str:
        lines: List[str] = [f"# Memory search results (query={query}, source=local_fallback)"]
        if reason:
            lines.append(f"> {reason}")
        for idx, item in enumerate(matches, start=1):
            category = item.get("category") or "local"
            abstract = item.get("abstract") or "N/A"
            path = item.get("path") or ""
            lines.append(f"{idx}. [{category}] {abstract}")
            if path:
                lines.append(f"   - source: {path}")
        return "\n".join(lines)

    async def memory_search_tool(self, query: str, limit: int = 5) -> str:
        query = (query or "").strip()
        if not query:
            return "Error: query cannot be empty."
        if not self.enable_memory:
            return "Memory is disabled."
        ready = await self.ensure_ready(self.ov_session_id)
        if not ready:
            return "Memory is unavailable (initialization failed or configuration missing)."

        lim = max(1, min(int(limit or 5), 20))
        try:
            result = await asyncio.wait_for(
                self.ov_client.search(
                    query=query,
                    target_uri="viking://user/memories",
                    session=self.ov_session,
                    limit=lim,
                ),
                timeout=self.memory_retrieve_timeout,
            )
            memories = result.memories if result else []
            self._last_search_mode = "openviking"
            self._last_search_error = ""
        except asyncio.TimeoutError:
            self._last_search_mode = "local_fallback"
            self._last_search_error = f"openviking search timed out after {self.memory_retrieve_timeout:.1f}s"
            fallback = self._local_memory_fallback_search(query, limit=lim)
            if fallback:
                return self._format_local_memory_results(
                    query=query,
                    matches=fallback,
                    reason=f"OpenViking search timed out after {self.memory_retrieve_timeout:.1f}s; used local fallback.",
                )
            return f"Memory search timed out after {self.memory_retrieve_timeout:.1f}s, and local fallback found nothing."
        except Exception as e:
            err = str(e).strip() or e.__class__.__name__
            self._last_search_mode = "local_fallback"
            self._last_search_error = err
            fallback = self._local_memory_fallback_search(query, limit=lim)
            if fallback:
                return self._format_local_memory_results(
                    query=query,
                    matches=fallback,
                    reason=f"OpenViking search failed ({err}); used local fallback.",
                )
            return f"Memory search failed: {err}"

        if not memories:
            fallback = self._local_memory_fallback_search(query, limit=lim)
            if fallback:
                self._last_search_mode = "local_fallback"
                self._last_search_error = ""
                return self._format_local_memory_results(
                    query=query,
                    matches=fallback,
                    reason="OpenViking returned no hits; used local fallback.",
                )
            return "No relevant memories found."

        lines = [f"# Memory search results (query={query}, limit={lim})"]
        for idx, mem in enumerate(memories[:lim], start=1):
            abstract = getattr(mem, "abstract", "") or "N/A"
            category = getattr(mem, "category", "") or "N/A"
            overview = getattr(mem, "overview", "") or ""
            reason = getattr(mem, "match_reason", "") or ""
            uri = getattr(mem, "uri", "") or ""
            lines.append(f"{idx}. [{category}] {abstract}")
            if overview:
                lines.append(f"   - overview: {overview}")
            if reason:
                lines.append(f"   - match_reason: {reason}")
            if uri:
                lines.append(f"   - uri: {uri}")
        return "\n".join(lines)

    async def memory_commit_tool(self) -> str:
        if not self.enable_memory:
            return "Memory is disabled."
        ready = await self.ensure_ready(self.ov_session_id)
        if not ready:
            return "Memory is unavailable (initialization failed or configuration missing)."
        try:
            await asyncio.wait_for(
                self.commit_session(),
                timeout=self.memory_commit_timeout,
            )
            self._memory_pending_count = 0
            return "Memory committed."
        except Exception as e:
            err = str(e).strip() or e.__class__.__name__
            return f"Memory commit failed: {err}"

    def memory_status_tool(self) -> str:
        local_memory_count = 0
        try:
            local_memory_count = sum(
                1 for _ in (Path(self.ov_data_path) / "viking" / "user" / "memories").rglob("*.md")
            )
        except Exception:
            local_memory_count = 0
        return json.dumps(
            {
                "enable_memory": self.enable_memory,
                "has_openviking": self.has_openviking,
                "session_id": self.ov_session_id,
                "ready": bool(self.ov_client and self.ov_session),
                "last_search_mode": self._last_search_mode,
                "last_search_error": self._last_search_error,
                "local_memory_count": local_memory_count,
                "pending_count": self._memory_pending_count,
                "memory_top_k": self.memory_top_k,
                "retrieve_every_n": self.memory_retrieve_every_n,
                "retrieve_min_chars": self.memory_retrieve_min_chars,
                "retrieve_timeout": self.memory_retrieve_timeout,
                "commit_every_n": self.memory_commit_every_n,
                "commit_min_chars": self.memory_commit_min_chars,
            },
            ensure_ascii=False,
            indent=2,
        )
