# utils/logger.py
"""
对话日志记录器

记录 LLM 对话历史，包括：
- 用户输入
- LLM 响应（每次迭代）
- 工具调用
- 最终答案
"""

import os
import json
import time
import re
from datetime import datetime
from typing import Optional, Dict, List, Any


class ConversationLogger:
    """
    对话日志记录器
    """
    
    def __init__(
        self,
        log_dir: str = "logs",
        save_interval_seconds: float = 2.0,
        max_observation_chars: int = 8000,
    ):
        """
        初始化日志记录器
        
        Args:
            log_dir: 日志保存目录
            save_interval_seconds: 会话写盘间隔（秒）
            max_observation_chars: Observation 最大字符数
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.save_interval_seconds = max(0.0, float(save_interval_seconds))
        self.max_observation_chars = max(0, int(max_observation_chars))
        self._last_save_time = 0.0
        self._dirty = False
        
        # 当前会话
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.conversations: List[Dict] = []
        self.current_conversation: Optional[Dict] = None
        
    def start_conversation(self, user_input: str, model: str):
        """开始新对话"""
        self.current_conversation = {
            "id": len(self.conversations) + 1,
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "user_input": user_input,
            "iterations": [],
            "final_answer": None,
            "total_iterations": 0,
            "status": "running"
        }
        self.conversations.append(self.current_conversation)
        self._mark_dirty()
        self.save_session()
    
    def log_iteration(
        self,
        iteration: int,
        llm_response: str,
        action: Optional[str] = None,
        action_input: Optional[Dict] = None,
        observation: Optional[str] = None
    ):
        """记录一次迭代"""
        if not self.current_conversation:
            return

        if observation:
            # 与 LLM 侧一致的清理规则
            observation = observation.replace("\n", " ").replace("\"", "")
            observation = re.sub(r"[ \t]+", " ", observation)
        if observation and self.max_observation_chars > 0 and len(observation) > self.max_observation_chars:
            observation = observation[: self.max_observation_chars] + "\n...[truncated]"
        
        iteration_log = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "llm_response": llm_response,
            "action": action,
            "action_input": action_input,
            "observation": observation
        }
        
        self.current_conversation["iterations"].append(iteration_log)
        self.current_conversation["total_iterations"] = iteration
        self._mark_dirty()
        self._maybe_save()
    
    def end_conversation(self, final_answer: str, status: str = "success"):
        """结束对话"""
        if not self.current_conversation:
            return
        
        self.current_conversation["final_answer"] = final_answer
        self.current_conversation["status"] = status
        self.current_conversation["end_timestamp"] = datetime.now().isoformat()

        self._mark_dirty()
        self.save_session()
        self.current_conversation = None

    def _mark_dirty(self) -> None:
        self._dirty = True

    def _maybe_save(self) -> None:
        if not self._dirty or self.save_interval_seconds <= 0:
            return
        now = time.monotonic()
        if (now - self._last_save_time) < self.save_interval_seconds:
            return
        self._write_session()
        self._last_save_time = now
        self._dirty = False
    
    def save_session(self):
        """保存整个会话"""
        if not self.conversations:
            return

        self._write_session()
        self._last_save_time = time.monotonic()
        self._dirty = False

    def _write_session(self):
        session_data = {
            "session_id": self.session_id,
            "start_time": self.conversations[0]["timestamp"] if self.conversations else None,
            "end_time": datetime.now().isoformat(),
            "total_conversations": len(self.conversations),
            "conversations": self.conversations
        }
        
        filename = f"session_{self.session_id}.json"
        filepath = os.path.join(self.log_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)
    
    def get_log_summary(self) -> str:
        """获取日志摘要"""
        return f"会话 {self.session_id}: {len(self.conversations)} 次对话"