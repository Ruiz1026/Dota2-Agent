"""
Skill loader for two-layer skill injection.

Layer 1: short descriptions for system prompt.
Layer 2: full content returned by load_skill tool.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Tuple


class SkillLoader:
    def __init__(self, skills_dir: Path):
        self.skills_dir = Path(skills_dir)
        self.skills: Dict[str, Dict[str, str | Dict[str, str]]] = {}
        self.reload()

    def reload(self) -> None:
        self.skills = {}
        if not self.skills_dir.exists():
            return
        for path in sorted(self.skills_dir.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            meta, body = self._parse_frontmatter(text)
            self.skills[path.stem] = {"meta": meta, "body": body}

    @staticmethod
    def _parse_frontmatter(text: str) -> Tuple[Dict[str, str], str]:
        """
        Parse markdown frontmatter:
        ---
        key: value
        ---
        body...
        """
        if not text:
            return {}, ""
        normalized = text.replace("\r\n", "\n")
        match = re.match(r"^---\n(.*?)\n---\n?(.*)$", normalized, re.DOTALL)
        if not match:
            return {}, normalized.strip()
        meta_raw = match.group(1).strip()
        body = match.group(2).strip()
        meta: Dict[str, str] = {}
        for line in meta_raw.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
        return meta, body

    def get_descriptions(self) -> str:
        lines = []
        for name, skill in self.skills.items():
            meta = skill.get("meta") or {}
            desc = str(meta.get("description") or "No description")
            lines.append(f"- {name}: {desc}")
        return "\n".join(lines)

    def get_content(self, name: str) -> str:
        key = (name or "").strip()
        if not key:
            return "Error: Missing required field 'name' for load_skill."
        skill = self.skills.get(key)
        if not skill:
            return f"Error: Unknown skill '{key}'."
        body = str(skill.get("body") or "").strip()
        return f"<skill name=\"{key}\">\n{body}\n</skill>"

