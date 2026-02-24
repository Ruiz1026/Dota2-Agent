---
description: OpenViking 记忆工作流（检索/提交/状态）
---
当你需要跨轮记忆支持时，按以下流程：

1. 先判断是否需要历史信息：
- 用户提到“之前/刚才/上次/继续/他们”等指代时，优先用 `memory_search` 检索。

2. 检索方式：
- `memory_search` 传入明确关键词（对象名 + 任务类型 + 时间线索）。
- 若检索为空，明确说明“未检索到相关记忆”，不要编造。

3. 写入与提交：
- 工具调用轨迹与最终答案会被系统自动记录。
- 若用户要求“立即保存/落盘”，调用 `memory_commit`。

4. 状态观察：
- 需要诊断记忆是否可用时，调用 `memory_status` 查看 `ready/session_id/pending_count`。

5. 回答约束：
- 记忆内容仅作为辅助线索；与当前 Observation 冲突时，以当前 Observation 为准。
