---
description: 获取战队选手列表（account_id、name、games_played、wins、is_current_team_member）
---
# get_team_players

## 工具用途
获取战队选手列表（account_id、name、games_played、wins、is_current_team_member）

## 参数详解
- `team_id`: `integer`，必填。战队ID

## 输入 Schema
```json
{
  "type": "object",
  "properties": {
    "team_id": {
      "type": "integer",
      "description": "战队ID"
    }
  },
  "required": [
    "team_id"
  ]
}
```

## ReAct 调用示例
Action: get_team_players
Action Input: {"team_id": 0}

## 结果说明
该工具返回 Observation 文本。回答时仅可基于返回内容得出结论；若字段缺失，必须明确说明“数据未提供/无法确认”。
