---
description: 获取指定玩家的统计总计数据（总击杀、死亡、助攻、GPM等）
---
# get_player_totals

## 工具用途
获取指定玩家的统计总计数据（总击杀、死亡、助攻、GPM等）

## 参数详解
- `account_id`: `integer`，必填。Steam 32位账号ID

## 输入 Schema
```json
{
  "type": "object",
  "properties": {
    "account_id": {
      "type": "integer",
      "description": "Steam 32位账号ID"
    }
  },
  "required": [
    "account_id"
  ]
}
```

## ReAct 调用示例
Action: get_player_totals
Action Input: {"account_id": 0}

## 结果说明
该工具返回 Observation 文本。回答时仅可基于返回内容得出结论；若字段缺失，必须明确说明“数据未提供/无法确认”。
