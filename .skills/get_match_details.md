---
description: 获取比赛摘要 + 原始数据（matches/{match_id} 响应字段）
---
# get_match_details

## 工具用途
获取比赛摘要 + 原始数据（matches/{match_id} 响应字段）

## 参数详解
- `match_id`: `integer`，必填。Dota 2 比赛ID，例如 8650430843

## 输入 Schema
```json
{
  "type": "object",
  "properties": {
    "match_id": {
      "type": "integer",
      "description": "Dota 2 比赛ID，例如 8650430843"
    }
  },
  "required": [
    "match_id"
  ]
}
```

## ReAct 调用示例
Action: get_match_details
Action Input: {"match_id": 0}

## 结果说明
该工具返回 Observation 文本。回答时仅可基于返回内容得出结论；若字段缺失，必须明确说明“数据未提供/无法确认”。
