---
description: 获取最近的公开比赛列表（高段位）
---
# get_public_matches

## 工具用途
获取最近的公开比赛列表（高段位）

## 参数详解
- `min_rank`: `integer`，可选。最低段位等级，默认70（神话），范围10-85
- `limit`: `integer`，可选。返回的比赛数量，默认20

## 输入 Schema
```json
{
  "type": "object",
  "properties": {
    "min_rank": {
      "type": "integer",
      "description": "最低段位等级，默认70（神话），范围10-85"
    },
    "limit": {
      "type": "integer",
      "description": "返回的比赛数量，默认20"
    }
  },
  "required": []
}
```

## ReAct 调用示例
Action: get_public_matches
Action Input: {}

## 结果说明
该工具返回 Observation 文本。回答时仅可基于返回内容得出结论；若字段缺失，必须明确说明“数据未提供/无法确认”。
