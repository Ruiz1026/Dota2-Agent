---
description: 搜索玩家（按昵称）
---
# search_players

## 工具用途
搜索玩家（按昵称）

## 参数详解
- `query`: `string`，必填。搜索关键词（玩家昵称）

## 输入 Schema
```json
{
  "type": "object",
  "properties": {
    "query": {
      "type": "string",
      "description": "搜索关键词（玩家昵称）"
    }
  },
  "required": [
    "query"
  ]
}
```

## ReAct 调用示例
Action: search_players
Action Input: {"query": ""}

## 结果说明
该工具返回 Observation 文本。回答时仅可基于返回内容得出结论；若字段缺失，必须明确说明“数据未提供/无法确认”。
