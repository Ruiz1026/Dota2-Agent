---
description: 批量提交比赛录像解析请求
---
# request_match_parses

## 工具用途
批量提交比赛录像解析请求

## 参数详解
- `match_ids`: `array[integer]`，必填。比赛ID列表

## 输入 Schema
```json
{
  "type": "object",
  "properties": {
    "match_ids": {
      "type": "array",
      "items": {
        "type": "integer"
      },
      "description": "比赛ID列表"
    }
  },
  "required": [
    "match_ids"
  ]
}
```

## ReAct 调用示例
Action: request_match_parses
Action Input: {"match_ids": []}

## 结果说明
该工具返回 Observation 文本。回答时仅可基于返回内容得出结论；若字段缺失，必须明确说明“数据未提供/无法确认”。
