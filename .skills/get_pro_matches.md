---
description: 获取最近的职业比赛列表
---
# get_pro_matches

## 工具用途
获取最近的职业比赛列表

## 参数详解
- `limit`: `integer`，可选。返回的比赛数量，默认10

## 输入 Schema
```json
{
  "type": "object",
  "properties": {
    "limit": {
      "type": "integer",
      "description": "返回的比赛数量，默认10"
    }
  },
  "required": []
}
```

## ReAct 调用示例
Action: get_pro_matches
Action Input: {}

## 结果说明
该工具返回 Observation 文本。回答时仅可基于返回内容得出结论；若字段缺失，必须明确说明“数据未提供/无法确认”。
