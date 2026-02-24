---
description: 获取战队列表（按评分排序）
---
# get_teams

## 工具用途
获取战队列表（按评分排序）

## 参数详解
- `limit`: `integer`，可选。返回的战队数量，默认20

## 输入 Schema
```json
{
  "type": "object",
  "properties": {
    "limit": {
      "type": "integer",
      "description": "返回的战队数量，默认20"
    }
  },
  "required": []
}
```

## ReAct 调用示例
Action: get_teams
Action Input: {}

## 结果说明
该工具返回 Observation 文本。回答时仅可基于返回内容得出结论；若字段缺失，必须明确说明“数据未提供/无法确认”。
