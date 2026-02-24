---
description: 获取联赛列表
---
# get_leagues

## 工具用途
获取联赛列表

## 参数详解
- `limit`: `integer`，可选。返回的联赛数量，默认20

## 输入 Schema
```json
{
  "type": "object",
  "properties": {
    "limit": {
      "type": "integer",
      "description": "返回的联赛数量，默认20"
    }
  },
  "required": []
}
```

## ReAct 调用示例
Action: get_leagues
Action Input: {}

## 结果说明
该工具返回 Observation 文本。回答时仅可基于返回内容得出结论；若字段缺失，必须明确说明“数据未提供/无法确认”。
