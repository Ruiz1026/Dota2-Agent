---
description: 获取指定字段的记录排行
---
# get_records

## 工具用途
获取指定字段的记录排行

## 参数详解
- `field`: `string`，必填。记录字段名（如 kills, gpm, xpm, hero_damage）
- `limit`: `integer`，可选。返回数量，默认20

## 输入 Schema
```json
{
  "type": "object",
  "properties": {
    "field": {
      "type": "string",
      "description": "记录字段名（如 kills, gpm, xpm, hero_damage）"
    },
    "limit": {
      "type": "integer",
      "description": "返回数量，默认20"
    }
  },
  "required": [
    "field"
  ]
}
```

## ReAct 调用示例
Action: get_records
Action Input: {"field": ""}

## 结果说明
该工具返回 Observation 文本。回答时仅可基于返回内容得出结论；若字段缺失，必须明确说明“数据未提供/无法确认”。
