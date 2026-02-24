---
description: 获取装备时机胜率统计
---
# get_scenarios_item_timings

## 工具用途
获取装备时机胜率统计

## 参数详解
- `item`: `string`，可选。物品名（如 spirit_vessel）
- `hero_id`: `integer`，可选。英雄ID
- `limit`: `integer`，可选。返回数量，默认20

## 输入 Schema
```json
{
  "type": "object",
  "properties": {
    "item": {
      "type": "string",
      "description": "物品名（如 spirit_vessel）"
    },
    "hero_id": {
      "type": "integer",
      "description": "英雄ID"
    },
    "limit": {
      "type": "integer",
      "description": "返回数量，默认20"
    }
  },
  "required": []
}
```

## ReAct 调用示例
Action: get_scenarios_item_timings
Action Input: {}

## 结果说明
该工具返回 Observation 文本。回答时仅可基于返回内容得出结论；若字段缺失，必须明确说明“数据未提供/无法确认”。
