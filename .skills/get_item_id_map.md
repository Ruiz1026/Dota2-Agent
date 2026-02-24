---
description: 查询装备ID映射（基于本地 constants_items_map.json）
---
# get_item_id_map

## 工具用途
查询装备ID映射（基于本地 constants_items_map.json）

## 参数详解
- `item_ids`: `array[integer]`，必填。物品ID列表

## 输入 Schema
```json
{
  "type": "object",
  "properties": {
    "item_ids": {
      "type": "array",
      "items": {
        "type": "integer"
      },
      "description": "物品ID列表"
    }
  },
  "required": [
    "item_ids"
  ]
}
```

## ReAct 调用示例
Action: get_item_id_map
Action Input: {"item_ids": []}

## 结果说明
该工具返回 Observation 文本。回答时仅可基于返回内容得出结论；若字段缺失，必须明确说明“数据未提供/无法确认”。
