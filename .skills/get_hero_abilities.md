---
description: 根据英雄ID获取技能列表（含天赋/命石）
---
# get_hero_abilities

## 工具用途
根据英雄ID获取技能列表（含天赋/命石）

## 参数详解
- `hero_id`: `integer`，必填。英雄ID
- `include_talents`: `boolean`，可选。是否包含天赋，默认True

## 输入 Schema
```json
{
  "type": "object",
  "properties": {
    "hero_id": {
      "type": "integer",
      "description": "英雄ID"
    },
    "include_talents": {
      "type": "boolean",
      "description": "是否包含天赋，默认True"
    }
  },
  "required": [
    "hero_id"
  ]
}
```

## ReAct 调用示例
Action: get_hero_abilities
Action Input: {"hero_id": 0}

## 结果说明
该工具返回 Observation 文本。回答时仅可基于返回内容得出结论；若字段缺失，必须明确说明“数据未提供/无法确认”。
