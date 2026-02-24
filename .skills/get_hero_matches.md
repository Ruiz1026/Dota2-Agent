---
description: 获取指定英雄的最近比赛记录（对阵、阵营、胜负、时长）
---
# get_hero_matches

## 工具用途
获取指定英雄的最近比赛记录（对阵、阵营、胜负、时长）

## 参数详解
- `hero_id`: `integer`，必填。英雄ID
- `limit`: `integer`，可选。返回的比赛数量，默认20

## 输入 Schema
```json
{
  "type": "object",
  "properties": {
    "hero_id": {
      "type": "integer",
      "description": "英雄ID"
    },
    "limit": {
      "type": "integer",
      "description": "返回的比赛数量，默认20"
    }
  },
  "required": [
    "hero_id"
  ]
}
```

## ReAct 调用示例
Action: get_hero_matches
Action Input: {"hero_id": 0}

## 结果说明
该工具返回 Observation 文本。回答时仅可基于返回内容得出结论；若字段缺失，必须明确说明“数据未提供/无法确认”。
