---
description: 获取分路角色胜率统计
---
# get_scenarios_lane_roles

## 工具用途
获取分路角色胜率统计

## 参数详解
- `lane_role`: `integer`，可选。分路角色 1-4
- `hero_id`: `integer`，可选。英雄ID
- `limit`: `integer`，可选。返回数量，默认20

## 输入 Schema
```json
{
  "type": "object",
  "properties": {
    "lane_role": {
      "type": "integer",
      "description": "分路角色 1-4"
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
Action: get_scenarios_lane_roles
Action Input: {}

## 结果说明
该工具返回 Observation 文本。回答时仅可基于返回内容得出结论；若字段缺失，必须明确说明“数据未提供/无法确认”。
