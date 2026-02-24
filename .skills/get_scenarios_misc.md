---
description: 获取场景胜率统计
---
# get_scenarios_misc

## 工具用途
获取场景胜率统计

## 参数详解
- `scenario`: `string`，可选。场景名称
- `limit`: `integer`，可选。返回数量，默认20

## 输入 Schema
```json
{
  "type": "object",
  "properties": {
    "scenario": {
      "type": "string",
      "description": "场景名称"
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
Action: get_scenarios_misc
Action Input: {}

## 结果说明
该工具返回 Observation 文本。回答时仅可基于返回内容得出结论；若字段缺失，必须明确说明“数据未提供/无法确认”。
