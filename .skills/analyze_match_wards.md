---
description: 分析比赛眼位并生成可视化图表（静态图片和交互式HTML时间线）
---
# analyze_match_wards

## 工具用途
分析比赛眼位并生成可视化图表（静态图片和交互式HTML时间线）

## 参数详解
- `match_id`: `integer`，必填。Dota 2 比赛ID
- `generate_html`: `boolean`，可选。是否生成交互式HTML页面，默认True
- `generate_image`: `boolean`，可选。是否生成静态图片，默认True

## 输入 Schema
```json
{
  "type": "object",
  "properties": {
    "match_id": {
      "type": "integer",
      "description": "Dota 2 比赛ID"
    },
    "generate_html": {
      "type": "boolean",
      "description": "是否生成交互式HTML页面，默认True"
    },
    "generate_image": {
      "type": "boolean",
      "description": "是否生成静态图片，默认True"
    }
  },
  "required": [
    "match_id"
  ]
}
```

## ReAct 调用示例
Action: analyze_match_wards
Action Input: {"match_id": 0}

## 结果说明
该工具返回 Observation 文本。回答时仅可基于返回内容得出结论；若字段缺失，必须明确说明“数据未提供/无法确认”。
