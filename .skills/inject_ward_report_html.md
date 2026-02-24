---
description: 将视野分析报告HTML片段写入已生成的眼位时间线网页
---
# inject_ward_report_html

## 工具用途
将视野分析报告HTML片段写入已生成的眼位时间线网页

## 参数详解
- `match_id`: `integer`，必填。Dota 2 比赛ID
- `report_html`: `string`，可选。视野分析报告HTML片段（不要包含完整HTML文档）
- `report_path`: `string`，可选。视野分析报告HTML文件路径（报告较长时使用）

## 输入 Schema
```json
{
  "type": "object",
  "properties": {
    "match_id": {
      "type": "integer",
      "description": "Dota 2 比赛ID"
    },
    "report_html": {
      "type": "string",
      "description": "视野分析报告HTML片段（不要包含完整HTML文档）"
    },
    "report_path": {
      "type": "string",
      "description": "视野分析报告HTML文件路径（报告较长时使用）"
    }
  },
  "required": [
    "match_id"
  ]
}
```

## ReAct 调用示例
Action: inject_ward_report_html
Action Input: {"match_id": 0}

## 结果说明
该工具返回 Observation 文本。回答时仅可基于返回内容得出结论；若字段缺失，必须明确说明“数据未提供/无法确认”。
