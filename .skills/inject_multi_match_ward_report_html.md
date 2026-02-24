---
description: 读取多场比赛视野汇总JSON并写入汇总分析报告
---
# inject_multi_match_ward_report_html

## 工具用途
读取多场比赛视野汇总JSON并写入汇总分析报告

## 参数详解
- `summary_path`: `string`，可选。汇总 JSON 文件路径（可选）
- `report_html`: `string`，可选。汇总分析报告HTML片段（不要包含完整HTML文档）
- `report_path`: `string`，可选。汇总分析报告HTML文件路径（报告较长时使用）
- `html_path`: `string`，可选。多场比赛网页路径（可选）

## 输入 Schema
```json
{
  "type": "object",
  "properties": {
    "summary_path": {
      "type": "string",
      "description": "汇总 JSON 文件路径（可选）"
    },
    "report_html": {
      "type": "string",
      "description": "汇总分析报告HTML片段（不要包含完整HTML文档）"
    },
    "report_path": {
      "type": "string",
      "description": "汇总分析报告HTML文件路径（报告较长时使用）"
    },
    "html_path": {
      "type": "string",
      "description": "多场比赛网页路径（可选）"
    }
  },
  "required": []
}
```

## ReAct 调用示例
Action: inject_multi_match_ward_report_html
Action Input: {}

## 结果说明
该工具返回 Observation 文本。回答时仅可基于返回内容得出结论；若字段缺失，必须明确说明“数据未提供/无法确认”。
