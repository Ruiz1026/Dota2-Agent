---
description: 通过战队名搜索战队并获取其最近比赛（支持模糊匹配）
---
# search_team

## 工具用途
通过战队名搜索战队并获取其最近比赛（支持模糊匹配）

## 参数详解
- `team_name`: `string`，必填。战队名称或标签（如 'Team Spirit', 'TSpirit', 'OG', 'LGD'）

## 输入 Schema
```json
{
  "type": "object",
  "properties": {
    "team_name": {
      "type": "string",
      "description": "战队名称或标签（如 'Team Spirit', 'TSpirit', 'OG', 'LGD'）"
    }
  },
  "required": [
    "team_name"
  ]
}
```

## ReAct 调用示例
Action: search_team
Action Input: {"team_name": ""}

## 结果说明
该工具返回 Observation 文本。回答时仅可基于返回内容得出结论；若字段缺失，必须明确说明“数据未提供/无法确认”。
