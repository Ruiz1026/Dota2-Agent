---
description: 多场比赛眼位/击杀/推塔汇总与热力图（含眼位存活时长分析）
---
# analyze_multi_match_wards

## 工具用途
多场比赛眼位/击杀/推塔汇总与热力图（含眼位存活时长分析）

## 参数详解
- `team_id`: `integer`，可选。战队ID（与 account_id 二选一）
- `account_id`: `integer`，可选。玩家账号ID（与 team_id 二选一）
- `match_ids`: `array[integer]`，可选。指定比赛ID列表（优先级最高）
- `limit`: `integer`，可选。自动获取比赛数量，默认5
- `sigma`: `number`，可选。热力图高斯 sigma
- `alpha`: `number`，可选。热力图最大透明度

## 输入 Schema
```json
{
  "type": "object",
  "properties": {
    "team_id": {
      "type": "integer",
      "description": "战队ID（与 account_id 二选一）"
    },
    "account_id": {
      "type": "integer",
      "description": "玩家账号ID（与 team_id 二选一）"
    },
    "match_ids": {
      "type": "array",
      "items": {
        "type": "integer"
      },
      "description": "指定比赛ID列表（优先级最高）"
    },
    "limit": {
      "type": "integer",
      "description": "自动获取比赛数量，默认5"
    },
    "sigma": {
      "type": "number",
      "description": "热力图高斯 sigma"
    },
    "alpha": {
      "type": "number",
      "description": "热力图最大透明度"
    }
  },
  "required": []
}
```

## ReAct 调用示例
Action: analyze_multi_match_wards
Action Input: {}

## 结果说明
该工具返回 Observation 文本。回答时仅可基于返回内容得出结论；若字段缺失，必须明确说明“数据未提供/无法确认”。
