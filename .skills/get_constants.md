---
description: 获取 constants 静态数据（优先读取本地缓存）
---
# get_constants

## 工具用途
获取 constants 静态数据（优先读取本地缓存）

## 参数详解
- `resource`: `string`，必填。资源名（如 heroes, items, abilities；也可用 list 获取资源列表）

## 输入 Schema
```json
{
  "type": "object",
  "properties": {
    "resource": {
      "type": "string",
      "description": "资源名（如 heroes, items, abilities；也可用 list 获取资源列表）"
    }
  },
  "required": [
    "resource"
  ]
}
```

## ReAct 调用示例
Action: get_constants
Action Input: {"resource": ""}

## 结果说明
该工具返回 Observation 文本。回答时仅可基于返回内容得出结论；若字段缺失，必须明确说明“数据未提供/无法确认”。
