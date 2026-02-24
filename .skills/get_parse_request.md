---
description: 查询解析请求状态
---
# get_parse_request

## 工具用途
查询解析请求状态

## 参数详解
- `job_id`: `string`，必填。解析请求的 jobId

## 输入 Schema
```json
{
  "type": "object",
  "properties": {
    "job_id": {
      "type": "string",
      "description": "解析请求的 jobId"
    }
  },
  "required": [
    "job_id"
  ]
}
```

## ReAct 调用示例
Action: get_parse_request
Action Input: {"job_id": ""}

## 结果说明
该工具返回 Observation 文本。回答时仅可基于返回内容得出结论；若字段缺失，必须明确说明“数据未提供/无法确认”。
