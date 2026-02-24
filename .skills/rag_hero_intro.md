---
description: 从本地 heroes_txt 知识库检索英雄介绍（RAG）
---
# rag_hero_intro

## 工具用途
从本地 heroes_txt 知识库检索英雄介绍（RAG）

## 参数详解
- `query`: `string`，必填。英雄名或包含英雄名的提问
- `top_k`: `integer`，可选。返回候选数量，默认1
- `max_chars`: `integer`，可选。每条返回内容最大字符数，默认0（不截断）

## 输入 Schema
```json
{
  "type": "object",
  "properties": {
    "query": {
      "type": "string",
      "description": "英雄名或包含英雄名的提问"
    },
    "top_k": {
      "type": "integer",
      "description": "返回候选数量，默认1"
    },
    "max_chars": {
      "type": "integer",
      "description": "每条返回内容最大字符数，默认0（不截断）"
    }
  },
  "required": [
    "query"
  ]
}
```

## ReAct 调用示例
Action: rag_hero_intro
Action Input: {"query": ""}

## 结果说明
该工具返回 Observation 文本。回答时仅可基于返回内容得出结论；若字段缺失，必须明确说明“数据未提供/无法确认”。
