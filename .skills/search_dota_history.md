---
description: 使用 SerpApi 进行 Dota 历史信息检索，并可选抓取网页全文摘要
---
# search_dota_history

## 工具用途
使用 SerpApi 进行 Dota 历史信息检索，并可选抓取网页全文摘要

## 参数详解
- `query`: `string`，必填。搜索关键词
- `num_results`: `integer`，可选。返回结果数量，1-10，默认 5
- `include_liquipedia`: `boolean`，可选。是否强制加入 Liquipedia 站点过滤，默认 true
- `sites`: `array[string]`，可选。自定义站点过滤列表，如 ["liquipedia.net/dota2", "gosugamers.net"]
- `fetch_fulltext`: `boolean`，可选。是否抓取结果页正文内容，默认 true
- `fulltext_max_chars`: `integer`，可选。正文最大字符数，默认 8000

## 输入 Schema
```json
{
  "type": "object",
  "properties": {
    "query": {
      "type": "string",
      "description": "搜索关键词"
    },
    "num_results": {
      "type": "integer",
      "description": "返回结果数量，1-10，默认 5"
    },
    "include_liquipedia": {
      "type": "boolean",
      "description": "是否强制加入 Liquipedia 站点过滤，默认 true"
    },
    "sites": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "自定义站点过滤列表，如 [\"liquipedia.net/dota2\", \"gosugamers.net\"]"
    },
    "fetch_fulltext": {
      "type": "boolean",
      "description": "是否抓取结果页正文内容，默认 true"
    },
    "fulltext_max_chars": {
      "type": "integer",
      "description": "正文最大字符数，默认 8000"
    }
  },
  "required": [
    "query"
  ]
}
```

## ReAct 调用示例
Action: search_dota_history
Action Input: {"query": ""}

## 结果说明
该工具返回 Observation 文本。回答时仅可基于返回内容得出结论；若字段缺失，必须明确说明“数据未提供/无法确认”。
