# Improvements

记录针对 RAG 效果提升所做的改动。

## 1. 知识库数据重建（2026-08，commit `8c5a087`）

从司法部官网原文重新解析生成 `data/data_minfa.json`（1260 条，逐条校验编号连续）：

- 新增层级字段：每条法条带 编/分编/章/节（`book` / `sub_book` / `chapter` / `section`）

**效果**：检索内容更完整、准确；层级信息为章节定位和后续元数据过滤打基础。

## 2. 元数据进入 embedding / LLM 上下文（2026-08，commit `e3ce52a`）

- 向量化文本从「纯正文」升级为「标题 + 层级 + 正文」（`full_title` + 非空 `book` / `sub_book` / `chapter` / `section`）
- 噪音字段（`source_file`、`content_type`）及冗余字段（`law_name`、`article`，已含在 `full_title` 中）通过 `excluded_embed_metadata_keys` / `excluded_llm_metadata_keys` 排除在向量与 prompt 之外，但仍保留在 `node.metadata` 供 UI 展示
- UI「支持依据」新增「章节定位」行

**效果**：「物权登记」「哪一编/哪一章」类问题的召回更好；LLM 能看到条款标题与章节定位，回答可引用具体条文。
