# Enterprise RAG Workbench / 企业知识库RAG工作台

面向企业文档问答与知识管理的全栈RAG系统，支持多格式文档入库、知识片段管理、混合检索、流式问答与来源溯源。

**技术栈：** Python / FastAPI / LangChain / LangGraph / PostgreSQL / Redis / Milvus / Vue3 / TypeScript / DashScope

## 功能

- **文档解析：** 支持TXT、Markdown、PDF、DOCX，结合原生解析、OCR路由和HTML表格保护。
- **知识管理：** Document → Section → Chunk → Vector处理链，支持片段观测、编辑、启停和索引维护。
- **混合检索：** Milvus向量与BM25并行召回、RRF融合；评测链路结合查询改写与重排。
- **Agent问答：** 预检索、知识库工具调用、多轮会话与带来源引用的回答。
- **流式交互：** SSE推送文本增量、执行状态、工具过程和来源，前端渲染Markdown。
- **效果评测：** 提供题集、逐题结果、评测脚本与结果复算入口。

## 评测结果

| 指标 | 优化前 | 优化后 | 对照设置 |
|---|---:|---:|---|
| Top5完整依据覆盖率 | 45.83% | 91.67% | 同24语义题，向量检索→查询改写＋混合召回＋重排 |
| 回答忠实度 | 0.351 | 0.965 | 同24语义题，自由回答→证据抽取与原文引用 |
| FAQ回答相关性 | 0.853 | 0.918 | 同24短FAQ，回答表达优化 |
| 复杂文档可用入库率 | 50% | 95% | 同20份PDF／DOCX，内容显示等价标准 |

[查看评测方法与结果文件](docs/evaluation.md)

## 目录结构

```text
backend/                     FastAPI后端、Agent与RAG链路
front/rag_front/             Vue3管理工作台
docs/                        启动说明与评测方法
evaluation/scripts/          评测、分析与复算脚本
evaluation/datasets/         题目、依据映射与版本记录
evaluation/results/          逐题和逐文件评测结果
evaluation/corpus_sources/   语料来源与固定版本
evaluation/verify_summary.py 结果复算入口
```

## 快速开始

按照[启动说明](docs/setup.md)配置基础服务、后端和前端。

复算评测结果：

```sh
python evaluation/verify_summary.py
```
