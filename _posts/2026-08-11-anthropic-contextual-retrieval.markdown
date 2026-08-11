---
title: "Introducing contextual retrieval（发布上下文检索）"
date: 2026-08-11
layout: post
liquid: false
tags: [语雀, 计算机, Anthropic Engineering 翻译]
categories: [计算机]
yuque: true
---

> 原文作者：Anton Troynikov、Barry Zhang | 发布于 2024-09-19 | 原文链接：https://www.anthropic.com/engineering/contextual-retrieval

# Introducing contextual retrieval

A method that dramatically reduces retrieval errors in RAG systems by adding context to each chunk before embedding and indexing.

一种方法：在嵌入和索引之前为每个文本块补充上下文，从而大幅降低 RAG 系统中的检索错误。

## The problem with naive RAG

## 朴素 RAG 的问题

In retrieval-augmented generation, documents are split into chunks, embedded, and retrieved by similarity. But chunks often lose the context they were cut from—a chunk that says "the company's revenue grew 3%" is meaningless without knowing *which* company and *which* year.

在检索增强生成中，文档被切分成块、做嵌入，并按相似度检索。但文本块常常丢失被切处的上下文——一句"公司营收增长了 3%"，若不知道是*哪家*公司、*哪*一年，就毫无意义。

## Contextual retrieval

## 上下文检索

The fix: before embedding a chunk, prepend a short, generated context that explains where the chunk came from.

解法：在嵌入一个文本块之前，先在前面拼上一段简短的、生成的上下文，说明这个块来自哪里。

```
Original chunk:
  "Its profit margin improved to 14% in Q2."

Contextualized chunk:
  "This chunk is from the 2023 annual report of Acme Corp, in the section
   on financial performance. Its profit margin improved to 14% in Q2."
```

A lightweight model call generates the context from the full document, then the contextualized chunk is embedded and indexed. At query time, retrieval returns chunks that already carry the context needed to interpret them.

用一个轻量模型调用，基于完整文档生成这段上下文，然后对这个带上下文的块做嵌入和索引。在查询时，检索返回的文本块已经携带了解读所需的上下文。

## Results

## 结果

Across evaluations, contextual retrieval reduced retrieval failure rates substantially compared to naive chunking, especially on documents with many similar short passages.

在各项评估中，相比朴素切块，上下文检索大幅降低了检索失败率，尤其在包含许多相似短段落的文档上。

## Pairing with reranking

## 与重排序搭配

Contextual retrieval works even better combined with [reranking](https://www.anthropic.com/news/3-5-models-and-claude): after the initial retrieval, a reranker reorders candidates by relevance, pushing the best chunk to the top.

上下文检索与[重排序](https://www.anthropic.com/news/3-5-models-and-claude)搭配效果更好：在初步检索之后，一个重排序器按相关性对候选重排，把最佳文本块顶到最前。

## Summary

## 总结

Most retrieval errors come from lost context at chunk boundaries. Adding a sentence of generated context per chunk is a cheap, high-leverage fix that compounds when paired with reranking.
