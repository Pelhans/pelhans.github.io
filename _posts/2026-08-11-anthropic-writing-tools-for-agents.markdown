---
title: "Writing tools for agents（为智能体编写工具）"
date: 2026-08-11
layout: post
liquid: false
tags: [语雀, 计算机, Anthropic Engineering 翻译]
categories: [计算机]
yuque: true
---

> 原文作者：Jessica Wang | 发布于 2025-06-26 | 原文链接：https://www.anthropic.com/engineering/writing-tools-for-agents

# Writing tools for agents

A practical guide to designing tools that agents can use reliably and effectively.

一份关于设计智能体能够可靠、高效使用的工具的实用指南。

## Why tool design matters

## 工具设计为何重要

Agents accomplish tasks by calling tools. The quality of those tools—how clearly they're specified, how well they handle errors, how predictable their outputs are—directly determines how capable the agent is. A poorly-designed tool can confuse even a strong model; a well-designed one amplifies it.

智能体通过调用工具来完成任务。这些工具的质量——规范是否清晰、错误处理是否到位、输出是否可预测——直接决定了智能体的能力上限。一个糟糕的工具即便面对强模型也会让它困惑；一个好工具则能放大模型的能力。

## Principles for tool design

## 工具设计的原则

### Make tools narrow and well-named

### 让工具聚焦且命名良好

Each tool should do one thing, with a name that describes the action. Avoid Swiss-army-knife tools that accept a `mode` parameter to do many unrelated things—they're harder for the model to reason about.

每个工具应当只做一件事，名字要描述这个动作。避免那种用 `mode` 参数做许多不相关事情的"瑞士军刀"式工具——模型更难对它们做推理。

### Write descriptions like a human would search

### 像人类搜索那样写描述

The model uses tool descriptions to decide what to call. Write them the way a teammate would describe the tool in conversation: what it does, when to use it, and any gotchas.

模型借助工具描述来决定调用什么。像队友在对话里介绍工具那样写：它做什么、何时用、以及任何坑。

```
name: search_invoices
description: Find invoices by customer name, date range, or amount. Use this when
  a user asks about billing history or wants to locate a specific invoice.
```

### Return structured, machine-readable output

### 返回结构化、机器可读的输出

Agents work best when tool outputs are parseable. Prefer JSON over prose. Include both the data and a short human-readable summary so the model can reason and report.

工具输出可被解析时，智能体表现最佳。优先用 JSON 而非自然语言。同时包含数据和一段简短的人类可读摘要，让模型既能推理也能汇报。

```
{
  "invoices": [ { "id": "INV-001", "amount": 120.00, "status": "paid" } ],
  "summary": "Found 1 invoice for Acme Corp, $120.00, paid."
}
```

### Fail loudly and helpfully

### 失败要响亮且有帮助

When a tool fails, return a clear error message the model can act on—not a generic exception. Include what went wrong and what to try next.

工具失败时，返回一个模型能据此行动的清晰错误信息——而非泛化的异常。说明出了什么问题、下一步该试什么。

```
{
  "error": "customer_not_found",
  "message": "No customer named 'Acme Corp' exists. Did you mean 'Acme Corporation'?",
  "suggestion": "Try search_customers with a partial name."
}
```

### Avoid tools that require the model to guess

### 避免需要模型去猜的工具

If a tool needs an ID the model can't know, provide a lookup tool first. Don't make the model guess primary keys.

如果工具需要一个模型无法知道的 ID，先提供一个查找工具。不要让模型去猜主键。

## Testing tools

## 测试工具

Treat tools like production code: unit-test their edge cases, integration-test them with the agent, and monitor which tools the agent calls most and where it gets stuck.

把工具当作生产代码对待：单元测试其边界情形、与智能体做集成测试、并监控智能体最常调用哪些工具、卡在何处。

## Summary

## 总结

Good tools are narrow, clearly described, structured in output, and honest about failure. Invest in tool design and your agent will do more with less prompting.

好的工具应当聚焦、描述清晰、输出结构化，并对失败坦诚。在工具设计上投资，你的智能体就能用更少的提示做更多的事。
