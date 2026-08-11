---
title: "Reliably building with managed agents（用受管智能体可靠地构建）"
date: 2026-08-11
layout: post
liquid: false
tags: [语雀, 计算机, Anthropic Engineering 翻译]
categories: [计算机]
yuque: true
---

> 原文作者：Justin Young、Jake Eaton | 发布于 2026-02-13 | 原文链接：https://www.anthropic.com/engineering/managed-agents

# Reliably building with managed agents

A practical guide to the Claude Agent SDK's managed agents—subagents that run in their own context with their own tools, and how to use them to build more reliable systems.

一份关于 Claude Agent SDK 受管智能体（在其自身上下文、用自身工具运行的子智能体）的实用指南，以及如何使用它们构建更可靠的系统。

## What are managed agents?

## 什么是受管智能体？

Managed agents are subagents spawned by a parent agent. Each runs in its own isolated context with its own set of tools, and reports back a final result to the parent. The parent stays focused; the subagent does the digging.

受管智能体是由父智能体派生的子智能体。每个都运行在自己隔离的上下文中，拥有自己的工具集，并向父智能体回报最终结果。父智能体保持聚焦；子智能体负责深挖。

## Why use them?

## 为何使用它们？

### Context isolation

### 上下文隔离

Long, messy sub-tasks can pollute the main context. Offloading them to a subagent keeps the parent clean and on-task.

冗长、杂乱的子任务会污染主上下文。把它们卸载给子智能体，能让父智能体保持干净、专注。

### Parallelism

### 并行

Independent sub-tasks can run as separate managed agents, either truly in parallel or sequenced, without tangling their contexts.

相互独立的子任务可以作为独立的受管智能体运行——无论是真正并行还是串行——而不会缠结彼此的上下文。

### Tool scoping

### 工具限定

Give a research subagent read-only tools and a coding subagent write tools. Limiting tools per role reduces mistakes.

给研究子智能体只读工具、给编码子智能体写工具。按角色限定工具能减少错误。

## A worked example

## 一个实例

Suppose you want to answer: "Which of our open GitHub issues are caused by the recent auth refactor?"

假设你想回答："我们开放的 GitHub issue 中，哪些是由近期 auth 重构引起的？"

```
Lead agent
  └─ spawns managed agent "researcher"
       tools: github read, search
       task: find issues mentioning auth failures since the refactor
       → returns: list of 7 candidate issues

Lead agent
  └─ spawns managed agent "analyzer"
       tools: read repo, read issues
       task: for each candidate, determine if root cause is the refactor
       → returns: 3 confirmed, with reasoning
```

The lead agent never sees the messy investigation; it only gets the structured result.

父智能体从不看到那团乱麻般的调查过程；它只拿到结构化的结果。

## Best practices

## 最佳实践

*   **Give subagents a single, bounded task.** Vague missions lead to vague results.
*   **Specify the output format.** Tell the subagent exactly what to return so the parent can use it.
*   **Limit tools to what's needed.** Less surface area, fewer mistakes.
*   **Keep the parent as orchestrator, not doer.** The parent decomposes; subagents execute.

*   **给子智能体单一、有边界的任务。** 模糊的使命导致模糊的结果。
*   **指定输出格式。** 明确告诉子智能体返回什么，父智能体才能用它。
*   **把工具限定在所需范围。** 表面积越小，错误越少。
*   **让父智能体当编排者而非执行者。** 父智能体拆解；子智能体执行。

## Summary

## 总结

Managed agents turn a single overloaded context into a clean divide-and-conquer structure. Use them to isolate context, scope tools, and parallelize independent work.
