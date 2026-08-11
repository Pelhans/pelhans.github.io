---
title: "How we built our multi-agent research system（我们如何构建多智能体研究系统）"
date: 2026-08-11
layout: post
liquid: false
tags: [语雀, 计算机, Anthropic Engineering 翻译]
categories: [计算机]
yuque: true
---

> 原文作者：Jeremy Hadfield | 发布于 2025-04-23 | 原文链接：https://www.anthropic.com/engineering/multi-agent-research-system

# How we built our multi-agent research system

An inside look at the architecture behind our multi-agent research system, which coordinates specialized agents to produce thorough, well-sourced research reports.

本文深入剖析我们多智能体研究系统背后的架构——它协调多个专门化智能体，产出详尽、有据可查的研究报告。

## Why multiple agents?

## 为何用多个智能体？

Research tasks are broad and open-ended. A single agent trying to cover every angle tends to go shallow or lose focus. Splitting the work across specialized agents—each with a clear sub-question—lets the system go deeper and stay organized.

研究任务面广且开放。单个智能体试图覆盖每个角度，往往会流于表面或失去焦点。把工作拆分给专门化智能体——每个负责一个清晰的子问题——让系统能挖得更深且保持条理。

## Architecture

## 架构

The system has three tiers:

系统分为三层：

1.  **Lead agent (orchestrator).** Receives the user's query, breaks it into sub-questions, and assigns each to a research agent.
2.  **Research agents.** Each independently searches, reads, and synthesizes a single sub-question, writing its findings to a shared workspace.
3.  **Synthesis agent.** Reads all findings and writes the final, cited report.

1.  **主导智能体（编排者）。** 接收用户查询，将其拆为子问题，分派给各研究智能体。
2.  **研究智能体。** 各自独立地搜索、阅读并综合单个子问题，把发现写入共享工作区。
3.  **综合智能体。** 读取所有发现，写出最终带引用的报告。

```
        User query
             │
             ▼
      ┌─────────────┐
      │  Lead agent │  decomposes query
      └──────┬──────┘
             │  spawns N research agents
     ┌───────┼───────────┐
     ▼       ▼           ▼
  [R1]     [R2]  ...   [Rn]   each: search + read + write
     └───────┼───────────┘
             ▼
      ┌─────────────┐
      │  Synthesis  │  reads all findings → cited report
      └─────────────┘
```

## Key design decisions

## 关键设计决策

### Shared, isolated context per agent

### 每个智能体共享且隔离的上下文

Each research agent runs in its own context, so a rabbit hole one agent goes down doesn't pollute the others. The only shared state is the workspace where findings are written.

每个研究智能体运行在自己的上下文里，因此一个智能体钻的牛角尖不会污染其他智能体。唯一的共享状态是写发现的工作区。

### Tool access per role

### 按角色分配工具访问权

Research agents get search and read tools; the synthesis agent gets only read access to the workspace. Limiting tools reduces mistakes and keeps each agent on-task.

研究智能体拥有搜索与阅读工具；综合智能体只对工作区有读权限。限制工具能减少错误，让每个智能体各司其职。

### Citation enforcement

### 强制引用

Every claim in the final report must trace back to a source the research agents collected. The synthesis agent is prompted to attach a citation to each factual sentence.

最终报告里的每一条论断都必须可追溯到研究智能体收集的来源。综合智能体被提示为每句事实性陈述附上引用。

## Lessons learned

## 经验教训

*   Orchestration overhead is real: spawning many agents costs latency and tokens, so the lead agent should only split when the query genuinely benefits.
*   Quality of the final report depends heavily on the synthesis agent's ability to reconcile conflicting findings.
*   Giving research agents a clear, bounded sub-question is the single biggest lever on output quality.

*   编排开销是真实的：派发许多智能体要花延迟与 token，所以主导智能体应只在查询确实受益时才拆分。
*   最终报告的质量，很大程度上取决于综合智能体调和相互冲突发现的能力。
*   给研究智能体一个清晰、有边界的子问题，是提升输出质量最大的杠杆。

## Summary

## 总结

A multi-agent research system trades extra compute for depth and organization. The orchestrator's job is to decompose well; the research agents' job is to dig; the synthesis agent's job is to unify—with citations.

多智能体研究系统用额外的算力换取深度与条理性。编排者的职责是拆得好；研究智能体的职责是挖得深；综合智能体的职责是带引用地统一起来。
