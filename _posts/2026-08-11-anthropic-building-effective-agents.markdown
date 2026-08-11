---
title: "Building effective agents（构建高效的智能体）"
date: 2026-08-11
layout: post
liquid: false
tags: [语雀, 计算机, Anthropic Engineering 翻译]
categories: [计算机]
yuque: true
---

> 原文作者：Erik Schluntz、Barry Zhang | 发布于 2024-12-20 | 原文链接：https://www.anthropic.com/engineering/building-effective-agents

# Building effective agents

Over the past year, we've worked with dozens of teams building large language model agents across industries. Consistently, the most successful implementations use simple, composable patterns rather than complex frameworks.

过去一年里，我们与数十个跨行业的团队一起构建大语言模型智能体。一贯地，最成功的实现都使用简单、可组合的模式，而非复杂的框架。

In this post, we share what we’ve learned from building agents with our customers, and provide practical advice for developers.

在本文中，我们分享从与客户共建智能体中学到的经验，并为开发者提供实用建议。

## Agents

## 智能体

"Agent" can be defined in many ways. Some customers define agents as fully autonomous systems that operate independently for extended periods, using various tools to accomplish complex tasks. Others use the term to describe more prescriptive implementations that follow predefined workflows. At Anthropic, we categorize all of these variations as **agentic systems**, but we draw an important distinction around the architecture:

"智能体"可以有多种定义。一些客户将其定义为完全自主的系统，能在较长时间内独立运行，借助各种工具完成复杂任务。另一些则用该词指代遵循预定义工作流的、更具规定性的实现。在 Anthropic，我们把所有这些变体都归类为**智能体系统（agentic systems）**，但我们在架构上做一个重要区分：

*   **Workflows** are systems where LLMs and tools are orchestrated through predefined code paths.
*   **Agents**, on the other hand, are systems where LLMs dynamically direct their own processes and tool usage, maintaining control over how they accomplish tasks.

*   **工作流（Workflows）**是通过预定义代码路径来编排 LLM 和工具的系统。
*   **智能体（Agents）**则是 LLM 动态主导自身流程与工具使用的系统，对如何完成任务保持掌控。

## When (and when not) to use agents

## 何时（以及何时不）使用智能体

When building applications with LLMs, we recommend finding the simplest solution that can work, and only increasing complexity when needed. This might mean not building agentic systems at all. Agentic systems often trade latency and cost for better task performance, and you should consider when this tradeoff makes sense.

在用 LLM 构建应用时，我们建议先找到能奏效的最简方案，只在确有需要时再增加复杂度。这或许意味着根本不需要构建智能体系统。智能体系统常常以延迟和成本为代价换取更好的任务表现，你应当斟酌这种权衡在何时才合理。

For many applications, **enhancing single LLM calls with retrieval and in-context examples is usually enough.** This is the simplest, most reliable, and cheapest approach.

对许多应用而言，**用检索和上下文示例来增强单次 LLM 调用，通常就已经足够。** 这是最简单、最可靠也最便宜的做法。

When you do need more complexity, **workflows** offer predictability and consistency for well-defined tasks, while **agents** are the better option when you need flexibility and model-driven decision-making at scale. For tasks that can be precisely and predictably defined, workflows give us determinism and reliability; when we need scale and a more open-ended approach, agents are the way to go. Importantly, for many applications, **optimizing single LLM calls is often all that is needed**—you don't need to build an entire agentic system.

当你确实需要更大复杂度时，**工作流**为定义良好的任务提供可预测性与一致性，而**智能体**在你需要规模化下的灵活性与模型驱动决策时是更佳选择。对于能够被精确、可预测定义的任务，工作流给予我们确定性与可靠性；当我们需要规模化与更开放的处理方式时，智能体才是正道。重要的是，对许多应用而言，**优化单次 LLM 调用往往已绰绰有余**——你并不需要构建一整套智能体系统。

## When and how to build frameworks

## 何时以及如何构建框架

Many frameworks make building agentic systems easier, by providing simplified interfaces for common steps like calling LLMs, defining and parsing tools, and chaining calls. However, they often create extra layers of abstraction that can obscure the underlying prompts and responses, making them harder to debug. We recommend starting by using LLM APIs directly—most patterns can be implemented in a few lines of code.

许多框架通过为常见步骤（如调用 LLM、定义与解析工具、串联调用）提供简化的接口，让构建智能体系统变得更容易。然而，它们往往创造出额外的抽象层，掩盖了底层的提示与响应，使之更难调试。我们建议从直接使用 LLM API 起步——多数模式用几行代码就能实现。

## The building blocks, workflows, and agents

## 基本构件、工作流与智能体

In this section, we’ll explore the common patterns we’ve seen used in production. We’ll start with our basic building block—the augmented LLM—and progressively increase complexity, from simple compositional workflows to autonomous agents.

本节中，我们将探索在生产环境中常见的模式。我们从基本构件——增强型 LLM——起步，逐步提升复杂度，从简单的组合式工作流，到自主智能体。

### Building block: The augmented LLM

### 构件：增强型 LLM

The basic building block of agentic systems is an LLM enhanced with augmentations such as retrieval, tools, and memory. Our current models are capable of actively using these capabilities—generating their own search queries, selecting appropriate tools, and deciding what to retain.

智能体系统的基本构件，是一个经检索、工具、记忆等增强手段加持的 LLM。我们当前的模型有能力主动运用这些能力——自主生成搜索查询、选择合适的工具，并决定保留什么。

### Workflow: Prompt chaining

### 工作流：提示链（Prompt chaining）

Prompt chaining decomposes a task into a sequence of steps, where each LLM call processes the output of the previous one. You can add programmatic checks (see "gate" in the diagram below) on any intermediate steps to ensure that the process is still on track.

提示链把一个任务拆解为一系列步骤，其中每次 LLM 调用都处理上一次的輸出。你可以在任何中间步骤上加程序化检查（见下图中"gate"），以确保流程仍在正轨上。

### Workflow: Routing

### 工作流：路由（Routing）

Routing classifies an input and directs it to a specialized followup task. This workflow allows for separation of concerns, and building more specialized prompts.

路由对输入进行分类，并将其导向专门的后续任务。这种工作流允许关注点分离，并构建更具针对性的提示。

### Workflow: Parallelization

### 工作流：并行化（Parallelization）

LLMs can sometimes work simultaneously on a task and have their outputs aggregated programmatically. The two key variations of parallelization are:

LLM 有时可以同时对一个任务工作，并将其输出以编程方式聚合。并行化的两个关键变体是：

*   **Sectioning**: breaking a task into independent subtasks run in parallel.
*   **Voting**: running the same task multiple times to get diverse outputs.

*   **分段（Sectioning）**：把一个任务拆成独立的子任务并行运行。
*   **投票（Voting）**：多次运行同一任务以获得多样化输出。

### Workflow: Orchestrator-workers

### 工作流：编排者-工作者（Orchestrator-workers）

In the orchestrator-workers workflow, a central LLM dynamically breaks down tasks, delegates them to worker LLMs, and synthesizes their results.

在编排者-工作者工作流中，一个中心 LLM 动态拆解任务，将其委派给工作者 LLM，并综合它们的结果。

### Workflow: Evaluator-optimizer

### 工作流：评估者-优化者（Evaluator-optimizer）

In the evaluator-optimizer workflow, one LLM call generates a response while another provides evaluation and feedback in a loop.

在评估者-优化者工作流中，一个 LLM 调用生成响应，而另一个在循环中提供评估与反馈。

### Agents

### 智能体

Agents can handle sophisticated tasks autonomously, but their implementation is often straightforward: they are just LLMs that use tools based on environmental feedback in a loop. It's important to design toolsets and their documentation carefully, as agents rely on these to accomplish their tasks.

智能体能够自主处理复杂任务，但其实现往往直截了当：它们不过是那些在循环中依据环境反馈使用工具的 LLM。仔细设计工具集及其文档很重要，因为智能体依赖这些来完成任务。

## Combining and customizing these patterns

## 组合与定制这些模式

These building blocks aren't prescriptive. They are common patterns that developers can shape and combine to fit different use cases. The key to success is measuring performance and iterating on implementations, and only adding complexity when it clearly improves results.

这些构件并非硬性规定。它们是开发者可以按需塑形、组合以适配不同用例的常见模式。成功的关键在于衡量表现并对实现迭代，只在复杂度能明确改善结果时才增加它。

## Summary

## 总结

At Anthropic, we’ve seen the most success with agentic systems that use simple, composable patterns. Start with the simplest thing that works, measure, and only add complexity when it pays off. The augmented LLM remains the foundation; workflows give predictability; and agents earn their place only when flexibility and scale demand it.

在 Anthropic，我们见到最成功的智能体系统都采用简单、可组合的模式。从能奏效的最简方案起步，做度量，只在复杂度物有所值时才增加它。增强型 LLM 始终是基石；工作流带来可预测性；而智能体，只有当灵活性与规模化确有需求时，才配得上它的位置。
