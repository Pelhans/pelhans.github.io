---
title: "Effective context engineering for AI agents（面向 AI 智能体的有效上下文工程）"
date: 2026-08-11
layout: post
liquid: false
tags: [语雀, 计算机, Anthropic Engineering 翻译]
categories: [计算机]
yuque: true
---

> 原文作者：Barry Zhang、Sholto Douglas | 发布于 2025-05-21 | 原文链接：https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents

# Effective context engineering for AI agents

When building agents, context engineering—the process of filling the model's context window with just the right information for the task at hand—is critical for performant, efficient, and reliable agents. Here we outline three patterns for effective context engineering.

在构建智能体时，上下文工程——即往模型的上下文窗口中填入恰好适合当前任务的信息这一过程——对于打造高性能、高效率且可靠的智能体至关重要。在此，我们勾勒出有效上下文工程的三种模式。

Until recently, LLM applications have mostly been limited to a handful of relatively simple, single-turn tasks. In these cases, traditional prompt engineering—the art of writing the perfect static prompt—suffices. However, as agentic applications have grown more capable and are increasingly executing multi-step tasks, we’ve had to adapt our approach.

直到最近，LLM 应用大多还局限于少数相对简单、单轮的任务。在这些情况下，传统的提示工程——即写出完美静态提示的艺术——已经足够。然而，随着智能体应用能力增强、越来越多地执行多步任务，我们不得不调整方法。

Even a surprisingly simple agent, such as [Claude](https://claude.com/) playing [Pokemon](https://www.youtube.com/watch?v=ZyCg1MzeD9U), can require a context window of over 1 million tokens. For these longer-horizon tasks, the agent’s context is no longer a single static prompt, but a dynamically constructed and updated system that we call *context engineering*. Done well, context engineering is the difference between agents that spin in loops from confusion, and agents that accomplish tasks reliably and efficiently.

即便是一个相当简单的智能体，例如玩 [Pokemon](https://www.youtube.com/watch?v=ZyCg1MzeD9U) 的 [Claude](https://claude.com/)，也可能需要超过 100 万 token 的上下文窗口。对于这些更长程的任务，智能体的上下文不再是一个静态提示，而是一个被动态构建与更新的系统，我们称之为*上下文工程*。做得好，上下文工程就是"在困惑中原地打转的智能体"与"可靠高效完成任务的智能体"之间的分水岭。

## The "what" of context engineering

## 上下文工程之"是什么"

Context engineering is the process of filling the *context window* with the *right* information and *right* tools, at the *right* time, for the task at hand. It is about making sure that the model is not missing context (which leads to the model guessing, or asking the user for information they’ve already provided), and is not overloaded with information (which drives up cost and latency, and distracts the model).

上下文工程，就是为当前任务，在*恰当*的时间，往*上下文窗口*中填入*恰当*的信息与*恰当*的工具的过程。它关乎确保模型不缺失上下文（缺失会导致模型猜测，或向用户索要其已经提供过的信息），也不被信息过载（过载会推高成本与延迟，并分散模型注意力）。

## The "why" of context engineering

## 上下文工程之"为什么"

The key observation is that the generative model itself is usually not the bottleneck. Most issues arise not because the model is too small or not trained well enough, but because the context isn’t set up correctly. We list some common failure modes:

关键观察在于：生成式模型本身通常并不是瓶颈。多数问题并非源于模型太小或训练不足，而是上下文没有正确搭建。我们列举一些常见失败模式：

*   **Missing context.** The model is missing information it needs to do the task well, so it guesses.
*   **Overloaded context.** The model has too much information and gets distracted, or the important bits get lost in the noise.
*   **Wrong format.** The information is present but in a format the model can’t easily use.
*   **Wrong structure.** The information is present but structured in a way that makes it hard for the model to act on.

*   **上下文缺失。** 模型缺失了做好任务所需的信息，于是开始猜测。
*   **上下文过载。** 模型信息过多而分心，或重要内容淹没在噪声中。
*   **格式错误。** 信息存在，但格式模型难以利用。
*   **结构错误。** 信息存在，但结构方式让模型难以据此行动。

## Three patterns for effective context engineering

## 有效上下文工程的三种模式

In this post, we describe three patterns that we have found to be particularly effective for context engineering:

在本文中，我们描述三种我们发现对上下文工程特别有效的模式：

1.  **Write workflow state to externalized memory** so that the model can read it at the right time, instead of having to carry it in the context window.
2.  **Select only the most relevant tools** so that the model is not overwhelmed by choices.
3.  **Pre-fill the agent’s context with a "landmark"** to help the model stay oriented across long, multi-step tasks.

1.  **把工作流状态写入外部化记忆**，使模型能在恰当的时间读取它，而不必一直把它扛在上下文窗口里。
2.  **只选择最相关的工具**，让模型不被过多选择淹没。
3.  **用"路标（landmark）"预填智能体上下文**，帮助模型在漫长多步任务中保持方位感。

### Pattern 1: Write workflow state to externalized memory

### 模式 1：把工作流状态写入外部化记忆

The first pattern is about moving information out of the context window and into a persistent external store that the model can read and write to. This keeps the context window small and focused, while still letting the agent accumulate state over a long task.

第一种模式是：把信息从上下文窗口移出，存入一个持久化的外部存储，供模型读写。这让上下文窗口保持小巧而聚焦，同时仍让智能体能在长任务中积累状态。

When an agent does a task that spans many steps, it naturally builds up state: what it has tried, what worked, what it learned. If you try to keep all of that in the context window, the window fills up and the model gets distracted. Instead, write it to a file or database the agent can read back when needed.

当智能体执行跨越许多步骤的任务时，会自然积累起状态：它试过什么、什么奏效、学到了什么。如果你试图把这些都留在上下文窗口里，窗口会被填满，模型也会分心。相反，把它写入一个文件或数据库，让智能体在需要时读回即可。

This is exactly the `progress.md` + git history pattern we used in the long-running app harness: the agent writes a short summary of what it did, and the next session reads it back instead of re-deriving everything.

这正是我们在长时运行应用支撑框架中使用的 `progress.md` + git 历史模式：智能体写下它所做工作的简短摘要，下一会话读回它，而非重新推导一切。

### Pattern 2: Select only the most relevant tools

### 模式 2：只选择最相关的工具

The second pattern is about tool selection. Agents that are given too many tools at once tend to perform worse, because the model has to reason about which tool to use among many, and can get confused.

第二种模式关乎工具选择。一下子被赋予过多工具的智能体，表现往往更差，因为模型不得不在众多工具中斟酌该用哪个，从而陷入困惑。

The fix is to give the model only the tools relevant to the current step. You can do this with a few approaches:

解决之道是只给模型与当前步骤相关的工具。你可以采用几种做法：

*   **Static narrowing**: pick a fixed subset of tools per agent type (e.g., a "researcher" agent only gets read/search tools).
*   **Dynamic selection**: use a separate model call to choose the top-k tools for the current task, and only expose those to the main agent.
*   **Hierarchical tools**: expose high-level "router" tools that, when called, reveal a more specific set of sub-tools.

*   **静态收窄**：为每种智能体类型选定固定的工具子集（例如，"研究员"智能体只获得读/搜索类工具）。
*   **动态选择**：用一次独立的模型调用来为当前任务挑选 top-k 工具，只把这些暴露给主智能体。
*   **层次化工具**：暴露高层的"路由"工具，被调用时再展开更具体的一组子工具。

### Pattern 3: Pre-fill the agent's context with a "landmark"

### 模式 3：用"路标"预填智能体上下文

The third pattern helps with long, multi-step tasks where the model can lose track of where it is. A "landmark" is a short, stable piece of text—like a running to-do list or a one-line statement of the current goal—that you pre-fill into the context at the start of each session.

第三种模式有助于解决漫长多步任务中模型"迷失方位"的问题。"路标"是一段简短、稳定的文本——好比一份持续推进的待办清单，或一句话的当前目标陈述——你在每个会话开始时把它预填进上下文。

In our long-running harness, the `features.json` file served as the landmark: each session opened with a clear view of what was done and what was next, so the model never had to reconstruct the project from scratch.

在我们的长时运行支撑框架中，`features.json` 文件就扮演了路标的角色：每个会话都以"什么已完成、下一步是什么"的清晰视图开场，模型因此永远不必从零重建项目全貌。

## Conclusion

## 结论

Context engineering is the discipline of putting the right information, in the right format, at the right time, in front of the model. The three patterns—externalized memory, relevant-tool selection, and landmarks—are simple but powerful levers for building agents that stay oriented, efficient, and reliable across long, complex tasks.

上下文工程，是一门"在恰当的时间、以恰当的格式、把恰当的信息摆在模型面前"的学问。这三种模式——外部化记忆、相关工具选择、路标——简单，却是强大的杠杆，能构建出在漫长复杂任务中保持方位感、高效且可靠的智能体。
