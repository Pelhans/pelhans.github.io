---
title: "Harness design for long-running application development（面向长时运行应用开发的支撑框架设计）"
date: 2026-08-11
layout: post
liquid: false
tags: [语雀, 计算机, Anthropic Engineering 翻译]
categories: [计算机]
yuque: true
---

> 原文作者：Cassidy Gray、Kyle Wood | 发布于 2026-07-29 | 原文链接：https://www.anthropic.com/engineering/harness-design-long-running-apps

# Harness design for long-running application development

This post shares details from our research into what it takes to build a harness that lets agents reliably complete long-running software projects. We cover the architecture, the key components we found made the biggest difference, and the tradeoffs involved.

本文分享我们关于"如何构建一个让智能体可靠完成长时运行软件项目的支撑框架"的研究细节。我们将介绍其架构、我们发现影响最大的关键组件，以及其中的权衡取舍。

## Building a long-running app harness

## 构建长时运行的应用支撑框架

Over the last year, AI agents have gone from unreliable demos to reliable tools that can write and edit code across an entire repository. For simpler tasks, agents work well out of the box. But when a project takes hours or days, agents tend to lose the thread and stall out. We set out to identify the core components of a harness that let agents complete long-running projects—in our case, building a full, production-quality clone of the Claude.ai chat interface.

过去一年里，AI 智能体从不可靠的演示，演进为能够跨整个代码库编写与编辑代码的可靠工具。对于较简单的任务，智能体开箱即用即可表现良好。但当项目需要数小时乃至数天时，智能体会渐渐迷失主线、陷入停滞。我们着手识别一套支撑框架的核心组件，使智能体能够完成长时运行的项目——在我们的案例中，是构建一个完整、生产质量的 Claude.ai 聊天界面克隆版。

Before getting into the details, here is a summary of what we found mattered most:

在展开细节之前，先给出我们发现最重要事项的概要：

*   **Initialization**: A dedicated first-step that sets up the environment, tools, and a feature spec. This gives the agent a clear blueprint to work from.
*   **Iteration loop**: A structured work cycle where the agent picks one feature, implements it, verifies it, and records progress.
*   **State persistence**: Structured notes and git history so each new context window can pick up where the last one left off.
*   **Verification**: Real end-to-end testing (not just unit tests) so the agent catches broken features instead of declaring false victory.

*   **初始化**：一个专门的"第一步"，负责搭建环境、工具和一份功能规格。这为智能体提供了一份清晰可循的蓝图。
*   **迭代循环**：一个结构化的工作周期，智能体在其中挑选一个功能、实现它、验证它并记录进展。
*   **状态持久化**：结构化的笔记与 git 历史，使每个新上下文窗口都能接续上一个窗口的工作。
*   **验证**：真正的端到端测试（而非仅单元测试），让智能体捕获失效功能，而非虚假地宣告胜利。

The rest of this post explains each piece and the tradeoffs we hit along the way.

本文余下部分将逐一解释每个组件，以及我们在此过程中遇到的权衡取舍。

### Architecture overview

### 架构概览

We built two processes: an **outer loop** that manages the overall project and a **coding agent** (the inner loop) that does the actual implementation work. The outer loop decides what to build next and when to stop; the coding agent focuses on one feature at a time.

我们构建了两个进程：一个负责管理整个项目的**外层循环（outer loop）**，以及一个真正执行实现工作的**编码智能体（内层循环）**。外层循环决定下一步构建什么、何时停止；编码智能体则一次只聚焦一个功能。

```
┌─────────────────────────────────────────┐
│            Outer Loop                     │
│  Manages project state, decides next     │
│  feature, checks completion              │
│                                           │
│  ┌─────────────────────────────────────┐ │
│  │         Coding Agent (inner loop)    │ │
│  │  Implements ONE feature per session  │ │
│  │  Reads progress, writes code, tests  │ │
│  └─────────────────────────────────────┘ │
│                                           │
│  Git commits + progress notes ──────────┐ │
└─────────────────────────────────────────┘ │
                                            ▼
                                   Persisted State
                                   (git + progress.md)
```

A diagram of the two-process harness. The outer loop delegates work to the coding agent; both write to a shared, persisted state.

这是双进程支撑框架的示意图。外层循环将工作委派给编码智能体；二者都写入同一份共享的、持久化的状态。

This separation matters because the outer loop can stay focused on project management while the coding agent stays focused on implementation. Each coding agent session starts fresh, reads the persisted state, does one piece of work, and writes its results back.

这种分离很重要，因为外层循环可以专注于项目管理，而编码智能体专注于实现。每一次编码智能体会话都以全新状态开始，读取持久化状态，完成一块工作，并把结果写回。

### The initialization step

### 初始化步骤

The first agent session is different from all the others. Instead of implementing a feature, it sets up the foundation everything else builds on.

第一个智能体会话与其余所有会话都不同。它不实现某个功能，而是搭建其余一切工作所依赖的基础。

We give the initializer a prompt that asks it to:

我们给初始化智能体一份提示，要求它：

*   Scaffold a minimal but working app (so there's always a runnable base).
*   Write an `init.sh` script that starts the dev server and runs a smoke test.
*   Generate a feature spec (`features.json`) breaking the request into testable features, each marked with `"passes": false`.
*   Create a `progress.md` file and make an initial git commit.

*   搭建一个最小但可运行的应用（这样始终有一个可运行的基础）。
*   编写一个 `init.sh` 脚本，用于启动开发服务器并跑一次冒烟测试。
*   生成一份功能规格（`features.json`），把请求拆解为可测试的功能，每个都标记为 `"passes": false`。
*   创建 `progress.md` 文件，并做一次初始 git 提交。

```
{
  "category": "functional",
  "description": "New chat button creates a fresh conversation",
  "steps": [
    "Navigate to main interface",
    "Click the 'New Chat' button",
    "Verify a new conversation is created"
  ],
  "passes": false
}
```

A sample feature entry from `features.json`. Each feature lists concrete steps and a pass/fail flag.

这是 `features.json` 中的一条功能示例。每个功能都列出具体步骤和一个通过/未通过标记。

The feature spec is the single most important artifact the initializer produces. It converts a vague request like "build a Claude.ai clone" into a concrete checklist the coding agent can work through one item at a time. Marking everything `false` upfront prevents the agent from declaring premature victory.

功能规格是初始化智能体产出的最重要产物。它把一个像"构建 Claude.ai 克隆版"这样模糊的请求，转化为一份具体的检查清单，编码智能体可以一次处理其中一项。一开始就将所有项标记为 `false`，能防止智能体过早宣告胜利。

### The iteration loop

### 迭代循环

Every coding agent session follows the same loop:

每个编码智能体会话都遵循同一套循环：

1.  **Orient**: Read `progress.md`, the git log, and `features.json` to understand the current state.
2.  **Select**: Pick the highest-priority `passes: false` feature.
3.  **Implement**: Write the code for that single feature.
4.  **Verify**: Start the dev server via `init.sh`, then test the feature end-to-end (we used browser automation).
5.  **Record**: Mark the feature `passes: true` in `features.json`, append a summary to `progress.md`, and commit to git.

1.  **定位（Orient）**：阅读 `progress.md`、git 日志和 `features.json`，了解当前状态。
2.  **选择（Select）**：挑选优先级最高且 `passes: false` 的功能。
3.  **实现（Implement）**：为该单一功能编写代码。
4.  **验证（Verify）**：通过 `init.sh` 启动开发服务器，然后对该功能做端到端测试（我们使用浏览器自动化）。
5.  **记录（Record）**：在 `features.json` 中将该功能标记为 `passes: true`，向 `progress.md` 追加摘要，并提交到 git。

```
[Assistant] Reading progress.md and features.json to pick the next feature...
[Tool Use] <read - progress.md>
[Tool Use] <read - features.json>
[Assistant] I'll implement "user can delete a conversation from the sidebar."
<implements feature>
[Tool Use] <bash - ./init.sh>   # starts dev server + smoke test
<tests feature end-to-end via browser>
[Assistant] Feature works. Marking passes:true and committing.
```

A compressed view of one coding agent session.

这是一个编码智能体会话的压缩视图。

Working one feature at a time sounds slow, but it was dramatically more reliable than letting the agent attempt many features at once. It kept each context window focused and left a clean, committable state behind.

一次只做一个功能听起来很慢，但相比让智能体试图一次性完成多个功能，它的可靠性要高得多。它让每个上下文窗口都保持聚焦，并留下一个干净、可提交的状态。

### Why state persistence works

### 状态持久化为何有效

The trick that makes long-running agents feasible is that *no individual session needs to be smart about the whole project*. Each session only needs to:

让长时运行智能体可行的关键诀窍在于：*没有任何单个会话需要对整个项目有全局智能*。每个会话只需：

*   Read a short summary of what happened before (`progress.md`).
*   Read a structured list of what's done and what's not (`features.json` + git log).
*   Do one small, well-defined thing.
*   Write a short summary of what it did.

*   阅读此前发生了什么的简短摘要（`progress.md`）。
*   阅读一份"已完成/未完成"的结构化清单（`features.json` + git 日志）。
*   做一件小而定义明确的事。
*   写下它所做工作的简短摘要。

Because git captures the actual code changes and `progress.md` captures the narrative, a fresh session can reconstruct enough context to continue without re-deriving everything from scratch.

由于 git 记录了实际的代码改动，而 `progress.md` 记录了来龙去脉，一个新会话便能重建足够的上下文以接续工作，无需从零重新推导一切。

### Verification is the hard part

### 验证是难点

The biggest gap between "code that looks done" and "code that actually works" is testing. We found two things:

"看起来完成"的代码与"真正可用"的代码之间最大的鸿沟就是测试。我们发现了两点：

*   **Unit tests aren't enough.** An agent can write passing unit tests while the feature is broken end-to-end. We required the agent to test like a user: open the app, click through the flow, confirm the result.
*   **Browser automation was the unlock.** Giving the agent a way to actually drive the UI (via Puppeteer MCP) turned vague "it should work" confidence into concrete pass/fail signals.

*   **单元测试不够。** 智能体可能写出通过的单元测试，而该功能在端到端层面却是坏的。我们要求智能体像用户那样测试：打开应用、点击流程、确认结果。
*   **浏览器自动化是关键突破。** 给智能体一种真正驱动 UI 的能力（通过 Puppeteer MCP），把"应该能用"的模糊信心，转化成了具体的通过/未通过信号。

There were still limits—agent vision and browser tooling can't catch every bug (e.g., native alert modals were invisible to the automation). But real end-to-end testing eliminated the most common failure: declaring victory on a broken app.

仍然存在局限——智能体的视觉和浏览器工具无法捕获每一类 bug（例如，原生 alert 弹窗对自动化工具不可见）。但真正的端到端测试消除了最常见的失败：在一个坏掉的应用上宣告胜利。

### Tradeoffs and what we'd do differently

### 权衡取舍，以及我们会做出的不同选择

A few things we learned that didn't make it into the clean architecture above:

一些我们学到的、却未纳入上述"干净架构"的经验：

*   **JSON over Markdown for the feature list.** Models are less likely to silently rewrite or corrupt a structured JSON file than freeform Markdown. Small, but it prevented a frustrating failure mode.
*   **Keep the initializer's spec broad, not deep.** The initializer should enumerate features, not implement them. Letting it write code led to a messy first commit.
*   **No premature multi-agent split.** We considered splitting into specialized agents (tester, reviewer, etc.) but found a single well-prompted coding agent was simpler and good enough for this scope. That may change for larger projects.

*   **功能列表用 JSON 而非 Markdown。** 相比自由格式的 Markdown，模型更不容易悄悄改写或损坏结构化的 JSON 文件。这是个小细节，但它避免了一种令人抓狂的失败模式。
*   **让初始化智能体的规格"广而浅"。** 初始化智能体应当罗列功能，而非实现它们。让它写代码会导致一个混乱的初始提交。
*   **不要过早拆分多智能体。** 我们曾考虑拆分为专门化智能体（测试员、审查员等），但发现一个提示得当的单一编码智能体更简单，且对这一范围而言已足够好。对于更大的项目，情况或许会不同。

## Conclusion

## 结论

Reliable long-running agents aren't about a smarter model—they're about a better harness. The combination of a dedicated initialization step, a tight iteration loop, persisted state, and real end-to-end verification let our coding agent build a production-quality app across many context windows without losing the thread.

可靠的长时运行智能体，靠的不是更聪明的模型，而是更好的支撑框架。将专门的初始化步骤、紧凑的迭代循环、持久化状态，以及真正的端到端验证结合起来，使我们的编码智能体能够在多个上下文窗口中构建一个生产质量的应用，而不迷失主线。

The patterns here generalize beyond web apps. Any long-running agentic task—research, data pipelines, financial modeling—benefits from the same backbone: a clear spec, one-thing-at-a-time execution, and evidence that the work actually succeeded.

这里的模式可推广到 Web 应用之外。任何长时运行的智能体任务——科研、数据流水线、金融建模——都能从同样的骨干中获益：清晰的规格、一次只做一件事的执行方式，以及工作确实成功的证据。
