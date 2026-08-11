---
title: "Equipping agents for the real world with agent skills（用智能体技能让智能体胜任真实世界）"
date: 2026-08-11
layout: post
liquid: false
tags: [语雀, 计算机, Anthropic Engineering 翻译]
categories: [计算机]
yuque: true
---

> 原文作者：Devin Péray、Carl Vondrick | 发布于 2025-08-14 | 原文链接：https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills

# Equipping agents for the real world with agent skills

Agent skills let you package domain knowledge, workflows, and tools into reusable capabilities that agents can discover and use when relevant.

智能体技能让你把领域知识、工作流和工具打包成可复用的能力，智能体能在相关时自行发现并使用它们。

For the last few years, LLM applications have improved by scaling context windows, better retrieval, and more capable models. But there's a limit to how much you can stuff into a prompt. As agents take on more complex, multi-step real-world tasks, the context window becomes a bottleneck: domain knowledge gets stale, workflows get buried, and the model spends tokens re-deriving things it should already know.

过去几年，LLM 应用通过扩大上下文窗口、更好的检索和更强模型得到改进。但你能塞进提示的内容有其上限。随着智能体承担更复杂、多步的真实世界任务，上下文窗口成了瓶颈：领域知识变得陈旧、工作流被埋没，模型把 token 花在重新推导本应早已知道的东西上。

Agent skills are our answer to this. A skill is a self-contained, reusable package—typically a folder with a `SKILL.md` file plus supporting scripts, references, and assets—that gives a model the domain knowledge and workflows it needs to accomplish a specialized task. When an agent encounters a relevant situation, it can discover and load the skill on demand, rather than carrying every capability in its context all the time.

智能体技能就是我们的答案。技能是一个自包含、可复用的包——通常是一个带有 `SKILL.md` 文件加上支撑脚本、参考资料与资源的文件夹——它给模型提供完成专门任务所需的领域知识与工作流。当智能体遇到相关情境时，可以按需发现并加载该技能，而非始终把所有能力都扛在上下文里。

## What is a skill?

## 什么是技能？

A skill is a folder containing:

技能是一个包含以下内容的文件夹：

*   A `SKILL.md` file with the skill's name, description, and instructions. The description is what the model uses to decide whether to use the skill; the body is the actual guidance.
*   Supporting files: scripts, reference docs, templates, datasets, or anything else the skill needs.

*   一个 `SKILL.md` 文件，含技能的名称、描述和指令。描述用于模型判断要不要使用该技能；正文才是真正的指引。
*   支撑文件：脚本、参考文档、模板、数据集，或技能所需的任何东西。

```
my-skill/
├── SKILL.md          # name, description, instructions
├── scripts/          # executable helpers
├── references/       # detailed docs loaded on demand
└── assets/           # templates, images, data
```

A skill's `SKILL.md` is deliberately lightweight—just enough for the model to know *when* the skill applies and *how* to start using it. Heavy reference material lives in the `references/` folder and is loaded only when needed. This keeps the always-on context small while making deep capability available on demand.

技能的 `SKILL.md` 刻意保持轻量——仅够让模型知道技能*何时*适用、以及如何*开始*使用它。厚重的参考资料放在 `references/` 文件夹里，只在需要时才加载。这让"常驻上下文"保持小巧，同时让深层能力可按需获取。

## Why skills?

## 为什么用技能？

Skills solve three problems that prompt-stuffing can't:

技能解决了"硬塞提示"无法解决的三大问题：

1.  **Stale knowledge.** Domain facts change. With a skill, you update a file; with a prompt, you re-train or re-paste.
2.  **Bursty context.** Most tasks only need specialized knowledge occasionally. Loading it always wastes tokens; loading it on demand saves them.
3.  **Composable capability.** Skills are modular. You can mix and match them per project without bloating any single prompt.

1.  **知识陈旧。** 领域事实会变。用技能，你只需更新一个文件；用提示，你得重新训练或重新粘贴。
2.  **突发式上下文。** 多数任务只是偶尔需要专门知识。一直加载浪费 token；按需加载才省。
3.  **可组合能力。** 技能是模块化的。你可以按项目混搭，而不必撑大任何单个提示。

## Designing effective skills

## 设计有效的技能

A good skill is discoverable, focused, and self-contained.

一个好的技能应当可被发现、聚焦且自包含。

### Write a description the model can match on

### 写一句模型能据此匹配的 description

The description is the single most important line in a skill. It should describe *when* to use the skill in the user's own words—the situations, not the implementation.

描述是技能里最重要的一行。它应当用*用户自己的话*描述技能*何时*使用——是情境，而非实现方式。

```
name: pdf-extraction
description: Use when extracting text, tables, or figures from PDF files, especially
  scanned or image-heavy PDFs where standard text extraction fails.
```

### Keep the body tight and procedural

### 正文紧凑、步骤化

The body should read like a recipe: what to do, in what order, with what checks. Include scripts as files, not pasted inline, so they can be executed rather than re-typed.

正文应像配方：做什么、按什么顺序、做什么检查。把脚本作为文件包含，而非粘贴在正文里，这样它们能被执行而非重新键入。

### Prefer references over long bodies

### 用 references 而非长正文

If a skill needs detailed reference material, put it in `references/` and point to it. The model loads it only when the task calls for it.

如果技能需要详尽的参考资料，把它放进 `references/` 并指向它。模型只在任务需要时加载。

## Skills in practice

## 技能的实践

We've found skills especially useful for: domain-specific workflows (e.g., a "release-management" skill that knows your tagging conventions), tool wrappers (e.g., a skill that knows the quirks of your internal API), and repetitive multi-step tasks (e.g., "triage-and-file-bug").

我们发现技能在以下场景尤其有用：领域特定的工作流（例如一个了解你 tagging 约定的 "release-management" 技能）、工具封装（例如了解你内部 API 怪癖的技能）、以及重复的多步任务（例如 "triage-and-file-bug"）。

Because skills are just files, they're version-controllable, reviewable, and shareable across a team—the same way you treat code.

因为技能只是文件，它们可版本控制、可审查、可在团队间共享——就像你对待代码一样。

## Summary

## 总结

Agent skills are a shift from cramming everything into the prompt to giving agents a filesystem of capabilities they can reach into when needed. They keep the always-on context small, make domain knowledge maintainable, and turn specialized expertise into reusable, discoverable packages.

智能体技能是一次范式转变：从把所有东西硬塞进提示，转为给智能体一份"能力文件系统"，在需要时伸手取用。它们让常驻上下文保持小巧、让领域知识可维护，并把专门知识变成可复用、可发现的包。
