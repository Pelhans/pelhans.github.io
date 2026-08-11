---
title: "Auto mode in Claude Code（Claude Code 的自动模式）"
date: 2026-08-11
layout: post
liquid: false
tags: [语雀, 计算机, Anthropic Engineering 翻译]
categories: [计算机]
yuque: true
---

> 原文作者：Jake Eaton | 发布于 2026-01-23 | 原文链接：https://www.anthropic.com/engineering/claude-code-auto-mode

# Auto mode in Claude Code

Auto mode lets Claude Code work autonomously on well-scoped tasks—running, editing, and testing without stopping for permission at each step. Here's how it works and when to use it.

自动模式让 Claude Code 能在边界清晰的任务上自主工作——运行、编辑、测试，而无需在每步停下来请求许可。本文介绍它的原理与适用场景。

## What is auto mode?

## 什么是自动模式？

Auto mode is a permission mode where Claude proceeds through a task without pausing for approval on each action. It combines two ideas we've discussed before: a tight iteration loop and a sandboxed environment.

自动模式是一种权限模式：Claude 推进任务时，不在每个动作上停下来等批准。它结合了我们先前讨论过的两个理念：紧凑的迭代循环与沙箱化的环境。

## How it differs from default mode

## 与默认模式的区别

| | Default mode | Auto mode |
|---|---|---|
| Permission prompts | Per action | Minimal / none |
| Best for | Exploratory, risky edits | Scoped, repetitive work |
| Safety reliance | Human approval | Sandbox + tool scoping |

| | 默认模式 | 自动模式 |
|---|---|---|
| 权限确认 | 每个动作 | 极少 / 无 |
| 最适合 | 探索性、高风险编辑 | 边界清晰、重复性工作 |
| 安全依赖 | 人工批准 | 沙箱 + 工具限定 |

## When to use it

## 何时使用

*   **Scoped refactors** across many files where the change is mechanical.
*   **Test-driven loops** where Claude writes a test, implements, and iterates until green.
*   **Batch migrations** with a clear, repeatable pattern.

*   **跨多文件的边界清晰重构**，改动是机械性的。
*   **测试驱动循环**，Claude 写测试、实现、迭代直到通过。
*   **批量迁移**，模式清晰且可重复。

Avoid auto mode for ambiguous tasks or when the blast radius is large and unsupervised.

对模糊任务，或破坏范围大且无人监督时，避免使用自动模式。

## Safety

## 安全性

Auto mode is safe *because* it runs inside a sandbox: filesystem and network boundaries contain any mistake. Combined with tool scoping—Claude can edit the workspace but not touch credentials—the risk is bounded even without per-step approval.

自动模式之所以安全，*是因为*它运行在沙箱之内：文件系统和网络边界兜住了任何错误。再结合工具限定——Claude 能改工作区但碰不到凭据——即便没有逐步确认，风险也被限定住了。

## Summary

## 总结

Auto mode trades per-step human approval for autonomous progress, relying on the environment—not the user—to enforce safety. Use it for scoped, repetitive work where the sandbox can contain mistakes.
