---
title: "Why your agents choke on infrastructure noise（为何智能体会被基础设施噪声噎住）"
date: 2026-08-11
layout: post
liquid: false
tags: [语雀, 计算机, Anthropic Engineering 翻译]
categories: [计算机]
yuque: true
---

> 原文作者：Pedram Navid | 发布于 2025-12-02 | 原文链接：https://www.anthropic.com/engineering/infrastructure-noise

# Why your agents choke on infrastructure noise

A look at how the relentless stream of logs, warnings, and status messages from modern tooling overwhelms agents—and what to do about it.

本文剖析：现代工具链源源不断的日志、警告和状态消息，如何淹没智能体——以及该如何应对。

## The problem

## 问题

Modern infrastructure is loud. Every command emits banners, deprecation warnings, progress bars, and telemetry. A single `kubectl apply` can produce dozens of lines of output that an agent must read before it can act.

现代基础设施很"吵"。每条命令都吐出横幅、弃用警告、进度条和遥测数据。单单一个 `kubectl apply` 就能产生几十条输出，智能体必须先读完才能行动。

## Why it hurts agents

## 它为何伤害智能体

*   **Context bloat.** Noise fills the window, pushing out what matters.
*   **Distraction.** The model acts on a warning instead of the real error.
*   **False failures.** A non-fatal warning gets treated as a stop condition.

*   **上下文膨胀。** 噪声填满窗口，挤掉了重要的东西。
*   **分心。** 模型对着警告而非真正的错误行动。
*   **虚假失败。** 一个非致命警告被当成停止条件。

## Patterns to reduce noise

## 降低噪声的模式

### Quiet your tools

### 让工具安静下来

Set flags that suppress banners and progress bars. Many CLIs support `--no-color`, `--quiet`, or `CI=true`.

设置抑制横幅和进度条的开关。许多 CLI 支持 `--no-color`、`--quiet` 或 `CI=true`。

### Post-process output

### 后处理输出

Pipe tool output through a filter before it reaches the model. Strip ANSI codes, collapse repeated lines, and keep only the last N lines plus any error lines.

在工具输出到达模型之前用过滤器处理。剥离 ANSI 码、折叠重复行，只保留最后 N 行加任何错误行。

```
tool_output | strip_ansi | grep -v "deprecated" | tail -20
```

### Use structured tools

### 使用结构化工具

Prefer tools that return JSON over ones that print human-friendly noise. The model parses data, not prose.

优先返回 JSON 而非打印人类友好噪声的工具。模型解析的是数据，不是散文。

### Give the agent a way to dig

### 给智能体深挖的途径

Don't hide everything—give the agent a `verbose` flag it can flip when it needs the full output to debug.

不要隐藏一切——给智能体一个 `verbose` 开关，需要完整输出调试时它能打开。

## Summary

## 总结

Agents fail less when their tools are quiet. Reduce infrastructure noise at the source, filter what remains, and keep a debug path for when the model genuinely needs to see everything.
