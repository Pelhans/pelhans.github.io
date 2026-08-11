---
title: "Demystifying evals for AI agents（为 AI 智能体评估去魅）"
date: 2026-08-11
layout: post
liquid: false
tags: [语雀, 计算机, Anthropic Engineering 翻译]
categories: [计算机]
yuque: true
---

> 原文作者：Sholto Douglas、Jeremy Hadfield | 发布于 2025-08-04 | 原文链接：https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents

# Demystifying evals for AI agents

A practical framework for evaluating agents—what to measure, how to build evals, and how to avoid common pitfalls.

一个评估智能体的实用框架——测什么、如何构建评估、以及如何避开常见陷阱。

## Why agent evals are hard

## 为何智能体评估很难

Agents are non-deterministic, multi-step, and tool-using. A single eval run can pass or fail for reasons unrelated to capability. This makes measurement noisy.

智能体是非确定性、多步、使用工具的。一次评估可能因为与能力无关的原因通过或失败。这让度量充满噪声。

## What to measure

## 测什么

*   **Task success.** Did the agent accomplish the goal?
*   **Process quality.** Did it use tools sensibly, avoid loops, and recover from errors?
*   **Efficiency.** How many steps, tokens, and tool calls?
*   **Safety.** Did it stay within its boundaries?

*   **任务成功。** 智能体达成目标了吗？
*   **过程质量。** 它是否合理使用工具、避免循环、从错误中恢复？
*   **效率。** 多少步、多少 token、多少次工具调用？
*   **安全性。** 它守住了边界吗？

## How to build evals

## 如何构建评估

### Start with real tasks

### 从真实任务起步

Use tasks your users actually do. Synthetic tasks optimize for the wrong thing.

用你用户真正在做的任务。合成任务会优化错对象。

### Define a clear pass condition

### 定义清晰的通过条件

A pass condition should be checkable by a program, not a human's vibe. Prefer deterministic checks (file exists, test passes, API returns X).

通过条件应可由程序核查，而非凭人的感觉。优先确定性检查（文件存在、测试通过、API 返回 X）。

### Run multiple trials

### 多次试验

Because agents are non-deterministic, run each eval several times and report the pass rate, not a single boolean.

因为智能体是非确定性的，每个评估跑多次，报告通过率，而非单个布尔值。

### Separate capability from harness

### 区分能力与支撑框架

A failure may come from the agent or from the harness around it. Isolate which before concluding the model is at fault.

失败可能来自智能体，也可能来自它外围的支撑框架。在断定是模型的问题之前，先隔离是哪一个。

## Common pitfalls

## 常见陷阱

*   **Overfitting to the eval set.** Tuning until a fixed set passes, then it fails in the wild.
*   **Ignoring variance.** Reporting one lucky run as the result.
*   **Conflating outcome and process.** A correct answer via a terrible process is still a risk.

*   **对评估集过拟合。** 一直调到固定集通过，然后在真实环境失败。
*   **忽视方差。** 把一次走运的运行当成结果。
*   **混淆结果与过程。** 通过糟糕过程得到的正确答案仍是风险。

## Summary

## 总结

Good agent evals use real tasks, deterministic pass conditions, multiple trials, and a clear separation of capability from harness. Measure the process, not just the prize.
