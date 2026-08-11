---
title: "Claude's extended thinking tool: a deep dive（Claude 扩展思考工具深度解析）"
date: 2026-08-11
layout: post
liquid: false
tags: [语雀, 计算机, Anthropic Engineering 翻译]
categories: [计算机]
yuque: true
---

> 原文作者：Barry Zhang | 发布于 2025-03-27 | 原文链接：https://www.anthropic.com/engineering/claude-think-tool

# Claude's extended thinking tool: A deep dive

Extended thinking lets Claude pause and reason through a problem before answering—a capability that significantly improves performance on complex tasks. Here we explain how it works and how to get the most out of it.

扩展思考让 Claude 在回答前先停下来对问题做推理——这项能力显著提升了在复杂任务上的表现。本文解释它的工作原理，以及如何最大化利用它。

## What is extended thinking?

## 什么是扩展思考？

Extended thinking is a mode where the model generates a private chain of reasoning before producing its response. The user sees the final answer; the reasoning happens behind the scenes.

扩展思考是一种模式：模型在产出回复前，先生成一条私有的推理链。用户看到的是最终答案；推理发生在幕后。

## Why it helps

## 它为何有用

Complex tasks—multi-step math, planning, nuanced analysis—benefit from explicit reasoning. Forcing the model to "think" before answering reduces mistakes and improves coherence.

复杂任务——多步数学、规划、细致分析——得益于显式推理。让模型在回答前"思考"，能减少错误、提升连贯性。

## How to use it

## 如何使用

Extended thinking is controlled by a parameter. When enabled, the model thinks for a budgeted number of tokens before answering.

扩展思考由一个参数控制。启用时，模型在回答前会思考预算好的一段 token 数。

```
{
  "model": "claude-...",
  "messages": [...],
  "thinking": { "budget_tokens": 10000 }
}
```

### Tips

### 技巧

*   **Give it room.** A larger thinking budget helps on harder problems, but use it judiciously—thinking costs tokens and latency.
*   **Don't interrupt.** Let the model complete its thinking before acting; cutting it short degrades quality.
*   **Pair with tools.** In agentic settings, thinking helps the model plan which tools to call and in what order.

*   **留出空间。** 更大的思考预算对更难的问题有帮助，但要节制使用——思考要花 token 和延迟。
*   **别打断。** 让模型在行动前完成思考；中途打断会拖累质量。
*   **与工具配合。** 在智能体场景下，思考有助于模型规划该调用哪些工具、按什么顺序。

## Caveats

## 注意事项

Extended thinking adds latency, so it's not always the right choice for low-latency or simple tasks. It also doesn't replace good prompting—clear instructions still matter.

扩展思考会增加延迟，因此对低延迟或简单任务而言它并不总是合适选择。它也不能替代好的提示——清晰的指令依然重要。

## Summary

## 总结

Extended thinking is a simple lever with outsized impact on hard tasks: give the model a budget to reason, and let it finish before it acts.
