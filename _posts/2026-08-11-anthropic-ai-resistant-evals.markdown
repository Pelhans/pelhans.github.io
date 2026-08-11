---
title: "Building AI-resistant technical evaluations（构建抗 AI 的技术评估）"
date: 2026-08-11
layout: post
liquid: false
tags: [语雀, 计算机, Anthropic Engineering 翻译]
categories: [计算机]
yuque: true
---

> 原文作者：Henry S.、David H. | 发布于 2025-11-18 | 原文链接：https://www.anthropic.com/engineering/AI-resistant-technical-evaluations

# Building AI-resistant technical evaluations

How we design internal technical interviews and assessments that remain meaningful even as models get better at solving them.

我们如何设计内部技术面试与考核，使其在模型越来越擅长解题的当下依然有意义。

## The challenge

## 挑战

As models improve, any evaluation that's easy to memorize or pattern-match becomes a weak signal. We need assessments that measure genuine engineering judgment, not just the ability to produce a correct answer.

随着模型进步，任何容易被记忆或模式匹配的评估都会变成弱信号。我们需要度量真正工程判断力的考核，而非仅仅"产出正确答案"的能力。

## Principles

## 原则

### Novelty

### 新颖性

Use problems that aren't on the internet. A question with a thousand Stack Overflow answers measures search, not skill.

用互联网上没有的问题。一个在 Stack Overflow 有上千答案的问题，度量的是搜索而非技能。

### Open-endedness

### 开放性

Good problems have many valid approaches and no single "right" implementation. They reward discussion of tradeoffs.

好的问题有多种有效解法、没有唯一"正确"实现。它们奖励对权衡取舍的讨论。

### Defense of decisions

### 为决策辩护

Ask candidates (or models) to justify their choices. The reasoning reveals more than the answer.

要求候选者（或模型）为自己的选择辩护。推理过程比答案本身揭示得更多。

### Iterative refinement

### 迭代精炼

Present a problem, then evolve it based on the solution. Adaptability under changing requirements is the real test.

抛出一个问题，再基于解法演化它。在变化需求下的适应能力才是真正的考验。

## Applying this to model evals

## 把这套用于模型评估

The same principles apply when evaluating agents: prefer private, evolving tasks over public static benchmarks, and score the process (reasoning, tool use, recovery from failure) not just the outcome.

同样的原则适用于评估智能体：偏好私有、演化的任务，而非公开静态基准；对过程（推理、工具使用、从失败中恢复）而非仅对结果打分。

## Summary

## 总结

AI-resistant evaluation means designing for novelty, open-endedness, and defensible reasoning—so the score reflects judgment, not memorization.
