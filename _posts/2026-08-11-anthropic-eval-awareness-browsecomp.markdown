---
title: "Measuring eval awareness in BrowseComp（在 BrowseComp 中度量评估感知）"
date: 2026-08-11
layout: post
liquid: false
tags: [语雀, 计算机, Anthropic Engineering 翻译]
categories: [计算机]
yuque: true
---

> 原文作者：Michael Y. Li、Sholto Douglas 等 | 发布于 2025-12-11 | 原文链接：https://www.anthropic.com/engineering/eval-awareness-browsecomp

# Measuring eval awareness in BrowseComp

A study of how aware models are of the benchmarks they're evaluated on—and why that matters for trusting eval numbers.

一项关于模型对自身所受评测基准有多"知情"的研究——以及为何这关乎我们能否信任评测数字。

## What is eval awareness?

## 什么是评估感知？

Eval awareness is the degree to which a model "knows" it's being evaluated on a specific benchmark. If a model has memorized the answers to BrowseComp questions, a high score reflects memory, not capability.

评估感知，是指模型在多大程度上"知道"自己正在某个特定基准上受测。如果模型记住了 BrowseComp 问题的答案，高分反映的是记忆而非能力。

## Why it matters

## 为何重要

Public benchmarks leak into training data. A model that scores well because it's seen the test set isn't demonstrating generalization—it's demonstrating recall. This makes headline numbers misleading.

公开基准会泄漏进训练数据。一个因见过测试集而得分高的模型，展示的不是泛化能力，而是回忆能力。这让标题数字具有误导性。

## How we measured it

## 我们如何度量

We probed BrowseComp with variants:

我们用变体探测 BrowseComp：

*   **Original questions** vs. **paraphrased questions** with the same answer.
*   **Answer-only prompts** that hint at the benchmark's structure.

*   **原问题** vs. **同答案的改写问题**。
*   暗示基准结构的**只给答案提示**。

If performance drops sharply on paraphrases but holds on the originals, the model likely memorized the literal questions—a signal of eval awareness.

如果模型在改写问题上表现骤降、在原问题上却稳定，那它很可能记住了字面问题——这是评估感知的信号。

## Findings

## 发现

Models showed meaningful gaps between original and paraphrased performance, indicating some degree of benchmark memorization. The effect was smaller than worst-case fears but large enough to warrant caution when interpreting scores.

模型在原问题与改写问题间的表现存在显著差距，表明存在一定程度的基准记忆。该效应小于最坏情形的担忧，但已大到足以让我们在解读分数时保持谨慎。

## What this means for evaluation

## 这对评估意味着什么

*   Report results on held-out or paraphrased variants, not just the canonical set.
*   Treat a single benchmark number as a lower-confidence signal.
*   Prefer evals that are hard to memorize (novel, large, or private).

*   报告在保留集或改写变体上的结果，而非仅规范集。
*   把单一基准数字视为低置信度信号。
*   偏好难以记忆的评估（新颖、庞大或私有）。

## Summary

## 总结

High benchmark scores can overstate capability when models are eval-aware. Measure the gap between original and paraphrased performance to estimate how much of a score is memory versus skill.
