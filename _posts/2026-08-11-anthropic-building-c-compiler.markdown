---
title: "How we built a C compiler with Claude Code（我们如何用 Claude Code 构建 C 编译器）"
date: 2026-08-11
layout: post
liquid: false
tags: [语雀, 计算机, Anthropic Engineering 翻译]
categories: [计算机]
yuque: true
---

> 原文作者：Boris Cherny | 发布于 2025-11-21 | 原文链接：https://www.anthropic.com/engineering/building-c-compiler

# How we built a C compiler with Claude Code

An end-to-end account of using Claude Code to implement a C compiler—from lexer to code generation—and what it taught us about agentic software engineering.

一份端到端的记录：如何用 Claude Code 实现一个 C 编译器——从词法分析器到代码生成——以及它教给我们的关于智能体软件工程的事。

## Why a C compiler?

## 为何是 C 编译器？

A compiler is a great stress test for agentic coding: it's a large, well-specified project with many interdependent parts and a clear notion of correctness (it must compile real C programs).

编译器是智能体编码的绝佳压力测试：它是一个庞大、规格明确的项目，有许多相互依赖的部分，且有清晰的"正确性"定义（它必须能编译真实的 C 程序）。

## The approach

## 方法

We used a long-running agent harness (the same pattern from our earlier posts): an initializer sets up the project skeleton and a feature list; coding agents implement one feature at a time; each session ends with a git commit and progress note.

我们用了长时运行智能体支撑框架（即之前文章里的同一套模式）：初始化智能体搭建项目骨架和功能列表；编码智能体一次实现一个功能；每个会话以一次 git 提交和进度记录结束。

## Stages

## 各阶段

1.  **Lexer.** Turn source text into tokens. Easy to test with unit cases.
2.  **Parser.** Build an AST from tokens, following the C grammar.
3.  **Semantic analysis.** Type checking and scope resolution.
4.  **Code generation.** Emit assembly or LLVM IR from the AST.
5.  **Testing.** Compile real small C programs and run them.

1.  **词法分析器。** 把源文本变成 token。用单元用例易测。
2.  **语法分析器。** 依据 C 文法从 token 构建 AST。
3.  **语义分析。** 类型检查与作用域解析。
4.  **代码生成。** 从 AST 生成汇编或 LLVM IR。
5.  **测试。** 编译真实的小 C 程序并运行它们。

## What worked

## 奏效之处

*   **A feature list as the landmark.** The agent always knew what was done and what was next.
*   **Real end-to-end tests.** Compiling actual C programs caught bugs that unit tests missed.
*   **Small, incremental commits.** Each session left a clean, reviewable state.

*   **功能列表当路标。** 智能体始终知道完成了什么、下一步是什么。
*   **真实的端到端测试。** 编译真实的 C 程序捕获了单元测试漏掉的 bug。
*   **小而增量的提交。** 每个会话都留下干净、可审查的状态。

## What didn't

## 不奏效之处

*   **Letting the agent skip the feature list.** When it tried to "just finish" a stage, it produced tangled, untestable code.
*   **Under-specifying the C subset.** Vague scope led to rework when edge cases appeared.

*   **让智能体跳过功能列表。** 当它想"直接做完"某个阶段时，产出了纠缠、不可测试的代码。
*   **C 子集规格不足。** 模糊的范围导致边界情况出现时返工。

## Takeaways

## 要点

Building a compiler with an agent is feasible, but only with the discipline we've preached: a clear spec, one feature at a time, and evidence the work actually compiles and runs.

用智能体构建编译器是可行的，但只有借助我们一直倡导的纪律才可行：清晰的规格、一次一个功能、以及"代码确实能编译运行"的证据。

## Summary

## 总结

A C compiler is a hard, well-defined target that rewards exactly the harness patterns—landmarks, incremental progress, real verification—that make long-running agents reliable.
