---
title: "How we built our SWE-bench workflow（我们如何构建 SWE-bench 工作流）"
date: 2026-08-11
layout: post
liquid: false
tags: [语雀, 计算机, Anthropic Engineering 翻译]
categories: [计算机]
yuque: true
---

> 原文作者：Erik Schluntz | 发布于 2025-02-26 | 原文链接：https://www.anthropic.com/engineering/swe-bench-sonnet

# How we built our SWE-bench workflow

A look at the agentic workflow we used to score strong results on SWE-bench, the benchmark for real-world software engineering tasks.

本文剖析我们用于在 SWE-bench（真实世界软件工程任务基准）上取得强劲成绩的智能体工作流。

## What is SWE-bench?

## SWE-bench 是什么？

SWE-bench is a benchmark of GitHub issues and their pull requests from real open-source projects. A model is given the repo and an issue, and must produce a patch that passes the associated tests.

SWE-bench 是一个由真实开源项目的 GitHub issue 及其 pull request 构成的基准。给定代码库和一个 issue，模型必须产出一个能通过相关测试的补丁。

## Our workflow

## 我们的工作流

The workflow is an agentic loop:

工作流是一个智能体循环：

1.  **Explore.** The agent reads the issue and navigates the codebase to locate the relevant code.
2.  **Hypothesize.** It forms a theory about the root cause.
3.  **Edit.** It writes a candidate patch.
4.  **Test.** It runs the relevant tests; if they fail, it loops back to hypothesize with the error in context.
5.  **Submit.** Once tests pass, it returns the patch.

1.  **探索。** 智能体阅读 issue 并浏览代码库，定位相关代码。
2.  **假设。** 它形成关于根因的推测。
3.  **编辑。** 它写出一个候选补丁。
4.  **测试。** 它运行相关测试；若失败，带着错误信息回到假设环节。
5.  **提交。** 一旦测试通过，返回补丁。

```
issue ─▶ explore ─▶ hypothesize ─▶ edit ─▶ test
                                  ▲          │
                                  └── fail ──┘
                                         │ pass
                                         ▼
                                       submit
```

## What moved the needle

## 什么带来了提升

*   **Good exploration tools.** Letting the agent search the codebase and read files freely was more important than fancy planning. The agent needs to find the right place to change.
*   **Test feedback in context.** Running the relevant tests and feeding failures back into the loop let the agent self-correct.
*   **Patience.** Allowing multiple edit-test iterations per issue, rather than one-shotting, dramatically improved pass rates.

*   **好的探索工具。** 让智能体自由搜索代码库、阅读文件，比花哨的规划更重要。智能体需要找到正确的改动位置。
*   **上下文中的测试反馈。** 运行相关测试并把失败喂回循环，让智能体自我纠正。
*   **耐心。** 对每个 issue 允许多次"编辑—测试"迭代，而非一遍过，显著提升了通过率。

## Lessons for production

## 对生产环境的启示

The same loop that works on SWE-bench works in real repos: explore, hypothesize, edit, test, repeat. The difference is that production code needs the patch to be clean and reviewable, not just test-passing.

在 SWE-bench 上奏效的循环，在真实代码库里同样奏效：探索、假设、编辑、测试、重复。区别在于生产代码需要补丁干净、可审查，而不只是通过测试。

## Summary

## 总结

Strong SWE-bench results came not from a clever model trick but from a disciplined agentic loop with good tools and honest test feedback.
