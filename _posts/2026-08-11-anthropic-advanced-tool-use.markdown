---
title: "Building effective agents: advanced tool use（构建高效智能体：高级工具使用）"
date: 2026-08-11
layout: post
liquid: false
tags: [语雀, 计算机, Anthropic Engineering 翻译]
categories: [计算机]
yuque: true
---

> 原文作者：Barry Zhang、Erik Schluntz | 发布于 2024-11-21 | 原文链接：https://www.anthropic.com/engineering/advanced-tool-use

# Building effective agents: Advanced tool use

A follow-up to our "Building effective agents" post, focused on the patterns that make tool use reliable at scale.

我们"构建高效智能体"一文的后继，聚焦于那些让工具使用在规模化下依然可靠的范式。

## Recap: the basics

## 回顾：基础

We previously covered that agents are LLMs that use tools in a loop. The simplest reliable pattern is one tool call, observe the result, and decide the next step. Most failures come from violating this discipline.

我们此前讲过，智能体是在循环中使用工具的 LLM。最简单可靠的范式是：一次工具调用、观察结果、决定下一步。多数失败来自违背这一纪律。

## Advanced patterns

## 高级范式

### Tool use as structured I/O

### 把工具使用当作结构化 I/O

Treat each tool as a typed function: clear inputs, structured outputs. The model's job is to map the task state to the right call. Good schemas make this easy.

把每个工具当作带类型的函数：输入清晰、输出结构化。模型的职责是把任务状态映射到正确的调用。好的 schema 让这很容易。

### Batching independent calls

### 批量处理独立调用

When several calls don't depend on each other, make them in one turn. This cuts latency. But keep dependencies visible—don't batch calls whose results the next step needs.

当多个调用彼此不依赖时，在同一轮里发出。这能降低延迟。但要让依赖关系可见——不要批处理那些下一步需要其结果才能进行的调用。

### Retry with feedback

### 带反馈的重试

On tool failure, feed the error back into the loop rather than giving up. A well-prompted agent will adjust its arguments or try a different tool.

工具失败时，把错误喂回循环而非放弃。提示得当的智能体会调整参数或换一个工具。

### Tool-choice gating

### 工具选择门控

For risky tools (e.g., send email, delete file), require an explicit confirmation step or a separate "planner" agent to authorize. Don't let the action agent decide unilaterally.

对高风险工具（如发邮件、删文件），要求一个显式的确认步骤，或一个独立的"规划者"智能体来授权。不要让执行智能体单方面决定。

### Progressive disclosure of tools

### 工具的渐进式披露

Don't load every tool upfront. Let the agent discover tools as needed (via search or a filesystem of tool definitions), keeping the context small.

不要预先加载所有工具。让智能体按需发现工具（通过搜索或工具定义的文件系统），保持上下文小巧。

## Anti-patterns

## 反范式

*   **The kitchen-sink tool.** One tool that does everything via a `mode` flag. Hard to reason about, easy to misuse.
*   **Silent failures.** Tools that return "ok" on error. The agent can't recover if it doesn't know something went wrong.
*   **Human-only outputs.** Tools that print prose the model must re-parse. Return JSON.

*   **瑞士军刀工具。** 用 `mode` 标志做所有事的单一工具。难推理、易误用。
*   **静默失败。** 出错时返回 "ok" 的工具。智能体若不知道出了错，就无法恢复。
*   **只给人看的输出。** 打印散文让模型重新解析的工具。应返回 JSON。

## Summary

## 总结

Reliable tool use at scale comes from treating tools as typed functions, batching only independent calls, retrying with feedback, gating risky actions, and disclosing tools progressively. Discipline beats cleverness.
