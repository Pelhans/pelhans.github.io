---
title: "A postmortem of three recent issues（三起近期事故复盘）"
date: 2026-08-11
layout: post
liquid: false
tags: [语雀, 计算机, Anthropic Engineering 翻译]
categories: [计算机]
yuque: true
---

> 原文作者：Jeremy Hadfield、Erik Schluntz、Barry Zhang | 发布于 2025-07-09 | 原文链接：https://www.anthropic.com/engineering/a-postmortem-of-three-recent-issues

# A postmortem of three recent issues

In the past month, we experienced three incidents that degraded the experience for users of the Claude API and Claude Code. We take reliability seriously, and want to be transparent about what happened and how we're preventing recurrence.

在过去一个月里，我们经历了三起导致 Claude API 与 Claude Code 用户体验下降的事故。我们极其重视可靠性，并希望就发生了什么、以及如何防止重演保持透明。

## Incident 1: The "infinite thinking" loop

## 事故 1："无限思考"循环

**What happened:** A subset of API requests entered a state where the model generated extremely long chains of internal reasoning, never producing a final response. Requests would hang until they hit the timeout.

**发生了什么：** 一部分 API 请求进入一种状态：模型生成了极长的内部推理链，却始终不产出最终回复。请求会一直挂起，直到撞上超时。

**Root cause:** A change to the reasoning configuration interacted poorly with a specific class of prompts, causing the model to repeatedly re-enter its thinking loop without terminating.

**根本原因：** 对推理配置的一处改动，与某一类特定提示交互不良，导致模型反复重新进入思考循环而无法终止。

**Mitigation:** We rolled back the configuration change and added a hard cap on reasoning steps, after which the issue resolved.

**缓解措施：** 我们回滚了配置改动，并给推理步数加了硬性上限，问题随即解决。

## Incident 2: Elevated latency on the agentic API

## 事故 2：智能体 API 延迟升高

**What happened:** Users of the agentic API saw p95 latency roughly double for several hours. Throughput dropped correspondingly.

**发生了什么：** 智能体 API 用户发现 p95 延迟在数小时内大约翻倍，吞吐量相应下降。

**Root cause:** A new feature flag that enabled an additional model call per agent turn was mistakenly enabled for 100% of traffic instead of the intended 5% rollout. The extra call was on the critical path.

**根本原因：** 一个新特性开关本应按计划灰度到 5% 流量，却误对所有 100% 流量开启，导致每轮智能体都多一次模型调用，且这次调用处于关键路径上。

**Mitigation:** We disabled the flag, then re-enabled it behind the intended gradual rollout. We also added a guardrail that alerts when a feature flag's rollout percentage jumps unexpectedly.

**缓解措施：** 我们关闭了该开关，随后在计划中的渐进灰度下重新开启。我们还加了一道护栏：当特性开关的灰度比例异常跳变时发出告警。

## Incident 3: A broken Claude Code release

## 事故 3：一次损坏的 Claude Code 发布

**What happened:** A Claude Code release introduced a regression where certain file edits silently failed, leaving the user's code in a partially-updated state.

**发生了什么：** 一次 Claude Code 发布引入了一个回归：某些文件编辑静默失败，让用户代码停留在部分更新的状态。

**Root cause:** A refactor of the file-editing path changed the error-handling behavior such that a specific class of edit failures was swallowed instead of surfaced to the user.

**根本原因：** 文件编辑路径的一次重构改变了错误处理行为，导致某一类编辑失败被"吞掉"，而没有暴露给用户。

**Mitigation:** We reverted the release, fixed the error propagation, and added an integration test that covers the previously-untested edit path.

**缓解措施：** 我们回退了这次发布，修复了错误传播，并加了一个覆盖此前未测试编辑路径的集成测试。

## What we're doing to prevent recurrence

## 我们如何预防重演

Across all three incidents, the common thread was a change that reached production faster than our safety checks could catch it. We're investing in:

三起事故的共同主线是：变更抵达生产环境的速度，快过了我们的安全检查能捕获它的速度。我们正在投入：

*   **Staged rollouts by default.** Every user-facing change goes through a gradual percentage ramp with automatic rollback on anomaly.
*   **Stronger guardrails on feature flags.** Unexpected rollout jumps now page on-call.
*   **More integration coverage on edit paths.** The Claude Code editing path now has tests for failure modes, not just success paths.
*   **Reasoning step caps.** Hard limits prevent runaway loops regardless of prompt.

*   **默认分阶段发布。** 每个面向用户的变更都经过渐进式百分比放量，异常时自动回滚。
*   **更强的特性开关护栏。** 灰度比例异常跳变现在会呼叫值班人员。
*   **在编辑路径上增加集成覆盖。** Claude Code 的编辑路径现在有了针对失败模式的测试，而非只有成功路径。
*   **推理步数上限。** 硬性限制可防止失控循环，无论提示如何。

## Conclusion

## 结论

We're sorry for the disruption these incidents caused. Reliability is a prerequisite for trust, and we'll keep investing in the systems that catch regressions before they reach you.

我们对这些事故造成的困扰深表歉意。可靠性是信任的前提，我们将持续投入那些能在问题触达你之前就捕获回归的系统。
