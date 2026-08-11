---
title: "An update on recent Claude Code quality reports（近期 Claude Code 质量报告复盘）"
date: 2026-08-11
layout: post
liquid: false
tags: [语雀, 计算机, Anthropic Engineering 翻译]
categories: [计算机]
yuque: true
---

> 原文：Anthropic Engineering | 发布于 2026-04-23 | 原文链接：https://www.anthropic.com/engineering/april-23-postmortem

# An update on recent Claude Code quality reports

We traced recent reports of Claude Code quality issues to three separate changes. Here's what happened and what we're changing.

我们把近期关于 Claude Code 质量下滑的报告，溯源到了三处独立的改动。以下是发生了什么，以及我们将做出什么改变。

Over the past month, we've been looking into reports that Claude's responses have worsened for some users. We've traced these reports to three separate changes that affected Claude Code, the Claude Agent SDK, and Claude Cowork. The API was not impacted. All three issues have now been resolved as of April 20 (v2.1.116).

过去一个月，我们一直在调查一些用户反映 Claude 的回复变差了的报告。我们将这些报告溯源到三处独立改动，影响了 Claude Code、Claude Agent SDK 和 Claude Cowork。API 未受影响。截至 4 月 20 日（v2.1.116），三处问题均已解决。

After investigation, we identified three different issues:

调查后，我们识别出三个不同的问题：

1.  On March 4, we changed Claude Code's default reasoning effort from `high` to `medium` to reduce very long latency. This was the wrong tradeoff. We reverted this change on April 7. This impacted Sonnet 4.6 and Opus 4.6.
2.  On March 26, we shipped a change to clear Claude's older thinking from sessions idle over an hour, to reduce latency. A bug caused this to keep happening every turn for the rest of the session, making Claude seem forgetful and repetitive. We fixed it on April 10.
3.  On April 16, we added a system prompt instruction to reduce verbosity. In combination with other prompt changes, it hurt coding quality and was reverted on April 20.

1.  3 月 4 日，我们把 Claude Code 默认推理强度从 `high` 改为 `medium` 以减少过长延迟。这是个错误的权衡。我们于 4 月 7 日回滚。影响 Sonnet 4.6 和 Opus 4.6。
2.  3 月 26 日，我们发布了一项改动，清除闲置超过一小时的会话里 Claude 较旧的思考，以减少延迟。一个 bug 导致它在会话剩余每轮都持续发生，使 Claude 显得健忘且重复。我们于 4 月 10 日修复。
3.  4 月 16 日，我们加了一条系统提示指令来减少啰嗦。结合其他提示改动，它损害了编码质量，于 4 月 20 日回滚。

## A change to Claude Code's default reasoning effort

## Claude Code 默认推理强度的改动

When we released Opus 4.6 in Claude Code in February, we set the default reasoning effort to `high`. Soon after, we received feedback that Opus 4.6 in high effort mode would occasionally think for too long, causing the UI to appear frozen.

2 月在 Claude Code 发布 Opus 4.6 时，我们把默认推理强度设为 `high`。不久后收到反馈：Opus 4.6 在 high 模式下偶尔思考过久，导致 UI 看似卡死。

In general, the longer the model thinks, the better the output. Effort levels are how Claude Code lets users set that tradeoff—more thinking versus lower latency. After internal evals, medium effort achieved slightly lower intelligence with significantly less latency. We rolled out a change making medium the default.

一般而言，模型思考越久，输出越好。effort 级别是 Claude Code 让用户设定该权衡的方式——更多思考 vs 更低延迟。内部评估后，medium 以略低的智能换来显著更低的延迟。我们推出了让 medium 成为默认值的改动。

Soon after rolling out, users began reporting that Claude Code felt less intelligent. After hearing more feedback, we reversed this decision on April 7. All users now default to `xhigh` effort for Opus 4.7, and `high` effort for all other models.

推出后不久，用户开始反映 Claude Code 显得没那么聪明。在听到更多反馈后，我们于 4 月 7 日推翻了这个决定。现在所有用户默认对 Opus 4.7 用 `xhigh` 强度，对其他模型用 `high` 强度。

## A caching optimization that dropped prior reasoning

## 一个丢弃先前推理的缓存优化

On March 26, we shipped what was meant to be an efficiency improvement. We use prompt caching to make back-to-back API calls cheaper. The design: if a session has been idle over an hour, clear old thinking sections to reduce resume cost.

3 月 26 日，我们发布了一项本应是效率改进的东西。我们用 prompt caching 让连续的 API 调用更便宜。设计是：若会话闲置超过一小时，清除旧的思考段落以降低恢复成本。

The implementation had a bug. Instead of clearing thinking history once, it cleared it on every turn for the rest of the session. This compounded: even reasoning from the current turn was dropped. Claude would continue executing, but increasingly without memory of why it had chosen to do what it was doing. This surfaced as forgetfulness, repetition, and odd tool choices.

实现有个 bug。它不是清除一次思考历史，而是在会话剩余每轮都清除。这层层叠加：连当前轮次的推理也被丢弃。Claude 会继续执行，但越来越不记得自己为何做那些事。这表现为健忘、重复和奇怪的工具选择。

We fixed this bug on April 10 in v2.1.101.

我们在 4 月 10 日 v2.1.101 修复了此 bug。

## A system prompt change to reduce verbosity

## 一条减少啰嗦的系统提示改动

Our latest model, Claude Opus 4.7, tends to be quite verbose. A few weeks before release, we started tuning. One addition to the system prompt caused an outsized effect:

我们最新的模型 Claude Opus 4.7 倾向于相当啰嗦。发布前几周我们开始调优。系统提示里一处新增造成了过大影响：

> "Length limits: keep text between tool calls to ≤25 words. Keep final responses to ≤100 words unless the task requires more detail."

> "长度限制：工具调用间的文本保持在 ≤25 词。最终回复保持在 ≤100 词，除非任务需要更多细节。"

After multiple weeks of internal testing and no regressions, we shipped it alongside Opus 4.7 on April 16. As part of this investigation, we ran more ablations using a broader set of evaluations. One showed a 3% drop for both Opus 4.6 and 4.7. We immediately reverted the prompt as part of the April 20 release.

内部测试数周、评估集无回归后，我们于 4 月 16 日随 Opus 4.7 一同发布。作为本次调查的一部分，我们用更广的评估集跑了更多消融实验。其中一个显示 Opus 4.6 和 4.7 都下降了 3%。我们立即在 4 月 20 日发布中回滚了该提示。

## Going forward

## 今后

We are going to do several things differently: we'll ensure a larger share of internal staff use the exact public build of Claude Code; we'll add tighter controls on system prompt changes, running a broad suite of per-model evals for every change, continuing ablations, and building tooling to make prompt changes easier to review and audit.

我们将做些不同的事：确保更大比例的内部员工使用 Claude Code 的精确公开版本；对系统提示改动加更严的控制，为每个改动跑一套广覆盖的按模型评估、持续消融，并构建工具让提示改动更易审查和审计。

Today we are resetting usage limits for all subscribers. We're immensely grateful for your feedback and for your patience.

今天我们将为所有订阅者重置使用上限。我们无比感谢你们的反馈与耐心。
