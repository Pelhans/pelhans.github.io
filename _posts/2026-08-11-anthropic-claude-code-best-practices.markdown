---
title: "Best practices for Claude Code（Claude Code 最佳实践）"
date: 2026-08-11
layout: post
liquid: false
tags: [语雀, 计算机, Anthropic Engineering 翻译]
categories: [计算机]
yuque: true
---

> 原文：Claude Code Docs | 原文链接：https://www.anthropic.com/engineering/claude-code-best-practices

# Best practices for Claude Code

Practical guidance for getting the most out of Claude Code, drawn from how effective engineers actually use it day to day.

关于如何最大化利用 Claude Code 的实用指引，源自高效工程师日常使用的真实方式。

## Getting started

## 入门

### Set up your environment

### 搭建你的环境

Run `/init` to generate a starter CLAUDE.md, then refine it over time. A good CLAUDE.md gives Claude persistent context it can't infer from the code alone.

运行 `/init` 生成一个 CLAUDE.md 起点文件，随后持续打磨。一份好的 CLAUDE.md 能为 Claude 提供它无法仅从代码推断出的持久上下文。

CLAUDE.md is a special file that Claude reads at the start of every conversation. Include Bash commands, code style, and workflow rules.

CLAUDE.md 是一份特殊文件，Claude 会在每次对话开始时读取它。在其中写入 Bash 命令、代码风格与工作流规则。

```
# Code style
- Use ES modules (import/export) syntax, not CommonJS (require)
- Destructure imports when possible (eg. import { foo } from 'bar')

# Workflow
- Be sure to typecheck when you're done making a series of code changes
- Prefer running single tests, and not the whole test suite, for performance
```

Keep CLAUDE.md short and human-readable. If Claude keeps doing something you don't want despite having a rule against it, the file is probably too long and the rule is getting lost. Treat CLAUDE.md like code: review it when things go wrong, prune it regularly.

保持 CLAUDE.md 简短、人类可读。如果即便有规则约束，Claude 仍反复做出你不想要的行为，文件很可能太长了，那条规则被淹没在噪声中。把 CLAUDE.md 当作代码对待：出问题就审查它，定期修剪。

### Give Claude the right tools

### 给 Claude 合适的工具

*   **CLI tools**: tell Claude to use `gh`, `aws`, `gcloud`, `sentry-cli` for external services.
*   **MCP servers**: connect tools like Notion, Figma, or your database with `claude mcp add`.
*   **Hooks**: run scripts automatically at specific workflow points (e.g., eslint after every edit).
*   **Skills**: create `SKILL.md` files in `.claude/skills/` for domain knowledge and repeatable workflows.
*   **Subagents**: define specialized assistants in `.claude/agents/` that run in isolated context.
*   **Plugins**: install bundles of skills, hooks, and MCP servers via `/plugin`.

*   **CLI 工具**：让 Claude 使用 `gh`、`aws`、`gcloud`、`sentry-cli` 等对接外部服务。
*   **MCP 服务器**：用 `claude mcp add` 连接 Notion、Figma 或你的数据库等工具。
*   **钩子（Hooks）**：在特定工作流节点自动运行脚本（如每次编辑后跑 eslint）。
*   **技能（Skills）**：在 `.claude/skills/` 下创建 `SKILL.md`，承载领域知识与可复用工作流。
*   **子智能体（Subagents）**：在 `.claude/agents/` 中定义运行于隔离上下文的专门助手。
*   **插件（Plugins）**：通过 `/plugin` 安装技能、钩子与 MCP 服务器的打包组合。

## Communicate effectively

## 有效沟通

### Be specific

### 具体明确

The more precise your instructions, the fewer corrections you'll need. Scope the task, point to source files, reference existing patterns, and describe the symptom (not just "fix the bug").

你的指令越精确，所需的纠偏就越少。界定任务范围、指向源文件、引用既有模式，并描述症状（而非只说"修一下 bug"）。

**Before**: "add tests for foo.py"
**After**: "write a test for foo.py covering the edge case where the user is logged out. avoid mocks."

**改前**："add tests for foo.py"
**改后**："write a test for foo.py covering the edge case where the user is logged out. avoid mocks."

### Provide rich content

### 提供丰富内容

Use `@` to reference files, paste screenshots/images, give URLs, pipe in data with `cat error.log | claude`, or let Claude fetch context itself.

用 `@` 引用文件、直接粘贴截图/图片、给出 URL、用 `cat error.log | claude` 灌入数据，或让 Claude 自行拉取上下文。

## Manage your session

## 管理你的会话

*   **Course-correct early and often**: `Esc` to stop, `Esc+Esc` or `/rewind` to restore, `/clear` to reset between unrelated tasks.
*   **Manage context aggressively**: run `/clear` between unrelated tasks; use `/compact <instructions>` for control.
*   **Use subagents for investigation**: delegate research to keep your main context clean.
*   **Rewind with checkpoints**: every prompt creates a checkpoint you can restore from.
*   **Resume conversations**: name sessions with `/rename` and treat them like branches.

*   **尽早且频繁纠偏**：`Esc` 停止，`Esc+Esc` 或 `/rewind` 恢复，`/clear` 在不同任务间重置。
*   **积极管理上下文**：在不同任务间运行 `/clear`；用 `/compact <instructions>` 精细控制。
*   **用子智能体做调研**：委派研究工作，保持主上下文干净。
*   **用检查点回退**：每条提示都会创建一个可恢复的检查点。
*   **恢复对话**：用 `/rename` 给会话命名，像分支一样对待它们。

## Automate and scale

## 自动化与规模化

*   **Non-interactive mode**: `claude -p "prompt"` for CI, pre-commit hooks, scripts.
*   **Multiple sessions**: run parallel CLI sessions in worktrees or the desktop app; use a Writer/Reviewer pattern.
*   **Fan out across files**: loop `claude -p` over a task list with `--allowedTools` scoping.
*   **Auto mode**: `claude --permission-mode auto -p "fix all lint errors"` for unattended runs.
*   **Adversarial review**: have a subagent review the diff in a fresh context before counting work done.

*   **非交互模式**：用 `claude -p "prompt"` 接入 CI、pre-commit 钩子、脚本。
*   **多会话**：在 worktree 或桌面应用中并行运行 CLI 会话；采用 Writer/Reviewer 模式。
*   **跨文件扇出**：用 `--allowedTools` 限定范围，对任务列表循环调用 `claude -p`。
*   **自动模式**：用 `claude --permission-mode auto -p "fix all lint errors"` 做无人值守运行。
*   **对抗式审查**：在宣告工作完成前，让子智能体在全新上下文中审查 diff。

## Avoid common failure patterns

## 规避常见失败模式

*   **The kitchen sink session** — mixing unrelated tasks. Fix: `/clear` between tasks.
*   **Correcting over and over** — polluted context. Fix: after two failed corrections, `/clear` and rewrite the prompt.
*   **The over-specified CLAUDE.md** — Claude ignores half of it. Fix: ruthlessly prune.
*   **The trust-then-verify gap** — plausible code that misses edge cases. Fix: always provide verification.
*   **The infinite exploration** — unsoped "investigate" reads hundreds of files. Fix: scope narrowly or use subagents.

*   **大杂烩会话**——混杂不相关任务。修复：任务之间 `/clear`。
*   **反复纠偏**——上下文被污染。修复：两次纠偏失败后，`/clear` 并重写提示。
*   **过度臃肿的 CLAUDE.md**——Claude 忽略其中一半。修复：无情修剪。
*   **信任却未验证的鸿沟**——看起来合理却遗漏边界情况的代码。修复：始终提供验证。
*   **无限探索**——无范围限定的"调研"读入数百文件。修复：收窄范围或使用子智能体。

## Develop your intuition

## 培养你的直觉

The patterns in this guide aren't set in stone. They're starting points that work well in general, but might not be optimal for every situation. Pay attention to what works. Over time, you'll develop intuition that no guide can capture—knowing when to be specific and when to be open-ended, when to plan and when to explore, when to clear context and when to let it accumulate.

本指南中的模式并非金科玉律。它们是普遍奏效的起点，却未必对每种情形都最优。留意哪些做法有效。随着时间推移，你将培养出任何指南都无法囊括的直觉——懂得何时该具体、何时该开放，何时该规划、何时该探索，何时该清空上下文、何时让它积累。

### Related resources

### 相关资源

*   [How Claude Code works](/docs/en/how-claude-code-works): the agentic loop, tools, and context management
*   [Extend Claude Code](/docs/en/features-overview): skills, hooks, MCP, subagents, and plugins
*   [Common workflows](/docs/en/common-workflows): step-by-step recipes for debugging, testing, PRs, and more
*   [CLAUDE.md](/docs/en/memory): store project conventions and persistent context

*   [How Claude Code works](/docs/en/how-claude-code-works)：智能体循环、工具与上下文管理
*   [Extend Claude Code](/docs/en/features-overview)：技能、钩子、MCP、子智能体与插件
*   [Common workflows](/docs/en/common-workflows)：调试、测试、PR 等分步配方
*   [CLAUDE.md](/docs/en/memory)：存储项目约定与持久上下文
