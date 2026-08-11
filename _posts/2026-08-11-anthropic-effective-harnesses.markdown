---
title: "Effective harnesses for long-running agents（长时运行智能体的有效支撑框架）"
date: 2026-08-11
layout: post
liquid: false
tags: [语雀, 计算机, Anthropic Engineering 翻译]
categories: [计算机]
yuque: true
---

# Effective harnesses for long-running agents

> 原文作者：Justin Young | 发布于 2025-11-26 | 原文链接：https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents

# Effective harnesses for long-running agents

Agents still face challenges working across many context windows. We looked to human engineers for inspiration in creating a more effective harness for long-running agents.

智能体在跨越多个上下文窗口工作时仍面临挑战。我们向人类工程师汲取灵感，为长时运行的智能体打造了一套更有效的支撑框架（harness）。

As AI agents become more capable, developers are increasingly asking them to take on complex tasks requiring work that spans hours, or even days. However, getting agents to make consistent progress across multiple context windows remains an open problem.

随着 AI 智能体能力增强，开发者越来越要求它们承担需要数小时甚至数天才能完成的复杂任务。然而，如何让智能体在多个上下文窗口之间保持连贯的进展，仍是一个悬而未决的问题。

The core challenge of long-running agents is that they must work in discrete sessions, and each new session begins with no memory of what came before. Imagine a software project staffed by engineers working in shifts, where each new engineer arrives with no memory of what happened on the previous shift. Because context windows are limited, and because most complex projects cannot be completed within a single window, agents need a way to bridge the gap between coding sessions.

长时运行智能体的核心挑战在于：它们必须在离散的会话中工作，而每一个新会话开始时都没有此前的记忆。想象一个由轮班工程师负责的项目，每位新到的工程师都对上一班发生了什么一无所知。由于上下文窗口有限，且大多数复杂项目无法在单个窗口内完成，智能体需要一种方式来弥合各次编码会话之间的鸿沟。

We developed a two-fold solution to enable the [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) to work effectively across many context windows: an **initializer agent** that sets up the environment on the first run, and a **coding agent** that is tasked with making incremental progress in every session, while leaving clear artifacts for the next session. You can find code examples in the accompanying [quickstart.](https://github.com/anthropics/claude-quickstarts/tree/main/autonomous-coding)

我们开发了一套双重解决方案，使 [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) 能够在多个上下文窗口中高效工作：一个在首次运行时搭建环境的**初始化智能体（initializer agent）**，以及一个被赋予"在每次会话中增量推进、并为下一次会话留下清晰产物"任务的**编码智能体（coding agent）**。相关代码示例可在附带的 [quickstart](https://github.com/anthropics/claude-quickstarts/tree/main/autonomous-coding) 中找到。

## The long-running agent problem

## 长时运行智能体问题

The Claude Agent SDK is a powerful, general-purpose agent harness adept at coding, as well as other tasks that require the model to use tools to gather context, plan, and execute. It has context management capabilities such as compaction, which enables an agent to work on a task without exhausting the context window. Theoretically, given this setup, it should be possible for an agent to continue to do useful work for an arbitrarily long time.

Claude Agent SDK 是一个强大的通用型智能体支撑框架，擅长编码，也能胜任其他需要模型借助工具收集上下文、进行规划与执行的任务。它具备诸如压缩（compaction）之类的上下文管理能力，使智能体能够在不耗尽上下文窗口的情况下处理任务。理论上，在此配置下，智能体应当能够无限期地持续做有用的工作。

However, compaction isn’t sufficient. Out of the box, even a frontier coding model like Opus 4.5 running on the Claude Agent SDK in a loop across multiple context windows will fall short of building a production-quality web app if it’s only given a high-level prompt, such as “build a clone of [claude.ai](http://claude.ai/redirect/website.v1.6a7d6767-46a9-40a7-9dbd-0f190febb431).”

然而，仅靠压缩是不够的。开箱即用时，即便是像 Opus 4.5 这样前沿的编码模型，在 Claude Agent SDK 上跨多个上下文窗口循环运行，如果只给定一个高层提示（例如"构建一个 [claude.ai](http://claude.ai/redirect/website.v1.6a7d6767-46a9-40a7-9dbd-0f190febb431) 的克隆版"），也无法建出一个生产质量的 Web 应用。

Claude’s failures manifested in two patterns. First, the agent tended to try to do too much at once—essentially to attempt to one-shot the app. Often, this led to the model running out of context in the middle of its implementation, leaving the next session to start with a feature half-implemented and undocumented. The agent would then have to guess at what had happened, and spend substantial time trying to get the basic app working again. This happens even with compaction, which doesn’t always pass perfectly clear instructions to the next agent.

Claude 的失败表现为两种模式。首先，智能体倾向于一次性做太多事情——本质上是在试图"一遍过"完成整个应用。这通常导致模型在实现过程中耗尽上下文，使下一会话从一个功能只做了一半且毫无文档记录的状态开始。智能体随后不得不猜测之前发生了什么，并耗费大量时间试图让基础应用重新跑起来。即便有压缩机制，这种情况仍会发生，因为压缩并不总是能把清晰无误的指令传递给下一个智能体。

A second failure mode would often occur later in a project. After some features had already been built, a later agent instance would look around, see that progress had been made, and declare the job done.

第二种失败模式往往出现在项目稍后阶段。在部分功能已经建成后，后续的某个智能体实例环顾四周，看到已有进展，便宣布任务完成。

This decomposes the problem into two parts. First, we need to set up an initial environment that lays the foundation for *all* the features that a given prompt requires, which sets up the agent to work step-by-step and feature-by-feature. Second, we should prompt each agent to make incremental progress towards its goal while also leaving the environment in a clean state at the end of a session. By “clean state” we mean the kind of code that would be appropriate for merging to a main branch: there are no major bugs, the code is orderly and well-documented, and in general, a developer could easily begin work on a new feature without first having to clean up an unrelated mess.

这把问题拆解成两部分。首先，我们需要搭建一个初始环境，为给定提示所要求的*所有*功能奠定基础，使智能体能够按步骤、按功能逐个推进。其次，我们应当提示每个智能体朝目标增量推进，同时在会话结束时将环境保持为干净状态。所谓"干净状态"，是指那种适合合并到主分支的代码：没有重大 bug、代码有序且有良好文档，总体而言，开发者无需先清理一堆无关乱局就能轻松着手新功能。

When experimenting internally, we addressed these problems using a two-part solution:

在内部实验中，我们用一套两部分的方案来解决这些问题：

1.  Initializer agent: The very first agent session uses a specialized prompt that asks the model to set up the initial environment: an `init.sh` script, a claude-progress.txt file that keeps a log of what agents have done, and an initial git commit that shows what files were added.
2.  Coding agent: Every subsequent session asks the model to make incremental progress, then leave structured updates.1

1.  初始化智能体：第一个智能体会话使用专门提示，要求模型搭建初始环境：一个 `init.sh` 脚本、一个记录各智能体已完成工作的 `claude-progress.txt` 文件，以及一个展示新增了哪些文件的初始 git 提交。
2.  编码智能体：此后的每一次会话都要求模型做增量推进，然后留下结构化的更新。1

The key insight here was finding a way for agents to quickly understand the state of work when starting with a fresh context window, which is accomplished with the claude-progress.txt file alongside the git history. Inspiration for these practices came from knowing what effective software engineers do every day.

这里的关键洞见在于：找到一种方法，让智能体在从一个全新上下文窗口开始时，能迅速理解工作所处的状态——这通过 `claude-progress.txt` 文件配合 git 历史来实现。这些实践方法的灵感，来源于我们对高效软件工程师日常工作的观察。

## Environment management

## 环境管理

In the updated [Claude 4 prompting guide](https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/claude-4-best-practices#multi-context-window-workflows), we shared some best practices for multi-context window workflows, including a harness structure that uses “a different prompt for the very first context window.” This “different prompt” requests that the initializer agent set up the environment with all the necessary context that future coding agents will need to work effectively. Here, we provide a deeper dive on some of the key components of such an environment.

在更新后的 [Claude 4 提示工程指南](https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/claude-4-best-practices#multi-context-window-workflows) 中，我们分享了一些多上下文窗口工作流的最佳实践，包括一种"为第一个上下文窗口使用不同提示"的支撑框架结构。这个"不同提示"要求初始化智能体搭建环境，并载入未来编码智能体高效工作所需的全部必要上下文。在此，我们深入剖析此类环境中的一些关键组成部分。

### Feature list

### 功能列表

To address the problem of the agent one-shotting an app or prematurely considering the project complete, we prompted the initializer agent to write a comprehensive file of feature requirements expanding on the user’s initial prompt. In the [claude.ai](http://claude.ai/redirect/website.v1.6a7d6767-46a9-40a7-9dbd-0f190febb431) clone example, this meant over 200 features, such as “a user can open a new chat, type in a query, press enter, and see an AI response.” These features were all initially marked as “failing” so that later coding agents would have a clear outline of what full functionality looked like.

为解决智能体"一遍过"式构建应用、或过早认为项目已完成的问题，我们提示初始化智能体编写一份详尽的功能需求文件，对用户初始提示进行展开。在 [claude.ai](http://claude.ai/redirect/website.v1.6a7d6767-46a9-40a7-9dbd-0f190febb431) 克隆示例中，这意味着超过 200 项功能，例如"用户可以打开新对话、输入查询、按回车并看到 AI 回复"。这些功能最初都被标记为"未通过（failing）"，以便后续编码智能体对完整功能长什么样有清晰的蓝图。

```
{
    "category": "functional",
    "description": "New chat button creates a fresh conversation",
    "steps": [
      "Navigate to main interface",
      "Click the 'New Chat' button",
      "Verify a new conversation is created",
      "Check that chat area shows welcome state",
      "Verify conversation appears in sidebar"
    ],
    "passes": false
}
```

We prompt coding agents to edit this file only by changing the status of a passes field, and we use strongly-worded instructions like “It is unacceptable to remove or edit tests because this could lead to missing or buggy functionality.” After some experimentation, we landed on using JSON for this, as the model is less likely to inappropriately change or overwrite JSON files compared to Markdown files.

我们提示编码智能体只能通过修改 `passes` 字段的状态来编辑该文件，并使用措辞强硬的指令，例如"删除或编辑测试是不可接受的，因为这可能导致功能缺失或存在 bug"。经过一些实验，我们最终选择使用 JSON 格式，因为相比 Markdown 文件，模型更不容易不恰当地改动或覆盖 JSON 文件。

### Incremental progress

### 增量进展

Given this initial environment scaffolding, the next iteration of the coding agent was then asked to work on only one feature at a time. This incremental approach turned out to be critical to addressing the agent’s tendency to do too much at once.

有了这套初始环境脚手架后，下一版编码智能体被要求一次只处理一个功能。这种增量式方法被证明对纠正智能体"一次做太多"的倾向至关重要。

Once working incrementally, it’s still essential that the model leaves the environment in a clean state after making a code change. In our experiments, we found that the best way to elicit this behavior was to ask the model to commit its progress to git with descriptive commit messages and to write summaries of its progress in a progress file. This allowed the model to use git to revert bad code changes and recover working states of the code base.

即便采用增量方式工作，模型在做出代码改动后仍将环境保持为干净状态，依然至关重要。在我们的实验中，引出这种行为的最佳方式是：要求模型把进展以描述性的提交信息提交到 git，并在进度文件中写下进展摘要。这让模型能够借助 git 回退糟糕的代码改动，并恢复代码库的可工作状态。

These approaches also increased efficiency, as they eliminated the need for an agent to have to guess at what had happened and spend its time trying to get the basic app working again.

这些方法也提升了效率，因为它们不再需要智能体去猜测之前发生了什么、并耗费时间试图让基础应用重新跑起来。

### Testing

### 测试

One final major failure mode that we observed was Claude’s tendency to mark a feature as complete without proper testing. Absent explicit prompting, Claude tended to make code changes, and even do testing with unit tests or `curl` commands against a development server, but would fail recognize that the feature didn’t work end-to-end.

我们观察到的最后一个主要失败模式，是 Claude 倾向于在没有充分测试的情况下就把功能标记为完成。在没有明确提示时，Claude 往往会做代码改动，甚至用单元测试或针对开发服务器的 `curl` 命令做测试，却没能识别出该功能在端到端层面根本不可用。

In the case of building a web app, Claude mostly did well at verifying features end-to-end once explicitly prompted to use browser automation tools and do all testing as a human user would.

在构建 Web 应用的情形下，一旦明确要求 Claude 使用浏览器自动化工具、并像人类用户那样进行所有测试，它在端到端验证功能方面大多表现良好。

Screenshots taken by Claude through the Puppeteer MCP server as it tested the claude.ai clone.

Claude 在测试 claude.ai 克隆版时，通过 Puppeteer MCP 服务器所拍摄的截图。

Providing Claude with these kinds of testing tools dramatically improved performance, as the agent was able to identify and fix bugs that weren’t obvious from the code alone.

为 Claude 提供这类测试工具显著提升了表现，因为智能体能够识别并修复那些仅看代码无法发现的 bug。

Some issues remain, like limitations to Claude’s vision and to browser automation tools making it difficult to identify every kind of bug. For example, Claude can’t see browser-native alert modals through the Puppeteer MCP, and features relying on these modals tended to be buggier as a result.

一些问题依然存在，例如 Claude 的视觉能力和浏览器自动化工具的局限性，使得它难以识别每一类 bug。举例来说，通过 Puppeteer MCP，Claude 无法看到浏览器原生的 alert 弹窗，因此依赖这类弹窗的功能往往更容易出现 bug。

## Getting up to speed

## 快速进入状态

With all of the above in place, every coding agent is prompted to run through a series of steps to get its bearings, some quite basic but still helpful:

将上述一切就位后，每个编码智能体都会被提示执行一系列步骤来摸清状况，其中一些相当基础但仍很有用：

1.  *Run `pwd` to see the directory you’re working in. You’ll only be able to edit files in this directory.*
2.  *Read the git logs and progress files to get up to speed on what was recently worked on.*
3.  *Read the features list file and choose the highest-priority feature that’s not yet done to work on.*

1.  *运行 `pwd` 查看你所在的工作目录。你只能编辑该目录下的文件。*
2.  *阅读 git 日志和进度文件，以了解最近做了哪些工作。*
3.  *阅读功能列表文件，选择尚未完成且优先级最高的功能来处理。*

This approach saves Claude some tokens in every session since it doesn’t have to figure out how to test the code. It also helps to ask the initializer agent to write an init.sh script that can run the development server, and then run through a basic end-to-end test before implementing a new feature.

这种方法在每个会话中为 Claude 节省了一些 token，因为它不必再去琢磨如何测试代码。让初始化智能体编写一个能启动开发服务器的 `init.sh` 脚本，并在实现新功能前跑一遍基础端到端测试，也是有帮助的。

In the case of the claude.ai clone, this meant that the agent always started the local development server and used the Puppeteer MCP to start a new chat, send a message, and receive a response. This ensured that Claude could quickly identify if the app had been left in a broken state, and immediately fix any existing bugs. If the agent had instead started implementing a new feature, it would likely make the problem worse.

以 claude.ai 克隆版为例，这意味着智能体总是先启动本地开发服务器，然后使用 Puppeteer MCP 开启新对话、发送消息并接收回复。这确保了 Claude 能迅速识别出应用是否处于损坏状态，并立即修复已有的 bug。如果智能体转而先去实现新功能，问题可能会变得更糟。

Given all this, a typical session starts off with the following assistant messages:

综上，一个典型会话会以如下助手消息开场：

```
[Assistant] I'll start by getting my bearings and understanding the current state of the project.
[Tool Use] <bash - pwd>
[Tool Use] <read - claude-progress.txt>
[Tool Use] <read - feature_list.json>
[Assistant] Let me check the git log to see recent work.
[Tool Use] <bash - git log --oneline -20>
[Assistant] Now let me check if there's an init.sh script to restart the servers.
<Starts the development server>
[Assistant] Excellent! Now let me navigate to the application and verify that some fundamental features are still working.
<Tests basic functionality>
[Assistant] Based on my verification testing, I can see that the fundamental functionality is working well. The core chat features, theme switching, conversation loading, and error handling are all functioning correctly. Now let me review the tests.json file more comprehensively to understand what needs to be implemented next.
<Starts work on a new feature>
```

## Agent failure modes and solutions

## 智能体失败模式与解决方案

| Problem | Initializer Agent Behavior | Coding Agent Behavior |
|---------|----------------------------|-----------------------|
| Claude declares victory on the entire project too early. | Set up a feature list file: based on the input spec, set up a structured JSON file with a list of end-to-end feature descriptions. | Read the feature list file at the beginning of a session. Choose a single feature to start working on. |
| Claude leaves the environment in a state with bugs or undocumented progress. | An initial git repo and progress notes file is written. | Start the session by reading the progress notes file and git commit logs, and run a basic test on the development server to catch any undocumented bugs. End the session by writing a git commit and progress update. |
| Claude marks features as done prematurely. | Set up a feature list file. | Self-verify all features. Only mark features as “passing” after careful testing. |
| Claude has to spend time figuring out how to run the app. | Write an `init.sh` script that can run the development server. | Start the session by reading `init.sh`. |

| 问题 | 初始化智能体行为 | 编码智能体行为 |
|------|------------------|----------------|
| Claude 过早宣布整个项目已完成。 | 搭建功能列表文件：基于输入规格，建立一个包含端到端功能描述列表的结构化 JSON 文件。 | 在会话开始时阅读功能列表文件，选择一个功能着手处理。 |
| Claude 让环境处于有 bug 或进展无文档记录的状态。 | 写入一个初始 git 仓库和进度记录文件。 | 会话开始时阅读进度记录文件和 git 提交日志，并在开发服务器上跑基础测试以捕获任何无文档记录的 bug。会话结束时写一次 git 提交和进度更新。 |
| Claude 过早把功能标记为完成。 | 搭建功能列表文件。 | 对所有功能自行验证。只有经过仔细测试后才标注为"通过"。 |
| Claude 需要花时间搞清楚如何运行应用。 | 编写一个能启动开发服务器的 `init.sh` 脚本。 | 会话开始时阅读 `init.sh`。 |

Summarizing four common failure modes and solutions in long-running AI agents.

以上总结了长时运行 AI 智能体中四种常见失败模式及其解决方案。

## Future work

## 未来工作

This research demonstrates one possible set of solutions in a long-running agent harness to enable the model to make incremental progress across many context windows. However, there remain open questions.

这项研究展示了在长时运行智能体支撑框架中，让模型跨多个上下文窗口实现增量推进的一组可行方案。然而，仍有开放性问题待解。

Most notably, it’s still unclear whether a single, general-purpose coding agent performs best across contexts, or if better performance can be achieved through a multi-agent architecture. It seems reasonable that specialized agents like a testing agent, a quality assurance agent, or a code cleanup agent, could do an even better job at sub-tasks across the software development lifecycle.

最值得注意的是，目前仍不清楚单一通用编码智能体是否在各种情境下都表现最佳，还是通过多智能体架构能取得更好性能。诸如测试智能体、质量保证智能体或代码清理智能体这类专门化智能体，似乎有理由在软件开发生命周期的各个子任务上做得更好。

Additionally, this demo is optimized for full-stack web app development. A future direction is to generalize these findings to other fields. It’s likely that some or all of these lessons can be applied to the types of long-running agentic tasks required in, for example, scientific research or financial modeling.

此外，本演示针对全栈 Web 应用开发做了优化。未来的一个方向是将这些发现推广到其他领域。部分乃至全部经验，很可能可以应用于科研或金融建模等所需的各类长时运行智能体任务。

### Acknowledgements

### 致谢

Written by Justin Young. Special thanks to David Hershey, Prithvi Rajasakeran, Jeremy Hadfield, Naia Bouscal, Michael Tingley, Jesse Mu, Jake Eaton, Marius Buleandara, Maggie Vo, Pedram Navid, Nadine Yasser, and Alex Notov for their contributions.

由 Justin Young 撰写。特别感谢 David Hershey、Prithvi Rajasakeran、Jeremy Hadfield、Naia Bouscal、Michael Tingley、Jesse Mu、Jake Eaton、Marius Buleandara、Maggie Vo、Pedram Navid、Nadine Yasser 和 Alex Notov 的贡献。

This work reflects the collective efforts of several teams across Anthropic who made it possible for Claude to safely do long-horizon autonomous software engineering, especially the code RL & Claude Code teams. Interested candidates who would like to contribute are welcome to apply at [anthropic.com/careers](http://anthropic.com/careers).

这项工作凝聚了 Anthropic 多个团队的集体努力，他们让 Claude 能够安全地开展长程自主软件工程，尤其是 code RL 与 Claude Code 团队。欢迎有兴趣的候选人申请加入 [anthropic.com/careers](http://anthropic.com/careers)。

### Footnotes

### 脚注

1\. We refer to these as separate agents in this context only because they have different initial user prompts. The system prompt, set of tools, and overall agent harness was otherwise identical.

1\. 我们在此将它们称为不同的智能体，仅是因为它们有着不同的初始用户提示。除此之外，系统提示、工具集以及整体智能体支撑框架都是相同的。
