---
title: "How we contain Claude across products（我们如何在各产品中遏制 Claude）"
date: 2026-08-11
layout: post
liquid: false
tags: [语雀, 计算机, Anthropic Engineering 翻译]
categories: [计算机]
yuque: true
---

> 原文作者：Max McGuinness 等 | 发布于 2026-05-25 | 原文链接：https://www.anthropic.com/engineering/how-we-contain-claude

# How we contain Claude across products

As agents grow more capable, so does their potential blast radius. The engineering question is how to cap it. Here's what we've learned building containment for claude.ai, Claude Code, and Cowork.

随着智能体能力增强，其潜在的破坏范围（blast radius）也随之扩大。工程问题是如何给它设上限。以下是我们为 claude.ai、Claude Code 和 Cowork 构建遏制机制时的所学。

Twelve months ago, we'd have rejected out of hand the idea of granting Claude access sufficient to take down an internal Anthropic service. Today that level of access is routine, and Anthropic developers are more productive for it. The risk of these deployments has two components: how likely a failure is, and how much damage one could do. Progress on safeguards and model training has steadily driven down the first; the second—the theoretical blast radius—only grows as capabilities and access expand.

十二个月前，我们会毫不犹豫地否决"授予 Claude 足以搞垮某个 Anthropic 内部服务的访问权"这种想法。如今这种级别的访问已是常态，Anthropic 开发者也因此更高效。这些部署的风险有两个组成部分：故障发生的可能性，以及一旦发生能造成多大损害。安全护栏与模型训练的进展持续压低了前者；而后者——理论上的破坏范围——随着能力与访问权的扩展只增不减。

There are broadly two ways to do this. The first is to supervise the agent's behavior via a human-in-the-loop. Claude Code previously protected against agents taking unintended actions by asking users for permission at each turn. Theoretically that works, but we've found the approach to be fallible. Our telemetry showed users approved roughly 93% of permission prompts. The more approvals a user sees, the less attention they pay to each.

大体上有两种方式。第一种是通过人在环（human-in-the-loop）监督智能体行为。Claude Code 过去靠每轮向用户请求许可来防止智能体做出非预期动作。理论上这管用，但我们发现这种方式并不可靠。遥测显示用户大约批准了 93% 的权限确认。用户看到的确认越多，对每一个的关注就越少。

The second approach to capping the blast radius—and the focus of much of this post—is containment. Rather than supervising what the agent does, we supervise what it's *able* to do by enforcing access boundaries through, for example, sandboxes, virtual machines, and egress controls.

限制破坏范围的第二种方式——也是本文的重点——是遏制。我们不去监督智能体*做了*什么，而是通过对访问边界的强制（例如沙箱、虚拟机、出口管控）来监督它*能*做什么。

Over the past two years, we've shipped three primary agentic products: claude.ai, Claude Code, and Claude Cowork. Each serves a different audience, requiring a different containment architecture.

过去两年，我们发布了三个主要的智能体产品：claude.ai、Claude Code 和 Claude Cowork。每个面向不同的受众，需要不同的遏制架构。

## Three types of risk, three components of defense

## 三类风险，三道防线

Security risks to agents fall into one of three categories: **User misuse**, **Model misbehavior**, and **External attackers**.

智能体的安全风险分为三类：**用户滥用**、**模型失当行为**、**外部攻击者**。

When building containment and defense systems, we apply defenses to three main components: **The environment in which the agent runs**, **The model the agent consults**, and **The external content the agent can reach**.

构建遏制与防御系统时，我们在三个主要组件上施加防御：**智能体运行的环境**、**智能体所咨询的模型**、**智能体能触及的外部内容**。

## Patterns for containing agents

## 遏制智能体的模式

### Pattern 1: The ephemeral container (claude.ai code execution)

### 模式 1：临时容器（claude.ai 代码执行）

When Claude runs code inside claude.ai, it does so in a gVisor container on isolated infrastructure. The agent is entirely server-side; the filesystem is ephemeral (per-session). The blast radius is minimal.

当 Claude 在 claude.ai 内运行代码时，它运行在隔离基础设施上的 gVisor 容器里。智能体完全在服务器端；文件系统是临时的（按会话）。破坏范围极小。

### Pattern 2: The human-in-the-loop sandbox (Claude Code)

### 模式 2：人在环沙箱（Claude Code）

Claude Code runs on a user's machine and has access to their filesystem, shell, and network. We shipped an OS-level sandbox (Seatbelt on macOS, bubblewrap on Linux) that hardens the boundary: reads are allowed, writes are allowed inside the workspace, but network is denied by default.

Claude Code 运行在用户机器上，可访问其文件系统、shell 和网络。我们发布了一个 OS 级沙箱（macOS 上的 Seatbelt、Linux 上的 bubblewrap）来加固边界：允许读、允许在工作区内写，但默认拒绝网络。

**Risk we missed: Everything before the trust dialog.** Between mid-2025 and January 2026, we received reports of vulnerabilities in Claude Code through our responsible disclosure program. Three of these targeted code that executes *before* the user has consented to anything—for example, a repository's `.claude/settings.json` defining a hook would execute automatically during startup, before the "Do you trust this folder?" prompt. The fix: defer parsing and execution of project-local configuration until after the user accepts the trust prompt.

**我们忽视的风险：信任对话框之前的一切。** 2025 年中到 2026 年 1 月，我们通过负责任披露计划收到了 Claude Code 漏洞报告。其中三个针对的是在用户*同意任何事之前*就执行的代码——例如仓库的 `.claude/settings.json` 定义的 hook 会在启动时、在"你信任这个文件夹吗？"提示出现之前自动执行。修复方法：把项目本地配置的解析与执行推迟到用户接受信任提示之后。

**Risk we missed: The user as an injection vector.** In February 2026, a researcher successfully phished an employee into launching Claude Code with a malicious prompt. The prompt asked Claude to read `~/.aws/credentials`, encode the contents, and POST them to an external endpoint. Across 25 retries, Claude completed the exfiltration 24 times. The only defense that holds here is the environment—egress controls and filesystem boundaries.

**我们忽视的风险：用户作为注入载体。** 2026 年 2 月，一名研究员成功通过钓鱼让一名员工用恶意提示启动 Claude Code。该提示要求 Claude 读取 `~/.aws/credentials`、编码其内容并 POST 到外部端点。在 25 次重试中，Claude 完成了 24 次外泄。这里唯一站得住脚的防御是环境层——出口管控与文件系统边界。

### Pattern 3: The local VM (Claude Cowork)

### 模式 3：本地虚拟机（Claude Cowork）

Claude Cowork runs on a user's desktop with access to a workspace folder. Because the platform is built for general knowledge work, not software engineering, the average user is much less likely to be fluent in bash. So admins should set a boundary that is absolute and always-on.

Claude Cowork 运行在用户桌面上，可访问一个工作区文件夹。由于该平台面向通用知识工作而非软件工程，普通用户不太可能精通 bash。因此管理员应设置一个绝对且始终开启的边界。

The VM has its own Linux kernel, filesystem, and process table. The user's selected workspace and `.claude` folder are mounted; nothing else on the host is visible. Credentials stay in the host's keychain and never enter the guest machine.

虚拟机拥有自己的 Linux 内核、文件系统和进程表。用户选定的工作区和 `.claude` 文件夹被挂载；主机上其他内容不可见。凭据留在主机的钥匙串里，绝不入客机。

**Risk we missed: Exfiltration through an approved domain.** A malicious file in the workspace carried hidden instructions and an attacker-controlled API key. Claude, following instructions, uploaded files to the attacker's Anthropic account via our Files API. The egress proxy saw `api.anthropic.com` and let it through. Fix: a man-in-the-middle proxy inside the VM that only passes requests carrying the VM's own session token.

**我们忽视的风险：通过已批准域名外泄。** 工作区里的一个恶意文件带有隐藏指令和一个攻击者控制的 API key。Claude 遵循指令，通过我们的 Files API 把文件上传到攻击者的 Anthropic 账户。出口代理看到 `api.anthropic.com` 就放行了。修复：在 VM 内加一个中间人代理，只放行携带 VM 自身会话 token 的请求。

## Looking ahead

## 展望未来

**Persistent memory poisoning.** The share of agent context that persists across sessions keeps growing; an injection that lands in any of these is reloaded each time the agent starts.

**持久化记忆污染。** 跨会话持久化的智能体上下文占比持续增长；落在其中任何一处的注入都会在每次启动时重新加载。

**Multi-agent trust escalation.** If a sub-agent's output is treated as higher-trust than raw tool results, a new vector for prompt injection is introduced.

**多智能体信任升级。** 如果子智能体的输出被视为比原始工具结果更可信，就引入了新的提示注入向量。

**Agent identity.** Should an agent possess its own principal identity, or act as an extension of the user? The answer may be a blend of the two.

**智能体身份。** 智能体应拥有自己的主体身份，还是作为用户的延伸？答案或许是两者的融合。

## Summary

## 总结

**Design for containment at the environment layer first, then steer behavior at the model layer.** The deterministic boundary is what gets hit when everything probabilistic misses.

**先在设计上做环境层的遏制，再在模型层引导行为。** 当一切概率性防御都失手时，命中的是那道确定性边界。

**Match isolation strength to the user's capacity for oversight.** A developer who can read bash and a knowledge worker who can't are not running the same threat model.

**让隔离强度匹配用户的监督能力。** 能读 bash 的开发者与不能读的知识工作者，面对的不是同一种威胁模型。

**Be wary of custom components.** Battle-tested hypervisors, syscall filters, and container runtimes have survived more adversarial attention than anything you'll build.

**警惕自定义组件。** 久经考验的虚拟机监视器、系统调用过滤器和容器运行时，经受的对抗性关注比你自建的任何东西都多。
