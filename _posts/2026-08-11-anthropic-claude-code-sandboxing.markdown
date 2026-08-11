---
title: "Beyond permission prompts: making Claude Code more secure and autonomous（超越权限确认：让 Claude Code 更安全且更自主）"
date: 2026-08-11
layout: post
liquid: false
tags: [语雀, 计算机, Anthropic Engineering 翻译]
categories: [计算机]
yuque: true
---

> 原文作者：David Dworken、Oliver Weller-Davies | 发布于 2025-10-20 | 原文链接：https://www.anthropic.com/engineering/claude-code-sandboxing

# Beyond permission prompts: making Claude Code more secure and autonomous

Claude Code's new sandboxing features, a bash tool and Claude Code on the web, reduce permission prompts and increase user safety by enabling two boundaries: filesystem and network isolation.

Claude Code 的新沙箱特性——沙箱 bash 工具，以及云端 Claude Code——通过启用两道边界（文件系统隔离与网络隔离），减少了权限确认并提升了用户安全性。

In [Claude Code](https://www.claude.com/product/claude-code), Claude writes, tests, and debugs code alongside you, navigating your codebase, editing multiple files, and running commands to verify its work. Giving Claude this much access to your codebase and files can introduce risks, especially in the case of prompt injection.

在 [Claude Code](https://www.claude.com/product/claude-code) 中，Claude 与你并肩编写、测试和调试代码，浏览你的代码库、编辑多个文件，并运行命令来验证自己的工作。给 Claude 如此多的代码库与文件访问权会带来风险，尤其在提示注入的情形下。

To help address this, we’ve introduced two new features in Claude Code built on top of sandboxing, both of which are designed to provide a more secure place for developers to work, while also allowing Claude to run more autonomously and with fewer permission prompts. In our internal usage, we've found that sandboxing safely reduces permission prompts by 84%. By defining set boundaries within which Claude can work freely, they increase security and agency.

为应对这一点，我们在 Claude Code 中基于沙箱引入了两个新特性，二者都旨在为开发者提供更安全的工作场所，同时让 Claude 能更自主地运行、减少权限确认。在我们的内部使用中，沙箱安全地减少了 84% 的权限确认。通过定义 Claude 可自由工作的边界，它们同时提升了安全性与自主性。

## Keeping users secure on Claude Code

## 在 Claude Code 上保障用户安全

Claude Code runs on a permission-based model: by default, it's read-only, which means it asks for permission before making modifications or running any commands. There are some exceptions to this: we auto-allow safe commands like echo or cat, but most operations still need explicit approval.

Claude Code 运行在基于权限的模型上：默认是只读的，意味着它在做修改或运行命令前会请求许可。这里有一些例外：我们会自动放行 echo、cat 这类安全命令，但多数操作仍需明确批准。

Constantly clicking "approve" slows down development cycles and can lead to 'approval fatigue', where users might not pay close attention to what they're approving, and in turn making development less safe.

不停地点"批准"拖慢了开发周期，并可能导致"审批疲劳"——用户不再仔细关注自己批准了什么，反过来又让开发更不安全。

To address this, we launched sandboxing for Claude Code.

为应对这一点，我们为 Claude Code 推出了沙箱。

## Sandboxing: a safer and more autonomous approach

## 沙箱：更安全、更自主的方式

Sandboxing creates pre-defined boundaries within which Claude can work more freely, instead of asking for permission for each action. With sandboxing enabled, you get drastically fewer permission prompts and increased safety.

沙箱创建了预定义的边界，在其中 Claude 可以更自由地工作，而不必为每个动作请求许可。启用沙箱后，权限确认大幅减少，安全性提升。

Our approach to sandboxing is built on top of operating system-level features to enable two boundaries:

我们的沙箱方法构建在操作系统级特性之上，启用两道边界：

1.  **Filesystem isolation**, which ensures that Claude can only access or modify specific directories. This is particularly important in preventing a prompt-injected Claude from modifying sensitive system files.
2.  **Network isolation**, which ensures that Claude can only connect to approved servers. This prevents a prompt-injected Claude from leaking sensitive information or downloading malware.

1.  **文件系统隔离**：确保 Claude 只能访问或修改特定目录。这对防止被提示注入的 Claude 修改敏感系统文件尤为重要。
2.  **网络隔离**：确保 Claude 只能连接被批准的服务器。这能防止被提示注入的 Claude 泄露敏感信息或下载恶意软件。

It is worth noting that effective sandboxing requires _both_ filesystem and network isolation. Without network isolation, a compromised agent could exfiltrate sensitive files like SSH keys; without filesystem isolation, a compromised agent could easily escape the sandbox and gain network access. It’s by using both techniques that we can provide a safer and faster agentic experience for Claude Code users.

值得注意的是，有效的沙箱需要*同时*具备文件系统隔离与网络隔离。没有网络隔离，被攻陷的智能体可能外泄 SSH 密钥等敏感文件；没有文件系统隔离，被攻陷的智能体可能轻易逃出沙箱获得网络访问。正是两者结合，才能为 Claude Code 用户提供更安全、更快速的智能体体验。

## Two new sandboxing features in Claude Code

## Claude Code 中的两个新沙箱特性

### Sandboxed bash tool: safe bash execution without permission prompts

### 沙箱 bash 工具：无需权限确认的 bash 安全执行

We're introducing [a new sandbox runtime](https://docs.claude.com/en/docs/claude-code/sandboxing), available in beta as a research preview, that lets you define exactly which directories and network hosts your agent can access, without the overhead of spinning up and managing a container. This can be used to sandbox arbitrary processes, agents and MCP servers. It is also available as [an open source research preview](https://github.com/anthropic-experimental/sandbox-runtime).

我们推出[一个新的沙箱运行时](https://docs.claude.com/en/docs/claude-code/sandboxing)（beta 研究预览），让你能精确定义智能体可访问哪些目录和网络主机，而无需承担启动和管理容器的开销。它可用于沙箱化任意进程、智能体与 MCP 服务器。它也作为[开源研究预览](https://github.com/anthropic-experimental/sandbox-runtime)提供。

In Claude Code, we use this runtime to sandbox the bash tool, which allows Claude to run commands within the defined limits you set. Inside the safe sandbox, Claude can run more autonomously and safely execute commands without permission prompts. If Claude tries to access something _outside_ of the sandbox, you'll be notified immediately, and can choose whether or not to allow it.

在 Claude Code 中，我们用这个运行时来沙箱化 bash 工具，让 Claude 在你设定的限制内运行命令。在安全的沙箱内部，Claude 能更自主地安全运行命令而无需权限确认。如果 Claude 试图访问沙箱*之外*的东西，你会立即收到通知，并可以决定是否允许。

We’ve built this on top of OS level primitives such as [Linux bubblewrap](https://github.com/containers/bubblewrap) and MacOS seatbelt to enforce these restrictions at the OS level. They cover not just Claude Code's direct interactions, but also any scripts, programs, or subprocesses that are spawned by the command.

我们把它构建在 [Linux bubblewrap](https://github.com/containers/bubblewrap)、MacOS seatbelt 等 OS 级原语之上，在操作系统层面强制执行这些限制。它们覆盖的不仅是 Claude Code 的直接交互，还包括命令所派生的任何脚本、程序或子进程。

As described above, this sandbox enforces both:

如上所述，这个沙箱同时强制执行：

1.  **Filesystem isolation,** by allowing read and write access to the current working directory, but blocking the modification of any files outside of it.
2.  **Network isolation,** by only allowing internet access through a unix domain socket connected to a proxy server running outside the sandbox. This proxy server enforces restrictions on the domains that a process can connect to, and handles user confirmation for newly requested domains. And if you’d like further-increased security, we also support customizing this proxy to enforce arbitrary rules on outgoing traffic.

1.  **文件系统隔离**：允许对当前工作目录的读写访问，但阻止对其外部任何文件的修改。
2.  **网络隔离**：只允许通过一个连接到沙箱外代理服务器的 unix domain socket 访问互联网。该代理服务器对进程可连接的域名施加限制，并处理对新请求域名的用户确认。若你想进一步提升安全性，我们还支持自定义该代理，对出站流量施加任意规则。

Both components are configurable: you can easily choose to allow or disallow specific file paths or domains.

两个组件都可配置：你可以轻松选择允许或禁止特定文件路径或域名。

Sandboxing ensures that even a successful prompt injection is fully isolated, and cannot impact overall user security. This way, a compromised Claude Code can't steal your SSH keys, or phone home to an attacker's server.

沙箱确保即便是成功的提示注入也被完全隔离，无法影响整体用户安全。这样，被攻陷的 Claude Code 无法窃取你的 SSH 密钥，也无法回连到攻击者的服务器。

To get started with this feature, run /sandbox in Claude Code and check out [more technical details](https://docs.claude.com/en/docs/claude-code/sandboxing) about our security model.

要开始使用此特性，在 Claude Code 中运行 /sandbox，并查看关于我们安全模型的[更多技术细节](https://docs.claude.com/en/docs/claude-code/sandboxing)。

To make it easier for other teams to build safer agents, we have [open sourced](https://github.com/anthropic-experimental/sandbox-runtime) this feature. We believe that others should consider adopting this technology for their own agents in order to enhance the security posture of their agents.

为让其他团队更容易构建更安全的智能体，我们[开源了](https://github.com/anthropic-experimental/sandbox-runtime)这一特性。我们相信其他团队也应考虑为自己的智能体采用这项技术，以提升其智能体的安全姿态。

### Claude Code on the web: running Claude Code securely in the cloud

### 云端 Claude Code：在云中安全运行 Claude Code

Today, we're also releasing [Claude Code on the web](https://docs.claude.com/en/docs/claude-code/claude-code-on-the-web) enabling users to run Claude Code in an isolated sandbox in the cloud. Claude Code on the web executes each Claude Code session in an isolated sandbox where it has full access to its server in a safe and secure way. We've designed this sandbox to ensure that sensitive credentials (such as git credentials or signing keys) are never inside the sandbox with Claude Code. This way, even if the code running in the sandbox is compromised, the user is kept safe from further harm.

今天，我们还发布[云端 Claude Code](https://docs.claude.com/en/docs/claude-code/claude-code-on-the-web)，让用户能在云中隔离沙箱里运行 Claude Code。云端 Claude Code 在隔离沙箱中执行每个会话，以安全方式完全访问其服务器。我们设计的这个沙箱确保敏感凭据（如 git 凭据或签名密钥）绝不与 Claude Code 同处沙箱内。这样，即便沙箱中运行的代码被攻陷，用户也不会遭到进一步伤害。

Claude Code on the web uses a custom proxy service that transparently handles all git interactions. Inside the sandbox, the git client authenticates to this service with a custom-built scoped credential. The proxy verifies this credential and the contents of the git interaction (e.g. ensuring it is only pushing to the configured branch), then attaches the right authentication token before sending the request to GitHub.

云端 Claude Code 使用一个自定义代理服务透明地处理所有 git 交互。在沙箱内部，git 客户端用定制的作用域凭据向该服务认证。代理验证该凭据与 git 交互的内容（例如确保只推送到配置的分支），然后附加正确的认证 token 再向 GitHub 发送请求。

## Getting started

## 开始使用

Our new sandboxed bash tool and Claude Code on the web offer substantial improvements in both security and productivity for developers using Claude for their engineering work.

我们新的沙箱 bash 工具与云端 Claude Code，为使用 Claude 做工程工作的开发者在安全性和生产力上都带来了显著提升。

To get started with these tools:

开始使用这些工具：

1.  Run `/sandbox` in Claude and check out [our docs](https://docs.claude.com/en/docs/claude-code/sandboxing) on how to configure this sandbox.
2.  Go to [claude.com/code](http://claude.ai/redirect/website.v1.7661c61a-15b2-4c13-b882-d127a232973f/code) to try out Claude Code on the web.

1.  在 Claude 中运行 `/sandbox`，并查看关于如何配置该沙箱的[文档](https://docs.claude.com/en/docs/claude-code/sandboxing)。
2.  前往 [claude.com/code](http://claude.ai/redirect/website.v1.7661c61a-15b2-4c13-b882-d127a232973f/code) 试用云端 Claude Code。

Or, if you're building your own agents, check out our [open-sourced sandboxing code](https://github.com/anthropic-experimental/sandbox-runtime), and consider integrating it into your work. We look forward to seeing what you build.

或者，如果你在构建自己的智能体，查看我们[开源的沙箱代码](https://github.com/anthropic-experimental/sandbox-runtime)，并考虑把它集成到你的工作中。我们期待看到你构建的东西。
