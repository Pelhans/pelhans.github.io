---
title: "Harness Engineering for Self-Improvement（Agent Harness 综述翻译）"
date: 2026-08-11
layout: post
liquid: false
tags: [语雀, 计算机, Agent Harness 论文解读]
categories: [计算机]
yuque: true
---

# Harness Engineering for Self-Improvement

> 原文作者：Lilian Weng（Lil'Log）| 发布于 2026-07-04 | 原文链接：https://lilianweng.github.io/posts/2026-07-04-harness/

# Harness Engineering for Self-Improvement

- Harness Design Patterns
                        
                
                    Pattern 1: Workflow Automation
- Pattern 2: File System as Persistent Memory
- Pattern 3: Sub-agent and Backend Jobs
- Case study: Coding Agent Harness
- Harness Layer vs Core Intelligence?

- Context Engineering
- Workflow Design
- Self-Improving Harness
- Evolutionary Search
- Joint Optimization with Model Weights

The concept of **recursive self-improvement (RSI)** dates back to [I. J. Good (1965)](https://philpapers.org/rec/GOOSCT), where he defined an "ultraintelligent machine" as a system that can surpass humans in all intellectual activities and design better machines to improve itself. [Yudkowsky (2008)](https://www.lesswrong.com/posts/JBadX7rwdcRFzGuju/recursive-self-improvement) used the phrase "recursive self-improvement" for a specific feedback loop: an AI uses its current intelligence to improve the cognitive machinery that produces its intelligence.

**递归式自我改进（RSI）** 的概念可以追溯至 [I. J. Good (1965)](https://philpapers.org/rec/GOOSCT)，他将"超智能机器"定义为一个在所有智力活动上都能超越人类、并能设计出更好的机器来改进自身的系统。[Yudkowsky (2008)](https://www.lesswrong.com/posts/JBadX7rwdcRFzGuju/recursive-self-improvement) 用"递归式自我改进"一词描述了一个特定的反馈环：AI 利用它当前的智能去改进产生其智能的认知机制。


This feedback loop in modern AI may indicate the model rewriting its own weights directly, or more broadly the model improves the *training pipeline* and the *deployment system*, which in turn enables a better successor model with improved performance across economically valuable tasks. The speed of research development in AI has been shown to drastically accelerated in frontier labs ([Anthropic](https://www.anthropic.com/institute/recursive-self-improvement); [OpenAI](https://openai.com/index/how-agents-are-transforming-work/)).

在现代 AI 中，这一反馈环可能意味着模型直接重写自身权重，或者更广义地说，模型改进*训练流水线*与*部署系统*，进而催生出在各类高经济价值任务上表现更优的后继模型。已有研究表明，前沿实验室中 AI 研发的速度因此得到了大幅加速（[Anthropic](https://www.anthropic.com/institute/recursive-self-improvement)；[OpenAI](https://openai.com/index/how-agents-are-transforming-work/)）。


I explicitly mention *"deployment system"* because the layer between the raw model and the real-world context seems to be as important as the model's raw intelligence (i.e. the evals right after pretraining). Harnesses are important components of AI deployment, as shown by successful coding agent products such as Claude Code and Codex. A **harness** is the system surrounding a base model that orchestrates execution and decides how the model thinks and plans, calls tools and acts, perceives and manages context, stores artifacts, and evaluates results.

我特意提到*"部署系统"*，是因为在原始模型与真实世界上下文之间的这层，其重要性似乎不亚于模型本身的原始智能（即预训练刚结束时的那些评测）。Harness 是 AI 部署的重要组成部分，Claude Code、Codex 等成功的编码 agent 产品便证明了这一点。**Harness** 是包裹在基座模型之外的系统，它负责编排执行、决定模型如何思考与规划、调用工具并采取行动、感知与管理上下文、存储产物并评估结果。


# Harness Design Patterns
Compared with [early agent frameworks](https://lilianweng.github.io/posts/2023-06-23-agent/), "agent = LLM + memory + tools + planning + action", harnesses engineering additionally include *workflow design (e.g. loop engineering), evaluation, permission controls, and persistent state management*. It is no longer only prompt templates, but closer to runtime and software system design: how the model observes, acts, memorizes, checks itself, and improves.

# Harness 设计模式
与[早期的 agent 框架](https://lilianweng.github.io/posts/2023-06-23-agent/)（"agent = LLM + 记忆 + 工具 + 规划 + 行动"）相比，harness 系统工程还额外包含*工作流设计（例如循环工程）、评估、权限控制以及持久化状态管理*。它不再只是提示词模板，而是更接近运行时与软件系统设计：模型如何观察、行动、记忆、自我检查并加以改进。


The design should be deliberately simple and generic to enable generalization, likely with reference to existing software engineering practices to benefit from prertaining knowlege. There is also a strong analogy between operating systems and harnesses. Similar to an OS, a harness should encapsulate complicated logic while keeping the interface simple. Meanwhile, configs, tool interfaces and other protocols may gradually become standardized across the industry.

设计应当刻意保持简单与通用，以便实现泛化，并很可能借鉴既有的软件工程实践、从而受益于预训练时学到的知识。Harness 与操作系统之间也存在很强的类比：如同 OS 一样，harness 应当封装复杂的内部逻辑，同时令接口保持简洁。与此同时，配置、工具接口以及其他协议，可能会在整个行业内逐渐走向标准化。


## Pattern 1: Workflow Automation
Defining a workflow in which the model can operate, test, and iterate is a key design for automation. Karpathy's autoresearch repo ([https://github.com/karpathy/autoresearch](https://github.com/karpathy/autoresearch)) is a clean example of how such a workflow can be constructed. A common workflow follows a goal-oriented loop of plan, execute, observe/test, improve, and execute again *until* the goal is achieved. The process may trigger proactive requests to users for clarity in task specification or execution preference.

## 模式 1：工作流自动化
为模型定义一套能够运行、测试并迭代的工作流，是自动化的关键设计。Karpathy 的 autoresearch 仓库（[https://github.com/karpathy/autoresearch](https://github.com/karpathy/autoresearch)）清晰地展示了这类工作流该如何构建。一种常见的工作流遵循一个以目标为导向的循环：规划、执行、观察/测试、改进，然后再次执行，*直到*目标达成。这一过程可能会主动向用户发起请求，以澄清任务定义或执行偏好。


![openai-agent-loop.png](/img/yuque/Agent%20harness/harness-2026-07-04/openai-agent-loop.png)
*图：A simplified Codex agent loop: the agent calls tools and tool responses affect the model's next generation.(Image source: OpenAI codex agent post)*

The workflow graph also emphasizes the model analyzing its own trajectories and failure cases and then iterating on its progress through an "agent runtime" rather than a static prompt template.

这张工作流图还强调：模型应分析自身的执行轨迹与失败案例，并通过"agent 运行时（agent runtime）"而非静态提示词模板来推进迭代。


## Pattern 2: File System as Persistent Memory
A recurring pattern in long-horizon agent systems is simple control over rich states and artifacts. A harness should not carry the entire workflow and all logs in context; instead, it should keep durable state in files. In long-horizon agentic rollout, artifacts such as experiment logs, code diffs, paper summaries, error traces, and past rollout trajectories often grow much longer than the context window that the model has trained for.

## 模式 2：以文件系统作为持久化记忆
在长周期（long-horizon）agent 系统中，一个反复出现的模式是：对丰富状态与产物进行简单可控的管理。Harness 不应把整个工作流和全部日志都塞进上下文，而应将持久化状态保存在文件里。在长周期 agent 的 rollout 过程中，实验日志、代码 diff、论文摘要、错误追踪以及过去的 rollout 轨迹等产物，其长度往往会超出模型训练时所用的上下文窗口。


Learning how to read, write, and edit the file system (commonly via `bash` commands) is a foundation skill for LLMs, and thus managing persistent memory in the simple form of files naturally benefits from improvements in core model capability.

学会读取、写入和编辑文件系统（通常通过 `bash` 命令）是 LLM 的一项基础能力，因此以文件这种简单形式来管理持久化记忆，自然能从模型核心能力的提升中获益。


## Pattern 3: Sub-agent and Backend Jobs
A harness can spawn multiple subagents to execute in parallel and monitor backend jobs. This is useful when the main agent needs to search multiple hypotheses, run experiments concurrently, or delegate isolated subtasks without polluting the main context. The parent agent then needs a small process manager: launch jobs, inspect logs, cancel failed runs, and merge results back into the main agent thread.

## 模式 3：子 agent 与后端任务
Harness 可以生成多个子 agent 并行执行，并监控后端任务。当主 agent 需要搜索多个假设、并发地运行实验，或将孤立的子任务委派出去而不污染主上下文时，这一模式非常有用。此时父 agent 需要一个轻量的进程管理器：启动任务、检查日志、取消失败运行，并将结果合并回主 agent 的线程。


The key design choice is to make parallelism explicit and inspectable. If subagent outputs only live in a transient chat context, they quickly become obselete and hidden. If they are stored as files, logs, and status records, the model can recover after interruptions and reason over its own execution history.

关键的设计选择是让并行过程变得**显式且可观测**。如果子 agent 的输出只存在于转瞬即逝的聊天上下文中，它们会很快变得过时、隐没。而如果它们以文件、日志和状态记录的形式存储，模型就能在中断后恢复，并能对自己的执行历史进行推理。


## Case study: Coding Agent Harness
The core interface of mainstream coding agents has become stabilized across Claude Code, Codex, OpenCode, and Cursor-style agents. They commonly use a loop like:

## 案例研究：编码 Agent Harness
主流编码 agent 的核心接口，在 Claude Code、Codex、OpenCode 以及 Cursor 风格的 agent 之间已经趋于稳定。它们通常采用如下的循环：


![coding-harness-loop.png](/img/yuque/Agent%20harness/harness-2026-07-04/coding-harness-loop.png)

With access to a set of tools, the coding agent is able to develop and debug issues in a given repository, similar to how human developers are equipped with IDEs.

通过一组工具，编码 agent 能够在给定的代码仓库中进行开发与调试，类似于人类开发者配备 IDE 的工作方式。


(Not a comprenhensive list; shown for demonstration. Read [this](https://github.com/yasasbanukaofficial/claude-code) if interested.)

（并非详尽列表，仅用于演示。如有兴趣可阅读[此文](https://github.com/yasasbanukaofficial/claude-code)。）


## Harness Layer vs Core Intelligence?
It is hard to forecast how much the future of RSI will rely on harness engineering, but the near-term path of RSI is unlikely to start as a model directly rewriting its weights. My prediction of a practical near-term path is:

## Harness 层与核心智能的关系？
很难预测 RSI 的未来在多大程度上会依赖 harness 工程，但 RSI 的近期路径不太可能从"模型直接重写自身权重"起步。我对一条务实的近期路径的预测是：


- Harness engineering will evolve in the direction of meta-methodology (i.e. improving the machinery for getting better answers, not just improving the answer itself). The harness system itself becomes an optimization target, with fewer heuristic rules and more general mechanisms.
- In turn, mature harnesses enable auto-research for model self-improvement loop and smarter models prevents harnesses from overengineering and keep the system sustainable.

- Harness 工程将朝着"元方法论"的方向演进（即改进获取更好答案的机制，而不只是改进答案本身）。Harness 系统本身成为一个被优化的对象，启发式规则越来越少，通用机制越来越多。
- 反过来，成熟的 harness 能让自动研究服务于模型的自我改进循环，而更聪明的模型又能防止 harness 过度工程化，使系统保持可持续。

Eventually it is possible that many harness improvements will be *internalized* into core model behavior, but the interface with external context and tools should remain. We have seen a softer version of this pattern with [prompt engineering](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/): manual prompt tricks became less central as instruction tuning and model reasoning improved, but *the need to specify goals, constraints, context, and evaluation did not disappear*.

最终，许多 harness 的改进有可能被*内化*进模型的核心行为中，但与外部上下文和工具的接口应当保留下来。我们在[提示词工程](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/)上已经见过这一模式的温和版本：随着指令微调和模型推理能力的提升，手工提示技巧变得不再那么核心，但*明确目标、约束、上下文与评估的需求从未消失*。


# Harness Optimization
The progression in the object being optimized in the harness system is roughly: instruction [prompts](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/) → structured context → workflow → harness code → optimizer code. As the model becomes more intelligent and powerful, we move toward more complex targets and generic methods.

# Harness 的优化
在 harness 系统中，被优化对象大致经历了这样的演进：指令 [prompts](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/) → 结构化上下文 → 工作流 → harness 代码 → 优化器代码。随着模型变得愈发智能和强大，我们转向更复杂的优化目标和更通用的方法。


## Context Engineering
Simply appending all the tool responses and model generations into the context can quickly grow out of control as the agentic job horizon increases significantly. Context management is a layer to construct a more structed and concise context for LLM and manage persistant states. There is no doubt that long-context research will keep on making progress but at the moment long-context intelligence and context engineering sometime intertwines.

## 上下文工程
随着 agent 任务周期显著拉长，简单地将所有工具响应和模型生成内容都塞进上下文，会迅速失控。上下文管理是一个用于为 LLM 构建更结构化、更精简的上下文，并管理持久化状态的层。毫无疑问，长上下文研究会持续取得进展，但就目前而言，长上下文智能与上下文工程有时仍交织在一起。


**Agentic Context Engineering** (ACE; [Zhang et al. 2025](https://arxiv.org/abs/2510.04618)) treats context as an evolving playbook rather than an increasingly lengthening prompt. It has three components to maintain one context playbook of bullet points, each with an identifier and a description.

- Generator: produces task trajectories, with reference to bullet points.
- Reflector: distills insights from successful and failed trajectories.
- Curator: updates the structured context with incremental, itemized entries.

**智能体上下文工程**（ACE；[Zhang et al. 2025](https://arxiv.org/abs/2510.04618)）将上下文视为一本不断演进的"操作手册"，而不是一段越来越长的提示词。它由三个组件构成，维护着一份由要点组成的上下文手册，每个要点都带有标识符和描述。

- 生成器（Generator）：参考这些要点，生成任务执行轨迹。
- 反思器（Reflector）：从成功与失败的轨迹中提炼洞见。
- 策展器（Curator）：以增量、逐项的方式更新结构化上下文。

![ace.png](/img/yuque/Agent%20harness/harness-2026-07-04/ace.png)
*图：Agentic Context Engineering (ACE) 框架。(Image source: Zhang et al. 2025)*

To prevent context collapse and brevity bias during iterative rewrites, one key design choice in ACE is that the curator does not rewrite a full prompt blob. It instead outputs a collection of structured, itemized bullets in the form of (identifier, description), and these bullets are merged into a structured context logbook with deterministic logic. The context items are refined and deduplicated periodically.

为防止在迭代重写过程中出现上下文坍缩和"简短偏见"，ACE 的一个关键设计选择是：策展器不会重写整段提示词，而是以（标识符，描述）的形式输出一组结构化的逐项要点，这些要点再以确定性逻辑合并进结构化的上下文日志本中。上下文条目会被定期精炼与去重。


The fact that ACE learns insights from rollouts helps us move toward self-managed memory, but the update rules and the overall workflow are still handcrafted. To move toward a more self-improving loop, **Meta Context Engineering** (MCE; [Ye et al. 2026](https://arxiv.org/abs/2601.21557)) separates the mechanism (how to manage context) from the artifact content (what is in context), running skill evolution at the meta-optimization level and context optimization at the base level.

ACE 从 rollout 中学习洞见，这一事实帮助我们迈向自我管理的记忆，但其更新规则和整体工作流仍是手工设计的。为了走向更具自我改进性质的循环，**元上下文工程**（MCE；[Ye et al. 2026](https://arxiv.org/abs/2601.21557)）将"机制（如何管理上下文）"与"产物内容（上下文中放什么）"分离开来，在元优化层运行技能演进，在基础层运行上下文优化。


An MCE skill $s \in \mathcal{S}$ defines a context function $c_s=(\rho_s,F_s)$ and maps an input $x$ to context $c = F_s(x;\rho_s)$, where:

一个 MCE 技能 $s \in \mathcal{S}$ 定义了一个上下文函数 $c_s=(\rho_s,F_s)$，并将输入 $x$ 映射为上下文 $c = F_s(x;\rho_s)$，其中：


- $\rho_s = \{\rho_1,\dots,\rho_m\}$ are static components (prompts, knowledge bases, code libraries).
- $F_s = \{F_1,\dots,F_k\}$ are dynamic operators (search, selection, filtering, formatting).

- $\rho_s = \{\rho_1,\dots,\rho_m\}$ 是静态组件（提示词、知识库、代码库）。
- $F_s = \{F_1,\dots,F_k\}$ 是动态算子（搜索、选择、过滤、格式化）。

The bi-level optimization is to find the best context $c_s^*$ given skill $s$ on the training data, while the outer loop finds the optimal skill that provides the best performance on the validation set:

其双层优化的目标是：在训练数据上给定技能 $s$ 时找到最优上下文 $c_s^*$，而外层循环则寻找能在验证集上取得最佳表现的技能：

The skill database tracks the history of previous skills, context functions and eval metrics $\mathcal{H}_{k-1} = \{(s_i,c_i,J_i^\text{train}, J_i^\text{val})\}_{i=1}^{k-1}$. A meta-level agent performs agentic [crossover](https://en.wikipedia.org/wiki/Crossover_(evolutionary_algorithm)) over prior skills to create a new skill given a task $\tau$: $s_k=\text{crossover}(\tau,\mathcal{H}_{k-1})$.

技能数据库记录了以往技能、上下文函数和评估指标的历史 $\mathcal{H}_{k-1} = \{(s_i,c_i,J_i^\text{train}, J_i^\text{val})\}_{i=1}^{k-1}$。一个元层 agent 会基于先前的技能进行智能体式的[交叉（crossover）](https://en.wikipedia.org/wiki/Crossover_(evolutionary_algorithm))，从而在给定任务 $\tau$ 时创造出新技能：$s_k=\text{crossover}(\tau,\mathcal{H}_{k-1})$。


Then a base-level context engineer executes the skill $s_k$ and learns the context function from rollout feedback $\mathcal{R}_k$, guided by the current skill: $c_k=\text{engineer}(\tau,s_k;c_{k-1}^*,\mathcal{R}_k)$.

随后，一个基础层的上下文工程师执行技能 $s_k$，并在当前技能的引导下，从 rollout 反馈 $\mathcal{R}_k$ 中学习上下文函数：$c_k=\text{engineer}(\tau,s_k;c_{k-1}^*,\mathcal{R}_k)$。


![mce.png](/img/yuque/Agent%20harness/harness-2026-07-04/mce.png)
*图：Meta Context Engineering (MCE) 框架：元层技能演进在上下文管理机制上搜索，基础层则优化任务上下文。(Image source: Ye et al. 2026)*

MCE does not enforce a heuristic rule for how to structure context as ACE does. It uses *free-form skills* to store the most important knowledge for a task, and evolves the skill and the skill-conditioned context iteratively together. Implementation-wise, a context function $c$ is instantiated as a collection of files in a dedicated directory, including both static (`skill.md`) and dynamic (context and data rollouts) components. Both meta-level and base-level optimization are executed in agentic coding envs with a standard tool set,

MCE 不像 ACE 那样强制规定如何结构化上下文的启发式规则。它使用*自由形式的技能*来存储任务最重要的知识，并让技能与"技能 conditioned 的上下文"一起迭代式地演进。在实现上，上下文函数 $c$ 被实例化为一个专用目录下的文件集合，既包括静态部分（`skill.md`），也包括动态部分（上下文与数据 rollout）。元层与基础层的优化都在具备标准工具集的智能体编码环境中执行。


**Meta-Harness** ([Lee et al. 2026](https://arxiv.org/abs/2603.28052)) moves another level deeper: the optimized object is the *code* that determines and optimizes what information should be stored, retrieved, and presented to the model. "Meta-" in its name means it is a harness for optimizing harnesses.

**元 Harness**（[Lee et al. 2026](https://arxiv.org/abs/2603.28052)）又往更深一层推进：被优化的对象是*代码*——这段代码决定并优化"哪些信息应当被存储、检索并呈现给模型"。"Meta-" 在它的名字中意味着：它是一个用于优化 harness 的 harness。

![meta-harness-outer-loop.png](/img/yuque/Agent%20harness/harness-2026-07-04/meta-harness-outer-loop.png)
*图：Meta-Harness 的外层循环优化算法。(Image source: Lee et al. 2026)*

The proposer for creating a new harness is itself a coding agent and the final output is a collection of harness candidates on the Pareto frontier.

提出"创建新 harness"这个动作的，本身就是一个编码 agent，而最终的输出是帕累托前沿（Pareto frontier）上的一组 harness 候选。


- The entire execution history is accessible via a file system, and thus the coding agent uses commands like grep or cat to read through it instead of shoveling everything into a single prompt context.
- The proposed harness is a dictionary in the file system containing its own source code, scores, rollout trajectories, and state updates.
- The mete-harness loop iteratively creates new harnesses, and only qualified ones are kept.

- 整个执行历史都可通过文件系统访问，因此编码 agent 使用 grep、cat 等命令去翻阅它，而不是把一切都塞进单一提示词上下文。
- 被提出的 harness 是文件系统中的一本"字典"，包含它自己的源代码、分数、rollout 轨迹与状态更新。
- 元 harness 循环会迭代地创建新 harness，并且只保留合格的那些。

![meta-harness.png](/img/yuque/Agent%20harness/harness-2026-07-04/meta-harness.png)
*图：Meta-Harness 在（左）少量迭代下的文本分类、（右）TerminalBench-2 上的表现。注意 TerminalBench-2 实验的搜索是从 Terminus-KIRA 与 Terminus-2 这两个很强的 harness 初始化的。(Image source: Lee et al. 2026)*

Still, the important lesson is clear: once harness design becomes an executable search space, a strong coding agent can exploit the same design space human engineers use.

不过，重要的一课已经清晰：一旦 harness 设计变成了一个可执行的搜索空间，一个强大的编码 agent 就能利用与人类工程师相同的设计空间。


## Workflow Design
Workflow design in harness engineering can be handcrafted by domain experts. Taking auto-research as an example, various frameworks have been proposed and tested. The **AI Scientist** system ([Lu et al. 2026](https://www.nature.com/articles/s41586-026-10265-5)) builds a pipeline to propose research ideas, write code, run experiments, analyze results, write a manuscript, and perform peer review. [Meng et al. (2026)](https://arxiv.org/abs/2605.26340) make verifiability the central design constraint in **ScientistOne**, where every claim (citation, numerical, methodological, conclusion) must trace to an evidence source and is audited by Chain-of-Evidence checks.

## 工作流设计
Harness 工程中的工作流设计，可以由领域专家手工完成。以自动研究为例，已有多种框架被提出并测试。**AI Scientist** 系统（[Lu et al. 2026](https://www.nature.com/articles/s41586-026-10265-5)）构建了一条流水线，用于提出研究想法、编写代码、运行实验、分析结果、撰写论文并进行同行评审。[Meng et al. (2026)](https://arxiv.org/abs/2605.26340) 在 **ScientistOne** 中将"可验证性"作为核心设计约束，其中每一条主张（引用、数值、方法、结论）都必须能追溯到证据来源，并由 Chain-of-Evidence 检查来审计。


![ai-scientist.png](/img/yuque/Agent%20harness/harness-2026-07-04/ai-scientist.png)
*图：AI Scientist 的流水线，涵盖想法生成、实验、论文写作与评审。(Image source: Lu et al. 2026)*

The **Autodata** agent ([Kulikov et al. 2026](https://arxiv.org/abs/2606.25996)) is designed to work as a data scientist for generating training and evaluation data. The main agent manages a *challenger* that proposes problems, a *weak solver*, a *strong solver*, and a *verifier/judge*, aiming to synthesize data at the "just right" level of difficulty, meaning that the strong solver succeeds but the weak solver fails.

**Autodata** agent（[Kulikov et al. 2026](https://arxiv.org/abs/2606.25996)）被设计成一名数据科学家，用于生成训练与评估数据。主 agent 管理着：提出问题的 *challenger*（挑战者）、*weak solver*（弱解算器）、*strong solver*（强解算器）以及 *verifier/judge*（验证者/评判者），目标是合成出难度"刚刚好"的数据——即强解算器能成功、而弱解算器会失败。


In Autodata, the challenger prompt is updated iteratively according to feedback from the solvers and verifier. The limitation here is that synthesized tasks are used to fine-tune weak solvers but not strong solvers; if the loop cannot iteratively improve the strong model, it is more like indirect distillation over a generated prompt distribution, with less RSI flavor.

在 Autodata 中，challenger 的提示词会根据来自解算器与验证者的反馈进行迭代更新。其局限在于：合成出的任务被用来微调弱解算器，而非强解算器；如果这个循环无法迭代地改进强模型，那它就更像是在生成的提示词分布上做间接蒸馏，RSI 的意味较弱。


![autodata.png](/img/yuque/Agent%20harness/harness-2026-07-04/autodata.png)
*图：Autodata 围绕 challenger、solver、verifier 角色生成合成训练与评估数据的智能体工作流设计。(Image source: Kulikov et al. 2026)*

The design space for workflow is *enormous*, and naturally we can think of workflow design as a search problem, and therefore we should be able to find good solutions by algorithms rather than only manually craft them. Following this direction, **Automated Design of Agentic Systems** (ADAS; [Hu et al. 2025](https://arxiv.org/abs/2408.08435)) formulates agent design itself as an optimization problem, "meta-agent search" where a meta-agent proposes new designs of agentic workflows.

工作流的设计空间*极其庞大*，我们自然可以把工作流设计视为一个搜索问题，从而应当能够用算法而非仅靠手工来找到好的解。沿着这一方向，**智能体系统的自动化设计**（ADAS；[Hu et al. 2025](https://arxiv.org/abs/2408.08435)）将 agent 设计本身形式化为一个优化问题——即"元 agent 搜索"，由元 agent 提出智能体工作流的新设计。


- Initialize an archive of agentic workflows with simple agents such as CoT and self-refine.
- Ask a meta-agent to program new agents, all in code, inspired by existing solutions in the archive.

- 用 CoT、self-refine 等简单 agent 初始化一个智能体工作流档案库（archive）。
- 让元 agent 受档案库中已有方案的启发，以代码形式为新的 agent 编程。

The meta-agent first generates a high-level description of the new workflow, and then implements it in code.

元 agent 首先生成新工作流的高层描述，再以代码形式实现它。

- The draft program then goes through two self-refine steps (i.e. ask the model to provide feedback and then ask the same model to refine the previously generated outputs based on the feedback; Madaan et al. 2023) by the meta-agent to check its novelty.
- Evaluate each new candidate and add successful ones back to the archive.
- Repeat steps 2-3 until the maximum iteration count is reached.

- 草稿程序随后由元 agent 经历两次 self-refine 步骤（即先让模型提供反馈，再让同一模型基于反馈精炼先前生成的输出；Madaan et al. 2023），以检查其新颖性。
- 评估每个新候选，将成功者加回档案库。
- 重复第 2–3 步，直到达到最大迭代次数。

![adas.png](/img/yuque/Agent%20harness/harness-2026-07-04/adas.png)
*图：Automated Design of Agentic Systems (ADAS) 示意图。(Image source: Hu et al. 2025)*

**AFlow** ([Zhang et al. 2025](https://arxiv.org/abs/2410.10762)) represents an agentic workflow as a graph, where nodes represent LLM-invoking actions and edges implement logical operations in code. The workflow optimization relies on [MCTS](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search) (Monte Carlo Tree Search):

- Initialize the starting workflow $W_0$ in the tree with a template.
- Select a workflow node using a soft mixture of score and uniform exploration.
- Expand it by asking an LLM to produce a modified workflow conditioned on its evaluation performance.
- Execute and evaluate the new workflow.
- Add it back to the tree if the new workflow shows improvement within a budget of $N$ rounds.
- Repeat steps 2-5 and stop when the top-$k$ average score plateaus or hit the budget.

![aflow.png](/img/yuque/Agent%20harness/harness-2026-07-04/aflow.png)
*图：AFlow optimization process over a tree of workflow candidates. (Image source: Zhang et al. 2025)*

Experiments of AFlow in QA, code, and math tasks showed decent improvement of AFlow over manually designed workflows and ADAS.

AFlow 在问答、代码与数学任务上的实验表明，它相比手工设计的工作流和 ADAS 都有不错的提升。


![aflow-exp.png](/img/yuque/Agent%20harness/harness-2026-07-04/aflow-exp.png)
*图：AFlow 与手工方法及 ADAS 的对比实验。(Image source: Zhang et al. 2025)*

## Self-Improving Harness
Either context engineering or workflow design is only one part of a harness. We need to search through the entire design space and optimize context-management logic, workflow, permissions, and many other harness components together. As we have seen in work like Meta-Harness, ADAS, and AFlow, **✨code✨** is a **universal language** for defining programs and systems. In simple words, a harness is code that programs how prompts, tool calls, subagents, control flow, memory, and workflow logic work together. If an LLM can optimize the code that executes agents, it can access a *much larger design space* than hand-written prompts.

## 自我改进的 Harness
无论是上下文工程还是工作流设计，都只是 harness 的一部分。我们需要搜索整个设计空间，把上下文管理逻辑、工作流、权限以及许多其他 harness 组件放在一起优化。正如我们在 Meta-Harness、ADAS、AFlow 等工作中看到的那样，**✨代码✨**是定义程序与系统的**通用语言**。简而言之，harness 就是一段代码，它编排了提示词、工具调用、子 agent、控制流、记忆与工作流逻辑如何协同工作。如果一个 LLM 能够优化执行 agent 的那段代码，它就能触及比手写提示词*大得多的设计空间*。


**Self-Taught Optimizer** (STOP; [Zelikman et al. 2023](https://arxiv.org/abs/2310.02304)) is one of the early examples of recursive scaffolding improvement. A seed improver $I_0$ at step $t=0$ takes an initial solution $s$, a utility function $u$, and a black-box language model $M$, and returns an improved solution $s'$, that is, $s' = I(u, s; M)$. The goal of STOP is not directly to improve $s$ but *to improve the improver $I$ itself*.

**自我教导优化器**（STOP；[Zelikman et al. 2023](https://arxiv.org/abs/2310.02304)）是递归式脚手架改进的早期例子之一。在步 $t=0$ 时，一个种子改进器 $I_0$ 接收初始解 $s$、效用函数 $u$ 和一个黑盒语言模型 $M$，并返回改进后的解 $s'$，即 $s' = I(u, s; M)$。STOP 的目标并非直接改进 $s$，而是*改进改进器 $I$ 本身*。

First, let's define the meta-utility as the average utility of a given improver function $I$ over a collection of downstream tasks $\mathcal{D}$:

首先，我们将元效用（meta-utility）定义为：给定改进器函数 $I$ 在一组下游任务集合 $\mathcal{D}$ 上的平均效用：


Because improving the improver function is an optimization problem itself, we can recursively get a new version of $I_t$ based on $I_{t-1}$'s performance measured by meta-utility via a self-improvement update:

由于改进"改进器函数"本身也是一个优化问题，我们可以通过自改进式的更新，基于 $I_{t-1}$ 在元效用上的表现，递归地得到新版本的 $I_t$：


![STOP-algo.png](/img/yuque/Agent%20harness/harness-2026-07-04/STOP-algo.png)
*图：Self-Taught Optimizer (STOP) 算法。(Image source: Zelikman et al. 2023)*

In their experiments, the improved improver discovered various strategies, such as genetic algorithms, decomposing and improving parts, multi-armed prompt bandits, simulated annealing, varying temperature, and beam/tree search. This is analogous to how a harness workflow can be represented as an object for optimization.

在他们的实验中，改进后的改进器发现了多种策略，例如遗传算法、分解并改进局部、多臂提示词赌博机（multi-armed prompt bandits）、模拟退火、调整温度，以及 beam/tree 搜索。这与"harness 工作流可被表示为一个待优化对象"的情形是类似的。


![STOP-patterns.png](/img/yuque/Agent%20harness/harness-2026-07-04/STOP-patterns.png)
*图：STOP 发现的自我改进策略示例。(Image source: Zelikman et al. 2023)*

A *cautionary* result in Zelikman et al. (2023)'s findings is that STOP improved mean downstream performance across iterations with GPT-4 but degraded with weaker models like GPT-3.5 and Mixtral. Recursive structure alone is not enough. The base model must be *capable enough* to improve the mechanism. This implies that harness improvement enables better deployment of the model but intelligence is still the core.

Zelikman 等人（2023）研究中一个*警示性*的结果是：STOP 在 GPT-4 上能跨迭代提升下游平均表现，但在 GPT-3.5、Mixtral 等较弱模型上反而退化。仅有递归结构是不够的，基座模型必须*足够有能力*才能改进这套机制。这意味着 harness 改进能让模型部署得更好，但智能本身仍是核心。


[Lin et al. (2026)](https://arxiv.org/abs/2605.30621) investigated the dependency of harness evolution on model capabilities in more details. They disentangled two axes: (1) *harness-updating* refers to the capability of producing useful harness edits and (2) *harness-benefit* denotes the capability of utilizing the updated harness, to achieve better task solving. Interestingly a range of model of different sizes and core intelligence, from Qwen3.5-9B to Claude Opus 4.6, were observed in their experiments to show similar harness updating capability; the 9B harness proposer/evolver is able to write a skill procedurally isomorphic to Opus. To best utilize a harness, a model needs to invoke skills/tools correctly and timely and be good at long-horizon instruction following.

[Lin et al. (2026)](https://arxiv.org/abs/2605.30621) 更细致地研究了 harness 演进对模型能力的依赖。他们把两个轴拆开：(1) *harness-updating*（harness 更新能力）指产出有用 harness 改动的能力；(2) *harness-benefit*（harness 收益能力）指利用更新后的 harness 以更好地求解任务的能力。有趣的是，在他们的实验中，从 Qwen3.5-9B 到 Claude Opus 4.6，一系列不同规模和核心智能的模型展现出了相近的 harness 更新能力——9B 的 harness 提出者/演进者甚至能写出与 Opus 在过程上同构（isomorphic）的技能。要最好地利用一个 harness，模型需要正确、及时地调用技能/工具，并擅长长周期指令跟随。


![harness-update.png](/img/yuque/Agent%20harness/harness-2026-07-04/harness-update.png)
*图：主要结果：(A) harness 更新能力在 Qwen2-32B 到 Opus 4.6 的一系列模型上持平；(B) harness 收益能力呈非单调性，中等层级模型受益最大。(Image source: Lin et al. 2026)*

A more recent work, **Self-Harness** ([Zhang et al. 2026](https://arxiv.org/abs/2606.09498)), relies on LLM agents to improve their own harness via a propose-evaluate-accept loop.

一项更新的工作 **Self-Harness**（[Zhang et al. 2026](https://arxiv.org/abs/2606.09498)）依靠 LLM agent，通过"提出-评估-接受"的循环来改进它们自己的 harness。


![self-harness.png](/img/yuque/Agent%20harness/harness-2026-07-04/self-harness.png)
*图：Self-Harness 通过"弱点挖掘—有界 harness 提出—验证"的循环来更新 harness。(Image source: Zhang et al. 2026)*

The loop in Self-Harness has three stages:

Self-Harness 的循环包含三个阶段：


- Weakness mining: cluster failures into verifier-grounded failure patterns.

- 弱点挖掘（Weakness mining）：将失败聚合成以验证器为依据的失败模式。

The current harness $h_t$ is used to evaluate on tasks and execution traces are collected for analysis.

当前的 harness $h_t$ 被用来在任务上评估，并收集执行轨迹以供分析。

- Note that two runs can share the same verifier outcome in the error logs on the surface, such as timeout or missing artifact, while having different causal mechanisms. Therefore we need a failure record of rich information, containing the terminal verifier-level cause, the causal status of the relevant agent behavior, and the abstract agent mechanism exposed by the trace, to uncover the root causes.
- Harness proposal: propose bounded harness edits based on mined failure patterns.

The same model is invoked under $h_t$ as a proposer.

- 注意：两次运行在错误日志的表面上可能共享相同的验证器结果（例如超时或缺失产物），却有着不同的因果机制。因此我们需要一份信息丰富的失败记录，包含终端验证器层级的原因、相关 agent 行为的因果状态，以及轨迹所暴露的抽象 agent 机制，才能挖出根本原因。
- Harness 提出（Harness proposal）：基于挖掘出的失败模式，提出有界的 harness 改动。

同一个模型在 $h_t$ 下被调用，扮演提出者的角色。

- The model is provided with a bounded proposal context: (1) the editable surfaces of the current harness, (2) the verifier-grounded failure patterns from the evaluation system, (3) records of passing behaviors that should be preserved, and (4) summaries of previously attempted edits.
- Harness edits should prefer recurrent error patterns that are addressable (e.g. not task-specific difficulty) and can be resolved by narrow changes.
- Harness edit candidates should be distinct and diverse.
- Proposal validation: validate and merge qualified edits to create a new harness $h_{t+1}$.

Candidate edits are evaluated by regression tests on held-in $D_\text{in}$ (for testing whether the weakness is resolved) and held-out $D_\text{out}$ (for checking whether other unknown issues were introduced) splits.

- 模型被赋予一个有界的提议上下文：(1) 当前 harness 可编辑的"面"；(2) 来自评估系统、以验证器为依据的失败模式；(3) 应当保留的"通过行为"记录；(4) 此前尝试过的改动的摘要。
- Harness 改动应优先选择那些可复现、可被解决（例如不是任务特定的难度）且能用小幅改动修好的错误模式。
- Harness 改动候选应当彼此不同且多样化。
- 提议验证（Proposal validation）：验证并合并合格的改动，生成新的 harness $h_{t+1}$。

候选改动通过回归测试在 held-in 的 $D_\text{in}$（检验弱点是否被解决）与 held-out 的 $D_\text{out}$（检查是否引入了其他未知问题）两个划分上评估。

- Candidates are accepted only if they have no regression on both held-in and held-out data.
- Accepted candidates are merged to update the harness to $h_{t+1}$, while rejected candidates are logged without changing the active harness.

- 只有当候选在 held-in 与 held-out 数据上都没有回归时，才被接受。
- 被接受的候选会被合并以更新 harness 到 $h_{t+1}$，而被拒绝的候选仅作日志记录，不改动当前活跃的 harness。

When running `MiniMax M2.5`, `Qwen3.5-35B-A3B`, and `GLM-5` on Terminal-Bench-2, Self-Harness was shown to learn model-specific harness instructions that target at different weaknesses of different base models and improve held-out pass rates.

在 Terminal-Bench-2 上运行 `MiniMax M2.5`、`Qwen3.5-35B-A3B` 与 `GLM-5` 时，Self-Harness 被证明能学到"针对特定模型的 harness 指令"——瞄准不同基座模型的不同弱点，并提升 held-out 的通过率。

Self-harness type of work does raise my concerns that if a program is allowed to edit the OS system, abstraction boundaries are broken. The editable surface needs to be properly designed and the permission control and security layers need to live outside this loop. All the challenges around [reward hacking](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/) still remain.

这类 self-harness 工作也引发了我的担忧：如果一个程序被允许编辑操作系统，抽象的边界就被打破了。可编辑的"面"需要被合理设计，权限控制与安全层也应当位于这个循环之外。围绕[奖励作弊（reward hacking）](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/)的所有挑战依然存在。


**Agentic Harness Engineering** (AHE; [Lin et al. 2026](https://arxiv.org/abs/2604.25850)) see the bottlenecks of harness evolution are around **observability**&mdash;that is, when a rollout fails, we need to know which component is responsible for that and every edit should be grounded by evidence.

**智能体 Harness 工程**（AHE；[Lin et al. 2026](https://arxiv.org/abs/2604.25850)）认为，harness 演进的瓶颈在于**可观测性**——也就是说，当一次 rollout 失败时，我们需要知道是哪个组件导致的，并且每一次改动都应有证据支撑。

The framework creates a closed loop with 3 observability pillars:

该框架构建了一个带有 3 个可观测性支柱的闭环：


- Component observability: every editable harness component has a representation in the file system so the action space is explicit and tracable.

- 组件可观测性（Component observability）：每个可编辑的 harness 组件在文件系统中都有对应表示，使动作空间显式且可追溯。

A harness contains 7 components: system prompt, tool description, tool implementation, middleware, skill, sub-agent configuration, and long-term memory.

一个 harness 包含 7 个组件：系统提示词、工具描述、工具实现、中间件（middleware）、技能（skill）、子 agent 配置，以及长期记忆。

- Each failure pattern is mapped to one component so the edit can be more targeted.
- Experience observability: analysize and summarize a large amount of raw trajectories into a hierarchy of evidence and failure patterns.

Each harness generates $k$ traces.

- 每个失败模式都被映射到某一个组件，从而让改动更有针对性。
- 经验可观测性（Experience observability）：将大量原始轨迹分析、汇总成一个"证据—失败模式"的层级结构。

每个 harness 生成 $k$ 条轨迹。

- Use an agent ("Agent debugger") to analysis the trajectories each stored in one file and generate per-task analysis report on the root cause for the failure or success.
- All the per-task reports are aggregated into a benchmark overview for the next step, and raw traces can be accessed if needed. This layered access structure is more token efficient.
- Decision observability: every edit is paired with a prediction for the next round to validate.

An agent ("Evolve agent") reads the repo and decides which component to edit, and then produces the edit and the reasoning behind it.

- 用一个 agent（"Agent debugger"）分析每条单独存于文件的轨迹，并生成关于"失败或成功的根本原因"的逐任务分析报告。
- 所有逐任务报告被聚合成一个供下一步使用的基准概览，原始轨迹在需要时可被访问。这种分层访问结构更省 token。
- 决策可观测性（Decision observability）：每一次改动都配有一条"对下一轮的预测"以供验证。

一个 agent（"Evolve agent"）读取仓库，决定编辑哪个组件，然后产出改动及其背后的推理。
- Every edit is a file-level, falsifiable claim and can be verified in the next round, under two constraints:

- 每一次改动都是一个文件级别的、可被证伪的断言（claim），并可在下一轮被验证，它受到两条约束：

(1) Edits are only applied to the harness workspace. the runs directory, tracer, verifier, and LLM configuration are read-only, which disables a set of reward hacking (e.g  disabling the verifier, swapping the model, or raising the reasoning budget) and thus it can keep every recorded gain attributable to harness edits.

(1) 改动只能应用于 harness 工作区。runs 目录、tracer、验证器与 LLM 配置都是只读的，这禁止了一整类奖励作弊（例如禁用验证器、替换模型或抬高推理预算），从而让每一个被记录的收益都可归因于 harness 改动。

- (2) Edits are evidence-driven, with a manifesto entry: the failure evidence's name, the inferred root cause, the targeted fix, and a predicted impact comprising both expected fixes and at-risk regressions.

- (2) 改动是证据驱动的，并带有一条"宣言式"条目：失败证据的名称、推断出的根本原因、针对性的修复，以及一条"预测影响"——既包括预期修复，也包括有回归风险之处。

On Terminal-Bench-2, AHE achieved better than human-designed harness (OpenCode, Terminus-2, Codex) except for Hard tier and a few other self-evolve baselines (ACE, TF-GRPO). The same frozen harness, without further evolving, transfers to SWE-bench-verified, indicating that the evolved harness is able to encode engineering experience into harness components rather than doing benchmark-specific optimization.

在 Terminal-Bench-2 上，AHE 的表现优于人类设计的 harness（OpenCode、Terminus-2、Codex），但在 Hard 层级以及少数其他自演进基线（ACE、TF-GRPO）上例外。同一个被"冻结"（不再演进）的 harness 还能迁移到 SWE-bench-verified，说明这个演进出来的 harness 能够把工程经验编码进 harness 组件，而非只做针对特定基准的优化。


## Evolutionary Search
Evolutionary search is an optimization method inspired by natural selection (see my old post on [evolutionary algorithm](https://lilianweng.github.io/posts/2019-09-05-evolution-strategies/)). It evolves a population of solutions by mutating them and only keeping those with high "fitness" in the crowd. Evolutionary search comes in handy when (1) the search space is extensive or weirdly shaped; and (2) it is hard to optimize directly with gradients but easy to evaluate solutions. Harness search seems to be a good fit here.

## 演化式搜索
演化式搜索是一种受自然选择启发的优化方法（参见我关于[演化算法](https://lilianweng.github.io/posts/2019-09-05-evolution-strategies/)的旧文）。它通过变异一个解的种群、只保留其中"适应度"高的个体来不断演进。当 (1) 搜索空间极其庞大或形状怪异，且 (2) 难以用梯度直接优化、却容易对解进行评估时，演化式搜索便能派上用场。Harness 搜索似乎正是它的用武之地。


Evolutionary search has been used in prompt engineering in the past studies. **Promptbreeder** ([Fernando et al. 2023](https://arxiv.org/abs/2309.16797)) optimizes task-specific prompts through a rich set of mutation operations, and interestingly the mutation prompts (i.e. instructions to an LLM to mutate a task prompt) are themselves also improved through evolution. **GEPA** ([Agrawal et al. 2025](https://arxiv.org/abs/2507.19457)) combines [reflection](https://lilianweng.github.io/posts/2023-06-23-agent/#self-reflection)-based prompting with evolutionary search and uses natural language reflection over trajectories of trial and error to propose prompt updates.

过去的研究中，演化式搜索已被用于提示词工程。**Promptbreeder**（[Fernando et al. 2023](https://arxiv.org/abs/2309.16797)）通过一套丰富的变异操作来优化特定任务的提示词，有趣的是，那些"变异提示词"（即指示 LLM 去变异任务提示词的指令）本身也通过演化得到改进。**GEPA**（[Agrawal et al. 2025](https://arxiv.org/abs/2507.19457)）将基于[反思（reflection）](https://lilianweng.github.io/posts/2023-06-23-agent/#self-reflection)的提示词与演化式搜索结合，并利用对试错轨迹的自然语言反思来提出提示词更新。


[Novikov et al. (2025)](https://arxiv.org/abs/2506.13131) introduced **AlphaEvolve** as a coding-agent evolutionary search system, which stores a pool of candidate programs and prompts frozen LLMs to generate diffs for improvement. As the system repeatedly evaluates child programs and keeps successful ones, it discovers better solutions in time.

[Novikov et al. (2025)](https://arxiv.org/abs/2506.13131) 提出的 **AlphaEvolve** 是一个编码 agent 的演化式搜索系统：它维护一个候选程序池，并提示冻结的 LLM 生成用于改进的 diff。随着系统不断评估子程序并保留成功者，它能随时间发现更优的解。


![alphaevolve.png](/img/yuque/Agent%20harness/harness-2026-07-04/alphaevolve.png)
*图：How AlphaEvolve works. (Image source: Novikov et al. 2025)*

A few details matter in the design of AlphaEvolve:

AlphaEvolve 的设计中有几个关键细节：


- The prompt includes parent programs, results, instructions, and sometimes meta information.
- The coding agent has access to the full repo, but code regions for improvement are explicitly marked with # EVOLVE-BLOCK-START and # EVOLVE-BLOCK-END.
- Meta-prompt co-evolves with instructions and context as suggested by LLM, in a similar way as how we evolve solution programs.

- 提示词中包含父程序、结果、指令，有时还有元信息（meta information）。
- 编码 agent 可以访问整个仓库，但待改进的代码区域会用 `# EVOLVE-BLOCK-START` 和 `# EVOLVE-BLOCK-END` 显式标注。
- 元提示词（meta-prompt）会与指令和上下文一起、按 LLM 的建议共同演进，方式与我们对"解程序"的演进类似。

Ablations show the evolution procedure, context in prompts, meta-prompts, full-file evolution and the use of stronger LLMs.

消融实验表明：演进流程、提示词中的上下文、元提示词、整文件演进，以及使用更强的 LLM，都各自有其价值。


![alphaevolve-plot.png](/img/yuque/Agent%20harness/harness-2026-07-04/alphaevolve-plot.png)
*图：Ablations show the value of everal designs in AlphaEvolve. (Image source: Novikov et al. 2025)*

Recent variants such as **ThetaEvolve** ([Wang et al. 2025](https://arxiv.org/abs/2511.23473)) combines evolutionary search with RL and in-context learning, and **DemoEvolve** ([Che, et al. 2026](https://arxiv.org/abs/2605.24539)) augments the self-rollout archive with human expert demonstrations as reference experience for harness-level diagnosis and editing. **ShinkaEvolve** ([Lange et al. 2025](https://arxiv.org/abs/2509.19349)), on the other hand, introduced three new components to improve LLM sampling efficiency:

近期的变体如 **ThetaEvolve**（[Wang et al. 2025](https://arxiv.org/abs/2511.23473)）将演化式搜索与 RL、上下文学习结合；**DemoEvolve**（[Che, et al. 2026](https://arxiv.org/abs/2605.24539)）则用人类专家演示来增强自我 rollout 档案库，作为 harness 层级诊断与编辑的参考经验。另一方面，**ShinkaEvolve**（[Lange et al. 2025](https://arxiv.org/abs/2509.19349)）引入了三个新组件来提升 LLM 采样效率：


- More sample-efficient exploration by designing parent sampling to balance performance rank and offspring count.
- Code-novelty rejection sampling by discarding candidates that are too similar to the existing population based on embedding-based cosine similarity.
- Identifying good patterns in successful solutions in a meta-scratchpad to guide future mutation.

- 通过设计"父代采样"来平衡性能排名与子代数量，从而实现更省样本的探索。
- 基于嵌入的余弦相似度，丢弃与现有种群过于相似的候选，即"代码新颖性拒绝采样"（code-novelty rejection sampling）。
- 在 meta-scratchpad 中识别成功解里的良好模式，以引导未来的变异。

Unlike the methods above, which focus on solution improvement, **Darwin Gödel Machine** (DGM; [Zhang et al. 2025](https://arxiv.org/abs/2505.22954)) explicitly targets the evolution of an editable harness-code repository with an LLM-based coding agent. Precisely, this agent is allowed to modify its own harness. A follow-up work on Hyperagents ([Zhang et al. 2026](https://arxiv.org/abs/2603.19461)) introduced a meta-agent to control how to modify existing task agents to create new ones.

与上述聚焦于"解改进"的方法不同，**达尔文·哥德尔机**（DGM；[Zhang et al. 2025](https://arxiv.org/abs/2505.22954)）明确以"可编辑的 harness 代码仓库"的演进为目标，使用一个基于 LLM 的编码 agent——确切地说，这个 agent 被允许修改它自己的 harness。其后续工作 Hyperagents（[Zhang et al. 2026](https://arxiv.org/abs/2603.19461)）引入了一个元 agent，来控制"如何修改已有的任务 agent 以创造新 agent"。


- Start with one coding agent in the pool.
- In each iteration, pick one parent with a probability proportional to its performance and inversely to the number of children it has, to modify and branch off to produce new agents.
- The selected parent agent examines its own benchmark evaluation log and then proposes improvements to its own harness codebase to generate a new version of the coding agent. Code editing is implemented with two basic tools: (1) bash (args: <bash_command>) and (2) editor (args: view/create/edit <file_path>).
- New coding agents are evaluated, and only those with sufficiently high performance are added back into the pool.
- Repeat steps 2-4 until some stop criteria hit.

- 从池中一个编码 agent 起步。
- 每次迭代，按"性能越高、已有子代越少则被选中的概率越大"的原则挑一个父代，对其进行修改并分叉，产生新的 agent。
- 被选中的父 agent 审视自己的基准评估日志，然后对自己的 harness 代码库提出改进，生成一个新版本的编码 agent。代码编辑通过两个基本工具实现：(1) bash（参数：<bash_command>）；(2) editor（参数：view/create/edit <file_path>）。
- 新的编码 agent 被评估，只有性能足够高者才会被加回池中。
- 重复第 2–4 步，直到满足某个停止条件。

DGM is harness evolution under a fixed model. In experiments with `Claude 3.5 Sonnet` as the base LLM and simple initial harness configs, the DGM-discovered agents are comparable to or outperform handcrafted agents on SWE-bench Verified (20% to 50%) and Polyglot (14.2% to 30.7%).

DGM 是在固定模型下的 harness 演进。在以 `Claude 3.5 Sonnet` 为基座 LLM、初始 harness 配置很简单时，DGM 发现的 agent 在 SWE-bench Verified（20%→50%）和 Polyglot（14.2%→30.7%）上，表现与手工设计的 agent 相当甚至更优。


This family of methods works well when candidate solutions are automatically evaluable and candidate fitness is easy to quantify, such as matrix multiplication, GPU kernel optimization, algorithm contests, datacenter scheduling. It struggles with domains where evaluation is slow, ambiguous, or mostly heuristic-based. The compute efficiency and effectiveness of evolution are also concerns.

这类方法在"候选解可自动评估、适应度易于量化"时表现良好，例如矩阵乘法、GPU 核函数优化、算法竞赛、数据中心调度。而在评估缓慢、模糊或主要依赖启发式规则的领域里，它会陷入困境。演化的计算效率与有效性，同样值得关注。


## Joint Optimization with Model Weights
Harness evolution changes the non-parametric system around the model. To enable full self-improvement, the model can totally be allowed to update its own weights at the same time. The weight update can be implemented via improvements in the model training pipeline or continual learning at test time. The topic of continual learning is worthy of its own post in the future.

## 与模型权重的联合优化
Harness 演进改变的是模型外围的"非参数"系统。要实现完整的自我改进，完全可以同时允许模型更新它自己的权重。权重的更新既可以通过改进模型训练流水线来实现，也可以通过测试时的持续学习来实现。持续学习这个主题，值得未来另写一篇文章专门讨论。


**SIA** ([Hebbar et al. 2026](https://arxiv.org/abs/2605.27276)) is an early attempt to combine harness improvement and model-parameter updates in the same optimization loop, with three components in the design:

- Meta-Agent: proposes the initial harness.
- Task-Specific Agent: executes the task.
- Feedback-Agent: chooses whether to update the harness or the model weights based on recent trajectories.

**SIA**（[Hebbar et al. 2026](https://arxiv.org/abs/2605.27276)）是将 harness 改进与模型参数更新放在同一个优化循环中的早期尝试，其设计包含三个组件：

- 元 Agent（Meta-Agent）：提出初始的 harness。
- 任务专用 Agent（Task-Specific Agent）：执行任务。
- 反馈 Agent（Feedback-Agent）：根据近期的轨迹，决定是更新 harness 还是更新模型权重。

![SIA.png](/img/yuque/Agent%20harness/harness-2026-07-04/SIA.png)
*图：SIA 中的 Feedback-Agent 决定下一轮迭代的类型。(Image source: Hebbar et al. 2026)*

There are a few confounding choices in SIA's experiments that make the results hard to interpret. For example, the task-specific agent is much weaker than the models used for the Meta-Agent and Feedback-Agent (`gpt-oss-120b` vs `Claude Sonnet 4.6`), and the baselines are too weak to cross-reference cleanly against related methods. I would consider the direction interesting, but the evidence provisional. Yet many challenges, such as training stability and Goodhart effect, still remain open.

SIA 的实验中有些混淆性的选择，使结果难以解释。例如，任务专用 agent 远弱于用于 Meta-Agent 和 Feedback-Agent 的模型（`gpt-oss-120b` vs `Claude Sonnet 4.6`），且基线太弱，无法与相关方法做干净的对照。我认为这个方向有趣，但证据尚属初步。不过，诸如训练稳定性、古德哈特定律（Goodhart effect）等许多挑战，仍然悬而未决。


**Continual Harness** ([Karten et al. 2026](https://arxiv.org/abs/2605.09998)) experimented in long-horizon gameplay setting with harness updating and co-learning a policy model by distilling a strong teacher model's labels on low-reward trajectories.

**Continual Harness**（[Karten et al. 2026](https://arxiv.org/abs/2605.09998)）在长周期游戏环境中做了实验，一边更新 harness，一边通过在低奖励轨迹上蒸馏强教师模型的标签，来协同学习一个策略模型。

# Future Challenges
The AI Scientist line of work is a strong demonstration that an expert-designed harness can coordinate a large portion of auto-research loop, experimented in the form of writing research papers. But paper production is not identical to scientific discovery. A system can write a plausible manuscript while still having fabricated citations, implementation drift, or weak experimental results.

# 未来挑战
AI Scientist 这一脉工作有力地证明：一个专家设计的 harness 能够协调整个自动研究循环的大部分环节，并以"撰写研究论文"的形式加以实践。但"产出论文"并不等同于"科学发现"。一个系统完全可以写出一篇看似合理的稿件，却依然包含伪造的引用、实现上的漂移，或薄弱的实验结果。

[Trehan & Chopra (2026)](https://arxiv.org/abs/2601.03315) tested whether LLMs can go from a research idea to a paper with minimal scaffolding and basic tools (i.e., `read_file`, `write_file`, `llm_search`, `list_files`). Each idea had a dedicated workspace where agents could generate and read documents as part of context. They experimented in three domains (world models, multi-agent RL, AI safety & alignment), with each domain containing 45-50 high-quality seed documents to inspire new ideas. Only four ideas were selected by human experts to run through the full pipeline, and only one was fully executed into a paper. They observed six recurring failure modes in the experiments:

[Trehan & Chopra (2026)](https://arxiv.org/abs/2601.03315) 测试了：在最少脚手架与基础工具（即 `read_file`、`write_file`、`llm_search`、`list_files`）的条件下，LLM 能否从一个研究想法走到一篇论文。每个想法都有一个专属工作区，agent 可以在其中生成并读取文档，作为上下文的一部分。他们在三个领域（世界模型、多 agent 强化学习、AI 安全与对齐）中做了实验，每个领域包含 45–50 篇高质量种子文档来激发新想法。只有 4 个想法被人类专家选中进入完整流水线，且只有 1 个被完整执行成论文。他们在实验中观察到了六种反复出现的失败模式：


- Bias toward training-data defaults: use old libraries, stale commands, standard formats, or assumptions not grounded in the actual repository or dataset.
- Implementation drift under execution pressure: when implementation becomes technically complex, the model may move toward a common simpler solution rather than the proposed method.
- Memory and context degradation: long-horizon projects lose critical details unless logs are written as persistent artifacts.
- Over-optimism: the model declares success despite noisy or failed experiments, similarly observed as "p-hacking and eureka-ing" pattern by Bubeck et al. (2025) where models can introduce "numerical duct tape" and declare victory when signals are still noise.
- Insufficient domain intelligence: the model lacks tacit craft knowledge, e.g. predicting implementation complexity, judging whether an experimental result is plausible, or knowing which baselines matter.
- Weak scientific taste: experiments may be executable but fail to answer the right question.

- 偏向训练数据中的默认做法：使用过时的库、陈旧的命令、标准格式，或基于并不符合实际仓库/数据集的假设。
- 执行压力下的实现漂移：当实现在技术上变复杂时，模型可能滑向一个常见的更简单解法，而非原本提出的方法。
- 记忆与上下文退化：长周期项目会丢失关键细节，除非日志被写成持久化产物。
- 过度乐观：即便实验充满噪声或已经失败，模型仍会宣布成功——这与 Bubeck 等人（2025）观察到的"p-hacking 与 eureka-ing"模式类似：模型可以引入"数值胶带（numerical duct tape）"，并在信号仍是噪声时就宣告胜利。
- 领域智能不足：模型缺乏默会的"手艺性知识"，例如预判实现复杂度、判断实验结果是否可信，或知道哪些基线才重要。
- 科学品味薄弱：实验或许能跑通，却没能回答正确的问题。

Toward full RSI, researchers have made real progress, but several bottlenecks remain.

在迈向完整 RSI 的道路上，研究者已取得了真正的进展，但若干瓶颈依然存在。


**1. Weak and fuzzy evaluators.** Many research claims do not have a fast and precise verifier, and the same is true for many real-world tasks. Current self-improvement loops work best for tasks when evaluation metrics are measurable and objective, similar as [how RL works](https://lilianweng.github.io/posts/2018-02-19-rl-overview/).

**1. 薄弱而模糊的评估器。** 许多研究主张并没有快速、精确的验证器，现实世界中的许多任务也是如此。当前的自我改进循环，在最擅长的是那些"评估指标可度量、客观"的任务，这与 [RL 的工作方式](https://lilianweng.github.io/posts/2018-02-19-rl-overview/) 类似。

Research taste, novelty, and long-term scientific value are much harder to measure. For example, research taste often mixes problem framing, experimental design, and judgment about which surprising results are worth pursuing and which failure cases are worth retries.

研究的"品味"、新颖性，以及长期的科学价值，则要难衡量得多。例如，研究品味往往混含着问题界定、实验设计，以及对"哪些令人惊讶的结果值得追下去、哪些失败案例值得重试"的判断。


**2. Context and memory lifecycle.** Memory grows as AI agents become more autonomous and independent. A useful harness needs to manage context and memory to complement existing limitation in long-context generation while still maximizing the success of long-horizon tasks. Since humans are able to maintain memory through our life time, I see an anoloy here that [context engineering](#context-engineering) will and should become a core part of intelligence, rather than staying in the software system layer.

**2. 上下文与记忆的生命周期。** 随着 AI agent 变得愈发自主、独立，记忆也在不断增长。一个有用的 harness 需要管理上下文与记忆，既补足当前长上下文生成能力的局限，又最大化长周期任务的成功率。既然人类能够终其一生地维持记忆，我在这里看到一个类比：[上下文工程](#context-engineering)将会、也应当成为智能的核心组成部分，而不只是停留在软件系统层。
**3. Negative results.** Researchers are incentivized to publish successful results and thus literature is biased toward successes. LLMs trained on a vast amount of data (mostly human created, at least for now, lol) may be bad at deciding when to abandon a hypothesis, report a negative result, or even acknowledge a failure due to the imablance of success vs failure cases in data. A research harness should make failed attempts easy to preserve, as learning from failure is the best way to trim down the task search space.

**3. 负面结果。** 研究者有动机去发表成功的结果，因此文献天然偏向成功。在海量数据（至少目前大多仍由人类创造）上训练的 LLM，可能因为数据中"成功 vs 失败"样本的不平衡，而不善于判断何时该放弃一个假设、何时该报告一个负面结果，甚至不愿承认失败。一个研究 harness 应当让失败的尝试易于被保留，因为"从失败中学习"是缩小任务搜索空间的最佳方式。

**4. Diversity collapse.** Evolutionary and RL loops tend to exploit known high-reward patterns. We need [mechanisms](https://lilianweng.github.io/posts/2020-06-07-exploration-drl/) to prevent the population from collapsing into variants of the same solution. This is especially critical for open-ended research, where the best path may initially look worse under the current evaluator.

**4. 多样性坍缩。** 演化式与 RL 循环倾向于利用那些已知的高奖励模式。我们需要[相应机制](https://lilianweng.github.io/posts/2020-06-07-exploration-drl/)来防止种群坍缩成同一解的各种变体。这一点对于开放式的（open-ended）研究尤为关键——因为在当前评估器下，最佳路径起初可能看起来更差。

**5. [Reward hacking](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/).** A self-improvement loop optimizes whatever signal it is given. If the reward comes from unit tests, the agent may overfit to tests; if it comes from a judge model, it may learn reward hacking tricks specific to this judge; if it comes from benchmark scores, it may exploit benchmark artifacts.

**5. [奖励作弊（Reward hacking）](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/)。** 自我改进循环会优化它所得到的任何信号。如果奖励来自单元测试，agent 可能过拟合于测试；如果来自评判模型，它可能学会只针对该评判器的作弊技巧；如果来自基准分数，它可能利用基准本身的瑕疵。

The evaluator and permission control should likely sit outside the loop that evolves harness, with held-out tests, trace audits, and human review at decision points that matter&mdash;how much oversight can be scaled up and automated remains an open research area.

评估器与权限控制，应当位于"演进 harness 的那个循环"之外，并配合留出测试集（held-out tests）、轨迹审计，以及在关键决策点上的人工复核——而"多大规模的监督可以被放大并自动化"仍是一个开放的研究课题。


**6. Long-term success.** An extrinsic loop of optimization works on rewards outside of individual rollouts that we can simulate in training sandbox.

**6. 长期成功。** 一个外在的优化循环，作用于那些超出"单个 rollout"之外的奖励——也就是我们在训练沙盒里可以模拟的东西。

Take coding agent as an example. Coding agents have already increased daily productivity in software engineering, but many optimization goals are still too short-term. It can often complete the task at hand, but less obvious how it should protect the long-term health of a repo collectively maintained by hundreds or thousands of engineers. Standard sandbox-based RLVR-style training rarely captures maintainability, ownership boundaries, migration cost, backwards compatibility, or future debugging burden.

以编码 agent 为例。编码 agent 已经提升了软件工程中的日常生产力，但许多优化目标仍过于短期。它常常能完成手头的任务，却很难说清它应该如何保护一个由成百上千名工程师共同维护的仓库的"长期健康"。标准的、基于沙盒的 RLVR 式训练，很少能捕捉到可维护性、所有权边界、迁移成本、向后兼容性，或未来的调试负担。


**7. The role of humans.** Humans should move up the stack, not be removed from the loop, meaning that human should provide oversight at the right time, at the right abstraction level and our system design should consider when and how to set up such touch points.

**7. 人类的角色。** 人类应当向上爬升到技术栈的更高层，而不是被移出循环之外——也就是说，人类要在恰当的时机、恰当的抽象层级上提供监督；我们的系统设计也应当思考"何时、以何种方式"设立这样的接触点。

Many challenges listed above need human's feedback and steering. After all, we are building the technology for better future of humanity, not other way around.

上面列出的诸多挑战，都需要人类的反馈与引导。毕竟，我们打造技术，是为了人类更美好的未来，而不是相反。


# Citation
Please cite this work as:

> Weng, Lilian. "Harness Engineering for Self-Improvement". Lil'Log (Jul 2026). https://lilianweng.github.io/posts/2026-07-04-harness/

Or use the BibTeX citation:

`@article{weng2026harness,
  title = {Harness Engineering for Self-Improvement},
  author = {Weng, Lilian},
  journal = {lilianweng.github.io},
  year = {2026},
  month = {July},
  url = &#34;https://lilianweng.github.io/posts/2026-07-04-harness/&#34;
}
# Appendix: Some Useful Benchmarks（附录：一些有用的基准测试）[#](#appendix-some-useful-benchmarks)

**[PaperBench](https://arxiv.org/abs/2504.01848)**: replicate 20 ICML 2024 Spotlight and Oral papers from scratch, including understanding paper contributions, developing a codebase, and successfully executing experiments.

Each replication task is decomposed into smaller, individually gradable tasks.
8,316 rubrics in total, co-developed with the paper authors.
The best model at the time (`Claude 3.5 Sonnet`, ~21%) does not outperform ML PhDs.
Includes PaperBench, PaperBench Code-Dev (a lighter version), and JudgeEval.


**[CORE-Bench](https://arxiv.org/abs/2409.11363)**: evaluate computational reproducibility of published research.

270 tasks based on 90 scientific papers across computer science, social science, and medicine.
Tasks involve reproducing results from provided code and data.
Includes multiple difficulty levels and both language-only and vision-language tasks.
The best reported agent at the time (`GPT-4o` and `GPT-4o-mini`) achieved only 21% accuracy on the hardest task.


**[ScienceAgentBench](https://arxiv.org/abs/2410.05080)**: evaluate LLM agents for data-driven scientific discovery.

Extracts 102 tasks from 44 peer-reviewed publications in four disciplines (math, chemistry, biology, geography).
Covers basic data-science tasks in these domains: data processing, model development, data analysis, and information visualization.


**[RE-Bench](https://arxiv.org/abs/2411.15114)**: evaluate frontier AI agents on realistic ML research-engineering envs against human experts.

7 challenging, open-ended ML research-engineering environments.
Each environment = (scoring function, starting solution, reference solution); each can be run with 8 or fewer H100 GPUs.
Examples: optimize a kernel, run a scaling-law experiment, fix an embedding, fine-tune GPT-2 for QA, etc.
Includes data from 71 eight-hour attempts by 61 distinct human experts.
Human experts achieved non-zero score in 82% of 8-hour attempts; 24% matched or exceeded strong reference solutions.
Best AI agents scored 4× higher than humans at a 2-hour budget, but humans had better returns to longer budgets and exceeded agents at 8-hour and 32-hour settings.


**[MLE-bench](https://arxiv.org/abs/2410.07095)**: evaluate ML engineering agents on offline Kaggle competitions.

Contains 75 ML-engineering competitions curated from Kaggle.
Tests training models, preparing datasets, running experiments, and submitting predictions to grading scripts.
Uses Kaggle public leaderboards as human baselines.
Best setup in the paper, `o1-preview` with AIDE scaffolding, reached at least Kaggle bronze-medal level in 16.9% of competitions.
Includes resource-scaling and contamination analyses.


**[KernelBench](https://arxiv.org/abs/2502.10517)**: evaluate correctness and speed for generated GPU kernels.

250 PyTorch tasks to evaluate whether LLM can write fast and correct kernels.
The evaluation metric fast_p = the percentage of generated kernels that are correct and faster than baseline.



# References（参考文献）[#](#references)
[1] Good, I. J. ["Speculations Concerning the First Ultraintelligent Machine."](https://philpapers.org/rec/GOOSCT) *Advances in Computers*, 6:31&ndash;88, 1965.

[2] Yudkowsky, Eliezer. ["Recursive Self-Improvement."](https://www.lesswrong.com/posts/JBadX7rwdcRFzGuju/recursive-self-improvement) LessWrong, 2008.

[3] Choi, et al. ["Anchored Self-Play for Code Repair."](https://openreview.net/forum?id=lTbBFAoPSA) ICML 2026.

[4] Zhao, et al. ["Absolute Zero: Reinforced Self-play Reasoning with Zero Data."](https://arxiv.org/abs/2505.03335) arXiv preprint arXiv:2505.03335, 2025.

[5] Yuan, et al. ["Self-Rewarding Language Models."](https://arxiv.org/abs/2401.10020) arXiv preprint arXiv:2401.10020, 2024.

[6] Chen, et al. ["Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models."](https://arxiv.org/abs/2401.01335) ICML 2024.

[7] Zhang, et al. ["Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models."](https://arxiv.org/abs/2510.04618) ICLR 2026.

[8] Ye, et al. ["Meta Context Engineering via Agentic Skill Evolution."](https://arxiv.org/abs/2601.21557) arXiv preprint arXiv:2601.21557, 2026.

[9] Lee, et al. ["Meta-Harness: End-to-End Optimization of Model Harnesses."](https://arxiv.org/abs/2603.28052) arXiv preprint arXiv:2603.28052, 2026.

[10] Lu, et al. ["Towards end-to-end automation of AI research."](https://www.nature.com/articles/s41586-026-10265-5) *Nature*, 651:914&ndash;919, 2026.

[11] Meng, et al. ["ScientistOne: Towards Human-Level Autonomous Research via Chain-of-Evidence."](https://arxiv.org/abs/2605.26340) arXiv preprint arXiv:2605.26340, 2026.

[12] Kulikov, et al. ["Autodata: An agentic data scientist to create high quality synthetic data."](https://arxiv.org/abs/2606.25996) arXiv preprint arXiv:2606.25996, 2026.

[13] Hu, Lu, and Clune. ["Automated Design of Agentic Systems."](https://arxiv.org/abs/2408.08435) ICLR 2025.

[14] Madaan, et al. ["Self-Refine: Iterative Refinement with Self-Feedback."](https://arxiv.org/abs/2303.17651) NeurIPS 2023.

[15] Zhang, et al. ["AFlow: Automating Agentic Workflow Generation."](https://arxiv.org/abs/2410.10762) ICLR 2025.

[16] Zelikman, et al. ["Self-Taught Optimizer (STOP): Recursively Self-Improving Code Generation."](https://arxiv.org/abs/2310.02304) COLM 2024.

[17] Zhang, et al. ["Self-Harness: Harnesses That Improve Themselves."](https://arxiv.org/abs/2606.09498) arXiv preprint arXiv:2606.09498, 2026.

[18] Fernando, et al. ["Promptbreeder: Self-Referential Self-Improvement Via Prompt Evolution."](https://arxiv.org/abs/2309.16797) arXiv preprint arXiv:2309.16797, 2023.

[19] Agrawal, A. et al. ["GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning."](https://arxiv.org/abs/2507.19457) arXiv preprint arXiv:2507.19457, 2025.

[20] Novikov, et al. ["AlphaEvolve: A coding agent for scientific and algorithmic discovery."](https://arxiv.org/abs/2506.13131) arXiv preprint arXiv:2506.13131, 2025.

[21] Lange, Imajuku, and Cetin. ["ShinkaEvolve: Towards Open-Ended And Sample-Efficient Program Evolution."](https://arxiv.org/abs/2509.19349) arXiv preprint arXiv:2509.19349, 2025.

[22] Wang, et al. ["ThetaEvolve: Test-time Learning on Open Problems."](https://arxiv.org/abs/2511.23473) arXiv preprint arXiv:2511.23473, 2025.

[23] Zhang, et al. ["Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents."](https://arxiv.org/abs/2505.22954) arXiv preprint arXiv:2505.22954, 2025.

[24] Zhang, et al. ["Hyperagents."](https://arxiv.org/abs/2603.19461) arXiv preprint arXiv:2603.19461, 2026.

[25] Yuksekgonul, et al. ["Learning to Discover at Test Time."](https://arxiv.org/abs/2601.16175) arXiv preprint arXiv:2601.16175, 2026.

[26] Riaz, et al. ["Epistemic Uncertainty for Test-Time Discovery."](https://arxiv.org/abs/2605.11328) arXiv preprint arXiv:2605.11328, 2026.

[27] Hebbar, et al. ["SIA: Self Improving AI with Harness & Weight Updates."](https://arxiv.org/abs/2605.27276) arXiv preprint arXiv:2605.27276, 2026.

[28] Trehan and Chopra. ["Why LLMs Aren't Scientists Yet: Lessons from Four Autonomous Research Attempts."](https://arxiv.org/abs/2601.03315) arXiv preprint arXiv:2601.03315, 2026.

[29] Bubeck, et al. ["Early science acceleration experiments with GPT-5."](https://arxiv.org/abs/2511.16072) arXiv preprint arXiv:2511.16072, 2025.

[30] Starace, et al. ["PaperBench: Evaluating AI's Ability to Replicate AI Research."](https://arxiv.org/abs/2504.01848) ICML 2025.

[31] Wijk, et al. ["RE-Bench: Evaluating frontier AI R&D capabilities of language model agents against human experts."](https://arxiv.org/abs/2411.15114) ICML 2025.

[32] Chan, et al. ["MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering."](https://arxiv.org/abs/2410.07095) arXiv preprint arXiv:2410.07095, 2024.

[33] Chen, et al. ["ScienceAgentBench: Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discovery."](https://arxiv.org/abs/2410.05080) ICLR 2025.

[34] Siegel, et al. ["CORE-Bench: Fostering the Credibility of Published Research Through a Computational Reproducibility Agent Benchmark."](https://arxiv.org/abs/2409.11363) TMLR 2024.

[35] Ouyang, et al. ["KernelBench: Can LLMs Write Efficient GPU Kernels?"](https://arxiv.org/abs/2502.10517) arXiv preprint arXiv:2502.10517, 2025.

[36] Lin, et al. ["Harness Updating Is Not Harness Benefit: Disentangling Evolution Capabilities in Self-Evolving LLM Agents."](https://arxiv.org/abs/2605.30621) arXiv preprint arXiv:2605.30621, 2026.

[37] Lin, et al. ["Agentic Harness Engineering: Observability-Driven Automatic Evolution of Coding-Agent Harnesses."](https://arxiv.org/abs/2604.25850) arXiv preprint arXiv:2604.25850, 2026.

[38] Karten, et al. ["Continual Harness: Online Adaptation for Self-Improving Foundation Agents."](https://arxiv.org/abs/2605.09998) arXiv preprint arXiv:2605.09998, 2026.

[39] Che, et al. ["DemoEvolve: Overcoming Sparse Feedback in Agentic Harness Evolution with Demonstrations."](https://arxiv.org/abs/2605.24539) arXiv preprint arXiv:2605.24539, 2026.

- Language-Model
- Agent
- Auto-Research
- Self-Improvement
- Prompting
