---
title: "Code execution with MCP: building more efficient agents（用代码执行 + MCP 构建更高效的智能体）"
date: 2026-08-11
layout: post
liquid: false
tags: [语雀, 计算机, Anthropic Engineering 翻译]
categories: [计算机]
yuque: true
---

> 原文作者：Adam Jones、Conor Kelly | 发布于 2025-11-04 | 原文链接：https://www.anthropic.com/engineering/code-execution-with-mcp

# Code execution with MCP: Building more efficient agents

[The Model Context Protocol (MCP)](https://modelcontextprotocol.io/) is an open standard for connecting AI agents to external systems. Connecting agents to tools and data traditionally requires a custom integration for each pairing, creating fragmentation and duplicated effort that makes it difficult to scale truly connected systems. MCP provides a universal protocol—developers implement MCP once in their agent and it unlocks an entire ecosystem of integrations.

[模型上下文协议（MCP）](https://modelcontextprotocol.io/) 是一项将 AI 智能体连接到外部系统的开放标准。传统上，将智能体连接到工具和数据需要为每一对组合做定制集成，造成碎片化与重复劳动，难以构建真正可扩展的互联系统。MCP 提供了一套通用协议——开发者只需在智能体中实现一次 MCP，就能解锁一整个集成生态。

Since launching MCP in November 2024, adoption has been rapid: the community has built thousands of [MCP servers](https://github.com/modelcontextprotocol/servers), [SDKs](https://modelcontextprotocol.io/docs/sdk) are available for all major programming languages, and the industry has adopted MCP as the de-facto standard for connecting agents to tools and data.

自 2024 年 11 月发布 MCP 以来，采用速度极快：社区已构建了数千个 [MCP 服务器](https://github.com/modelcontextprotocol/servers)，所有主流编程语言都有 [SDK](https://modelcontextprotocol.io/docs/sdk) 可用，业界已将 MCP 采纳为连接智能体与工具、数据的实际标准。

Today developers routinely build agents with access to hundreds or thousands of tools across dozens of MCP servers. However, as the number of connected tools grows, loading all tool definitions upfront and passing intermediate results through the context window slows down agents and increases costs.

如今开发者常常构建能访问数十个 MCP 服务器、数百乃至数千个工具的智能体。然而，随着连接工具数量增长，预先加载所有工具定义、并把中间结果经过上下文窗口传递，会拖慢智能体并推高成本。

In this blog we'll explore how code execution can enable agents to interact with MCP servers more efficiently, handling more tools while using fewer tokens.

在本文中，我们将探讨代码执行如何让智能体更高效地与 MCP 服务器交互，用更少的 token 处理更多的工具。

## Excessive token consumption from tools makes agents less efficient

## 工具带来的过量 token 消耗让智能体效率下降

As MCP usage scales, there are two common patterns that can increase agent cost and latency:

随着 MCP 使用规模扩大，有两种常见模式会推高智能体成本与延迟：

1.  Tool definitions overload the context window;
2.  Intermediate tool results consume additional tokens.

1.  工具定义挤占上下文窗口；
2.  中间工具结果消耗额外 token。

### 1. Tool definitions overload the context window

### 1. 工具定义挤占上下文窗口

Most MCP clients load all tool definitions upfront directly into context, exposing them to the model using a direct tool-calling syntax. These tool definitions might look like:

多数 MCP 客户端预先把全部工具定义直接载入上下文，用直接的工具调用语法暴露给模型。这些工具定义可能长这样：

```
gdrive.getDocument
     Description: Retrieves a document from Google Drive
     Parameters:
                documentId (required, string): The ID of the document to retrieve
                fields (optional, string): Specific fields to return
     Returns: Document object with title, body content, metadata, permissions, etc.
```

Tool descriptions occupy more context window space, increasing response time and costs. In cases where agents are connected to thousands of tools, they’ll need to process hundreds of thousands of tokens before reading a request.

工具描述占据更多上下文窗口空间，增加响应时间与成本。在智能体连接了数千个工具的情况下，它们在读到一条请求之前，可能就要先处理数十万 token。

### 2. Intermediate tool results consume additional tokens

### 2. 中间工具结果消耗额外 token

Most MCP clients allow models to directly call MCP tools. For example, you might ask your agent: "Download my meeting transcript from Google Drive and attach it to the Salesforce lead."

多数 MCP 客户端允许模型直接调用 MCP 工具。例如，你可能会让智能体："从 Google Drive 下载我的会议转录，并附到 Salesforce 的线索上。"

The model will make calls like:

模型会发起类似这样的调用：

```
TOOL CALL: gdrive.getDocument(documentId: "abc123")
        → returns "Discussed Q4 goals...\n[full transcript text]"
           (loaded into model context)

TOOL CALL: salesforce.updateRecord(
    objectType: "SalesMeeting",
    recordId: "00Q5f000001abcXYZ",
    data: { "Notes": "Discussed Q4 goals...\n[full transcript text written out]" }
    (model needs to write entire transcript into context again)
```

Every intermediate result must pass through the model. In this example, the full call transcript flows through twice. For a 2-hour sales meeting, that could mean processing an additional 50,000 tokens. Even larger documents may exceed context window limits, breaking the workflow.

每个中间结果都必须经过模型。在这个例子中，完整的通话转录流过了两次。一场 2 小时的销售会议，这可能意味着额外处理 50,000 个 token。更大的文档甚至可能超出上下文窗口上限，使工作流崩溃。

With large documents or complex data structures, models may be more likely to make mistakes when copying data between tool calls.

面对大文档或复杂数据结构，模型在工具调用间复制数据时更容易出错。

## Code execution with MCP improves context efficiency

## 用代码执行 + MCP 提升上下文效率

With code execution environments becoming more common for agents, a solution is to present MCP servers as code APIs rather than direct tool calls. The agent can then write code to interact with MCP servers. This approach addresses both challenges: agents can load only the tools they need and process data in the execution environment before passing results back to the model.

随着代码执行环境在智能体中愈发普及，一种解决方案是把 MCP 服务器表现为代码 API，而非直接的工具调用。智能体于是可以编写代码来与 MCP 服务器交互。这种方式同时解决了两大挑战：智能体只需加载当前所需的工具，并能在执行环境中先处理数据，再把结果回传给模型。

There are a number of ways to do this. One approach is to generate a file tree of all available tools from connected MCP servers. Here's an implementation using TypeScript:

实现方式有多种。一种是从已连接的 MCP 服务器生成所有可用工具的文件树。下面是一个 TypeScript 实现：

```
servers
├── google-drive
│   ├── getDocument.ts
│   ├── ... (other tools)
│   └── index.ts
├── salesforce
│   ├── updateRecord.ts
│   ├── ... (other tools)
│   └── index.ts
└── ... (other servers)
```

Then each tool corresponds to a file, something like:

每个工具对应一个文件，大致如下：

```
// ./servers/google-drive/getDocument.ts
import { callMCPTool } from "../../../client.js";

interface GetDocumentInput {
  documentId: string;
}

interface GetDocumentResponse {
  content: string;
}

/* Read a document from Google Drive */
export async function getDocument(input: GetDocumentInput): Promise<GetDocumentResponse> {
  return callMCPTool<GetDocumentResponse>('google_drive__get_document', input);
}
```

Our Google Drive to Salesforce example above becomes the code:

上面那个 Google Drive 到 Salesforce 的例子变成了这样的代码：

```
// Read transcript from Google Docs and add to Salesforce prospect
import * as gdrive from './servers/google-drive';
import * as salesforce from './servers/salesforce';

const transcript = (await gdrive.getDocument({ documentId: 'abc123' })).content;
await salesforce.updateRecord({
  objectType: 'SalesMeeting',
  recordId: '00Q5f000001abcXYZ',
  data: { Notes: transcript }
});
```

The agent discovers tools by exploring the filesystem: listing the `./servers/` directory to find available servers (like `google-drive` and `salesforce`), then reading the specific tool files it needs (like `getDocument.ts` and `updateRecord.ts`) to understand each tool's interface. This lets the agent load only the definitions it needs for the current task. This reduces the token usage from 150,000 tokens to 2,000 tokens—a time and cost saving of 98.7%.

智能体通过探索文件系统来发现工具：列出 `./servers/` 目录找到可用服务器（如 `google-drive`、`salesforce`），再读取它需要的特定工具文件（如 `getDocument.ts`、`updateRecord.ts`）来理解每个工具的接口。这让智能体只加载当前任务所需的工具定义。token 用量从 150,000 降到 2,000——节省 98.7% 的时间与成本。

Cloudflare [published similar findings](https://blog.cloudflare.com/code-mode/), referring to code execution with MCP as "Code Mode." The core insight is the same: LLMs are adept at writing code and developers should take advantage of this strength to build agents that interact with MCP servers more efficiently.

Cloudflare [也发表了类似发现](https://blog.cloudflare.com/code-mode/)，把代码执行 + MCP 称作 "Code Mode"。核心洞见相同：LLM 擅长写代码，开发者应当利用这一优势，构建与 MCP 服务器交互更高效的智能体。

## Benefits of code execution with MCP

## 代码执行 + MCP 的好处

Code execution with MCP enables agents to use context more efficiently by loading tools on demand, filtering data before it reaches the model, and executing complex logic in a single step. There are also security and state management benefits to using this approach.

代码执行 + MCP 让智能体通过按需加载工具、在到达模型前先过滤数据、一步执行复杂逻辑，从而更高效地使用上下文。这种方式在安全与状态管理上也有好处。

### Progressive disclosure

### 渐进式披露

Models are great at navigating filesystems. Presenting tools as code on a filesystem allows models to read tool definitions on-demand, rather than reading them all up-front.

模型非常擅长浏览文件系统。把工具表现为文件系统中的代码，让模型能按需读取工具定义，而非一次性全部读取。

Alternatively, a `search_tools` tool can be added to the server to find relevant definitions. For example, when working with the hypothetical Salesforce server used above, the agent searches for "salesforce" and loads only those tools that it needs for the current task. Including a detail level parameter in the `search_tools` tool that allows the agent to select the level of detail required (such as name only, name and description, or the full definition with schemas) also helps the agent conserve context and find tools efficiently.

此外，也可以给服务器加一个 `search_tools` 工具来查找相关定义。例如，处理上面假设的 Salesforce 服务器时，智能体搜索 "salesforce"，只加载当前任务需要的那些工具。在 `search_tools` 工具里加一个 detail level 参数，让智能体选择所需详细程度（如仅名称、名称+描述、或带 schema 的完整定义），也能帮助智能体节省上下文、高效找工具。

### Context efficient tool results

### 省上下文的工具结果

When working with large datasets, agents can filter and transform results in code before returning them. Consider fetching a 10,000-row spreadsheet:

处理大型数据集时，智能体可以在返回结果前用代码过滤和变换。考虑拉取一张 10,000 行的电子表格：

```
// Without code execution - all rows flow through context
TOOL CALL: gdrive.getSheet(sheetId: 'abc123')
        → returns 10,000 rows in context to filter manually

// With code execution - filter in the execution environment
const allRows = await gdrive.getSheet({ sheetId: 'abc123' });
const pendingOrders = allRows.filter(row =>
  row["Status"] === 'pending'
);
console.log(`Found ${pendingOrders.length} pending orders`);
console.log(pendingOrders.slice(0, 5)); // Only log first 5 for review
```

The agent sees five rows instead of 10,000. Similar patterns work for aggregations, joins across multiple data sources, or extracting specific fields—all without bloating the context window.

智能体看到的是 5 行而非 10,000 行。类似的模式适用于聚合、跨数据源 join、或抽取特定字段——都不会撑爆上下文窗口。

#### More powerful and context-efficient control flow

#### 更强大且省上下文的控制流

Loops, conditionals, and error handling can be done with familiar code patterns rather than chaining individual tool calls. For example, if you need a deployment notification in Slack, the agent can write:

循环、条件判断、错误处理可以用熟悉的代码模式完成，而非串联一个个工具调用。例如，要在 Slack 里等一个部署通知，智能体可以写：

```
let found = false;
while (!found) {
  const messages = await slack.getChannelHistory({ channel: 'C123456' });
  found = messages.some(m => m.text.includes('deployment complete'));
  if (!found) await new Promise(r => setTimeout(r, 5000));
}
console.log('Deployment notification received');
```

This approach is more efficient than alternating between MCP tool calls and sleep commands through the agent loop.

这种方式比在智能体循环里交替调用 MCP 工具与 sleep 命令更高效。

Additionally, being able to write out a conditional tree that gets executed also saves on "time to first token" latency: rather than having to wait for base model to evaluate an if-statement, the agent can let the code execution environment do this.

此外，把条件树写出来直接执行，也节省了"首 token 延迟"：智能体无需等模型去判断 if 语句，而是让代码执行环境来做。

### Privacy-preserving operations

### 隐私保护操作

When agents use code execution with MCP, intermediate results stay in the execution environment by default. This way, the agent only sees what you explicitly log or return, meaning data you don’t wish to share with the model can flow through your workflow without ever entering the model's context.

当智能体用代码执行 + MCP 时，中间结果默认留在执行环境里。这样，智能体只看到你显式 log 或返回的东西，意味着你不想分享给模型的数据可以流过工作流，却从不进入模型上下文。

For even more sensitive workloads, the agent harness can tokenize sensitive data automatically. For example, imagine you need to import customer contact details from a spreadsheet into Salesforce. The agent writes:

对更敏感的工作负载，支撑框架可以自动对敏感数据做 token 化。例如，假设你需要把客户联系方式从电子表格导入 Salesforce。智能体写：

```
const sheet = await gdrive.getSheet({ sheetId: 'abc123' });
for (const row of sheet.rows) {
  await salesforce.updateRecord({
    objectType: 'Lead',
    recordId: row.salesforceId,
    data: {
      Email: row.email,
      Phone: row.phone,
      Name: row.name
    }
  });
}
console.log(`Updated ${sheet.rows.length} leads`);
```

The MCP client intercepts the data and tokenizes PII before it reaches the model:

MCP 客户端拦截数据，并在到达模型之前对 PII 做 token 化：

```
// What the agent would see, if it logged the sheet.rows:
[
  { salesforceId: '00Q...', email: '[EMAIL_1]', phone: '[PHONE_1]', name: '[NAME_1]' },
  { salesforceId: '00Q...', email: '[EMAIL_2]', phone: '[PHONE_2]', name: '[NAME_2]' },
  ...
]
```

Then, when the data is shared in another MCP tool call, it is untokenized via a lookup in the MCP client. The real email addresses, phone numbers, and names flow from Google Sheets to Salesforce, but never through the model. This prevents the agent from accidentally logging or processing sensitive data. You can also use this to define deterministic security rules, choosing where data can flow to and from.

随后，当数据在另一次 MCP 工具调用中被共享时，通过 MCP 客户端的查找做去 token 化。真实的邮箱、电话、姓名从 Google Sheets 流向 Salesforce，却从不经过模型。这能防止智能体意外记录或处理敏感数据。你也可以用它来定义确定性的安全规则，指定数据可以从哪流向哪。

### State persistence and skills

### 状态持久化与技能

Code execution with filesystem access allows agents to maintain state across operations. Agents can write intermediate results to files, enabling them to resume work and track progress:

带文件系统访问的代码执行让智能体能跨操作维护状态。智能体可以把中间结果写入文件，从而恢复工作、跟踪进度：

```
const leads = await salesforce.query({
  query: 'SELECT Id, Email FROM Lead LIMIT 1000'
});
const csvData = leads.map(l => `${l.Id},${l.Email}`).join('\n');
await fs.writeFile('./workspace/leads.csv', csvData);

// Later execution picks up where it left off
const saved = await fs.readFile('./workspace/leads.csv', 'utf-8');
```

Agents can also persist their own code as reusable functions. Once an agent develops working code for a task, it can save that implementation for future use:

智能体还可以把自己的代码持久化为可复用函数。一旦智能体为某任务写出了能用的代码，就可以保存该实现供日后使用：

```
// In ./skills/save-sheet-as-csv.ts
import * as gdrive from './servers/google-drive';
export async function saveSheetAsCsv(sheetId: string) {
  const data = await gdrive.getSheet({ sheetId });
  const csv = data.map(row => row.join(',')).join('\n');
  await fs.writeFile(`./workspace/sheet-${sheetId}.csv`, csv);
  return `./workspace/sheet-${sheetId}.csv`;
}

// Later, in any agent execution:
import { saveSheetAsCsv } from './skills/save-sheet-as-csv';
const csvPath = await saveSheetAsCsv('abc123');
```

This ties in closely to the concept of [Skills](https://docs.claude.com/en/docs/agents-and-tools/agent-skills/overview), folders of reusable instructions, scripts, and resources for models to improve performance on specialized tasks. Adding a SKILL.md file to these saved functions creates a structured skill that models can reference and use. Over time, this allows your agent to build a toolbox of higher-level capabilities, evolving the scaffolding that it needs to work most effectively.

这与 [Skills](https://docs.claude.com/en/docs/agents-and-tools/agent-skills/overview) 的概念紧密相关——Skills 是存放可复用指令、脚本和资源的文件夹，帮助模型在专门任务上提升表现。给这些保存的函数加一个 SKILL.md 文件，就创建了一个结构化的技能，模型可以引用和使用。久而久之，这让你的智能体积累出一套更高层能力的工具箱，演进成它高效工作所需的脚手架。

Note that code execution introduces its own complexity. Running agent-generated code requires a secure execution environment with appropriate [sandboxing](https://www.anthropic.com/engineering/claude-code-sandboxing), resource limits, and monitoring. These infrastructure requirements add operational overhead and security considerations that direct tool calls avoid. The benefits of code execution—reduced token costs, lower latency, and improved tool composition—should be weighed against these implementation costs.

注意，代码执行也带来了自身的复杂性。运行智能体生成的代码需要一个安全的执行环境，具备合适的[沙箱](https://www.anthropic.com/engineering/claude-code-sandboxing)、资源限制与监控。这些基础设施要求增加了运维开销和安全考量，而直接工具调用无需面对。代码执行的好处——更低的 token 成本、更低的延迟、更好的工具组合能力——应当与其实现成本权衡。

## Summary

## 总结

MCP provides a foundational protocol for agents to connect to many tools and systems. However, once too many servers are connected, tool definitions and results can consume excessive tokens, reducing agent efficiency.

MCP 为智能体连接众多工具与系统提供了基础协议。然而，一旦连接的服务器过多，工具定义与结果就会消耗过量 token，降低智能体效率。

Although many of the problems here feel novel—context management, tool composition, state persistence—they have known solutions from software engineering. Code execution applies these established patterns to agents, letting them use familiar programming constructs to interact with MCP servers more efficiently. If you implement this approach, we encourage you to share your findings with the [MCP community](https://modelcontextprotocol.io/community/communication).

尽管这里许多问题看似新颖——上下文管理、工具组合、状态持久化——它们在软件工程里都有已知的解决方案。代码执行把这些成熟的模式应用到智能体上，让它们用熟悉的编程结构更高效地与 MCP 服务器交互。如果你实现了这种方式，我们鼓励你把发现分享给 [MCP 社区](https://modelcontextprotocol.io/community/communication)。
