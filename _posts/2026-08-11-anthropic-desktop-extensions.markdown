---
title: "Introducing desktop extensions (DXT)（发布桌面扩展 DXT）"
date: 2026-08-11
layout: post
liquid: false
tags: [语雀, 计算机, Anthropic Engineering 翻译]
categories: [计算机]
yuque: true
---

> 原文作者：David Dworken | 发布于 2025-05-21 | 原文链接：https://www.anthropic.com/engineering/desktop-extensions

# Introducing desktop extensions (DXT)

Desktop extensions (DXT) are a new open standard for packaging and distributing MCP servers, making it easy for users to install and use tools that connect Claude to their local data and systems.

桌面扩展（DXT）是一套用于打包和分发 MCP 服务器的全新开放标准，让用户能轻松安装并使用那些把 Claude 连接到本地数据与系统的工具。

## The problem with MCP installation today

## 当前 MCP 安装的问题

Today, installing an MCP server often means cloning a repo, installing dependencies, setting environment variables, and editing a config file by hand. That's a meaningful barrier for non-developers, and even for developers it's fiddly.

今天，安装一个 MCP 服务器往往意味着克隆仓库、安装依赖、设置环境变量、并手动编辑配置文件。这对非开发者是实实在在的门槛，对开发者而言也很琐碎。

## What is DXT?

## 什么是 DXT？

DXT is a single-file format (a `.dxt` zip containing a `manifest.json`, the server code, and any assets) that bundles everything an MCP server needs. Users install it with one click; Claude Desktop handles the rest.

DXT 是一种单文件格式（一个包含 `manifest.json`、服务器代码和任何资源的 `.dxt` zip），打包了 MCP 服务器所需的一切。用户一键安装，其余交给 Claude Desktop。

```
my-extension.dxt
├── manifest.json     # name, version, tools, config schema
├── server/           # the MCP server code
└── assets/           # icons, docs
```

The `manifest.json` declares the tools the extension provides and the configuration it needs:

`manifest.json` 声明了扩展提供的工具，以及它所需的配置：

```
{
  "name": "Google Drive",
  "version": "1.0.0",
  "tools": ["getDocument", "listFiles"],
  "config": {
    "apiKey": { "type": "string", "secret": true }
  }
}
```

## Why it matters

## 为何重要

DXT makes MCP servers as easy to install as a browser extension. It lowers the barrier for users to connect Claude to the tools and data they already use, and gives developers a standard distribution format.

DXT 让 MCP 服务器的安装像安装浏览器扩展一样简单。它降低了用户把 Claude 连接到自己已在使用的工具与数据的门槛，并给开发者一套标准的分发格式。

## Getting started

## 开始使用

Developers can package an existing MCP server as a `.dxt` file; users can install it directly in Claude Desktop. The spec is open, so any client can adopt it.

开发者可以把已有的 MCP 服务器打包成 `.dxt` 文件；用户可以直接在 Claude Desktop 中安装。规范是开放的，任何客户端都能采用。
