---
layout:     post
title:      "知识图谱入门 (八)" 
subtitle:   "语义搜索"
date:       2018-04-26 00:15:18
author:     "Pelhans"
header-img: "img/post-bg-universe.jpg"
header-mask: 0.3 
catalog:    true
tags:
    - Knowledge Graph
---


> 本节对语义搜索做一个简单的介绍，而后介绍语义数据搜索、混合搜索。最后介绍语义搜索的交互范式。

* TOC
{:toc}

# 语义搜索简介

不同的搜索模式之间的技术差异可以分为:

* 对用户需求的表示(query model)    
* 对底层数据的表示(data model)    
* 匹配方法(matching technique)

以前常用的搜索是基于文档的检索(document retrieval )。信息检索(IR)支持对文档的检索，它通过轻量级的语法模型表示用户的检索需求和资源内容，如 AND OR。即目前占主导地位的关键词模式：词袋模型。它对主题搜索的效果很好，但**不能应对更加复杂的信息检索需求**。

数据库(DB) 和知识库专家系统(Knowledge-based Expert System)可以提供更加精确的答案(data retrieval)。它使用表达能力更强的模型来表示用户的需求、利用数据之间的内在结构和语义关联、允许复杂的查询、返回精确匹配查询的具体答案。

语义搜索答题可分为两类：

* DB 和KB 系统属于重量级语义搜索系统，它对语义**显示的和形式化的建模**，例如 ER图或 RDF(S) 和OWL 中的知识模型。主要为**语义的数据检索系统**。

* 基于语义的IR 系统属于轻量级的语义搜索系统。采用轻量级的语义模型，例如分类系统或者辞典。语义数据(RDF)嵌入文档或者与文档关联。它是基于**语义的文档检索系统**。

语义搜索的流程图如下图所示：

![](/img/in-post/xiaoxiangkg_note8/xiaoxiangkg_note8_1.png)

# 语义数据搜索

语义是基于标准化的逻辑语言，从而
