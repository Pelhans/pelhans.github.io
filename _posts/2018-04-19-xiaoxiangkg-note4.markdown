---
layout:     post
title:      "知识图谱入门 (四)" 
subtitle:   "知识挖掘"
date:       2018-04-19 00:15:18
author:     "Pelhans"
header-img: "img/post-bg-universe.jpg"
header-mask: 0.3 
catalog:    true
tags:
    - Knowledge Graph
---


<span id="busuanzi_container_page_pv">
  本文总阅读量<span id="busuanzi_value_page_pv"></span>次
</span>

> 本节介绍了知识挖掘的相关技术，包含实体链接与消歧，知识规则挖掘，知识图谱表示学习。

* TOC
{:toc}

#  知识挖掘

知识挖掘是指从数据中获取实体及新的实体链接和新的关联规则等信息。主要的技术包含实体的链接与消歧、知识规则挖掘、知识图谱表示学习等。其中实体链接与消歧为知识的**实体**挖掘，知识规则挖掘属于结构挖掘，表示学习则是将知识图谱映射到向量空间而后进行挖掘。

## 实体消歧与链接

![](/img/in-post/xiaoxiangkg_note3/xiaoxiangkg_note3_5.png)

实体链接的流程如上图所示，这张图在前一章出现过，那里对流程进行了简要说明。此处对该技术做进一步的说明。

### 示例一: 基于生成模型的 entity-mention 模型


