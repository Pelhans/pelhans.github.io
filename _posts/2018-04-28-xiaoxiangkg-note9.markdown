---
layout:     post
title:      "知识图谱入门 (九)" 
subtitle:   "语义问答"
date:       2018-04-28 00:15:18
author:     "Pelhans"
header-img: "img/post-bg-universe.jpg"
header-mask: 0.3 
catalog:    true
tags:
    - Knowledge Graph
---


> 本节对知识问答的概念做一个概述并介绍KBQA实现过程中存在的挑战，而后对知识问答主流方法做一个介绍。

* TOC
{:toc}

# 知识问答简介

问答系统的历史如下图所示：

![](/img/in-post/xiaoxiangkg_note9/xiaoxiangkg_note9_1.png)

可以看出，整体进程由基于模板到信息检索到基于知识库的问答。基于信息检索的问答算法是基于关键词匹配+信息抽取、浅层语义分析。基于社区的问答依赖于网民贡献，问答过程依赖于关键词检索技术。基于知识库的问答则基于语义解析和知识库。

根据问答形式可以分为一问一答、交互式问答、阅读理解。一个经典的测评数据集为QALD，主要任务有三类：

* 多语种问答，基于Dbpedia    
* 问答基于链接数据    
* Hybrid QA，基于RDF and free text data

## 知识问答简单流程分类


