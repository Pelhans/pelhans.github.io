---
layout:     post
title:      "知识图谱入门 (三)" 
subtitle:   "知识抽取"
date:       2018-03-19 00:15:18
author:     "Pelhans"
header-img: "img/post-bg-universe.jpg"
header-mask: 0.3 
catalog:    true
tags:
    - Knowledge Graph
---


> 本节介绍了针对结构化数据、非结构化数据、半结构化数据的知识抽取方法。

* TOC
{:toc}

#  知识抽取的概念

知识抽取，即从不同来源、不同结构的数据中进行知识提取，形成知识(结构化数据)存入到知识图谱。大体的任务分类与对应技术如下图所示：

![](/img/in-post/xiaoxiangkg_note3/xiaoxiangkg_note3_1.png) 

## 知识抽取的子任务

- 命名实体识别    
    - 检测: 北京是忙碌的城市。        [北京]： 实体
    - 分类：北京是忙碌的城市。        [北京]:  地名    
- 术语抽取    
从语料中发现多个单词组成的相关术语。    
- 关系抽取    
王思聪是万达集团董事长王健林的独子。$$\rightarrow~$$ [王健林] <父子关系> [王思聪]    
- 事件抽取    
例如从一篇新闻报道中抽取出事件发生是触发词、时间、地点等信息，如图二所示。    
- 共指消解    
弄清楚在一句话中的代词的指代对象。例子如图三所示。

![](/img/in-post/xiaoxiangkg_note3/xiaoxiangkg_note3_2.png)

![P3](/img/in-post/xiaoxiangkg_note3/xiaoxiangkg_note3_3.png)

# 面向非结构化数据的知识抽取

## 实体抽取

实体抽取抽取文本中的原子信息元素，通常包含任命、组织/机构名、地理位置、时间/日期、字符值等标签，具体的标签定义可根据任务不同而调整。如：

![](/img/in-post/xiaoxiangkg_note3/xiaoxiangkg_note3_4.png)

单纯的实体抽取可作为一个序列标注问题，因此可以使用机器学习中的HMM、CRF、神经网络等方法解决。

## 实体识别与链接


