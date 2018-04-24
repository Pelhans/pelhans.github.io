---
layout:     post
title:      "知识图谱入门 (七)" 
subtitle:   "知识推理"
date:       2018-04-24 00:15:18
author:     "Pelhans"
header-img: "img/post-bg-universe.jpg"
header-mask: 0.3 
catalog:    true
tags:
    - Knowledge Graph
---


> 本节对本体任务推理做一个简单的介绍，并介绍本体推理任务的分类。而后对本体推理的方法和工具做一个介绍。

* TOC
{:toc}

# 知识推理简介

所谓推理就是通过各种方法**获取新的知识或者结论**，这些知识和结论满足语义。其具体任务可分为可满足性(satisfiability)、分类(classification)、实例化(materialization)。

可满足性可体现在本体上或概念上，在本体上即本体可满足性是检查一个本体是否可满足，即检查该本体是否有模型。如果本体不满足，说明存在不一致。概念可满足性即检查某一概念的可满足性，即检查是否具有模型，使得针对该概念的解释不是空集。

![](/img/in-post/xiaoxiangkg_note7/xiaoxiangkg_note7_1.png)

上图是两个不可满足的例子，第一个本体那个是说，Man 和 Women 的交集是空集，那么就不存在同一个本体Allen 既是Man 又是Women。 第二个概念是说概念Eternity是一个空集，那么他不具有模型，即不可满足。

分类，针对Tbox的推理，计算新的概念包含关系。如:

![](/img/in-post/xiaoxiangkg_note7/xiaoxiangkg_note7_2.png)

即若Mother 是 Women的子集，Women是 Person的子集，那么我们就可以得出 Mother是 Person的子集这个新类别关系。
