---
layout:     post
title:      "Metapath2vec"
subtitle:   "一种异构图神经网络的潜入方法"
date:       2021-12-13 00:15:18
author:     "Pelhans"
header-img: "img/attention.jpg"
header-mask: 0.3 
catalog:    true
tags:
    - NLP
---


* TOC
{:toc}

# 概览
最近看到了论文 www-2021 的最佳论文 Heterogeneous Graph Neural Network via Attribute Completion, 文章里提出了异构信息网络属性补全问题的解决方案。从文章内容来看，大体可以分为 2 个阶段：1）基于 Metapath2vec 方法对图节点进行 embedding，得到节点的向量表示；2）以原网络中有属性的节点作为监督信号来训练网络，用 attention 方法，对属性进行补全 。因为是第一次接触图相关的东西，所以这里对第一阶段的 Metapath2vec 方法做一个调研与了解。

## 问题定义

一个图 G，由顶点 V 和边的集合 E 组成, $$ G=(V,E), E\in (V\times V) $$。一个网络的表示就是将网络中的每个节点用一个固定维数的向量进行表示。该表示应该具有以下性质：    
* 适应性：网络的表示必须能适应网络的变化，当不断有新的节点和边添加进来时，网络表示需要适应网络的正常演化。    
* 属于同一个社区的节点有着类似的表示：网络中往往会出现一些特征相似的点构成的团状数据结构，这些节点表示成向量后必须相似。    
* 低维：维数不能过高    
* 其他 nlp embedding 具有的性质

从发展历程上来看，Metapath2vec 方法的发展经由 Word2vec --> Deepwalk --> Node2vec --> Metapath2vec 。接下来分别介绍一下

# word2vec 
这个可以看前面的文章 [word2vec](http://pelhans.com/2019/04/29/deep_learning-note11/)

# Deepwalk
对应论文 Online Learning of Social Representations    

直接将图结构进行嵌入不好做，因此 Deepwalk 参考了 Word2vec 的思想，将图网络进行采样，得到一个序列，采样方法就是在网络上不断的重复随机选择游走过程，记录所经过的路径。

# Node2vec
论文：node2vec: Scalable Feature Learning for Networks

Deepwalk 是全随机游走，但图嘛，出了全随机外，还有俩天生的游走方式：BFS 和 DFS。BFS 倾向于在初始节点的周围游走，可以反映出一个节点的邻居的微观特性；而 DFS 会跑的比较远，可以反应节点邻居的宏观特性。

Node2vec 就综合考虑了这两点，给出了一套游走方案（出发节点 t, 当前节点 v, 下一个节点 x)：    
* 如果 t 和 x 相等，那么采样概率为 $$\frac{1}{p}$$，p 叫返回概率，大于 1 时 倾向往远走，小于 1 时，倾向往回走    
* 如果 t 和 x 相连，则采样概率为 1    
* 如果 t 和 x 不相连，采样概率为 $$\frac{1}{q} $$， q 叫 出入参入，大于 1 时往远跑概率低，像 BFS，小于 1 就像 DFS

不同的 p 和 q 设置体现了对网络不同部分的重视。到这里引入两个概念，同质性和结构对等性：    
* 同质性：距离相近的节点的 embedding 应该尽量相似    
* 结构对等性：结构上相似的节点 embedding 应该尽量接近

论文中说，BFS 更多抓住了网络的结构性，DFS 更能体现网络的同质性。这个有点反直觉，毕竟 BFS 就采集周围的一圈，怎么体现结构性？这里借鉴 [探索node2vec同质性和结构性之谜](https://zhuanlan.zhihu.com/p/68453999) 的结论。那就是作者说的同质性和这个定义还不太一样，论文里的同质性是指模型能找出每个 簇团的边界，是的簇内彼此联系的紧密成都要超过簇外节点的联系，这就要求更大的感受野，DFS 能做到这个。至于结构性，论文里的结构性是只能够充分学习微观上的局部结构，这样的话，BFS 在周围游走会捕捉到这个。

# Metapath2vec
Metapath2vec 使用基于 Metapath 的随机游走方案。而后用 skip-gram 模型完成顶点的嵌入。所谓异质网络是指网络节点和边类型的总量大于 2 的那种。前面的几种游走方案在异构网络上不能直接用，比如作者 - 论文 - 研究院 这种网络。直接随机游走的效果很差，没有章法。而采用 Metapath 的方法，沿着预先定制的 路径，比如 作者 -- 论文 -- 作者，作者 -- 研究院 -- 作者  这种指定的 metapath 。在 metapath 上进行采样可以学到我们想要的网络结构信息，有了章法，网络表示会更准确。

具体来说：     
* 两个点间有边，且下一个节点属于我们定义好的 metapath 上的下一个类型的节点，则概率为 $$\frac{1}{周围指定类型节点的集合大小} $$    
* 两个点间右边，但类型不对，则概率为 0    
* 两个点间没有边，概率为 0


