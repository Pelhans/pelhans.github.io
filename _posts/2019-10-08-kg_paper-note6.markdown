---
layout:     post
title:      "实体链接(二)"
subtitle:   "LIMES 中的大规模数据链接方法"
date:       2019-10-08 00:15:18
author:     "Pelhans"
header-img: "img/tensorflow.jpg"
header-mask: 0.3 
catalog:    true
tags:
    - Knowledge Graph
---


* TOC
{:toc}

# LIMES — A Time-Efficient Approach for Large-Scale Link Discovery on theWeb of Data

论文提出并评价了度量空间中一种新的链接发现方法 LIMES(Link Discovery Framework for metric spaces)。该的方法在映射过程中利用度量空间的数学特性过滤掉大量不满足映射条件的实例对。 

LIMES利用度量空间中的三角不等式来计算实例相似性的悲观估计来解决链接发现的数据尺度(scale)问题。基于这些近似，LIMES 可以过滤出大量不能满足用户设置的匹配条件的实例对。然后计算其余实例对的真实相似性，并返回匹配的实例。最终结果表明， LIMES 所需的比较次数相比于传统暴力方明显减少，同时准确性得到的保留。


