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

LIMES利用度量空间中的三角不等式来计算实例相似性的悲观估计, 以此解决链接发现的数据尺度(scale)问题。基于这些近似，LIMES 可以过滤出大量不满足用户设置的匹配条件的实例对。然后计算其余实例对的真实相似性，并返回匹配的实例。最终结果表明， LIMES 所需的比较次数相比于传统暴力方明显减少，同时准确性得到的保留。

## 数学框架

给定度量空间$(A, m)$ 和 A 中的三个点 $x, y, z$， 则可以得到三角不等式：

$$ m(x, y) \leq m(x, z) + m (z, y) $$

没有其他限制的话， 还有

$$ m(x, z) \leq m(x, y) + m(y, z) $$

因此我们有

$$ m(x, y) - m(y, z) \leq m(x, z) \leq m(x, y) + m(y, z) $$

该式有两个主要含义：

* 当给定 x 到 y 的距离和 y 到 z 的距离时， 我们可以估算 x 到 z 的距离。 此时参考点 y 被称作 样本点(exemplar[Frey and Dueck, 2007])。样本点是度量空间 A 中的一部分， 我个人理解这个样本点类似于聚类后的中心点，我们用它来表达度量空间中的点，因此要求    
    * 它的数量是远小于目标空间的    
    * 在距离空间上均匀分布    
    * 各个样本点之间的距离尽可能的大    
    * 有了样本点， 我们可以大幅度减少比较次数，比如我们现在想做知识融合， 源空间 S 有10000个数据， 目标空间 T 也有 10000 个数据， 这样融合需要的比较次数就是 10000*10000。 但是我在目标空间选取了 100 个样本点， 现在比较的话， 比较次数是 10000* 1000 + 100 * 10000(一般不会这么大)。 这就小了两个数量级。    
* 假设 $m(x, y) - m(y, z) \gt \theta $。而 $m(x, z)$ 大于等于它， 因此可以得到 $m(x, z) \gt \theta$。

## 核心流程

LIMES 的核心流程为：

* Exemplar computation: 根据一定的方法生成样本点(exemplar)， LIMES 的方法将在下面说    
* Filtering: 采用三角不等式 $ m(s, y) - m(t, y) \leq \theta $ 过滤大量不满足要求的候选对    
* Similarity computation: 对满足要求的进一步计算精确的距离 $m(s, t)$， 若满足要求则匹配成功    
* Serialization: 最终满足要求的将会被存储为用户指定的格式

流程如下图所示

![](/img/in-post/kg_paper/el_limes_workflow.jpg)

### Exemplar computation

样本点生成算法如下图所示

![](/img/in-post/kg_paper/el_limes_exemplar_computation.jpg)

首先给定目标空间 T 和目标样本点数量 n。输出样本点集合 E 和将 T内点分配到距离最近的 $e\in E$ 的集合。

* 随机选取初始点 $e_{1} \in T $    
* 集合 $ E = E \cup {e_{1}},~~ \eta = e_{1} $    
* 计算样本点 $e_{1}$ 到目标空间 T 内所有点 t 的距离(方便后面计算，减少重复计算量)）    
* 当样本点数量少于 n 时， 进行循环：
    * 获取满足 $ e^{'} \in arg\max_{t}\sum_{t\in T}\sum_{e\in E}m(t, e)$. 即对于 T 内的所有点，计算它到所有E 内样本点的距离和， 并选取使该距离最大的那个作为新的样本点 $e^{'}$    
    * 将该点加入到样本点集合 E 中    
    * 计算 $e^{'}$ 到 T 内所有点得距离    
* 将 T 内的每个样本点 t 映射到离它最近的样本点上    

可以看到算法的时间复杂度为 
$O(|E||T|)$。

### Matching Based on Exemplars

这个其实前面说过， 这里直接贴一个论文中的算法

![](/img/in-post/kg_paper/el_comparison_algorithm.jpg)

最坏的情况下匹配算法的复杂度为 
$O(|S||T|)$， 总复杂度为
$O((|E| + |S|)|T|)$。 比暴力还大， 不过参数设置没问题的话， LIMES 还是能够大幅度减少比较次数和运行时间的。

## Evaluation

论文内评估主要回答四个问题：

* 样本点的数目 n 设为多少最好    
* 阈值 $\theta$ 和比较次数的关系是怎样的    
* 两个知识库， 那个作为 S, 那个作为 T 重要么？    
* LIMES 和其他的比表现如何

对于第一个问题， 论文指出， 当 $\theta \geq 0.9$ 时， 
$ n = \sqrt{|T|}$。即样本点的数量为目标知识库大小的平方根。 随着阈值 $\theta$ 的减小， 所需样本点的数量也逐渐增多， 不过变化相对于 T 的变化不明显。

第二个问题， 低阈值需要更多的样本点才能降低比较次数。

第三个问题，实验表明有影响， 但不大， 一般来说 目标空间小于等于源空间大小会稍微好一点， 但影响在 5% 以下

最后一个问题， LIMES 和 SILK 在合成数据和真实数据上比较， 表明 LIMES 比 SILK 运行时间少很多， 在阈值为 0.95 时， LIMES 的运行时间约为 SILK 的 1.6%。

