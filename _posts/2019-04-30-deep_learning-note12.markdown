---
layout:     post
title:      "深度学习笔记（十二）"
subtitle:   "Glove"
date:       2019-04-30 00:15:18
author:     "Pelhans"
header-img: "img/dl_background.jpg"
header-mask: 0.3 
catalog:    true
tags:
    - Deep Learning
---

* TOC
{:toc}

# 概述

Glove 是 Global Vectors 的缩写，是一种词向量训练方法。作者认为传统基于 SVD 的 word-word 共现计数模型的优点是利用了全局统计信息，训练速度快，但缺点是对高频词汇较为偏向，并且仅能概括词对间的相关性，对语义信息表示的并不好。同时我们常用的 word2vec 虽然对语义信息表示的更好，但是没有包含更多的统计信息。因此就想结合二者的优势。大体上就是抛弃 word2vec 那种简单向量加和的方式，转而设计了一个可以利用统计信息的损失函数。

# 预备知识-SVD 介绍

SVD 是 奇异值分解（singular Value Decomposition） 的简称，在机器学习领域被广泛应用。

对方阵求特征向量和特征值我们都比较熟悉。比如：

![](/img/in-post/ml_mianshi/fangzhen_exam1.jpg)

![](/img/in-post/ml_mianshi/fangzhen_exam2.jpg)

求得特征值和特征向量后，我们就可以得到如下分解：

$$ A = W\Sigma W^{-1} $$

其中 W 就是 n 个特征向量张成的 nxn 维矩阵，$$ \Sigma $$ 是特征值为主对角线的 nxn 维矩阵。当 W 为酉矩阵时（即$$W^{T}W = I $$），则有

$$ A = W\Sigma W^{T} $$

那当A 不是方阵的话该怎么办呢？就得用  SVD 了。假设 A 现在的shape 是 mxn。则我们定义矩阵 A 的 SVD 为：

$$ A = U\Sigma V^{T} $$

其中 U 是 mxm 矩阵，$$\Sigma$$ 是 mxn 的矩阵，除了主对角线上的元素以外全为0，主对角线上的每个元素都成为奇异值， V 是一个 nxn 的矩阵。U 和 V 都是酉矩阵。

怎么求 U、$$\Sigma$$ 、V 呢？首先对于 V，我们通过计算 $$ A^{T}A$$ 的特征向量得到，U 则通过计算 $$ AA^{T}$$ 的特征向量得到。中间的奇异值矩阵则通过$$Av_{i} = \sigma_{i}u_{i}$$ 得到。

## SVD 在 word-word 共现统计中的应用

可以看[知乎-AI 机动队](https://zhuanlan.zhihu.com/p/60208480) 的示例。

# Glove

有了上述了解，那 Glove 究竟做了哪些改动呢？先放结论，Glove 设计了如下损失函数：

$$ L = \sum_{i,j=1}^{V}f(X_{ij})(w_{i}^{T}\tilde{w}_{j} + b_{i} + \tilde{b}_{j} - log X_{ij})^{2} $$

其中 

$$ w_{i}w_{j} = log P(i|j) $$

$$ w_{i}$$ 和 $$\tilde{w}_{j}$$ 都是要学习的参数。$$X_{ij}$$ 是单词j 出现在单词 i 上下文的次数。b 是偏置。$$ f(X_{ij})$$ 是权重函数：

$$ f(x) = (\frac{x}{x_{max}})^{\alpha} ~~~if x < x_{max} ~~~else~~~~1 $$

其中 $$ x_{max}$$ 是人工设置的超参，原作者采用了 100.$$ \alpha$$ 也是超参，作者用的是 $$ 3/4$$。可以看到，f 其实相当于一个限制，给共现次数的权重加了一个上限。

接下来说这个Loss，较为正式的解释请看[Glove 详解](http://www.fanyeong.com/2018/02/19/glove-in-detail/)。这里说一下个人的理解。抛开两个偏置和权重，核心是 $$ w_{i}^{T}\tilde{w}_{j} - log X_{ij}$$。可以看到，这个差值是希望我们学习到的两个向量间的内积尽可能的接近共现次数的对数，这不就把统计信息用上了么。。。

# Ref

1. [Glove 详解](http://www.fanyeong.com/2018/02/19/glove-in-detail)    
2. [CS224N笔记(二)：GloVe](https://zhuanlan.zhihu.com/p/60208480)    
