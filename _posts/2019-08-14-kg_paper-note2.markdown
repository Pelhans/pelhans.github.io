---
layout:     post
title:      "关系抽取（二）"
subtitle:   "Enriching Pre-trained Language Model with Entity Information for Relation Classification"
date:       2019-08-14 00:15:18
author:     "Pelhans"
header-img: "img/tensorflow.jpg"
header-mask: 0.3 
catalog:    true
tags:
    - Knowledge Graph
---

> 多读书, 多看报,多吃零食, 多睡觉.

* TOC
{:toc}

# 论文概览

论文链接: [Enriching Pre-trained Language Model with Entity Information for Relation Classification](https://arxiv.org/abs/1905.08284?context=cs)

## 论文要解决的问题是什么

尝试使用预训练模型 BERT 到句子级别关系抽取任务上

## 这篇文章得主要贡献是什么

1) 率先将 BERT 用在了关系抽取任务上, 探索了实体和实体位置在预训练模型中的结合方式    
2) 可以通过在实体前后加标识符得方式表明实体位置, 代替传统位置向量得做法.论文也证实了这种方法得有效性

# 论文详解

首先上一个模型结构图:

![](/img/in-post/kg_paper/R-BERT.jpg)

模型整体分为几个部分: 输入, BERT, 输出整合. 下面逐个介绍.

## 输入

假设输入的句子为: "The kitchen is the last renovated part of the house .", 在送入 BERT 之前,它将受到以下处理:

* 开头添加CLS 符号: "[CLS] The kitchen is the last renovated part of the house ."    
* 第一个实体得前后添加 \$ 符号: "[CLS] The \$ kitchen \$ is the last renovated part of the house ."    
* 第二个实体前后添加 # 符号: "[CLS] The \$ kitchen \$ is the last renovated part of the # house # ."

两个实体前后添加特殊符号的目的是标识两个实体, 让模型能够知道这两个词的特殊性,相当于变相指出两个实体得位置. 此时输入的维度为[batch size n, max_length m, hidden size d]

## BERT 

这里对 BERT 就不做过多的介绍, 直接看它的输出, 这里需要用到它的 CLS 位置输出和序列输出. [CLS] 位置的输出可以作为句子的向量表示, 记作 $H_{0}$, 它的维度是 [n, d]. 它经过 tanh 激活和线性变换后得到, $W_{0}$ 的维度是 [d, d], 因此 $H^{'}$ 的维度就是[n, d]

$$H^{'}_{0} = W_{0}(tanh(H_{0})) + b_{0} $$

除了利用句向量之外, 论文还结合了两个实体得向量. 实体向量通过计算BERT 输出的实体各个字向量的平均得到, 假设BERT 输出的 实体1得开始和终止向量为 $H_{i}$, $H_{j}$. 实体2得为 $H_{k}$, $H_{m}$. 那么实体1 和 2得向量表示就是:

$$ e1 = \frac{1}{j-i+1}\sum_{t=i}^{j}H_{t} $$

$$ e2 = \frac{1}{m-k+1}\sum_{t=k}^{m}H_{t} $$

维度为 [n, d], 得到的实体向量也需要经过激活函数和线性层, $W_{1}$ 和 $W_{2}$ 的维度都是 [d, d]:

$$ H^{'}_{1} = W_{1}e_{1} + b_{1} $$

$$ H^{'}_{2} = W_{2}e_{2} + b_{2} $$

因此它俩得维度也都是 [n, d]. 最后把 $$H^{'}_{0}, H^{'}_{1}, H^{'}_{2}$$ 连接起来得到一个综合向量[n, 3d] 输入到线性层并做softmax 分类.

$$ h^{''} = W_{3}[concat(H^{'}_{0}, H^{'}_{1}, H^{'}_{2})] + b_{3} $$

$$ p = softmax(h^{''}) $$

其中 $W_{3}$ 的维度是 [关系数量 L, 3d], 因此 $h^{''}$ 得维度是 [n, L]. 经过得到了每句话得关系类别概率分布,完成分类.

# 效果

在 SemEval-2010 Task 8 dataset 上做了实验, 实验证明 R-BERT 比其他的模型如CR-CNN, ATTENTION- CNN 等效果都要好. 除此之外,作者的实验还表明:

* 移除实体前后得标识符会使模型得 F1 从 89.25% 降至 87.98%. 说明标识符确实可以帮助模型提供实体信息    
* 在 BERT 输出层仅利用 CLS 得句子向量而不利用实体向量会使得模型 F1 降至 87.98%(和标识符得影响差不多), 说明想办法主动明确实体信息对模型是有帮助的

# 个人启发

* 在 BERT 里采用这种方法标注实体位置确实是第一次见, 而且还蛮有效得, 之前一直想直接给 BERT 位置向量, 是不是可以 PK 一下或者结合一下?    
* 想办法明确实体给模型看对模型是有好处得

# 参考

论文链接: [Enriching Pre-trained Language Model with Entity Information for Relation Classification](https://arxiv.org/abs/1905.08284?context=cs)
