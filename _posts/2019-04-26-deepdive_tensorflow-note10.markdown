---
layout:     post
title:      "深度学习笔记（十）"
subtitle:   "Attention 基础"
date:       2019-04-26 00:15:18
author:     "Pelhans"
header-img: "img/tensorflow.jpg"
header-mask: 0.3 
catalog:    true
tags:
    - Deep Learning
---

> 眼睛看过只是别人的，整理出来并反复学习才是自己的。

* TOC
{:toc}

# 什么是 Attention
在传统的 encoder-decoder 模型中，encoder 读取输入的句子将其转换为一个定长的向量，然后decoder 再将这个向量解码为对应的输出。然而，此时信息被压缩到一个定长的向量中，内容比较复杂，同时对于较长的输入句子，转换成定长向量可能带来一定的损失，因此随着输入序列长度的上升，这种结构的效果面临着挑战。

Attention 机制可以解决这种由长序列到定长向量转化而造成的信息损失问题。Attention 即注意力，它和人看文章或者图片时的思路类似，即将注意力集中到某几个区域上来帮助当前决策。现在我们以翻译为例，演示 attention 的工作状态：

![](/img/in-post/tensorflow/attention_example.gif)

可以看到，在 decoder 每个输出时， attention 将重点放在几个相关的输入上。 Attention 的这种关注是通过对不同的输入分配不同的权重实现的。

# Attention 的原理

![](/img/in-post/tensorflow/attention_yuanli.png)

上图是一个比较经典的 attention 图，其中 $\overleftarrow{h_{.}}$ 和 $\overrightarrow{h_{.}}$ 是隐藏状态的输出，现在我们将做 decoder计算 decoder 状态 $s_{t}$：

* 计算每一个输入位置 j 与 当前输出位置的关联性$e_{t,j} = align(s_{t-1}, h_{j})$，写成向量形式为：$\overrightarrow{e_{t}} = (align(s_{t-1}, h_{1}),\dots,align(s_{t-1}, h_{T})) $。$e_{ij}$表示一个对齐模型，计算方式有很多种，不同的计算方式代表不同的 attention 模型，最简单的且最常用的对齐模型是矩阵相乘，常见的对齐方式为：    
$$
\alpha_{t,j}(h_{j}, s_{t-1}) =
\left\{
\begin{aligned}
h_{j}^{T}s_{t-1} && dot \\
h_{j}^{T}W_{a}s_{t-1} && general \\
v_{a}^{T}tanh(W_{a}[h_{j}^{T};s_{t-1}]) && concat
\end{aligned}
\right.
$$}    
* 对 $\overrightarrow{e_{t}}$进行 softmax 操作得到归一化的概率分布    
* 利用刚刚得到的概率分布，可以进行加权求和得到相应的 context vector $\overrightarrow{c_{t}} = \sum_{j=1}^{T}\alpha_{tj}h_{j} $。    
* 根据 $\overrightarrow{c_{t}}$ 和 $s_{t-1}$计算下一个状态 $s_{t} = f(s_{t-1}, y_{t-1}, c_{t})$

上面比较重要的步骤是计算关联性权重，得到 attention 分布，从而判断那些隐藏单元比较重要并赋予较大权重。通过引入 attention机制，我们在预测decoder 每一个状态时，综合考虑了全文序列，避免单一向量时的长程信息丢失的问题，使得模型效果得到极大改善。

# Attention 机制的分类
可以从多角度对 Attention 进行分类，如从信息选择的方式上，可以分为 Soft attention 和 Hard attention。从信息接收的范围上可分为 Global attention 和 Local attention。

## Soft attention 与 Hard attention
我们前面描述的传统 Attention 就是 Soft Attention，它选择的信息是所有输入信息在对齐模型分布下的期望。而 Hard Attention 只关注到某一位置上的信息，一般而言，Hard Attention 是实现有两种：一种是选取概率最高的输入信息，另一种是在对齐模型的概率分布上进行随机采样。硬性注意力的一个缺点是基于最大采样或随机采样的方式来选择信息。因此最终的损失函数与注意力分布之间的函数关系不可导，因此无法使用在反向传播算法进行训练。为了使用反向传播算法，一般使用软性注意力来代替硬性注意力。硬性注意力需要通过强化学习来进行训练。

## Global attention 与 Local attention
Global Attention 和传统的 注意力机制一样，所有的信息都用来计算 context vector 的权重。这会带来一个明显的缺点，即所有的信息都要参与计算，这样计算的开销就比较大，而别当encoder 的句子比较长时，如一段话或一篇文章。因此 Local Attention 就被提了出来，它是一种介于Kelvin Xu所提出的Soft Attention和Hard Attention之间的一种Attention方式，即把两种方式结合起来。下图是 Local 的图示

![](/img/in-post/tensorflow/local_attention.png)

上图中， $\hat{h}_{s}表示 全部的 encoder 向量，$h_{t}$表示 时间步 t 的 decoder 输出。Local Attention 首先会为 decoder 端当前的词预测一个 encoder 端对齐的位置(aligned position)$p_{t}$，而后基于 $p_{t}$选择一个窗口，用于计算 context vector $c_{t}$，$p_{t}$的计算公式为：

$$ p_{t} = S*sigmoid(v_{p}^{T}tanh(W_{p}h_{t})) $$

其中 S 表示 encoder 端的句子长度，$v_{p}$和 $w_{p}$是模型参数。得到 $p_{t}$ 后，$c_{t}$的计算将值关注窗口 $[p_{t}-D, p_{t}+D] 内的2D+1 个 encoder 输入。对齐向量 $a_{t}$的计算公式为：

$$ a_{t}(s) = align(h_{t}, \hat{h}_{s})exp(\frac{(s-p_{t})^{2}}{2\sigma^{2}}) $$

Global Attention 和 Local Attention 各有优劣，实际中 Global 的用的更多一点，因为：

* Local Attention 当 encoder 不长时，计算量并没有减少    
* 位置向量$p_{t}$的预测并不非常准确，直接影响到 Local Attention 的准确率


