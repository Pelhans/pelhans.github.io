---
layout:     post
title:      "Tensorflow 笔记（五）"
subtitle:   "常见损失函数"
date:       2019-04-15 00:15:18
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

# 概览

损失函数用来估计模型的预测值与真实值之间的不一致程度。模型通过预定的损失函数进行学习并更新参数，同时它也是一种评估模型好坏的方法。根据机器学习任务的不同，损失函数可分为分类和回归两大类，下面我们对常见的损失函数做一个总结。该文章将保持更新。

# 分类
## 0-1 损失

0-1 损失函数的形式是最简单的，它的公式为：

$$ l(y_{i}, \hat{y}_{i}) = 1- \delta_{y_{i}\hat{y}_{i}} $$

其中 $y_{i}$ 是真实值，$\hat{y}_{i}$ 是模型预测值。$\delta$为克罗内克符。上式含义为为当预测标签与真实标签一致时，损失为0，否则为1。

该损失函数因为不包含 x，因此无法用于反向传播。

##  交叉熵损失函数

我们知道信息熵的定义公式为 $-p\log p$。假设有两个概率分布 p(x) 和 q(x)，其中 p 是已知的分布(ground truth)，q 是未知的分布(预测分布)，则交叉熵函数是两个分布的互信息，可以反应两分布的相关程度：

$$ l(y_{i}, \hat{y}_{i}) = -\sum_{i=1}^{M}y_{i}\log\hat{y}_{i} $$

其中 M 表示类别总数。交叉熵损失函数在 logistic 回归中是权重 w 的凸函数，但在神经网络中不是凸的。更直观一些，上式可以用另一种方式表达：

$$ l_{i} = -(y_{i}\log\hat{y}_{i} + (1-y_{i})\log(1-\hat{y}_{i})) $$

当真实标签 $y_{i}$为 1时，函数后半部分消失，当真实标签为0时，后半部分消失。当预测值 $\hat{y}_{i}$和真实值相同时，损失函数为0.

### 交叉熵损失函数的导数

把相关公式先整理一下,首先假设输入x经由线性层得到输出z：

$$ z_{i} = w_{i}x_{i} + b $$

而后 z 通过softmax 转换为概率：

$$ \hat{y}_{i} = \frac{e^{z_{i}}}{\sum_{k}e^{z_{k}}} $$

结合 ground truth 计算损失函数：

$$ l = -\sum_{i=1}^{M}y_{i}\log\hat{y}_{i} $$

现在我们想求与之相关的各项导数：$$\frac{\partial{l}}{\partial{w}}$$, $$\frac{\partial{l}}{\partial{z}}$$, $$\frac{\partial{l}}{\partial{\hat{y}_{i}}}$$, $$\frac{\partial{\hat{y}_{i}}}{\partial{z}}$$,   $$\frac{\partial{z}}{\partial{w}}$$,   $$\frac{\partial{z}}{\partial{b}}$$。

$$\frac{\partial{z}}{\partial{w}} = x $$

$$\frac{\partial{z}}{\partial{b}} = 1 $$

$$
\begin{aligned}
\frac{\partial{\hat{y}_{i}}}{\partial z_{i}} & = \frac{\partial\frac{e^{z_{i}}}{\sum_{k}e^{z_{k}}}}{\partial z_{i}} \\
 & = \frac{e^{z_{i}}\sum_{k}e^{z_{k}} - e^{z_{i}}e^{z_{i}}}{(\sum_{k}e^{z_{k}})^{2}} \\
 & = \frac{e^{z_{i}}(\sum_{k}e^{z_{k}} - e^{z_{i}})}{(\sum_{k}e^{z_{k}})^{2}} \\
 & = \hat{y}_{i}(1 - \hat{y}_{i})
\end{aligned}
$$

上式有一个特别的说法，在其他博文里看到了，大体是说会存在 $$\frac{\partial{\hat{y}_{j}}}{\partial{ z_{i}}}$$ 的情况，此时上式第三步分步求导那就变了，因为 $e^{z_{j}}$对 $z_{i}$求导为0，所以分步求导第一项就消失了。虽然从公式上理解，但 ground truth 只有1个值为1，这种 j 对 i求导的情况是否真的有必要还是不太理解。这里取他们的说法。因此：

$$ \frac{\partial{\hat{y}_{i}}}{\partial z_{i}} = \hat{y}_{i}(\delta_{ij} - \hat{y}_{i}) $$

$$ \frac{\partial{l}}{\partial{\hat{y}_{i}}} = -\sum_{i=1}^{M}y_{i}\frac{1}{\hat{y}_{i}} $$

$$ \frac{\partial{l}}{\partial{z}} = \frac{\partial{l}}{\partial{\hat{y}_{i}}} * \frac{\partial{\hat{y}_{i}}}{\partial{z}} = -\sum_{i=1}^{M}y_{i}(\delta_{ij} - \hat{y}_{i}) $$

$$ \frac{\partial{l}}{\partial w} = \frac{\partial{l}}{\partial \hat{y}_{i}} * \frac{\partial{\hat{y}_{i}}}{\partial z} * \frac{\partial{z}}{\partial w} = -\sum_{i=1}^{M}y_{i}(\delta_{ij} - \hat{y}_{i}) x_{i} $$

## Softmax 损失函数


