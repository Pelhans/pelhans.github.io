---
layout:     post
title:      "深度学习笔记（五）"
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

## Softmax 损失函数
Softmax 损失函数的全程是 softmax with cross-entropy，其实就是 softmax 和交叉熵组合而成。损失函数公式如下：

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

原始的 softmax 损失函数非常优雅、简洁，广泛应用于分类问题。它的特点就是优化类间的距离非常棒，但是优化类内距离时比较弱。

## Softmax 损失函数的改进
### Focal loss

二分类问题中，交叉熵损失函数为：

$$ L = -y\log y^{'} - (1-y)\log(1-y^{'}) = 
\left\{
\begin{aligned}
-\log y^{'} && y=1 \\
-\log (1-y^{'}) && y=1
\end{aligned}
\right. $$

而 Focal loss 的公式如下所示：

$$ L_{fl} = 
\left\{
\begin{aligned}
    -\alpha(1-y^{'})^{r}\log y^{'} && y=1 \\
    -(1-\alpha)y^{'r}\log(1-y^{'}) && y=0
\end{aligned}
\right.
$$

首先在原基础上加了一个因子，其中 r>0 使得减少易分类样本的损失，使总损失函数更加专注于困难的、错分类的样本。此外，加入平衡因子 $\alpha$，用来平衡正负样本本身的比例不均。

###  Large-Margin Softmax Loss
Softmax loss 擅长学习类间的信息，因为它采用类间的竞争机制，只关心对于正确标签预测概率的准确性，忽略了其他非正确标签的差异，导致学习到的特征比较散。关于从最优化角度看 Softmax loss，可以看 [王峰的这篇文章](https://zhuanlan.zhihu.com/p/45014864)。那么怎么减小类内方差呢？下面给出 L-softmax loss的定义：

$$ L_{i} = -\log\left(\frac{e^{z_{i}}}{e^{z_{i}} + \sum_{j\neq y_{i}}e^{z_{j}}}      \right) $$

$$ z_{i} = |W_{y_{i}}| |x_{i}| \psi(\theta_{y_{i}}) $$

$$ z_{j} = |W_{y_{j}}| |x_{i}| \cos(\theta_{y_{j}}) $$

$$ \psi(\theta) = 
    \left\{
        \begin{aligned}
        \cos(m\theta), && 0 \leq \theta \leq \frac{\pi}{m} \\
        D(\theta), && \frac{\pi}{m} \lt \theta \leq \pi
        \end{aligned}
    \right. 
$$

其中 m 是整数，它决定类内的聚拢程度。m值越大则分类边缘越大，同时学习目标就越难。从下图可以看出m的效果。其中 第一列是m为1，也就是原始损失函数的时候，可以看到此时类间的距离还可以，但类内间距比较大。第二幅图是 m=2时，可以看到类内间距明显小了很多，而越往后随着m的增大类内间距越小。

![](/img/in-post/tensorflow/l_softmax.png)

那么为什么会有这种效果呢？个人理解，以前分类的类内角度搜索范围是 $\theta\in [0,\pi]$，在加了 m以后，它的范围缩小到 $\frac{\theta}{m}\in [0, \frac{\pi}{m}]$。因此类内间距就变小了。但带来的坏处也很明显，随着m的增大，搜索空间越来越小，学习难度也就越来越高。

![](/img/in-post/tensorflow/l_softmax_theta.png)

## KL 散度
}}}
KL 散度用来衡量两个分布之间的相似性，定义公式为：

$$ KL(p|q) = \sum_{i}p_{i}\log(\frac{p_{i}}{q_{i}}) $$

KL 散度是非负的，只有当 p 与 q 处处相等时，才会等于0。KL 散度也可以写成：

$$ KL(p|q) = \sum_{i}p_{i}\log p_{i} - p_{i}\log q_{i} = -l(p,p) + l(p,q) $$

因此 
$KL(p|q)$ 的散度也可以说是p与q的交叉熵和p信息熵的和。同时需要注意的时，KL散度对p、q是非对称的。

## Hinge 损失
Hinge 损失主要用于支持向量机中，用来解决SVM中的间距最大化问题。它的称呼来源于损失的形状，定义为：

$$ l(\hat{y}, y) = max(0, 1-y\hat{y}) $$

其中 y 的标签为1或-1. 如果分类正确，loss为0，否则为 $1-\hat{y}$。

## 指数损失

用于Adaboost集成学习算法中，特点是梯度比较大，定义为：

$$ l(\hat{y}, y) = e^{-\beta y \hat{y}} $$

## logistic 损失

logistic 取了 指数损失的对数形式，它的梯度变化相比于指数损失更加平缓：

$$ l(\hat{y}, y) = \frac{1}{\ln 2}\ln(1+e^{-y \hat{y}}) $$

# 回归问题
## 均方误差 MSE /L2损失
MSE，mean squared error的缩写，在 逻辑回归问题中用到过，公式定义为：

$$ MSE = \frac{1}{N}\sum_{i=1}^{N}(y_{i} - \hat{y_{i}})^{2} $$

L2 损失也常常作为正则项出现。当预测值与目标值差异很大时，梯度容易爆炸。

## 平均误差 MAE / L1 损失
Mean absolute loss(MAE)也被称为L1 Loss，是以绝对误差作为距离：

$$ MAE = \frac{1}{N}\sum_{i=1}^{N}|y_{i} - \hat{y_{i}}| $$

L1 损失具有稀疏性，为了惩罚较大的值，因此常常将其作为正则项添加到其他损失函数中作为约束，L1 损失的最大问题在于梯度在零点不平滑，会导致跳过极小值。

# 距离损失

知乎专栏中，一位博主总结了大量的距离损失[Tensorflow中的损失函数](https://zhuanlan.zhihu.com/p/45200767)，很值得参考。包含：

* 欧氏距离    
* 曼哈顿距离    
* 切比雪夫距离    
* 闵式距离    
* 标准化欧氏距离    
* 夹角余弦    
* 皮尔逊相关系数    
* 汉明距离    
* JS散度    
* Hellinger distance    
* 巴氏距离    
* MMD距离（Maximum mean discrepancy)
* Wasserstein distance

# 参考
[从最优化的角度看待Softmax损失函数](https://zhuanlan.zhihu.com/p/45014864)    
[Tensorflow中的损失函数](https://zhuanlan.zhihu.com/p/45200767)    
[【AI初识境】深度学习中常用的损失函数有哪些（覆盖分类，回归，风格化，GAN等任务）？](https://zhuanlan.zhihu.com/p/60302475)    
[【技术综述】一文道尽softmax loss及其变种](https://zhuanlan.zhihu.com/p/34044634)    
[损失函数改进之Large-Margin Softmax Loss](https://blog.csdn.net/u014380165/article/details/76864572)    
