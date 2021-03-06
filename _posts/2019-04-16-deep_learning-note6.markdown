---
layout:     post
title:      "深度学习笔记（六）"
subtitle:   "正则化项"
date:       2019-04-16 00:15:18
author:     "Pelhans"
header-img: "img/dl_background.jpg"
header-mask: 0.3 
catalog:    true
tags:
    - Deep Learning
---

* TOC
{:toc}

# 概览

**正则化的本质就是对参数的先验假设**。通过对参数的正则化，以偏差的增加换取方差的减少，从而使得机器学习算法的泛化性增加。**偏差度量着偏离真实函数或者参数的误差期望，而方差度量着数据上任意特定采样可能导致的估计期望的偏差**。因此高偏差相当于模型欠拟合，而高方差是过拟合，导致泛化能力弱。大多数机器学习算法则是在偏差-方差，经验风险-结构风险之间做权衡。将正则项 $\Omega(\theta)$加入目标函数 J，有：

$$ \tilde{J}(\theta; X, y) = J(\theta; X, y) + \alpha\Omega(\theta) $$

其中 $\alpha\in[0, \infty]$ 是权衡范数惩罚项$\Omega$和标准目标函数 J 相对贡献的超参数，当 $\alpha$为0时相当于没有正则化。添加正则化项可以减小某些衡量标准下参数的规模。

一般而言，在神经网络中的正则化项只对权重 w 做，而不对偏置 b 做，精确拟合偏置所需的数据通常比拟合权重少得多。这是因为每个权重 w 会指定两个变量如何相互作用，而每个偏置只控制单一变量，这意味着我们不对其进行正则化也不会导致太大的方差。另外正则化偏置参数可能会导致明显的欠拟合。

下面介绍常用的正则化方法，包含 L1、L2正则化、Dropout、早停止、Bagging等集成方法、参数绑定和参数共享。

# L2 正则化

L2正则也被称作岭回归或 Tikhonov 正则。L2 正则化项公式为：

$$ \Omega(\theta) = \frac{1}{2}||w||_{2}^{2} $$

该**正则化项使得权重更加接近原点(实际上可以令他接近任意点，也有正则化效果)。具体来说，在显著减小目标函数方向上的参数会保留的相对完好。在无助于目标函数减小的方向(对应Hessian矩阵较小的特征值)上改变参数不会显著增加梯度。这种不重要方向对应的分量会在训练过程中因正则化而衰减掉。**

我们可以从两个层次来理解，首先从单步更新来说：

$$ \tilde{J}(w; X, y) = \frac{\alpha}{2}w^{T}w + J(w; X, y) $$

与之对应的梯度为：

$$ \nabla_{w}\tilde{J}(w; X, y) = \alpha w + \nabla_{w}J(w; X, y) $$

因此更新策略为：

$$ w \leftarrow (1-\epsilon\alpha)w - \epsilon\nabla_{w}J(w; X, y) $$

可以看到，加入正则项后，每步执行通常的梯度更新之前先收缩权重向量，以此控制权重大小。

另一方面，也可以从整个训练过程中来看这个问题。首先假设 $$w^{*}$$ 为为正则化的目标函数取得最小训练误差时的权重。令目标函数 J 在 $$w^{*}$$ 处展开并忽略二阶以上项，有：

$$ \hat{J}(\theta) = J(w^{*}) + \frac{1}{2}(w-w^{*})^{T}H(w-w^{*}) $$

其中一阶项的导数为0没有写。当$\hat{J}$取得最小值时，它的梯度等于0：

$$ \nabla_{w}\hat{J}(w) = H(w-w^{*}) $$

现在考虑加入正则项后，它的梯度也应该为零，因此：

$$ \alpha \tilde{w} + H(\tilde{w}-w^{*}) = 0 $$

$$ \tilde{w} = (H + \alpha I)^{-1}Hw^{*} $$

可以看出，当 $\alpha$ 趋近于0时，趋近于没正则项。对上式更进一步，因为H是半正定的，对其分解成对角矩阵$\Lambda$和标准正交基Q，并且有 $$H = Q\Lambda Q^{T} $$，因此有：

$$
\begin{aligned}
\tilde{w} & = (Q\Lambda Q^{T} + \alpha I)^{-1}Q\Lambda Q^{T} w^{*} \\
 & = [Q(\Lambda + \alpha I)Q^{T}]^{-1}Q\Lambda Q^{T}w^{*} \\
 & = Q(\Lambda + \alpha I)^{-1}\Lambda Q^{T}w^{*}
\end{aligned}
$$

可以看到权重衰减的效果是沿着由 H 的特征向量所定义的轴缩放 $$w^{*}$$。具体来说，我们会根据 $$\frac{\lambda_{i}}{\lambda_{i}+\alpha}$$ 因子缩放与 H 第 i 个特征向量对齐的 $$w^{*}$$ 的分量。沿着 H 特征值较大的方向，正则化的影响较小，而 $$\lambda_{i} << \alpha$$ 的分量受正则化的影响会比较大，向零收缩.

![](/img/in-post/tensorflow/L2_shiyi.png)

还可以从参数的先验角度来看。现在我们假设n个参数服从正态分布 $w \sim N(0, \sigma^{2})$，那么它的负对数似然为：

$$ E_{w}(log g(w)) = -log(\prod_{i=1}^{n}\frac{1}{2\pi\sigma^{2}}e^{-\frac{w_{i}^{2}}{2\sigma^{2}}}) = \frac{1}{2}\log(2\pi \sigma^{2}) + \frac{1}{2\sigma^{2}}\sum_{i}w_{i}^{2} $$

忽略掉常数项，上式可以简化为 
$\lambda|w|^{2}$，就得到了 L2 正则化的表达式。

# L1 正则化
我们可以按照 L2 正则分析的逻辑来看 L1. 对模型参数 w 的 L1 正则化被定为为：

$$ \Omega(\theta) = |w|_{1} = \sum_{i}|w_{i}| $$

也就是各个参数的绝对值之和。加上正则项后的目标函数为：

$$ \tilde{J}(w; X, y) = \alpha |w|_{1} + J(w; X, y) $$

两侧求导计算梯度：

$$ \nabla_{w}\tilde{J}(w; X, y) = \alpha sign(w) + \nabla_{w}J(w; X, y) $$

其中 sign(w) 表示取 w 各元素的正负号。因此权重的更新公式就变成：

$$ w \leftarrow (w - \alpha sign(w)) + \nabla_{w}J(w; X, y) $$

可以看到正则化对梯度的影响不再是线性地缩放每个w，而是添加了一项与 w 正负号相关的常数。当 w 为正时，w就会减去 $\alpha$ 使得它减小向0移动，而当 w 为负时，w会加上 $\alpha$，使它变大，还是向零移动。这点和 L2 行为很不一样，该行为可以被用于特征选择，能大幅度简化模型，同时由于参数量的减少，也更稀疏，很好的减少了过拟合的可能性。而L2 的结果则是使得参数尽可能的靠近 0，它的解更具有平滑性。

从整体训练过程来看，首先像L2 一样进行泰勒展开：

$$ \hat{J}(\theta) = J(w^{*}) + \frac{1}{2}(w-w^{*})^{T}H(w-w^{*}) $$

其中 $w^{*}$ 和L2一样表示未正则化时的最优解w。加上正则项：

$$ \hat{J}(\theta) = J(w^{*}) + \frac{1}{2}(w-w^{*})^{T}H(w-w^{*}) + \alpha |w| $$

由于 L1 称发现该在完全一般化的 Hessian 的情况下，无法得到直接清晰的代数表达式，因此我们进一步假设 Hessian 是对角的，且对角元素全正。那么上式可以分解为关于参数的求和：

$$ \hat{J}(\theta) = J(w^{*}) + \sum_{i}[\frac{1}{2}H_{i,i}(w_{i}-w_{i}^{*})^{2} + \alpha |w_{i}|] $$

两侧求导而后移项可得到 w 的解：

$$ w_{i} = sign(w_{i}^{*})\max\left\{|w_{i}^{*}| - \frac{\alpha}{H_{i,i}}, 0\right\} $$

对于每个 i， 考虑 $w_{i}^{*}$ 大于零的情形(小于0的情况类似，对应分析即可)，会有两种可能的结果：

1). $w_{i}^{*} \leq \frac{\alpha}{H_{i,i}}$ 时，正则化后目标的最优值 $w_{i}$ 是 0,。这是因为在方向i上 $J(\theta) $对 $\hat{J}$的贡献被抵消，L1正则化将 $w_{i}$推到0.    

2). $w_{i}^{*} \gt \frac{\alpha}{H_{i,i}}$ 时，此时正则化不会将 $w_{i}$ 的最优值推到 0，而仅仅在那个方向上移动 $\frac{\alpha}{H_{i,i}} $。

从参数的先验分布角度来看，假设参数w服从拉普拉斯分布 $ w \sim Laplace(0,b)$，此时w分布的负对数似然为：

$$ L = -log (\prod_{i=1}^{n}\frac{1}{2b}e^{-\frac{|w_{i}|}{b}}) = n\log(2b) + \frac{1}{b}\sum_{i}|w_{i}| $$

可以看到，简化后得到 
$\lambda|w|_{1}$，这就是 L1 正则化的形式。

![](/img/in-post/tensorflow/l1_zhengze.jpg)

# Dropout

直译就是摘除，具体而言 Dropout 训练的集成包括所有从基础网络中除去非输出单元后形成的子网络。从实现上来说，最简单的方案是将一些单元的输出乘以0就可以当做删除这个神经元。因此每个迭代过程都会有不同的节点组合，从而导致不同的输出。这可以看做机器学习中的集成方法。更进一步的说，dropout 通过随机行为训练网络并平均多个随机决定进行预测，实现了一种参数共享的 bagging 形式。再进一步讲，Dropout 不仅仅是孙连一个 bagging 的集成模型，而且是共享隐藏单元的集成模型。

Dropout 强大的大部分原因来自施加到隐藏单元的掩码噪声，这可以看做对输入内容信息的高度智能化、自适应破坏的一种形式，而不是对输入原始值的破坏。Dropout 的另一个重要方面是噪声是乘性的。

# 提前终止

在训练中我们经常观察到，训练误差会随着时间的推移逐渐降低但验证集的误差会再次上升。这意味着我们只要返回使得验证集误差最低的参数设置，就可以获得验证集误差更低的模型(并且因此有希望获得更好的测试误差)。在每次验证集误差有所改善后，我们存储模型参数的副本。当训练算法终止时，我们返回这些参数而不是最新的参数。当验证集上的误差在事先指定的循环次数内没有进一步改善时，算法就会终止。这种策略被称为提前终止(early stopping)。

那么问题是为什么提前终止具有正则化效果呢？ Bishop 等认为提前终止可以将优化过程的参数空间限制在初始参数值 $\theta_{0}$ 的小邻域内。下图为提前终止效果的示意图，左图 实线轮廓线表示负对数似然的轮廓。虚线表示从原点开始的 SGD 所经过的轨迹。提前终止的轨迹在较早的点$\tilde{w}$ 处停止,而不是停止在最小化代价的点 $w^{*}$ 处。(右) 为了对比,使用 L2 正则化效果的示意图。虚线圆圈表示 L2 惩罚的轮廓,L2 惩罚使得总代价的最小值比非正则化代价的最小值更靠近原点。

![](/img/in-post/tensorflow/early_stop.png)

从公式推导来看，在二次误差的简单线性模型和简单的梯度下降情况下，可以证明提**前终止相当于 L2 正则化**。

# 数据增强

让机器学习模型泛化更好的最好办法是使用更多的数据进行训练。当然，在实际中的数据量往往是有限的，因此一种解决方法是创建假数据并添加到训练集中。如图像识别中的旋转、平移、添加噪声等操作。

# 参数绑定和参数共享

参数范数惩罚是正则化参数使其彼此接近的一种方式，而更流行的方法是使用约束：强迫某些参数相等，也就是参数共享。和正则化参数相比，参数共享的一个显著特点是，只有参数(唯一一个集合)的子集需要被存储在内存中，如在神经网络中，可显著减少内存。

# Bagging 和其他集成方法
Bagging 是通过结合几个模型降低繁华误差的技术，主要想法是分别训练几个不同的模型，然后让所有模型表决测试样例的输出。这是机器学习中常规策略的一个例子，被称为模型平均。采用这种策略的技术被称为集成方法。模型平均奏效的原因是不同的模型通常不会在测试集上产生完全相同的误差。

