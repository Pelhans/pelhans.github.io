---
layout:     post
title:      "深度学习笔记（四）"
subtitle:   "神经网络中的权值初始化"
date:       2019-04-12 00:15:18
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

那么为什么会有这么多初始化策略呢？深度学习算法的训练通常是迭代的，因此要求使用者给一些开始迭代的初始点，而有些深度模型会受到初始点的影响，使得算法遭遇数值困难，并完全失败。此外初始点可以决定学习收敛的多快，以及是否收敛到一个代价高或低的点，更何况代价相近的点也可能有极大的泛化差别。不同的网络结构和激活函数需要适配不同的初始化方法。目前常用的初始化方法包含随机均匀初始化、正态分布初始化、Xavier初始化、He初始化、预训练等。

一个好的初始化方法要求各层激活值不会出现饱和现象,同时各层得激活值不为0.

# 随机初始化

随机初始化包含均匀随机初始化和正态随机初始化，在 tensorflow 中对应的代码为：

* 均匀随机：tf.initializers.random_uniform(-0.1, 0.1)    
* 正态分布：tf.initializaers.random_normal(0, 1)，均值为0，方差为1的正态分布。    
* 正态分布带截断：tf.initializers.truncated_normal(0, 1)，生成均值为0，方差为1的正态分布，若产生随机数落到2$\sigma$外，则重新生成。    

下面以均匀分布为例说明随机初始化的缺点。

均匀损及初始化即在一定范围内随机进行初始化,首先权重矩阵初始化公式为：

$$ W_{ij} \~ U[-\frac{1}{\sqrt{n}}, \frac{1}{\sqrt{n}}] $$

易知该分布的均值为0，方差通过公式$\frac{(b-a)^{2}}{12}$ 得到方差为 $Var(W_{ij}) = \frac{1}{3n}$。

那个初始化公式怎么来的呢? 首先我们假设再[a,b] 内均匀分布,因此概率密度函数为$$ f(x) = \frac{1}{b-a}$$,当 $$x\in [a,b]$$, 否则为 0. 因为要对称, 所以 a = -b, 带入方差公式,消除 b 可得方差为:

$$ \sum_{i=1}^{n}\frac{a^{2}}{3} $$

要使得方差为常数, 如 $\frac{1}{3}$时, $$ a = -\frac{1}{\sqrt{n}} $$.也就得到了上面的分布.

现在把输入记为x，并假设它服从正态分布。并假设W与x独立，则线性隐藏层的输出的方差为：

$$ Var(y) = Var(\sum_{i=1}^{n}W_{ki}x_{i}) = \sum_{i=1}^{n}Var(W_{ki})Var(x_{i}) = \sum_{i=1}^{n} Var(W_{ki}) = \sum_{i=1}^{n}\frac{1}{3n} = \frac{1}{3} $$

因此标准初始化的隐层均值为0，方差为常量，和网络的层数无关，这意味着对于 sigmoid 来说，自变量落在有梯度的范围内。但是对于下一层来说情况就发生了变化，这是因为下一层的输入的均值经过sigmoid后不是0了，方差自然也跟着变了，好的性质没有了。输入输出的分布也不一致。

以 tanh做激活函数的神经网络为例，查看激活值状态、参数梯度的各层分布[xavier的论文](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)：

首先看激活值标准化以后的分布，如下图所示，激活值的方差是逐层递减的。这是因为

$$ z^{i+1} = f(s^{i}) $$

$$ s^{i} = z^{i}W^{i} + b^{i} $$

$$ Var(z^{i}) = Var(x)\prod_{i'=1}^{i-1}n_{i'}Var(W)^{i'} $$

其中f是激活函数，$z^{i+1}$是它的输出，$z^{i}$是该层的输入，可以看到z  累计了W的方差，因此逐渐降低：

![](/img/in-post/tensorflow/xavier_1.png)

对于反向传播状态的梯度的方差是逐层增加的，换句话说是在反向传播过程中逐层递减的。如下图所示：

![](/img/in-post/tensorflow/xavier_2.png)

因为：

$$
\begin{aligned}
\frac{\partial cost}{\partial z^{i}} & = \frac{\partial cost}{\partial s^{i+1}} \times \frac{\partial s^{i+1}}{\partial z^{i+1}}\times \frac{\partial z^{i+1}}{\partial s^{i}} \\
& = \frac{\partial cost}{\partial s^{i+1}}\times w^{i+1}\times \frac{\partial z^{i+1}}{\partial s^{i}}
\end{aligned}
$$

$$ \frac{\partial z^{i+1}}{\partial s^{i}} = f^{'}(s^{i}) \simeq 1 $$

将上面的结果进一步用到多层中,即可得到下式:

$$ Var(\frac{\partial Cost}{\partial z^{i}}) = Var(\frac{\partial Cost}{\partial z^{d}})[nVar(W)]^{d-i} $$

可以看到，和前面的原因相似，在反向传播的过程中累计了参数的方差，导致自身越来越小。

对于参数的梯度，它基本上与层数无关，如下图所示：

![](/img/in-post/tensorflow/xavier_3.png)

这是因为：

$$\begin{aligned}
\frac{\partial cost}{\partial w^{i}} &= \frac{\partial cost}{\partial s^{i}}\times \frac{\partial s^{i}}{\partial w^{i}}
& = \frac{\partial cost}{\partial s^{i}}\times z^{i}
\end{aligned}$$

将上面的结果进一步用到多层中,即可得到下式:

$$ Var[\frac{\partial Cost}{\partial w{i}}] = [nVar(W)]^{d}Var[w]Var[\frac{\partial Cost}{\partial s^{d}}] $$

和层数 i没关系，但和总层数d有关，因此太深的话会更容易出现梯度消失或者爆炸这种问题。

# Xavier 初始化

Xavier 作者 Glorot 认为，**优秀的初始化应该使得各层的激活值和状态梯度的方差在传播过程中保持一致**。

为什么会是这样？根据论文[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)，它虽然是探讨BN的，但个人感觉很相近。论文里说，由于前一层的参数更新，所以这一层的输入（前一层的输出）的分布会发生变化，这种现象被称之为ICS。同样，这篇文章的观点认为BN work的真正原因，在与其将数据的分布都归一化到均值为0，方差为1的分布上去。因此，每一层的输入（上一层输出经过BN后）分布的稳定性都提高了，故而整体减小了网络的ICS。那么对于 Xavier 的情况也很类似，它保持在训练过程中各层的激活值和状态梯度的方差不变，也相当于减小了网络的ICS。

根据  Glorot 的假设，有：

$$ n_{i}Var(W^{i+1}) = 1 $$

$$ n_{i+1}Var(W^{i+1}) = 1 $$

其中 $n_{i}$表示该单元的输入的数量，$n_{i+1}$表示输出的数量。但对于同一层来说，输入和输出的数量往往不一致，因此作者取二者的均值作为方差：

$$ Var(W^{i+1}) = \frac{2}{n_{i}+n_{i+1}} $$

根据均匀分布方差公式，反过来求分布的边界即可得到W的分布边界：

$$ W \~ U[-\frac{\sqrt{6}}{\sqrt{n_{i}+n_{i+1}}}, \frac{\sqrt{6}}{\sqrt{n_{i}+n_{i+1}}}] $$

假设各层的大小一样，那么就有 $Var(W) = \frac{1}{n}$，再带入状态的梯度和激活值的公式中可以看到这个$\frac{1}{n}$和前面乘积的n抵消了。也就保持方差的不变。

# He 初始化

He 初始化的基本思想是，当使用Relu 作为激活函数时， Xavier 的效果不好，这是因为对于 Relu，在训练过程中会有一部分神经元处于关闭状态，导致分布变了。因此作者做出了改进：

对于 ReLU 的初始化方法：

$$ W \~ N[0, \sqrt{\frac{2}{n_{i}}}] $$

推导过程为：

$$ y_{l} = W_{l}x_{l} + b_{l} $$

$$ x_{l} = f(y_{l-1}) $$

假设W和x都满足独立同分布 i.i.d。又假设$w_{l}$ 是零均值，w和x相互独立。那么有：

$$ Var[y_{l}] = n_{l}Var[w_{l}x_{l}] = n_{l}Var[w_{l}]E[x_{l}^{2}] $$

对于 ReLU, $x_{l} = \max(0, y_{l-1}) $，因此它的均值不是0。现在假设w是关于0对称分布且b为0，那么$y_{l-1}$也是0均值的同时分布关于0对称。因此：

$$ E[x_{l}^{2}] = \frac{1}{2}Var[y_{l-1}] $$

$$ Var[y_{l}] = \frac{1}{2}n_{l}Var[w_{l}]Var[y_{l-1}] $$

那么当到第L层时，我们有：

$$ Var[y_[L]] = Var[y_{l}](\prod_{l=2}^{L}\frac{1}{2}n_{l}Var[w_{l}]) $$

根据 Xavier中的经验，我们要消除连乘部分，也就是说：

$$ \frac{1}{2}n_{l}Var[w_{l}] = 1 $$

因此，当w为正态分布时，它的分布公式就是 $N[0, \sqrt{\frac{2}{n_{i}}}]$。

对于Leaky ReLU来说，它的初始化公式就是：

$$ W \~ N[0, \sqrt{\frac{2}{(1+\alpha^{2})\hat{n_{i}}}}] $$

其中$$ \hat{n_{i}} = h_{i}*w_{i}*d_{i} $$，而 $$h_{i}$$、$$w_{i}$$表示卷基层中卷积核的高和宽，$$d_{i}$$是当前卷积核的个数。

# 参考
[Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)    
[深度学习之参数初始化（一）——Xavier初始化](https://blog.csdn.net/victoriaw/article/details/73000632)    
[深度前馈网络与Xavier初始化原理](https://zhuanlan.zhihu.com/p/27919794)    
[NMT Tutorial 3扩展c. 神经网络的初始化](http://txshi-mt.com/2018/11/16/NMT-Tutorial-3c-Neural-Networks-Initialization/)    

