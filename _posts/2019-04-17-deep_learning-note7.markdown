---
layout:     post
title:      "深度学习笔记（七）"
subtitle:   "激活函数"
date:       2019-04-17 00:15:18
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

激活函数是神经网络的一个重要组成部分，它可以**将线性分类器转换为非线性分类器**，这已被证明是近年来在各种任务重所见到的高性能的关键。不同的激活函数在实践中经常表现出非常多样的行为。例如 **Sigmoid 函数，在过去非常流行。它可以将任意范围的输入转化为0-1之间的输出，逻辑回归中就用它来做二分类问题。在早起网络不深的时候用它效果很好，但随着网络的加深，函数两端饱和区域的劣势就显现出来，很容易导致梯度的消失。 tanh 函数相比于 sigmoid 函数，它可以将输出映射到-1 - 1内，激活函数可以为负值，网络表达能力可能会有提升。再后来有了ReLU 激活函数，它的形式非常简单，相比于前两个，它的导数在正值区恒定，梯度不易消失，同事它还相当于一个滤波器，可以降低网络信息冗余度。目前 RuLU 依旧很流行，虽然针对ReLU的一些缺点，如没有负输出、零点处导数突变等，一些大牛做出了改进，如 LReLU、PReLU、ELU、SELU、GELU等，但实际使用中差距并不大，尤其是在使用BN 后。**

除此之外，Ranacgabdran 等还根据自动搜索提出了一些激活函数，它们大都是基本函数的组合。其中效果比较好的如 swish。下图给出可见激活函数的公式和形状。

![](/img/in-post/tensorflow/activation_01.png)    
![](/img/in-post/tensorflow/activation_02.png)


# Sigmoid
sigmoid 函数的公式为：

$$ \sigma(a) = \frac{1}{1+e^{-a}} $$

sigmoid 函数可以将输出映射到 0-1 之间，单调连续，输出范围有限，优化稳定，可以用作输出层。但在两端存在饱和区域，会导致梯度消失，导致训练出现问题，同时输出不是以0为中心的。

它可以在二分类问题中，通过使用贝叶斯公式很自然的得到。首先我们有两个类比 C1 和C2，现在我们想计算
$ p(C_{1}|x) $，那么根据贝叶斯公式有：

$$ p(C_{1}|x) = \frac{p(x|C_{1})*p(C_{1})}{p(x|C_{1})p(C_{1}) + p(x|C_{2})p(C_{2})} $$

分子分母同时除以
$p(x|C_{1})p(C_{1})$，并令 
$a = \ln\frac{p(x|C_{1})p(C_{1})}{p(x|C_{2})p(C_{2})}$，因此我们得到sigmoid公式：

$$ \sigma(a) = \frac{1}{1+e^{-a}} $$

对于多分类问题，我们就可以得到 softmax 函数。

关于 sigmoid 函数，比较常用的两个计算是：

$$ \sigma(-a) = 1 - \sigma(a) $$

$$ \sigma^{'}(a) = (1-\sigma(a))\sigma(a) $$

都比较容易推导，这里就不展开了。

除此之外， sigmoid 函数还是偏移和缩放后的tanh函数：

$$ \sigma(x) = \frac{1}{2} + \frac{1}{2}tanh(\frac{x}{2}) $$

它的积分等于：

$$ \int \frac{e^{x}}{1+e^{x}}dx = \int \frac{1}{u}du = \log u = \log(1+e^{x}) $$

上面的函数被称为 softplus 函数。

Sigmoid 函数的优缺点：

优点：    
* 连续函数，便于求导    
* Sigmoid函数的输出在(0,1)之间，输出范围有限，优化稳定，可以用作输出层。    

缺点：    
* sigmoid函数的输出不是0均值的，会导致后层的神经元的输入是非0均值的信号，这会对梯度产生影响。    
* sigmoid函数在变量取绝对值非常大的正值或负值时会出现饱和现象，影响梯度更新    
* 计算复杂度高，因为sigmoid函数是指数形式。

# tanh 函数

tanh 函数的定义为：

$$ tanh(x) = \frac{e^{x}-e^{-x}}{e^{x}+e^{-x}} =  \frac{1-e^{-2x}}{1+e^{-2x}} $$

函数位于 [-1, 1] 的区间上，对应的图像为：

![](/img/in-post/tensorflow/activation_tanh.png)

几个比较有用的推导：

$$ tanh(-x) = -tanh(x) $$

$$ tanh(x) = \frac{e^{x}-e^{-x}}{e^{x}+e^{-x}} = \frac{e^{x}*(1-e^{-2x})}{e^{x}*(1+e^{-2x}) } = \sigma(2x) - \frac{e^{-2x}}{1+e^{-2x}} = \sigma(2x) - \frac{e^{-2x}+1-1}{1+e^{-2x}} = 2\sigma(2x) - 1 $$

$$ 
\begin{aligned}
\frac{d}{dx}tanh(x) & = \frac{d(2\sigma(2x) -1)}{dx} \\
        & = 4\sigma(2x)(1 - \sigma(2x)) \\
        & = 4\sigma(2x) - 4\sigma^{2}(2x)\\
        & = 1 - (4\sigma^{2}(2x) + 1 - 4\sigma(2x)) \\
        & = 1- (2\sigma(2x) - 1)^{2} \\
        & = 1-tanh^{2}(x) 
\end{aligned}
$$

tanh 函数相比于 sigmoid 函数的收敛速度更快，输出还是以0为中心的，缺点是饱和区域依旧存在，梯度消失问题没解决。

到这里可能会想，为什么会强调输出以0为中心？根据 CS231N 的的解释，如果我们用 sigmoid ，那么神经网络隐层的每个节点输出总是正数，也就意味着同一层的权重梯度方向相同，都为正或者都负。以下图为例，我们假想的优化参数曲线为 y=-x，它不在一三象限，因此在梯度方向相同的约束下，其中一个梯度为0，另一个梯度尽可能的走是最优的走法(走的最远)。这样一步一步的更新就出现 zaizag(锯齿形)现象。

![](/img/in-post/tensorflow/activation_zaizag.png)

优点：相比于 sigmoid，它是零均值的。

缺点：除了零均值，指数形式和饱和都还在。

# ReLU 整流线性单元

ReLU 的定义非常简单$ y = \max(0, x)$,对应的图像为：

![](/img/in-post/tensorflow/activation_relu.png)

它的优点是，相比于 sigmoid 和 tanh，ReLU 在SGD 中能够快速收敛(据说是因为线性和非饱和的形式)。sigmoid 和 tanh 的计算代价很大，ReLU的实现更简单，同时由于在正区域梯度恒定，没有饱和区域，因此可以缓解梯度消失问题。在无监督预训练中也能有较好的表现。除此之外还提供了神经网络的稀疏表达能力。缺点是随着训练的进行，可能出现神经元死亡，权重无法更新的情况。如果发生这种情况，那么神经元的梯度从这一点开始就永远为0，相当于永远死亡了。

上面的优点都比较容易理解，那这个神经元死亡是怎么回事呢？

假设有这么一个神经元，输入为x，经过线性层 $z = w*x$，而后经过 ReLU 激活函数 $a = ReLU(z)$。训练中，w 的更新公式为：

$$ W \leftarrow w - lr * \frac{\partial L}{\partial w} $$

如果在某次更新中，梯度太大，而学习率也被设置很大的话，那么权重就会更新过多。就可能会出现对于后续的任意样本输入，z 都小于0。经过 ReLU后，输出a 就会为0.

而后在反向传播时：

$$ \frac{\partial L}{\partial w} = \frac{\partial L}{\partial z}* x $$

$$ \frac{\partial L}{\partial z} = \frac{\partial L}{\partial a} * \frac{\partial a}{\partial z} $$

由于 a 的输出为常数 0，那么 $\frac{\partial a}{\partial z} = 0$，因此 $\frac{\partial L}{\partial z} = 0$。 也就是 $\frac{\partial L}{\partial w} = 0 $。我们看到，权重矩阵 W 不更新了，输出还是0，所以相当于死掉了。

优点：    
* 在x>0区域上，不会出现梯度饱和、梯度消失的问题。    
* 计算复杂度低，不需要进行指数运算，只要一个阈值就可以得到激活值。    
* 使用ReLU的SGD算法的收敛速度比 sigmoid 和 tanh 快

缺点：    
* ReLU的输出不是0均值的。    
* 神经元凋亡现象    
* 在x<0时，梯度为0。这个神经元及之后的神经元梯度永远为0，不再对任何数据有所响应，导致相应参数永远不会被更新；一般由参数初始化不好或者学习率太大导致。



# ReLU 的改进

前面说了一大堆的 ReLU 的缺点，有很多大牛在此基础上做了改进，如 Leaky ReLU、PReLU(Parametric ReLU)等。

Leaky ReLU 公式定义为：

$$ f(x) = 
\left\{
    \begin{aligned}
    \alpha x & ~~~ x<0 \\
    x & ~~~x\geq 0
    \end{aligned}
\right.
$$
}

其中 $\alpha$是一个很小的常数，这样在负轴上不再是常数，信息得以保留，神经元死亡的情况也得到解决。但缺点是 $\alpha$ 得人工指定，调起来比较麻烦.Parametric ReLU 中将它作为一个参数进行训练，效果更好。

# Gaussian Error Linerar Units(GELUS)
GELUs正是在激活中引入了随机正则的思想，是一种对神经元输入的概率描述，直观上更符合自然的认识，同时实验效果要比Relus与ELUs都要好。数学公式表达为:

$$ GELU(x) = xP(X<= x) = x\Phi(x) $$

其中 $$\Phi$$ 是正态分布的概率函数. 多余标准正态分布得情况,,论文中提供了近似计算得数学公式:

$$ GELU(x) = 0.5x(1 + tang[\sqrt(\frac{2}{\pi})(x+0.044715x^{3})]) $$

Bert 中的实现为:

```Python
def gelu(x):
	cdf = 0.5 * (1.0 + tf.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
  return x * cdf
```



# 什么样的函数能用来做激活函数

这里 博主 [Hengkai Guo](https://www.zhihu.com/question/67366051/answer/262087707) 总结的很好，除了单调性有争议外，其他的目前激活函数大都满足。大体说来需要满足：

* 非线性    
* 几乎处处可微    
* 计算简单    
* 非饱和性    
* 单调性    
* 输出范围有限    
* 接近恒等变换    
* 参数少    
* 归一化

# 参考
[The Activation Function in Deep Learning 浅谈深度学习中的激活函数](https://www.cnblogs.com/rgvb178/p/6055213.html)    
[深度学习中，使用relu存在梯度过大导致神经元“死亡”，怎么理解？](https://www.zhihu.com/question/67151971)   
[Hengkai Guo](https://www.zhihu.com/question/67366051/answer/262087707)    
[wiki](https://en.wikipedia.org/wiki/Activation_function)

