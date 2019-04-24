---
layout:     post
title:      "Tensorflow 笔记（九）"
subtitle:   "循环神经网络基础"
date:       2019-04-24 00:15:18
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

循环神经网络或RNN 是一类用于处理序列数据的神经网络，循环神经网络可以扩展到更长的序列，大多数循环网络也能处理可变长度的序列。同事，循环神经网络以不同的方式共享参数，输出的每一项是前一项的函数，输出的每一项对先前的输出应用相同的更新规则产生，这种循环方式导致参数通过很深的计算图共享。

# 展开计算图
循环神经网络的一般定义为：

$$ h^{t} = f(h^{t-1}, x^{t}; \theta) $$

其中 h 表示隐藏单元的值。现在我们用一个函数  $g^{t}$ 将过去序列作为输入来生成当前状态：

$$ h^{t} = f(h^{t-1}, x^{t}; \theta) = g^{t}(x^{t}, x^{t-1}, \dots, x^{2}, x^{1}) $$

下图表示没输出的循环网络和展开后的计算图：

![](/img/in-post/tensorflow/rnn_unfold.png)

展开图的优点为：

*  无论序列的长度，学成的模型始终具有相同的输入大小，因为它指定的是从一种状态到另一种状态的转移，而不是在可变长度的历史状态上的操作。    
* 我们可以再每个时间步使用相同参数的相同转移函数 f。    
* 展开图通过显示的信息流动路径帮助说明信息在时间上向前(计算输出和损失)和向后(计算梯度)的思想。

# 循环神经网络

基于上面的图展开和参数共享思想，可以设计出各种循环神经网络。其中比较重要的三种设计模式如下图所示：

下图这种循环网络是比较典型的结构，当前时间步接收上一时间步隐藏单元的输出和当前时间步的输入作为整体输入。同时每个时间步都有输出。任何图灵可计算的函数都可以通过这样一个有限维的循环网络计算，在这个意义上，该种循环网络是万能的。

![结构1](/img/in-post/tensorflow/rnn_mode1.png)

下面这个结构中的每个隐藏单元接收上一时间步输出层的输出和当前时间步的输入作为整体输入。这种RNN 没有上一个那么强大，因为它只能表示更小的函数集合。同时，由于输出 o 要用来预测当前时间步的输出，因此通常缺乏过去的重要信息，因此没那么强。但它更容易训练，因为每个时间步可以与其他时间步分离训练，允许训练期间更多的并行化。(关于它训练的并行化，是因为可以用训练集中的正确标签来作为上一时间步的输出，这样不同时间步之间在训练时解耦了)

![结构2](/img/in-post/tensorflow/rnn_mode2.png)

下面的网络结构在序列结束时有单个输出，这样的网络可以用于概括序列并产生用于进一步处理的固定大小的表示。

![结构3](/img/in-post/tensorflow/rnn_mode3.png)

# 循环网络的反向传播

下面我们基于第一个结构作为我们研究的对象。并假设网络的后续结构如下：

$$ a^{t} = b + Wh^{t-1} + Ux^{t} $$

$$ h^{t} = tanh(a^{t}) $$

$$ o^{t} = c + Vh^{t} $$

$$ \hat{y}^{t} = softmax(o^{t}) $$

$$ L = \sum_{t}L^{t} = -\sum_{t}\log p_{model}(y^{t} | {x^{1},\dots,x^{t}}) $$

我们想计算包含参数 U、V、W、b、c以及以时间t为索引的 x、h、o、L。

$$ \frac{\partial L}{\partial L^{t}} = 1 $$

$$ (\nabla_{o^{t}}L)_{i} = \frac{\partial L}{\partial L^{t}}*\frac{\partial L^{t}}{\partial o_{i}^{t}} = \hat{y}_{i}^{t} - 1_{i,y^{t}} $$

$$ \nabla_{h^{\tau}}L = V^{T}\nabla_{o^{\tau}}L $$

$$
\begin{aligned}
\nabla_{h^{t}}L &= \frac{\partial h^{t+1}}{\partial h^{t}}^{T}(\nabla_{h^{t+1}}L) + \frac{\partial o^{t}}{\partial h^{t}}^{T}(\nabla_{o^{t}}L) \\
& = W^{T}(\nabla_{h^{t+1}}L)diag(1-(h^{t+1})^{2})+V^{T}(\nabla_{o^{t}}L) 
\end{aligned}
$$

其中 diag()表示包含括号中元素的对角矩阵。接下来我们要求关于参数的梯度，但需要注意参数共享：

$$ \nabla_{c}L = \sum_{t}(\frac{\partial o^{t}}{\partial c})^{T}\nabla_{o^{t}}L = \sum_{t}\nabla_{o^{t}}L $$

$$ \nabla_{b}L = \sum_{t}diag(1-(h^{t})^{2})\nabla_{h^{t}}L $$

$$ \nabla_{V}L = \sum_{t}\sum_{i}(\frac{\partial L}{\partial o^{t}_{i}})\nabla_{V}o^{t}_{i} = \sum_{t}(\nabla_{o^{t}}L)h^{(t)^{T}} $$

$$ \nabla_{W}L = \sum_{t}\sum_{i}diag(1-(h^{t})^{2})(\nabla_{h^{t}}L)h^{(t-1)^{T}} $$

$$ \nabla_{U}L = \sum_{t}diag(1-(h^{t})^{2})(\nabla_{h^{t}}L)x^{(t)^{T}} $$

# 双向 RNN
双向循环神经网络结合时间上从序列起点开始移动的RNN和另一个时间上从序列末尾开始移动的RNN。下图是一个典型的双向RNN。

![](/img/in-post/tensorflow/rnn_birnn.png)
