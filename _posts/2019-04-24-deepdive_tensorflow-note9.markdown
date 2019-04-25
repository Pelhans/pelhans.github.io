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

$$ \nabla_{W}L = \sum_{t}diag(1-(h^{t})^{2})(\nabla_{h^{t}}L)h^{(t-1)^{T}} $$

$$ \nabla_{U}L = \sum_{t}diag(1-(h^{t})^{2})(\nabla_{h^{t}}L)x^{(t)^{T}} $$

# 双向 RNN
双向循环神经网络结合时间上从序列起点开始移动的RNN和另一个时间上从序列末尾开始移动的RNN。下图是一个典型的双向RNN：

* $h^{t}$ 代表通过时间向未来移动的子 RNN 状态，向右传播信息。    
* $g^{t}$ 代表通过时间向过去移动的子 RNN 的状态，向左传播信息。    
* 输出单元 $o^{t}$：同时依赖于过去、未来以及时刻 t 的输入 $x^{t}$。

![](/img/in-post/tensorflow/rnn_birnn.png)

# 长期依赖问题
长期依赖问题是深度学习中的一个主要挑战，其产生的根本原因是经过许多阶段传播之后，梯度趋向于消失或者爆炸。其中梯度消失占大部分情况，而梯度爆炸占少数情况。但是梯度爆炸一旦发生，对优化过程的影响巨大。**RNN 中涉及相同函数的多次组合，每个时间步一次，这些组合可以导致极端非线性行为，**因此在RNN 中长期以来问题特别突出。接下来上公式来验证一下：

现在假设循环联系是一个非常简单的、缺少非线性激活函数和输入 x 的循环神经网络(下面公式中如$h^{(t)}$表示t时间步的隐藏单元，而 $W^{t}$表示W的t次幂)：

$$ h^{(t)} = W^{T}h^{(t-1)} $$

现在我们算一下梯度：

$$ h^{(t)} = W^{t}h^{(0)} $$

$$ \frac{\partial h^{(t)}}{\partial h^{(t-1)}} = W $$

$$ \frac{\partial h^{(t)}}{\partial h^{(0)}} = W^{t} $$

$$ \nabla_{h^{(0)}}L = \frac{\partial h^{(t)}}{\partial h^{(0)}}\nabla_{h^{(t)}}L = W^{t}\nabla_{h^{(t)}}L $$

当 W 符合下列形式的特征分解时：

$$ W = Q\Lambda Q^{T} $$

其中Q 为正交矩阵，$\Lambda$为特征矩阵组成的三角矩阵。则：

$$ h^{(t)} = Q\Lambda^{t}Q^{T}h^{(0)} $$

$$ \nabla_{h^{(0)}} = \frac{\partial h^{(t)}}{\partial h^{(0)}}\nabla_{h^{(t)}}L = Q\Lambda^{t} Q^{T}\nabla_{h^{(t)}}L $$

**前向传播时，当特征值提升到t次时，幅值不到1的特征值衰减到0，而幅值大于1的就会激增。任何不与最大特征值向量对齐的 $h^{(0)}$的部分将会被丢弃。反向传播时，对于梯度也是如此，随着t的增加，特征值幅度不到1的梯度会衰减到0，而大于1的部分将会指数增长。**

现在考虑带激活函数的情形，假设激活函数为 $tanh()$，然后再加上偏置，因此RNN内部更新公式变成：

$$ h^{(t+1)} = tanh(b + W^{h^{(t)}} + Ux^{(t+1)}) $$

则对应的梯度为：

$$ \frac{\partial h^{(t+1)}}{\partial h^{(t)}} = diag(1-(h^{(t+1)})^{2})W $$

前向传播时，因为 tanh函数限制在(-1, 1)之间，因此前向传播时不会指数增长，这也是RNN中使用 tanh 而不是 ReLU的原因。

反向传播时，$diag(1-(h^{(t+1)})^{2})W$ 对W进行了一定程度上的缩小，$h^{(t+1)}$越大，则结果越小。如果 W 的特征值经过这样的缩小后，每个时刻都远小于1，则该梯度部分将衰减到0。如果缩小后永远大于1，那么该梯度部分将指数增长。而在缩小后，不同时刻有时候大于1，有时候小于1，那么梯度就比较理想。

# 长期依赖的处理
处理长期以来的一种方法是设计工作在多个时间尺度的模型，使得模型的某些部分在细力度时间尺度上操作并能处理小细节，而其他部分在粗时间尺度上操作并能把遥远过去的信息更有效地传递过来。这些策略包含：

* 跳跃链接：增加遥远过去的变量到当前变量的直接连接来得到粗粒度时间尺度，如 t 连到 t+d 单元。引入 d 延时的循环可以减轻梯度消失的的问题，现在梯度指数降低的速度与 $\frac{\tau}d$相关，而不是 $\tau$，允许算法捕捉到更长时间的依赖性，但梯度爆炸问题依然存在。    
* 删除连接：主动删除时间跨度为1的连接，用更长的连接替换它们。    
* 渗漏单元：前面看到长期依赖的问题根源是梯度的累计，因此我们可以使梯度接近1，这可以通过线性自连接单元实现。如 $ h^{(t)} = \alpha\h^{(t-1)}+(1-\alpha)x^{(t)} $$，当 $\alpha$接近1时，过去的信息被记住，当 $\alpha$ 接近0时，当前输入的信息被使用。其实这里和LSTM的思想很接近了。

对于梯度爆炸的问题，常用的解决方案是梯度截断。梯度截断有两种不同实例：

* 一种选择是在参数更新之前，逐元素地截断小批量产生的参数梯度。    
$$ g_{i} = 
\left\{
    \begin{aligned}
    g_{i}~~&& if g_{i} <= v\\
    sign(g_{i})\times v&&, else
    \end{aligned}
\right.
$$    
* 另一种实在参数更新之前截断梯度g的范数，该方案可以确保截断后的梯度仍然是在正确的梯度方向上,但实践表明，两种方式效果相近    

$$ \overrightarrow{g} = 
\left\{
    \begin{aligned}
    \overrightarrow{g}~~&& if ||\overrightarrow{g}|| <= v\\
    \frac{\overrightarrow{g}\times v}{||\overrightarrow{g}||}&&, else
    \end{aligned}
\right.
$$    

}}
# RNN 的变种
目前实际应用中，最有效的序列模型是门控RNN，包括LSTM和GRU。门控RNN的思路和渗漏单元一样，通过生成通过时间的路径，使得梯度既不消失，也不爆炸。但相比于渗漏单元，门控RNN能够学会何时清除信息，而不是手动决定。

## LSTM

LSTM 旨在避免长期以来性问题，关于LSTM，这篇文章讲的特别的棒--[Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)。我按照自己的记忆习惯整理一下。

下图是一个典型的 LSTM 单元结构，它相比与普通的RNN，内部结构变得更加复杂，同时增加了在图上方贯穿运行的细胞状态。下面分块去理解它。

![](/img/in-post/tensorflow/LSTM3-chain.png)

### Cell 状态

LSTM 最重要的就是cell 状态 C, 它以水平线在图上方贯穿运行，只有一些次要的线性交互，信息很容易沿着它不变的流动。

![](/img/in-post/tensorflow/LSTM3-C-line.png)

###  遗忘门
sigmoid 函数限制在可以把输出控制在(0, 1)之间，描述么米格部分有多少量可以通过，它起到门的作用。LSTM 有三个门：遗忘门、输入门、输出门。

遗忘门控制了 cell 上一个状态$C^{t-1}$中，有多少信息进入当前状态$C^{t}$。这由一个 sigmoid 函数控制，该函数接收上一隐藏单元的输出和当前时间步的输入，输出一个0-1之间的数来控制遗忘。：

$$ f_{t} = \sigma(b_{i}^{f} + \sum_{j}U_{i,j}^{f}x_{j}^{t} + \sum_{j} W_{i,j}^{f}h_{j}^{t-1}) $$

![](/img/in-post/tensorflow/LSTM3-focus-f.png)

### 输入门

输入门控制接下来哪些信息要被保存到Cell状态中，输入门的方程为：

$$ i_{t} = \sigma(b_{i}^{g} + \sum_{j}U_{i,j}^{g}x_{j}^{t} + \sum_{j}W_{i,j}^{g}h_{j}^{t-1} ) $$

同时，tanh 层创建可以添加到状态的新候选值 $\tilde{C}_{t}$的向量。

$$ \tilde{C}_{t} = tanh(b_{i} + \sum_{j}U_{i,j}x_{j}^{t} + \sum_{j}W_{i,j}h_{j}^{t-1}) $$

![](/img/in-post/tensorflow/LSTM3-focus-i.png)

最终新 $C^{t}_{i}$ 的状态方程为：

$$ C^{t}_{i} = f_{i}^{t}C_{i}^{t-1} + i_{i}^{t}\tilde{C}_{t} $$

![](/img/in-post/tensorflow/LSTM3-focus-C.png)

### 输出门

输出门控制了cell 状态$C^{t}$中有多少会进入输出。输出门的方程为：

$$ o^{t} = \sigma(b_{i}^{o} + \sum_{j}U_{i,j}^{o}x_{j}^{t} + \sum_{j}W_{i,j}^{o}h_{j}^{t-1} ) $$

接下来通过 tanh 函数将Cell状态的值压缩到 (-1, 1)，并将其乘以输出门的输出，得到隐藏单元的输出：

$$ h^{t}_{i} = tanh(C_{i}^{t})o_{t} $$

![](/img/in-post/tensorflow/LSTM3-focus-o.png)

## GRU
门控循环单元(GRU) 与 LSTM 的主要区别是：

* GRU 的单个门控单元同时作为遗忘门和输入门    
* GRU 不再区分Cell 状态 和隐藏单元输出 h

这样产生的模型比标准 LSTM 更简单，很受欢迎。GRU 中有两门：复位门、更新门。

![](/img/in-post/tensorflow/GRU.png)

对应的内部方程为：

$$ z_{t} = \sigma(W_{z}*[h_{t-1}, x_{t}] )$$

$$ r_{t} = \sigma(W_{r}*[h_{t-1}, x_{t}] ) $$

$$ \tilde{h}_{t} = tanh(W*[r_{t}*h_{t-1}, x_{t}] ) $$

$$ h_{t} = (1-z_{t})*h_{t-1} + z_{t}*\tilde{h}_{t} $$


LSTM 与  GRU 有两种非线性函数：sigmoid 和 tanh，其中：

* sigmoid 用于各种门，这是因为它可以输出 0-1 范围内的数，可以很好的模拟关闭程度    
* tanh 用于激活函数：    
    * tanh 相比于 sigmoid 收敛速度更快，且输出以0为中心。    
    * 相比于 ReLU，由于循环神经网络的特殊性，前向传播信息容易爆炸增长
### LSTM 为什么能缓解梯度消失？
按照惯例，上公式... 我们计算 $\frac{C_{t}}{\partial C_{t-1}} $：

$$ 
\begin{aligned}
\frac{C_{t}}{\partial C_{t-1}} & = \frac{\partial C_{t} }{\partial f_{t} }*\frac{\partial f_{t} }{\partial h_{t-1} }*\frac{\partial h_{t-1} }{\partial C_{t-1} } \\
& + \frac{\partial C_{t} }{\partial i_{t} }*\frac{\partial i_{t} }{\partial h_{t-1} }\frac{\partial h_{t-1} }{\partial C_{t-1} } \\ 
&+ \frac{\partial C_{t} }{\partial \tilde{C}_{t} }*\frac{\partial \tilde{C}_{t} }{\partial h_{t-1} }*\frac{\partial h_{t-1} }{\partial C_{t-1} }\\
& + \frac{\partial C_{t} }{\partial C_{t-1} } 
\end{aligned}
$$

代入有：

$$
\begin{aligned}
\frac{C_{t}}{\partial C_{t-1}} & = C_{t-1}\sigma^{'}(\dot)W_{f}*o_{t-1}tanh^{'}(C_{t-1}) \\
& + \tilde{C}_{t}\sigma^{'}(\dot)W_[i]*o_{t-1}tanh^{'}(C_{t-1}) \\
& + i_{t}tanh^{'}(\dot)W_{C}*o_{t-1}tanh^{'}(C_{t-1}) \\
& + f_{t}
\end{aligned}
$$

可以看到在 cell 状态这条路径上，$\frac{C_{t}}{\partial C_{t-1}}$ 在任何时间步上都可以取大于1的值或者[0,1]范围内的值，如果我们开始收敛到0，我们总是可以将$f_{t}$的值设置的更高，使得导数更接近1，从而缓解梯度消失。
