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

# Attention 机制的本质思想
可以把 Attention 机制抽象成下图所示模型：

![](/img/in-post/tensorflow/attention_chouxiang.jpg)

现在我们将source 中的元素想象成由一系列 <Key, Value> 的数据对构成，此时Target 中的某个元素 Query。通过计算 Query 和各个 Key 的相似性或相关性，得到每个 Key 对应 Value 的权重系数，然后对 Value 进行加权求和，即得到了最终的 Attention 数值。所以本质上 Attention 机制是对 Source 中元素的 Value 值进行加权求和，而 Query 和 Key 用来计算对应 Value 的权重系数。用公式表达为：

$$ Attention(Query, Source) = \sum_{i=1}^{S}align(Query, Key)*Value $$

其中 S 表示 Source 序列的长度。 在传统的 encoder-decoder 中， Source 的 Key 和 Value 指向相同的东西，而在Self-Attention 中，Key 、Value、Query 都指向相同的东西。

由上图也可以引出另一种理解，即 Attention 机制看做一种软寻址(Soft Addressing)：通过 Query 和 存储器内元素 Key 的地址计算相似性来寻址。之所以叫 Soft，是因为它不想一般寻址只从存储内容里面找出一条内容，而是可能从每个Key 地址都会取出内容。

# Attention 机制的分类
可以从多角度对 Attention 进行分类，如从信息选择的方式上，可以分为 Soft attention 和 Hard attention。从信息接收的范围上可分为 Global attention 和 Local attention。

## Soft attention 与 Hard attention
我们前面描述的传统 Attention 就是 Soft Attention，它选择的信息是所有输入信息在对齐模型分布下的期望。而 Hard Attention 只关注到某一位置上的信息，一般而言，Hard Attention 是实现有两种：一种是选取概率最高的输入信息，另一种是在对齐模型的概率分布上进行随机采样。硬性注意力的一个缺点是基于最大采样或随机采样的方式来选择信息。因此最终的损失函数与注意力分布之间的函数关系不可导，因此无法使用在反向传播算法进行训练。为了使用反向传播算法，一般使用软性注意力来代替硬性注意力。硬性注意力需要通过强化学习来进行训练。

## Global attention 与 Local attention
Global Attention 和传统的 注意力机制一样，所有的信息都用来计算 context vector 的权重。这会带来一个明显的缺点，即所有的信息都要参与计算，这样计算的开销就比较大，而别当encoder 的句子比较长时，如一段话或一篇文章。因此 Local Attention 就被提了出来，它是一种介于Kelvin Xu所提出的Soft Attention和Hard Attention之间的一种Attention方式，即把两种方式结合起来。下图是 Local 的图示

![](/img/in-post/tensorflow/local_attention.jpg)

上图中， $\hat{h}_{s}表示 全部的 encoder 向量，$h_{t}$表示 时间步 t 的 decoder 输出。Local Attention 首先会为 decoder 端当前的词预测一个 encoder 端对齐的位置(aligned position)$p_{t}$，而后基于 $p_{t}$选择一个窗口，用于计算 context vector $c_{t}$，$p_{t}$的计算公式为：

$$ p_{t} = S*sigmoid(v_{p}^{T}tanh(W_{p}h_{t})) $$

其中 S 表示 encoder 端的句子长度，$v_{p}$和 $w_{p}$是模型参数。得到 $p_{t}$ 后，$c_{t}$的计算将值关注窗口 $[p_{t}-D, p_{t}+D] 内的2D+1 个 encoder 输入。对齐向量 $a_{t}$的计算公式为：

$$ a_{t}(s) = align(h_{t}, \hat{h}_{s})exp(\frac{(s-p_{t})^{2}}{2\sigma^{2}}) $$

Global Attention 和 Local Attention 各有优劣，实际中 Global 的用的更多一点，因为：

* Local Attention 当 encoder 不长时，计算量并没有减少    
* 位置向量$p_{t}$的预测并不非常准确，直接影响到 Local Attention 的准确率

## Self-attention
前面我们说过，当 Key、Value、Query 指向相同时，就是 Self-Attention。比如现在想翻译"I arrived at the bank after crossing the river"，Self-Attention 利用了 Attention 机制，计算每个单词与其他所有单词之间的关联(也包含自身)，而后根据对齐模型分布得到加权表示作为该词的新的表示，这一表示很好的考虑到上下文的信息。下图展示在 encoder 和 decoder 时，一层层做  Self-Attention 时的流程：

![](/img/in-post/tensorflow/self_attention.gif)

很显然，Self-Attention 可以捕获句子中长距离的相互依赖特征，使远距离依赖被缩短，有利于有效利用这些特征，同时 Self-Attention 对于增加计算的并行性也有直接帮助。

# Transformer
谷歌提出的 Transformer 模型，用全 Attention 的结构代替的 LSTM，在翻译上取得了更好的成绩。这里基于[Attention Is All You Need](https://arxiv.org/abs/1706.03762)，对 Transformer 做一个介绍。

模型结构如下图所示，模型包含 encoder 和 decoder 两部分，其中 encoder 包含 Nx 个(6个)当前单元，decoder 部分包含 Nx 个框中单元。下面我们分块对其进行描述。

![](/img/in-post/tensorflow/transformer_detail.jpg)

## 输入
Encoder 的输入包含词向量和位置向量，词向量部分和正常的网络一样，通过学习获得，维度为 $d_{model}$。而位置向量则和以往不同，以往的位置向量是通过学习获得的，而这里谷歌提出了一种位置向量嵌入的方法：

$$ \left\{
\begin{aligned}
PE_{2i}(p) = sin(\frac{pos}{10000^{\frac{2i}{d_{model}}}}) \\
PE_{2i+1}(p) = cos(\frac{pos}{10000^{\frac{2i}{d_{model}}}}) 
\end{aligned}
\right.
$$}

其中 $d_{model}$ 是嵌入向量的维度， pos 表示当前词的位置， i表示嵌入向量中的 第 i个元素。谷歌还特意将这种方式构造的向量和学习得到的向量作对比，发现效果接近，然后谷歌就用这个构造式的，因为虽然效果接近，但这种构造式的更能在使用中适应不同长度序列。

除此之外，选择它的重要原因是这种构造的函数能够尝试学习相对位置信息，这是因为当我们固定位置 k时，相对位置 $PE_{pos+k}$ 是 $PE_{pos}$ 的线性函数，这就为表达相对位置提供了可能。

获得 位置向量后，将位置向量和词向量进行加和得到最终输入向量，所以前面我们看到词向量和位置向量维度是相同的。

## Encoder 部分

![](/img/in-post/tensorflow/transformer_encoder.png)

Encoder 部分由6个相同的子模块组成，每个子模块就是上面图中左侧那个方块了。包含几个子部分：

* Multi-Head Attention    
* Residual connection    
* Normalisation    
* Position-wise Feed-Forward Networks

在Encoder 内部又可以看做包含两个子层，一个是 Multi-Head Self-Attention为主，另一个是 Position-wise Feed-Forward Networks，每个子层内的运算可以总结为：

$$ sub_layer_output = LayerNorm(x + (SubLayer(x))) $$

接下来着重介绍 两个子层。

### Multi-Head Attention

定义上比较简单，我们知道，在 Attention 的本质那里我们定义 Attention 的公式为：

$$ Attention(Q, K, V) = \sum_{i}^{S}align(Q, K) * V $$

谷歌在文章里有一个具体的形式，公式为：

$$ Attention(Q, K, V) = softmax(\frac{QK^{T}}{\sqrt{d_{k}}})V $$

作者把这种叫Attention 叫做"Scaled Dot-Product Attention"，对应结构如下图所示，其中 Q 和 K 的维度是 $d_{k}$，V 的维度是 $d_{v}$。这个公式相比于正常的 Dot Attention 多了一个缩放因子 $\frac{1}{\sqrt{d_{k}}}$(这个缩放因子据说可以防止结果过大,但 无论多大,经过softmax不都变成归一化概率了么? 想不懂为什么)。除此之外作者还提到了 Additive Attention，这个没细看，等以后用到再说。。。

在这个 scaled dot-product attention 中,还有一个 mask部分, 在训练时它将被关闭,在测试或者实际使用中,它将被打开去遮蔽当前预测词后面的序列. 

![](/img/in-post/tensorflow/multi_head_attention.png)

所谓的 Multi-Head Attention 就是把 Q, K, V 通过参数矩阵映射一下，然后再做 Attention，把这个过程重复 h次，结果拼接起来。这个Multi-head 的 h就显而易见了。用公式表示为：

$$ MultiHead(Q, K, V) = Concat(head1, \dots, head_{h})W^{O} \\
        where head_{i} = Attention(QW_{i}^{Q}, KW_{i}^{K}, VW_{i}^{V}) $$

其中 $W_{i}^{Q} \in \mathbb{R}^{d_{model} \times d_{k}}$，$W_{i}^{K} \in \mathbb{R}^{d_{model} \times d_{k}}$, $W_{i}^{V} \in \mathbb{R}^{d_{model} \times d_{v}}$, $W_{i}^{O} \in \mathbb{R}^{d_{model} \times hd_{v}}$。

其中需要注意的是，一方面不同的head的矩阵时不同的,另一方面 multi-head Attention 可以并行计算，论文里 h=8, $d_{k} = d_{v} = d_{model}/h = 64.

### Position-wise Feed-Forward Networks
论文里说，它是一个前馈全连接网络，它被等同的应用到每一个位置上(pplied to each position separately and identically. )，它由两个线性变换和 ReLU 激活函数组成：

$$ FFN(x) = max(0, xW_{1} + b_{1})W_{2} + b_{2} $$

这个线性变换被应用到各个位置上，并且它们的参数是相同的。不过不同层之间的参数就不同了。这相当于一个核大小为1 的卷积。

### Transformer 内的数据流动

到这里 Transformer 核心模块得结构已经介绍差不多了,下面推导一下在一个block 内的数据流动. 

首先是输入部分,假设输入有 n 个词, 每个词的维度维 $d_{model}$(与论文保持一致), 每个词都有一个位置向量,它的维度也是 $d_{model}$.而后总的input 由词向量和 位置向量加和得到,因此 input 得维度是 $[n, d_{model}]$.

接下来数据被复制两份,一个用来做残差连接, 维度不变. 另一份进入 multi-head self-attention. 对于  Multi-head , $W_{i}^{Q}, W_{i}^{K}$ 的维度都是 $[d_{model}, d_{k}]$, 因此 $ Q W_{i}^{Q}$ 和  $ K W_{i}^{K}$ 的维度是$[n, d_{k}]$, 根据 attention 得计算公式,经过 self-attention 后得到的向量输出还是 $[n, d_{k}]$. multi-head 会把 h 个head 结果进行连接, 因为$d_{k}*h=d_{model}$,因此multi-head 得输出维度为 $[n, d_{model}]$. 至于 scaled 和 mask 不影响维度变化, 这里不考虑.

multi-head 出来后的向量回先和 残差传过来的做加和 和 LN, 维度不变, $[n, d_{k}]$. 之后会进入残差和 position-wise feed forward 网络. 这个网络的输出和输入维度一致, 内部的维度为2048. 因此 一个 block 得输出维度为 $[n, d_{k}]$, 和输入一样.

## Decoder
Decoder 部分相比于 Encoder ，结构上多了一个 Masked Multi-Head Attention 子层，它对decoder 端的序列做 attention. 相比于正常的  Scaled Dot-Product Attention，它在 Scale 后加了一个Mask 操作。这是因为在解码时并不是一下子出来的，它还是像传统 decoder 那样，一个时间步一个时间步的生成，因此在生成当前时间步的时候，我们看不到后面的东西，因此用 MASK 给后面的 遮住。

![](/img/in-post/tensorflow/multi_head_attention.png)

因此在解码时的流程为：

* 假设当前已经解码的序列为$s_{1}, s_{2}, \dots, s_{t-1}$，把该序列做词向量和位置向量嵌入    
* 对上述向量做 Masked Multi-Head Attention，把得到的结果作为 Q    
* Encoder 端的输出向量看做 K, V    
* 结合 Q, K, V 做 Multi-Head Attention 和 FFN等操作    
* 重复 decoder 部分 的子结构得到输出，而后解码得到输出词的概率

## 总结
Self-Attention 具有以下优点：

* 计算复杂度为 $O(n^{2}*d)$，而循环网络复杂度为 $O(n*d^{2})$，卷积的复杂度是 $O(k*n*d^{2})$，当n 远小于 d时，Self-Attention 更快    
* 可并行化    
* 可以更好的解决长时依赖问题，同时只需一步计算即可获取全局信息。    
* Self-Attention 更可解释，在翻译任务中学习到一些语法和语义信息

缺点是：

* 实践中 RNN 可以轻易解决的事，Transformer 没做到，如 复制 string，尤其是碰到比训练时的 序列更长时    
* 理论上， Transformer 非图灵完备

# 参考
[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)    
[https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)    
[《Attention is All You Need》浅读（简介+代码）](https://spaces.ac.cn/archives/4765/comment-page-1)
