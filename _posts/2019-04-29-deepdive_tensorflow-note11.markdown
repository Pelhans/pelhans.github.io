---
layout:     post
title:      "深度学习笔记（十一）"
subtitle:   "Word2Vec"
date:       2019-04-29 00:15:18
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

# 概述

Word2Vec 是谷歌的分布式词向量工具。使用它可以很方便的得到词向量。Word2Vec 分别使用两个模型来学习词向量，一个是 CBOW(Continuous bag-of-word)，另一个是 Skip-gram模型。

CBOW 模型是指根据上下文来预测当前单词。而Skip-gram 是根据给定的词去预测上下文。所以这两个模型的本质是让模型去学习词和上下文的 co-occurrence。有意思的是，我们需要的词向量只是两个模型的“副产物”。

# CBOW
## 正向传播
CBOW 模型是根据上下文去估算当前词语，其模型结构如下图所示：

![](/img/in-post/tensorflow/word2vec_cbow2.png)

输入为 C 个单词，它们是给定单词的上下文，记为 $$\mathbf{X} = [\overrightarrow{x}_{1}, \overrightarrow{x}_{2},\dots, \overrightarrow{x}_{C}] $$。每个输入 $\overrightarrow{x}_{i}$ 都采用 ont-hot 表示法，维度等于词典大小 V。输入 $\mathbf{X}$ 的维度为 $C \times V$。

对于每个输入单词，都会乘以同一个权值矩阵 $\mathbf{W}$。$\mathbf{W} \in \mathbb{R}^{V\times N}$，N 为隐藏单元大小。可以看到在输入时权值是共享的。

隐向量记为 $\overrightarrow{\mathbf{h} }$，它记为所有输入词向量映射结果的均值：

$$ \overrightarrow{\mathbf{h} } = \frac{1}{C}\mathbf{W}^{T}(\overrightarrow{x}_{1} + \overrightarrow{x}_{2} + \dots + \overrightarrow{x}_{C}) = \frac{1}{C}(\overrightarrow{\mathbf{w}}_{I_{1}} + \overrightarrow{\mathbf{w}}_{I_{2}} + \dots + \overrightarrow{\mathbf{w}}_{I_{C}}  ) $$

其中 $$ \overrightarrow{\mathbf{w}}_{I_{i}} $$ 表示 权值矩阵 $ \mathbf{W} $ 中第 $I_{1}$ 行对应的向量。$I_{i}$ 表示第 i 个输入单词在词汇表V 中的编号。上式基于 输入的每个单词是 onte-hot 表示的，因此可以直接从 W 中提取出 1 对应的行。最终得到的 $$\overrightarrow{\mathbf{h}} \in \mathbb{R}^{N} $$。

得到隐藏层表示后，在乘以输出层权值矩阵 $\mathbf{W}^{'}$，其中 $\mathbf{W}^{'} \in \mathbb{R}^{N\times V}$。得到一个 1xV 维的向量 $$\overrightarrow{\mathbf{u}} = W^{'T}\overrightarrow{\mathbf{h}} $$  。其中

$$ u_{j} = \overrightarrow{\mathbf{w}^{'}}_{j}* \overrightarrow{\mathbf{h}} $$

表示词表 V 中，第j个单词的得分。向量$$\overrightarrow{\mathbf{u}}$$ 经过 softmax 层就可以得到词表的概率分布。对于词汇表中第 j 个单词来讲，它对应的输出概率为：

$$ y_{j} = p(word_{j} | \overrightarrow{\mathbf{x}}) = \frac{exp(u_{j})}{\sum_{j^{'}=1}^{V}exp(u_{j^{'}})} ,~~~~j = 1, 2, \dots, V $$

假设我们的目标输出是词 $word_{O}$，它在词典中的位置为 $j^{*}$，那么我们我们的优化目标是最大化目标词的输出概率
$p({word_{O} | word_{I_{1}},  word_{I_{2}},\dots,  word_{I_{C}}})$。因为涉及到 e指数，因此我们采用最小化它的负对数作为优化目标：

$$
\begin{aligned}
L & = -log p({word_{O} | word_{I_{1}},  word_{I_{2}},\dots,  word_{I_{C}}}) \\
& = -log \frac{exp(u_{j^{*}})}{\sum_{i=1}^{V}exp(u_{i})} \\
& = -u_{j^{*}} + log\sum_{i=1}^{V}exp(u_{i})  \\
& = -\overrightarrow{\mathbf{w}}^{'}_{j^{*}}*\overrightarrow{\mathbf{h}} + log\sum_{i=1}^{V}exp(\overrightarrow{\mathbf{w}}^{'}_{i}*\overrightarrow{\mathbf{h}})
\end{aligned}
$$

## 反向传播
我们要更新 $\mathbf{W}$ 以及 $\mathbf{W}^{'}$。

根据 softmax 的导数公式，可以很容易的求出$\frac{\partial E}{\partial u_{j}}$：

$$ \frac{\partial L}{\partial u_{j}} = y_{j} - \delta_{jj^{*}} $$

其中$$\delta_{jj^{*}}$$ 表示当 $$j^{*}=j$$ 时为1，其余时为0。因为 

$$u_{j} = -\overrightarrow{\mathbf{w}}^{'}_{j}*\overrightarrow{\mathbf{h}}$$

所以 

$$\frac{\partial u_{j}}{\partial \overrightarrow{\mathbf{w}}^{'}_{j}} = \overrightarrow{\mathbf{h}} $$

因此

$$ \frac{\partial L}{\partial \overrightarrow{\mathbf{w}}^{'}_{j}} = \frac{\partial L}{\partial u_{j}}* \frac{\partial u_{j}}{\partial \overrightarrow{\mathbf{w}}^{'}_{j}} = (y_{j} - \delta_{jj^{*}})\overrightarrow{\mathbf{h}} $$

因此 $\overrightarrow{\mathbf{w}}^{'}_{j}$的更新共识为：

$$ \overrightarrow{\mathbf{w}}_{j}^{'(new)} = \overrightarrow{\mathbf{w}}_{j}^{'(old)} - \eta (y_{j} - \delta_{jj^{*}})\overrightarrow{\mathbf{h}} $$

接下来计算$\mathbf{W}$ 的梯度。

$$ \frac{\partial L}{\partial \overrightarrow{\mathbf{h}}} = \frac{\partial L}{\partial \overrightarrow{\mathbf{u}}}*\frac{\partial \overrightarrow{\mathbf{u}}}{\partial \overrightarrow{\mathbf{h}}} = \sum_{j=1}^{V}((y_{j} - \delta_{jj^{*}})\overrightarrow{\mathbf{h}})\mathbf{W}^{'} $$

因为 $$\overrightarrow{\mathbf{h}} = \mathbf{W}^{T}\overrightarrow{\mathbf{x}}$$，而 $\overrightarrow{\mathbf{h}}$是 one-hot 编码的，因此只有一个分量非零，所以：

$$ \frac{\partial L}{\mathbf{W}} = \overrightarrow{\mathbf{x}} \otimes \frac{\partial L}{\partial \overrightarrow{\mathbf{h}}} = \frac{\partial L}{\partial \overrightarrow{\mathbf{h}}} $$

所以 $\mathbf{W}$ 的更新方程为：

$$ \overrightarrow{\mathbf{w}}_{I}^{(new)} =  \overrightarrow{\mathbf{w}}_{I}^{(old)} - \frac{1}{C}\eta \sum_{j=1}^{V}((y_{j} - \delta_{jj^{*}})\overrightarrow{\mathbf{h}})\mathbf{W}^{'} $$

其中第I行表示 输入x 的one-hot 表示中的非零行，其他行保持不变。

# Skip-Gram
Skip-gram 模型是根据一个单词来预测其上下文，相比与 CBOW 的输入层权重共享，Skip-Gram 的输出权重$\mathbf{W}^{'}$是共享的。

一个比较有意思的问题是， Skip-Gram 这个名字怎么理解呢？下面是[咸菜坛子](https://www.zhihu.com/question/302594410/answer/535720380)的回答：

>谢邀。首先n-gram是一系列连续的词（tokens），而skip-gram，或者skip-n-gram，skip的是token之间的gap。比如，下面的句子：    
    "the fox jumps over the lazy dog"    
    jumps over the是一个3-gram，那么(jumps, the)刚好skip了一个gram (over)，而这恰恰是skip-gram model的定义，即是，用目标词汇（jumps）预测周边（window size=2范围内的）词汇(over, the, fox)。

另外需要注意的是，Skip-Gram 模型的输入不是我们想象中的输入一个单词，预测一个句子，而是一个一个的 词对，如上面示例中，如果我们以fox 目标词汇，window size=1，那么我们的训练语料就有：(fox, jumps)、(fox, the)两个。

Skip-gram 模型的网络结构如下图所示：

![](/img/in-post/tensorflow/word2vec_skipGram1.png)

网络的输入为 $\overrightarrow{\mathbf{x}}$，采用 one-hot 编码。网络输出 $$\overrightarrow{\mathbf{x}}_{1},\dots, \overrightarrow{\mathbf{x}}_{C}$$， 其中 $\overrightarrow{\mathbf{x}}_{C}$ 是第c个输出单词的词汇表概率分布。在输出时共享权值矩阵 $\mathbf{W}^{'}$。

Skip-Gram 网络的目标是网络的多个输出之间的联合概率最大。假设网络第c个输出的第j个分量为 $u_{j}^{c} = \overrightarrow{\mathbf{w}}_{j}^{'}*\overrightarrow{\mathbf{h}} $，则：

$$ y_{j}^{c} = p(word_{j}^{c} | \overrightarrow{\mathbf{x}} ) = softmax(u_{j}^{c}) ~~~, j=1,2,\dots, V$$

当网络有C个输出时，定义损失函数：

$$ L = -log p(word_{O_{1}}, word_{O_{2}},\dots, word_{O_{C}} | word_{I}) = -log\prod_{c=1}^{C}softmax(u_{j^{*}_{c}}^{c}) $$

上式和之前相比本质上并无区别，下面直接给出更新公式：

$$ \overrightarrow{\mathbf{w}}^{'(new)}_{j} = \overrightarrow{\mathbf{w}}^{'(old)}_{j} - \eta \times (\sum_{c=1}^{C}(y_{j}^{c} - \delta_{jj^{*}}^{c}))\times  \overrightarrow{\mathbf{h}} ~~~~, j=1,2,\dots, V$$

$$ \overrightarrow{\mathbf{w}}^{(new)}_{I} = \overrightarrow{\mathbf{w}}^{(old)}_{I} - \eta(\sum_{j=1}^{V}\sum_{c=1}^{C}(y_{j}^{c}-\delta_{jj^{*}}^{c} )  \overrightarrow{\mathbf{w}}^{'}_{j}) $$

其中$\overrightarrow{\mathbf{w}}_{I}$ 表示输入x中的非零行，其他行在更新中保持不变。

# 优化
前面两个模型在计算输出时需要用到 softmax 函数，涉及到词表大小 V 词计算。而且在计算误差时，也需要更新一个很大的权值矩阵。这是一个很大的消耗，因此Word2Vec 采取了一些优化手段，其**核心思想是限制输出单元的数量**：

* 通过分层 softmax 来高效计算 softmax 函数    
* 通过负采样来缩减输出单元数量

## 分层 softmax

在分层softmax 中，算法并不直接求解输出向量 $$ {\overrightarrow{\mathbf{w}}_{1}^{'}, \dots, \overrightarrow{\mathbf{w}}_{V}^{'}} $$，而是求解哈夫曼树的路径编码表达。




# 参考
[1] [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781.pdf)    
[2]  [词的表达](http://www.huaxiaozhuan.com/%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86/chapters/word_representation.html)    
[3]  [skip-gramm模型skip了什么？为什么叫skip-gramm模型？](https://www.zhihu.com/question/302594410/answer/535720380)    
[4]  [关于skip gram的输出？](https://www.zhihu.com/question/268674520/answer/342067053)    
[5]  [漫谈Word2vec之skip-gram模型](https://zhuanlan.zhihu.com/p/30302498)    
[6]  [Hierarchical Softmax（层次Softmax）](https://zhuanlan.zhihu.com/p/56139075)    
[7]  [（三）通俗易懂理解——Skip-gram的负采样](https://zhuanlan.zhihu.com/p/39684349)    
[8]  [word2vec原理推导与代码分析](http://www.hankcs.com/nlp/word2vec.html#respond)    
