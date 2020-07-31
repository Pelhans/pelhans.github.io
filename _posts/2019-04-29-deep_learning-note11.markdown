---
layout:     post
title:      "深度学习笔记（十一）"
subtitle:   "Word2Vec"
date:       2019-04-29 00:15:18
author:     "Pelhans"
header-img: "img/dl_background.jpg"
header-mask: 0.3 
catalog:    true
tags:
    - Deep Learning
---

* TOC
{:toc}

# 概述

Word2Vec 是谷歌的分布式词向量工具。使用它可以很方便的得到词向量。Word2Vec 分别使用两个模型来学习词向量，一个是 CBOW(Continuous bag-of-word)，另一个是 Skip-gram模型。

CBOW 模型是指根据上下文来预测当前单词。而Skip-gram 是根据给定的词去预测上下文。所以这两个模型的本质是让模型去学习词和上下文的 co-occurrence。有意思的是，我们需要的词向量只是两个模型的“副产物”。

Word2Vec 得到的词向量相比于传统的 one-hot 词向量相比，主要有以下两个优势：

* 低维稠密：Word2Vec 得到的词向量一般设置为50-500之间，而 one-hot 类型的词向量维度等于词表大小。    
* 蕴含语义信息：one-hot 表示法中，每一个词与其他词都是正交的，也就是说词与词之间没有任何关系，这不符合实际情况。而 Word2Vec 的假设“具有相同上下文的词语包含相似的语义信息”，使得语义相近的词保留了一定的几何关系。

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

$$ \frac{\partial L}{\partial \overrightarrow{\mathbf{h}}} = \frac{\partial L}{\partial \overrightarrow{\mathbf{u}}}*\frac{\partial \overrightarrow{\mathbf{u}}}{\partial \overrightarrow{\mathbf{h}}} = \sum_{j=1}^{V}((y_{j} - \delta_{jj^{*}}))\mathbf{W}^{'} $$

因为 $$\overrightarrow{\mathbf{h}} = \mathbf{W}^{T}\overrightarrow{\mathbf{x}}$$，而 $\overrightarrow{\mathbf{h}}$是 one-hot 编码的，因此只有一个分量非零，所以：

$$ \frac{\partial L}{\mathbf{W}} = \overrightarrow{\mathbf{x}} \otimes \frac{\partial L}{\partial \overrightarrow{\mathbf{h}}} = \frac{\partial L}{\partial \overrightarrow{\mathbf{h}}} $$

所以 $\mathbf{W}$ 的更新方程为：

$$ \overrightarrow{\mathbf{w}}_{I}^{(new)} =  \overrightarrow{\mathbf{w}}_{I}^{(old)} - \frac{1}{C}\eta \sum_{j=1}^{V}((y_{j} - \delta_{jj^{*}}))\mathbf{W}^{'} $$

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

![](/img/in-post/tensorflow/word2vec_softmax.png)

树中每个叶子节点代表词典中的一个词，于是每个词语都可以被 01 唯一编码，于是我们可以计算条件概率 
$p(w|context(x))$。我们约定以下符号：

* 路径中包含节点的个数为$N(w)$    
* 路径中的第j个节点为 $n(w,j)$    
* 路径中非叶子节点的参数向量 $\theta_{1}^{w}$    
* 路径中第j个节点对应的编码 $d(w,j)$

因此可以给出 w 的条件概率：

$$ p(w | context(w)) = \prod_{j=1}^{N(w)}p(d(w,j) | x_{w}, \theta_{j-1}^{w}) $$

也就是一条路径上从根节点到叶节点的编码概率。其中每一次分裂都是一个二分类问题，因此采用 sigmoid 函数来分类：

$$ p(d(w,j) | x_{w}, \theta_{j-1}^{w}) = 
\left\{
    \begin{aligned}
\sigma(x_{w}^{T} \theta_{j-1}^{w}),& d(w,j) = 0 \\
1-\sigma(x_{w}^{T}\theta_{j-1}^{w}),& d(w,j) = 1 
\end{aligned}
\right.
$$}

这样上式就可以简写为：

$$ p(d(w,j) | x_{w}, \theta_{j-1}^{w} ) = [\sigma(x_{w}^{T}\theta_{j-1}^{w} )]^{1-d(w,j)} * [1-\sigma(x_{w}^{T}\theta_{j-1}^{w}) ]^{d(w,j)} $$

因此负对数似然损失为：

$$ L = -\sum_{w\in C}\sum_{j=1}^{N(w)}\log \left\{ [\sigma(x_{w}^{T}\theta_{j-1}^{w} )]^{1-d(w,j)} * [1-\sigma(x_{w}^{T}\theta_{j-1}^{w}) ]^{d(w,j)} \right\} $$

因为前面两个求和，因此直接对求和后的分析，再加起来就好了，假定$l(w,j)$表示L中的词w对应的第j个结点的损失，则：

$$ l(w,j) = -\left\{ (1-d(w,j))log[\sigma(x_{w}^{T}\theta_{j-1}^{w})] +d(w,j)log[1-\sigma(x_{w}^{T}\theta_{j-1}^{w}) ]  \right\} $$

接下来计算 $\theta_{j-1}^{w}$ 的导数和 $ x_{w}$ 的导数。

$$ 
\begin{aligned}
\frac{\partial l(w,j)}{\partial \theta_{j-1}^{w}} & = -\left[ (1-d(w,j))\sigma(x_{w}^{T}\theta_{j-1}^{w})(1-\sigma(x_{w}^{T}\theta_{j-1}^{w}))*\frac{1}{\sigma(x_{w}^{T}\theta_{j-1}^{w})}*x_{w} -d(w,j)\frac{1}{1-\sigma(x_{w}^{T}\theta_{j-1}^{w})}*\sigma(x_{w}^{T}\theta_{j-1}^{w})(1-\sigma(x_{w}^{T}\theta_{j-1}^{w}))*x_{w} \right] \\
& = - \left[ (1-d(w,j))(1-\sigma(x_{w}^{T}\theta_{j-1}^{w}))x_{w} - d(w,j)\sigma(x_{w}^{T}\theta_{j-1}^{w})x_{w} \right] \\
& = -\left[1-\sigma(x_{w}^{T}\theta_{j-1}^{w})-d(w,j) + d(w,j)\sigma(x_{w}^{T}\theta_{j-1}^{w})- d(w,j)\sigma(x_{w}^{T}\theta_{j-1}^{w}) \right]*x_{w}\\
& = -[1-\sigma(x_{w}^{T}\theta_{j-1}^{w})-d(w,j)]x_{w}
\end{aligned}
$$

因此 $\theta_{j-1}^{w}$ 的更新公式为：

$$ \theta_{j-1}^{w(new)} = \theta_{j-1}^{w(old)} - \eta[1-\sigma(x_{w}^{T}\theta_{j-1}^{w})-d(w,j)]x_{w} $$

类似地，我们可以得到 $x_{w}$ 的梯度：

$$ \frac{\partial l(w,j)}{\partial x_{w}} = [1-d(w,j)-\sigma(x_{w}^{T}\theta_{j-1}^{w})]\theta_{j-1}^{w} $$

## 负采样
负采样是加快训练速度的一种方法，如对于训练样本(fox,the)，the 是正样本，剩下词表中其他的全是负样本。如果对所有负样本都输出概率那计算量将非常庞大。为此我们可以从所有负样本中随机选取一批负样本，仅仅利用这些负样本进行更新，这就叫负采样。根据谷歌的建议，一般挑选 5-20个负样本，挑选的公式为：

$$ p(word) = \frac{f(word)^{\frac{3}{4}}}{\sum_{i=1}^{V}(f(word)\frac{3}{4})} $$

其中 $f(word)$表示词出现的概率，计算得到的 p(word)越大则越有可能被选中。公式里面的 $\frac{3}{4}$完全是基于经验的。

最后借用[（三）通俗易懂理解——Skip-gram的负采样](https://zhuanlan.zhihu.com/p/39684349)里的一个实现相关技巧。

>负采样的C语言实现非常的有趣。unigram table有一个包含了一亿个元素的数组，这个数组是由词汇表中每个单词的索引号填充的，并且这个数组中有重复，也就是说有些单词会出现多次。那么每个单词的索引在这个数组中出现的次数该如何决定呢，有公式$P(w_i)*table\_size$，也就是说计算出的负采样概率*1亿=单词在表中出现的次数。    
有了这张表以后，每次去我们进行负采样时，只需要在0-1亿范围内生成一个随机数，然后选择表中索引号为这个随机数的那个单词作为我们的negative word即可。一个单词的负采样概率越大，那么它在这个表中出现的次数就越多，它被选中的概率就越大。

# word2vec 问题

参考 [word2vec问题（不涉及原理细节）](https://zhuanlan.zhihu.com/p/42080556)

# 参考
[1] [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781.pdf)    
[2]  [词的表达](http://www.huaxiaozhuan.com/%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86/chapters/word_representation.html)    
[3]  [skip-gramm模型skip了什么？为什么叫skip-gramm模型？](https://www.zhihu.com/question/302594410/answer/535720380)    
[4]  [关于skip gram的输出？](https://www.zhihu.com/question/268674520/answer/342067053)    
[5]  [漫谈Word2vec之skip-gram模型](https://zhuanlan.zhihu.com/p/30302498)    
[6]  [Hierarchical Softmax（层次Softmax）](https://zhuanlan.zhihu.com/p/56139075)    
[7]  [（三）通俗易懂理解——Skip-gram的负采样](https://zhuanlan.zhihu.com/p/39684349)    
[8]  [word2vec原理推导与代码分析](http://www.hankcs.com/nlp/word2vec.html#respond)    
