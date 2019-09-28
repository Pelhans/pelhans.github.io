---
layout:     post
title:      "命名实体识别（二）"
subtitle:   "NER 论文研读"
date:       2019-09-28 00:15:18
author:     "Pelhans"
header-img: "img/tensorflow.jpg"
header-mask: 0.3 
catalog:    true
tags:
    - Knowledge Graph
---


* TOC
{:toc}

# End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF

该文章首先使用卷积神经网络将单词的字符级信息编码成其字符级表示形式, 之后将字符级和单词级表示形式进行组合, 并将它们输入到双向 LSTM 中, 以对每个单词的上下文信息进行建模. 在 BLSTM 上, 使用顺序 CRF 联合解码整个句子的标签. 这个论文很经典, 值得细看.

![](/img/in-post/kg_paper/ner_task_cnn_blstm_crf_cnn.jpg)

上图是用 CNN 提取 字符级信息的示意图, 首先输入是一个一个的字符. 如 "Pad, P, l, a, y, i, n, g, Pad". 通过 lookup embedding 或者预训练得到的字向量, 将这些字符转化为向量表示"$W_{char} = w_{0}^{char}, w_{1}^{char}, w_{2}^{char}, w_{3}^{char}, w_{4}^{char}, w_{5}^{char}, w_{6}^{char}, w_{7}^{char}, w_{8}^{char}$". 对 $W_{char}$ 做卷积, 卷积核的深度是 $w_{i}^{char}$ 的维度, 宽度自己定, 一般可以取 3, 5 这种. 最终采用 均值池化或最大值池化 来获取固定维度的输出. 论文里卷积核宽度为3, 采用最大池化. 用了 30 个卷积核.

经过上面方法得到的字符级表示将会和词级别的表示连接在一起, 作为整个词的输入向量 $W = w_{0}, w_{1}, \dots, w_{8}$, 而后输入到 BLSTM 和CRF 做预测.  网络结构如下图所示

![](/img/in-post/kg_paper/ner_task_cnn_blstm_crf_total.jpg)

word embedding 用的是 Glove,在 60 亿 Wikipedia 和 网络文本上训练得到的 100 维词向量. char 的 embedding 采用的是在 $$ \left(-\sqrt{\frac{3.0}{wordDim}}, \sqrt{\frac{3.0}{wordDim}}  \right)$$ 范围内的随机初始化. 其中 wordDim 论文里用的是 30.

对于正向 LSTM, 它从前向后读取每个时间步的输入, 我们取得每个时间步的输出 $H^{正} = ( h_{1}^{正}, h_{2}^{正}, \dots, h_{n}^{正} )$ 作为正向 LSTM 的输出. 方向 LSTM 以相反的时间步获取文本输入而后得到每个时间步的输出 $H^{反} = ( h_{1}^{反}, h_{2}^{反}, \dots, h_{n}^{反}  )$. 最终我们把对应时间步的正向和反向的 LSTM 输出拼接得到 BLSTM 层的输出. $H = \{(H_{1}^{正}, H_{1}^{反}), (H_{2}^{正}, H_{2}^{反}), \dots, (H_{n}^{正}, H_{n}^{反}) \} $.

CRF 可以利用全局信息进行标记. 充分考虑当前标记受前面标记的影响. 下面介绍 CRF 在 BLSTM 后是怎么坐的.

BLSTM 的输出经过 softmax 后得到的是一个 $n\times k$ 的矩阵 P, 其中n 是序列长度, k 是类别数量. 因此 $P_{i, j}$ 表示第 i 个词的 第 j 个预测标签. 对于某一预测序列 $ y = (y_{1}, y_{2}, \dots, y_{n}) $, 可以定义如下分数:


$$ s(W, y) = \sum_{i=0}^{n+1}A_{y_{i}, y_{i+1}} + \sum_{i=1}^{n}P_{i, y_{i}} $$

其中 A 是转移分数, 其中 $ A_{i, j}$ 表示从标签 i 转移到 标签 j 的分数. $y_{0}$ 和 $y_{n+1}$ 表示 start 和 end 标签. 因此转移矩阵 A 是一个 $k+2$ 的方阵. 将每个可能的序列得分 s 综合起来 输入到  softmax 中, 得到每个序列对应的概率

$$ p(y|X) = \frac{e^{s(X, y)}}{\sum_{\tilde{y}\in Y_{X}}e^{s(X, \tilde{y})} } $$

在训练期间, 将会最大化正确标签的对数概率, 即

$$ \log(p(y | X)) = s(X, y) - \log\left(\sum_{\tilde{y}\in Y_{X}}e^{s(X, \tilde{y})} \right) $$

其中 $Y_{X}$ 表示输入序列 X 对应的所有可能的标签序列. 通过最大化上式, 模型将会学习有效正确的标签顺序, 避免如(IOB) 这样的输出. 在解码时, 求解使得分数 s 最高的标签序列

$$ y^{*} = arg\max_{\tilde{y}\in Y_{X}} s(X, \tilde{y}) $$

因为这里的转移矩阵只考虑了 bigram 的相互作用. 所以在优化和解码时可以直接用 DP 计算.


# Leveraging linguistic structures for named entity recognition with bidirectional recursive neural networks

论文使用基于文本语言学结构(linguistic structures)的 BRNN-CNN 网络(一个附加了卷积神经网络的特殊 RNN)来提升 NER 性能. 这种改进的动机是NER 和 语言学组分高度相关. 与传统的序列标注系统不同, 系统首先确定文本块是否是语言成分来识别哪些文本块是可能的命名实体. 然后通过句法和语义信息递归地传播到每个组成节点, 使用成分结构树(constituency tree) 对这些块进行分类. 该方法获得了当时的最优性能.

NER 可以看做是查找命名实体块和命名实体块分类两个任务的组合. 经典的顺序标注方法几乎不了解句子的短语结构。 但是，根据改论文的分析，大多数命名实体块实际上是语言成分，例如 名词短语。 这促使作者专注于NER的基于成分的方法，其中NER问题被转换为成分结构的每个节点上的命名实体分类任务。

标记流程为:

* 从选取结构中提取特征    
* 递归的对每个选取结构进行分类    
* 解决预测冲突

这里记一下成分分析. 所谓成分分析, 简单来说就是识别句子中的 如名词短语, 动词短语, 并以树结构的形式展现出来, 也就是成分结构树. 下图是一个例子

![](/img/in-post/kg_paper/ner_task_consti.png)

叶子节点就是输入的句子中的每个词,  NP 是名词短语, VP 是动词短语.

改论文就对输入的句子通过启发式规则等方法整理成类似成分树的样子. 其中每个节点包含: POS, word 和 head 信息. 对于每个word, 通过 Glove 得到嵌入向量. 同时为了捕捉形态学信息, 又通过一系列的卷积和 highway 层来生成利用了 char level 的表示, 最终把 char level 和 word level 的表示连接在一起作为 word 的最终表示. 

遗憾的是论文里没找到具体是怎么处理 char level 的, 只有上面那句话....有种被骗了的感觉....

给定一个成分分析树，其中每个节点代表一个组成部分. 网络为每个节点递归计算两个隐藏状态特征: $H_{bot}$, $H_{top}$. 如下图所示

![](/img/in-post/kg_paper/ner_task_consti_h.png)

最终对于每个节点, 根据 H 计算 NER 标记的概率分布预测类别.

# Neural Reranking for Named Entity Recognition

Reranking 是通过利用更多抽象特征来提高系统性能的框架。Reranking 系统可以充分利用全局特征，而在使用精确解码的基线序列标记系统中这是很难处理的。 重排序方法已用于许多NLP任务中, 如 parsing, 机器翻译, QAs.

在没有用深度学习进行 NER 的rerank 之前, Colins (2002) 尝试使用增强算法和投票感知器算法作为命名实体边界(没有实体分类)的 Reranking 模型. Nguyen(2010) 将带有内核的支持向量机 SVM 应用于模型重新排序, 从而在 CoNLL03 数据集上获得了较高的 F 值. Yoshida 和 Tsujii 在生物医学 NER 任务上使用了简单的对数线性模型重新排名模型, 也获得了一些改进.  但前面的这些方法均采用稀疏的人工特征, 因此该论文是深度学习用于 NER Reranking 的一次尝试.

改论文首先选择了两个 baseline 模型: CRF 和 BLSTM-CRF 模型来做 NER. 其中 CRF 采用了如下特征:

![](/img/in-post/kg_paper/ner_task_rerank_crf.png)

其中 shape 表示字符是否为数字/字母, captial 表示单词是否以大写开头. connect words 包含5个类型: "of", "and", "for", "-", other. Prefix 和 suffix 包含每个词的 4-level 前缀和后缀.

至于模型方面, 则比较基础, 它们的结构如下图所示:

![](/img/in-post/kg_paper/ner_task_rerank_baseline.png)

有了 baseline 模型, 接下来将获取它的输出作为排序模型的输入. 设 baseline 模型的 n-best 序列输出为 ${L_{1}, L_{2}, \dots, L_{n}}$. 其中的每个序列 ${l_{i1}, l_{i2}, \dots, l_{it}}$.  举个例子, 输入为 “Barack Obama was born in hawaii .”, 那么$L_{i}$ 就是 “B-PER I-PER O O O B-LOC O .”$C_{2}$ 或者 “LOC O O O O .” (C3) 这种.  接下来将 NER 的实体合并,并将 O 标记的用对应的单词替换掉. 如 C2 就可以得到 "PER was born in LOC .". 这样得到 rerank 模型的输入$C_{1}, C_{2}, \dots, C_{n}$.

模型采用固定窗口大小的 CNN 来获取词的字符级表示(和第一个论文里的一样),  CNN 窗口大小为3, 池化采用最大池化, 卷积核的数量为 50. 通过该方式得到的 word 表示和 SENNA 得到的 word 嵌入表示连接得到最终的 word 表示. 对于未登录词, 在 $$ \left(-\sqrt{\frac{3.0}{wordDim}}, \sqrt{\frac{3.0}{wordDim}}  \right) $$ 范围内进行随机初始化.

模型分别用 LSTM 和 CNN 来提取输入序列的特征. 对于 LSTM , 选取最后时间步的输出 $h_{LSTM}$ 作为 LSTM 的特征, CNN 的话, 就是普通的 CNN 对输入序列进行卷积和池化操作得到 $h_{CNN}$. 

接下来 LSTM 特征和 CNN 特征将会连接到一起得到 $h(C_{i})$, 而后通过 $s(C_{i}) = \sigma(Wh(C_{i}) + b)$ 得到该序列的分数.

解码通过联合 baseline 的输出概率$p(L_{i})$ 和 rerank 模型的得分的加权和作为输入序列的最终分数. 最终选取综合得分最高的序列作为 rerank 解码的序列.

$$ \hat{y}_{i} = rag\max_{C_{i}\in C(S)}(\alpha s(C_{i}) + (1-\alpha)p(L_{i})) $$

损失函数采用平方误差损失函数加 L2 正则项.

# Neural Architectures for Named Entity Recognition

传统的 NER 解决方案是把它当做一个序列标注问题来解决.  但在改论文中, 借鉴了 Dyer 的基于 转移的依存句法分析模型套路. 并将其用到了 NER 中. 

举个例子, 现在我们有输入 " Mark Watney visited Mars", 想对它进行 NER 标记. 

首先定义两个栈: stack 和 output. 其中 stack 用来存储存储待分类的 chunk, output 存储已经标记好的输出序列. 再定义一个缓存区 Buffer, 它里面装着没被处理的词.  再定义一些动作. 其中 SHIFT 表示将词从 Buffer 移动到 stack, OUT 表示将词从 Buffer 移动到 output. REDUCE(y) 表示将 stack 中的所有词打包成一个实体, 并赋予标记 y. 同时将该实体的表示放到 output 中.  算法处理的流程如下图所示

![](/img/in-post/kg_paper/ner_task_rnn_stack.png)

* 首先开始时,  output 和  stack 为空. Buffer 存了全部的词.     
* 第一步模型输出 SHIFT, Buffer 顶端元素 Mark 出栈, Stack 将该元素入栈.    
* 第二步模型输出 SHIFT, Buffer 顶端元素 Watney 出栈, Stack 将该元素入栈.    
* 第三步模型输出 REDUCE(PER), Stack 元素被打包成一个实体, (Mark Watney), 赋予类别 PER, 从 Stack 出栈, output 将该(Mark Watney)-PER 入栈    
* 第四步模型输出OUT, Buffer 顶端元素 visited 出栈, output 将该元素入栈    
* 第五步模型输出 SHIFT, Buffer 顶端元素 Mars 出栈, Stack 将该元素入栈.    
* 第六步模型输出 REDUCE(LOC), Stack 元素被打包成一个实体, (Mars), 赋予类别 PER, 从 Stack 出栈, output 将该(Mras)-LOC 入栈    
* Buffer 和 Stack 都为空, 算法终止.

给定当前 stack, buffer, output 和 动作的历史记录, 采用堆叠(stack) LSTMs 来分别计算它们的固定维度嵌入, 而后把它们四个连接起来作为综合的表示. 模型根据该综合表示得到每个时间步的概率分布, 并选贪婪的取概率最大那个作为当前动作, 尽管这种得到的结果不是最优的, 但论文里说, 效果还不错. 当输入序列长度为 n 时, 那么输出最多的时间步 为 2n.

那么怎么获得它们的向量表示呢? 论文分为两个部分 , 一个是基于字符的词表示, 另一个是基于 word 的表示. 对于 word 的表示, 论文采用 Word2vec 得到预训练的词向量. 对于 字符级表示, 论文采用 双向的 LSTM 来得到. 模型结构如下图所示(和前面 RNN 提取字符综述那的一样....):

![](/img/in-post/kg_paper/ner_task_rnn_stack_brnn.png)

论文认为, 以往虽然有用 CNN 提取字符级特征的, 但 LSTM 由于时间序列中最后输入的那些时间步影响更大, 因此前向 LSTM 可以获取后缀特征, 后向 LSTM 可以获取前向特征. 而这些特征是严重位置相关的, 所以 LSTM 在这时做的更好.
