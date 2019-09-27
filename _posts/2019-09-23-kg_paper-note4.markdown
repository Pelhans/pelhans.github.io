---
layout:     post
title:      "命名实体识别（一）"
subtitle:   "[笔记] 综述 A Survey on Deep Learning for Named Entity Recognition"
date:       2019-09-23 00:15:18
author:     "Pelhans"
header-img: "img/tensorflow.jpg"
header-mask: 0.3 
catalog:    true
tags:
    - Knowledge Graph
---


* TOC
{:toc}

# 摘要

**命名实体识别（NER）的任务是识别 mention 命名实体的文本范围，并将其分类为预定义的类别，例如人，位置，组织等**。NER 是各种自然语言应用（例如问题解答，文本摘要和机器翻译) 的基础。尽管早期的NER系统有着较好的识别精度，但是却严重依赖精心设计规则，需要大量的人力取整理和设计。近年来, 随着深度学习的使用, 使得 NER 系统的精度获得了质的提升. 在本文中，作者对 NER 的现有深度学习技术进行了全面回顾。

* 介绍NER资源，包括标记的NER语料库和现成的NER工具。    
* 将现有结构分为三部分：输入的分布式表示(Distributed representations for input)，上下文编码器(Context encoder) 和标签解码器(tag decoder)。    
* 介绍最具代表性的深度学习技术在 NER 中的应用, 包含 多任务学习, 迁移学习, 主动学习, 强化学习 和 对抗学习, Attention 等.    
* 介绍NER系统面临的挑战，并概述了该领域的未来方向。

## NER 技术概览

下图忽略网络中间细节, 展示了 NER 任务的目标. 首先网络的输入是一个一个的词, 记做 $w_{1}, w_{2}, \dots, w_{n}$. NER 任务的目标就是给出一个命名实体的起始和终止边界, 并给出该命名实体的类别. 如 Michael Jeffrey Jordan 就是一个命名实体, 它的起止位置为 $[w_{1}, w_{3}]$, 实体类型为 Person. 

![](/img/in-post/kg_paper/ner_task_ill.jpg)

一般来说, 做 NER 有四种方法, 和一般机器学习任务的方法一样:

* 基于规则: 手工制定符合什么条件的是什么词/类别. 优点是不需要标注数据, 缺点是制定规则和维护都很麻烦, 而且迁移成本高. 比较出名的有 LaSIE-II, NetOwl等    
* 无监督方法: 基于无监督算法, 不需要标注数据, 不过准确度一般有限.    
* 基于特征的机器学习方法: 需要标注数据, 同时一般结合精心设计的特征.常用的模型如 HMM, 决策树, 最大熵模型, CRF 等.常用的特征包含词级别特征(大小写, 词的形态, 词性标记), 文档和语料特征(局部语法和共现)等    
* 基于深度学习的方法: 需要标注数据, 自动学习特征, 可以端到端的搞

现在一般领域性比较强, 数据量特别少的会用规则, 其余基本上都是机器学习或者深度学习. 尤其是在数据量比较充足的时候, 深度学习一般都可以获得比较不错的指标, 有时也会加一些规则辅助.

# NER 数据资源和流行工具
## 资源

论文里给出了很多英文语料, 如下图所示:

![](/img/in-post/kg_paper/ner_task_corpus.jpg)

实际论文中, 用 CoNLL03 和 OntoNotes 两个的多一些. 

* CoNLL03包含两种语言的路透社新闻标注：英语和德语。     
    * 英语数据集包含大部分体育新闻，并在四种实体类型（人员，位置，组织和其他）中进行了标注。    
* OntoNotes项目的目标是标注大型语料库    
    * 包括各种类型（博客，新闻，脱口秀，广播，Usenet新闻组和对话电话语音）以及结构信息（语法和谓词参数结构）和浅语义（单词).    
    * 发行版1.0到发行版5.0共有5个版本。     
    * 这些文本用18种粗粒度实体类型（由89个子类型组成）进行标注。

## NER 工具

由学术界 提供的有 StanfordCoreNLP, OSU Twitter NLP, Illinois NLP, NeuroNER, NERsuite, Polyglot, and Gimli. 工业界提供的有 spaCy, NLTK, OpenNLP, LingPipe, AllenNLP, and IBM Watson. 

下图是工具的汇总和对应链接

![](/img/in-post/kg_paper/ner_task_tools.jpg)

对我个人来说, 一般中文项目用 HanNLP, StanfordCoreNLP, NLTK, spaCy 多一些. 

# NER 的性能评估指标

作者给出了精确匹配(Exact-match Evaluation) 和 宽松匹配(Relaxed-match Evaluation) 评估两种. 不过用的不多这里就不写了.

首先为了计算 F1, 定义一下 TP, FP, FN

* True Positive(TP): 实体被 NER 识别并标记为该类型 同时和 ground truth 对上了    
* False Positive(FP): 实体被 NER 识别并标记为该类型 但是和 ground truth 对不上    
* False Negative(FN):  实体没有被识别和标记为该类型, 但 ground truth 是

有了它们仨, 就可以算精确度(Precision), 召回率(Recall)和 F1 值了. 

* 精确率一般用来衡量查准率, 公式为: $ Precision = \frac{TP}{TP + FP} $    
* 召回率一般永来衡量查全率, 公式为: $ Recall = \frac{TP}{TP + FN} $    
* F 值是精确率和召回率的调和平均值, 公式为: $ F1 = 2 \times \frac{Precision\times Recall}{Precision + Recall} $

举个例子:

"张三 爱 北京 天安门 前 的 毛主席"###"Person O Location Location O O Person"###"Location O Person Location O Location Person"

上面###左侧是原, 中间是 ground truth, 左侧是 预测的标签. 这里需要注意的是, 我们的 TP, FP, FN 是针对单个类别的. 因此此时计算 Location 的F1的话, TP = 1(第四个), FP = 1(第一个), FN = 2(第三个和 第六个), Precision 就是 0.5, Recall 是 $\frac{1}{3}$, F1值就是 0.4. 

需要注意的是, 上面这个例子还没考虑实体边界的情况, 具体可以看这个代码 https://github.com/Pelhans/Entity_Linking/blob/master/evaluation.py (作者找不到了= =).

有了每个类别的指标后, 有两种办法把它们综合在一起:

* Macro averaged F-score: 根据每个类型的值来计算,得到平均值, 相当于把每个类型平等对待    
* Micro averaged F-score: 综合所有实体的所有类别的贡献来计算平均值, 相当于把每个实体平等看待.

一般 Micro 方法更容易受到样本不均衡的影响, 容易使得表现较好的大数样本掩盖表现不好的小数据量类别.

# NER 中的深度学习技术

## DL 为什么那么有效

NER 受益于 深度学习的好处主要有三点:

* NER 受益于 DL 的高度非线性, 相比于传统的线性模型(线性 HMM 和 线性链 CRF), 深度学习模型能够学到更复杂的特征    
* 深度学习能够自动学习到对模型有益的特征, 传统的机器学习方法需要需要繁杂的特征工程, 而深度学习则不需要    
* 深度学习可以端到端的搭建模型, 这允许我们搭建更复杂的 NER 系统.    

## 模型分层标准

传统的模型分层标准为: 字符层(character-level), 词层(word-level), 标签层(tag level). 该论文认为传统的模型分层标准不合理. 原因是 word level 这个表述不准确. 原始数据可以以 word 为单位进行输入,  也可以是在 char level 后, 有char 组合得到.因此论文提出新的分类方法:

* 输入的分布式表示( Distributed representations for input ): 基于 char 或者 word 嵌入的向量, 同时辅以 词性标签(POS), gazetter 等人工特征.    
* 语义编码(context encoder): 该层通过 CNN, RNN, LM, Transformer 等网络获取语义依赖.    
* 标签解码(tag decoder): 预测输入序列对应的标签, 常用的如 softmax, CRF, RNN, 指针网络(Point Network)

下图给出该分类的示意:

![](/img/in-post/kg_paper/ner_task_3tax.jpg)

## 输入的分布式表示

分布式表示通过把词映射到低维空间的稠密实值向量, 其中每个维度都表示隐含的特征维度. 一般 NER 系统的输入采用三种表示: word-level, char-level, 混合(hybrid) 表示.

需要注意的是, 该论文针对的是英语 NER, 因此这里的词是指 has, Jeff 这种, 字是指 a, b,  c 这种.

### Word-level 表示

很流行的一种方法, 通常使用无监督算法如 连续词袋模型(CBOW) 和 skip gram 模型对大量文本进行预训练, 得到每个词对应的向量表示. 其模型示意如下图所示:

![](/img/in-post/kg_paper/ner_task_word_level.jpg)

其中 CBOW 是给定周围词来预测中心词, skip gram 模型是给定中心词预测周围的词.更详细的了解请看博客[Word2Vec](http://pelhans.com/2019/04/29/deepdive_tensorflow-note11/).

Word level 比较好用的工具是 Word2Vec 和 Glove, 除此之外还有 fastText, SENNA等.

### Character-level 表示

除了词级别的, 还可以用基于字级别的向量表示, 现有的字符级标识对于显示利用子单词级信息(如前缀和后缀)很有用. 字符级表示的另一个优点是可以减轻未登录词(OOV)的问题. 所以字符级表示可以处理没见过的词,同时共享词素信息. 

通常有两个广泛使用的提取字符级表示的体系结构: 基于 CNN 的和 基于 RNN 的模型. 下图分别介绍它们.

#### CNN 用于 char-level 表示

论文(End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF) 利用 CNN 提取单词的字符级表示, 然后字符级表示与 word 级表示连在一起作为最终的词表示输入到 RNN 中.

论文(Leveraging linguistic structures for named entity recognition with bidirectional recursive neural networks) 应用了一系列的卷积和 highway 层来生成单词的字符级表示. 最终该表示被输入到双向递归网络中.

论文(Neural reranking for named entity recognition) 提出了一种用于 NER 的神经网络 reranking 模型, 其中使用了固定窗口大小的卷积层来提取词的字符级表示.

论文(Deep contextualized word representations) 提出了 ELMo 词表示, 它是通过在双层双向语言模型上进行字符级卷积运算得到的.

下面对这四篇论文做一个概述.

##### End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF

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


##### Leveraging linguistic structures for named entity recognition with bidirectional recursive neural networks

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

##### Neural Reranking for Named Entity Recognition

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

#### RNN 用于 char-level 表示

下图给出了一个常见的 RNN 用于提取 char level表示的结构. 其中输入为 "start, J, o, r, d, a, n, end". 通过 lookup embedding 得到字符的向量表示 "W = w0, w1, w2, w3, w4, w5, w6, w7". 该层后有一个双向的 RNN. 正向的 RNN那接收 W 的正序列, 并将最终时间步的输出$h^{正}$作为正向 RNN 的输出. 同理反向 RNN 接收 W 的倒序列, 得到最终时间步的输出 $h^{反}$. 最终词的表示有 $h^{正}$ 和 $h^{反}$ 拼接得到.

![](/img/in-post/kg_paper/ner_task_rnn.png)

论文(Neural architectures for named entity recognition) 利用双向 LSTM 提取单词的字符级表示, 和前面经典的 CNN+BLSTM+CNN 的做法类似, 将 BLSTM 得到的字符级词表示和预训练得到的词向量连接在一起, 作为最终的 词表示.

论文(Attending to characters in neural sequence labeling models) 使用 门控机制(gate)控制字符级信息和预训练得到的单词嵌入相结合, 通过 gate, 该模型能够动态地决定从字符或者单词级组分中使用多少信息.

论文(Named entity recognition with stack residual lstm and trainable bias decoding) 引入了具有堆叠残差(stack resdual) LSTM 和可训练偏差解码的神经网络 NER 模型, 其中输入的 word 表示来自从 RNN 中提取的字符级特征与词嵌入的结合.

论文(Multi-task cross-lingual sequence tagging from scratchMulti-task cross-lingual sequence tagging from scratch) 开发了一种以统一的方式处理跨语言和多任务联合训练的模型. 改论文采用了一个深层次的双向 GRU结构, 来从单词的字符序列中学习丰富的形态表示. 而后将字符级表示和单词嵌入连接起来得到单词的最终表示.

##### Neural Architectures for Named Entity Recognition

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

##### Attending to characters in neural sequence labeling models

慢慢补坑吧, 论文看太多优点恶心了, 先把原论文的内容整理完..

### 混合表示(Hybrid Representation)

除了基于字和词的表示外, 一些研究还使用了额外的信息, 如 gazetteers 和 lexical similarity 等 添加到 word 表示中, 添加这些人工特征虽然会增加模型的表现, 不过可能降低模型的迁移泛化能力.

在论文(Bidirectional lstm-crf models for sequence tagging) 中, 作者用了四个额外特征 : 拼写特征, 文本特征, 词向量和 gazetteer 特征. 他们的实验结果表明, 额外的特征能够提升标注的准确性.

论文(Named entity recognition with bidirectional lstm-cnns) 使用 BiLSTM 和 字符级的 CNN网络. 模型输入除了 词嵌入之外, 还包含词级别特征(开头大写, lexicons) , 字级别特征(每个字符使用了4维额外特征: 大写, 小写, 标签符号, 其他).

论文(Disease named entity recognition by combining conditional random fields and bidirectional recurrent neural networks) 使用了基于 CRF 的神经网络徐彤来识别疾病命名实体. 该系统使用了很多额外的特征: words, POS tags, chunking, word shape(dictionary 和 morphological 特征).

论文(Fast and accurate entity recognition with iterated dilated convolutions) 将 100 维的词向量与 5 维的 word shape 特征向量(是否全部大写, 不是全部大写, 开投字母大写, 包含大写字母)进行连接.

论文(Multi-channel bilstm-crf model for emerging named entity recognition in social media) 使用 词嵌入, 字嵌入 和 与词相关的语法嵌入(POS 标签, 一寸角色, 单词位置, head 位置) 来构造词表示.

论文(Robust lexical features for improved neural network named-entity recognition) 发现神经网络常常会抛弃大部分的词法(lexical) 特征. 他们提出一个可以离线训练的并且可以用于任何神经网络系统的新词法表示. 该词法表示的维度为 120, 每个词都有一个, 通过计算词与实体类型的相似度得到.

## Context  Encoder Architectures

神经网络 NER 系统的第二层通过接收上一层的嵌入向量来学习语义编码. 论文将该层分为四类: CNN, RNN, 递归神经网络, transformer.

### CNN

Collobert 等(Natural language processing (almost) from scratch) 用整个句子的信息来对词进行标记, 其网络结构如下图所示:

![](/img/in-post/kg_paper/ner_task_context_cnn.png)

每个词通过向量嵌入转化为对应的向量. 之后通过卷积层来得到局部的特征. 卷积层的输出大小与输入的句子长度有关. 为了获取固定维度的句子表示, 在卷积后增加了池化层. 池化可以用最大池化或者平均池化.最后 tag decoder 使用该句子表示来得到标签的概率分布.

论文(Named entity recognition in chinese clinical text using deep neural network) 使用卷积层和一系列的全局隐层节点来生成全局的特征表示. 之后局部特征和全局特征被联合输入到标准的仿射网络来进行临床实体识别.

论文(Joint extraction of multiple relations and entities by using a hybrid neural network) 观察到 RNN 靠后时间步的影响要大于前面时间步的词. 然而对于整个句子来说, 重要的特征有可能出现在各个角落. 因此他们提出 BLSTM-RE 模型, 其中 BLSTM 用来不做长期依赖特征, 同时 CNN 用来学习 高级别(high-level)表示. 之后特征被输入到 sigmoid 分类器. 最终 整个句子表示(由 BLSTM 生成) 和 关系表示(sigmoid 分类器生成的) 被输入到另一个 LSTM 网络来预测实体.

论文(Fast and accurate entity recognition with iterated dilated convolutions) 提出膨胀卷积神经网络(Iterated Dilated Convolutional Neural Networks, ID-CNNs), 该模型在大文本和结构化预测上拥有比传统 CNN 更好的表现. ID-CNNs 的时间复允许固定深度的卷积在整个文档上并行的跑. 下图给出膨胀 CNN 模块的结构. 其中展示了四个堆叠的宽度为3的膨胀卷积. 对于膨胀卷积, 输入宽度的影响会随着深度的增加指数型增长, 同时每层的分辨率没有什么损失. 实验表明, 相比于传统得到 Bi-LSTM-CRF 有 14-20x 倍的加速, 同时保持相当高的准确率.

下图为 膨胀 CNN 模块示意图, 最大膨胀深度为 4, 宽度为 3. 对最后一个神经元有贡献的神经元都被加黑显示.

![](/img/in-post/kg_paper/ner_task_context_pengzhang.png)

### RNN

RNN 以及它的变体 LSTM 和 GRU 被证明在序列数据中有较好的效果. 其中前向 RNN 能够有效利用过去的信息, 反向 RNN 能够利用未来的信息, 因此 双向 RNN 能够利用整个序列的信息, 因此 双向 RNN成为深度语义编码的标配结构. 一个传统的基于 RNN 的语义编码结构如下图所示

![](/img/in-post/kg_paper/ner_task_context_rnn.png)

Huang 等(Bidirectional lstm-crf models for sequence tagging) 最先使用 BiLSTM+CRF 结构用于 序列标注问题(POS, chunking 和 NER). 之后涌现出很多工作都使用 BiLSTM 作为基本结构来编码语义信息. 

论文(Toward mention detection robustness with recurrent neural networks) 表明, RNN 不仅在通用情况下超过传统系统, 同时在英语的跨领域情况下也能达到最优的性能(相对误差减少 9%). 在跨语言的情况下, 在类似的荷兰语 NER 任务中, RNN 明显优于传统方法(相对误差减少 22%).

论文(Neural Models for Sequence Chunking) 通过研究 DNN 用于序列分块的方法, 提出了一种替代方法, 并提出了三种神经网络模型, 以便每个块都可以作为完整的标记单元. 实验结果表明, 所提出的神经序列分块模型可以再文本分块和槽填充任务上实现最佳性能.

论文(Multi-task cross-lingual sequence tagging from scratch) 将深度 GRUs 用于 字符级和 词级别表示上来进行形态学和语义信息编码. 之后作者进一步扩展模型到跨语言和多任务的中.

论文(Named entity recognition with parallel recurrent neural networks) 在同一输入上采用了多个独立的双向 LSTM 单元. 他们的模型通过使用模型间的正则项促进了 LSTM 单元之间的多样性. 通过将计算分散到多个  LSTM 中, 他们发现模型的中参数减少了.

### 递归神经网络

递归神经网络是非线性自适应模型，能够通过以拓扑顺序遍历给定的结构来学习深度的结构化信息。命名实体与语言成分高度相关，例如名词短语。但是，典型的顺序标注方法很少考虑句子的短语结构。为此，论文(Leveraging linguistic structures for named entity recognition with bidirectional recursive neural networks, char level 那里有说) 提出对NER的成分结构中的每个节点进行分类。该模型递归地计算每个节点的隐藏状态向量，并通过这些隐藏向量对每个节点进行分类。下图显示了如何为每个节点递归计算两个隐藏状态特征。自下而上的方向计算每个节点的子树的语义成分，而自上而下的对应对象将包含子树的语言结构传播到该节点。给定每个节点的隐藏矢量，网络将计算命名实体类型加特殊非实体类型的概率分布。

![](/img/in-post/kg_paper/ner_task_context_recursive.png)

### 神经语言模型

语言模型用来描述序列的生成. 给定一个序列$(t_{1}, t_{2}, \dots, t_{N})$. 则得到该序列的概率为

$$ p(t_{1}, t_{2}, \dots, t_{N}) = \prod_{k=1}^{N}p(t_{k}| t_{1}, t_{2}, t_{k-1} ) $$

类似的, 一个后向模型得到该序列的概率

$$ p(t_{1}, t_{2}, \dots, t_{N}) = \prod_{k=1}^{N}p(t_{k} | t_{k+1}, t_{k+2}, \dots, t_{N}) $$

对于神经语言模型, 可以用 RNN 的每个时间步$t_{k}$的输出得到概率 $p(t_{k})$. 在每个位置上, 可以得到两个语义相关的表示(前向和后向), 之后将它们俩结合作为最终语言模型向量表示 $t_{k}$. 这种语言模型知识已经被很多论文证实在序列标注中很有用.

论文(Semi-supervised multitask learning for sequence labeling) 提出一个序列标注模型, 该模型要求模型除了预测标签意外, 还要预测它周围的词, 网络结构如下图所示

![](/img/in-post/kg_paper/ner_task_context_lm_surround.png)

在每个时间步, 要求模型预测当前词的标签和下一个词. 反向的网络就预测当前标签和前一个词, 这样网络就预测了当前标签和周围的词.

论文(Semisupervised sequence tagging with bidirectional language model) 提出 TagLM, 一个语言模型增强序列标注器. 该标注器同时考虑了预训练词嵌入和双向语言模型嵌入. 

下图展示了 LM-LSTM-CRF 的网络结构.

![](/img/in-post/kg_paper/ner_task_context_lmlstm.png)

其中左侧下方是 字符级的嵌入, 通过双向 LSTM 得到. 中间 三个黑点那个是预训练得到的词向量, 虚线右侧将上下文嵌入得到的 LM 部分. 它们三个连接得到一个综合的表示, 被输入到 BLSTM 和 CRF 中得到标记序列.

### Deep Transformer

Transformer 这个大家也很熟悉了, 它在论文中的结构如下图所示

![](/img/in-post/kg_paper/ner_task_context_transformer.png)

实际使用中用的是左侧的那个块中的东西. 很多任务表明 Transformer 在序列生成等多种任务上有很好的表现, 同时可以并行, 效率高.(该综述里介绍了 BERT, GPT, ELMo, 但没介绍用 Transformer 做 NER 的论文, 等以后自己补上吧)

## Tag decoder

Tag 是NER 的最后一层, 它接收语义表示输出标注序列. 常见的解码方式为: MLP + softmax, CRF, RNN 和指针网络(pointer network). 下图给出了它们的结构示意图

![](/img/in-post/kg_paper/ner_task_tag_four.png)

### MLP + softmax

这个太常见了, 比如 BLSTM 的输出后面跟一个全连接层进行降维, 再接 softmax 得到标签的概率分布.

### CRF

自从 CRF 被提出加到神经网络后面用于解码后, 几乎成了一个标配了. 主要是因为 CRF 可以利用全局信息进行标记. 下面以 BLSTM+CRF 那篇论文为网络结构, 说一下 CRF 的原理.

BLSTM 的输出经过 softmax 后得到的是一个 $n\times k$ 的矩阵 P, 其中n 是序列长度, k 是类别数量. 因此 $P_{i, j}$ 表示第 i 个词的 第 j 个预测标签. 对于某一预测序列 $ y = (y_{1}, y_{2}, \dots, y_{n}) $, 可以定义如下分数:


$$ s(W, y) = \sum_{i=0}^{n+1}A_{y_{i}, y_{i+1}} + \sum_{i=1}^{n}P_{i, y_{i}} $$

其中 A 是转移分数, 其中 $ A_{i, j}$ 表示从标签 i 转移到 标签 j 的分数. $y_{0}$ 和 $y_{n+1}$ 表示 start 和 end 标签. 因此转移矩阵 A 是一个 $k+2$ 的方阵. 将每个可能的序列得分 s 综合起来 输入到  softmax 中, 得到每个序列对应的概率

$$ p(y|X) = \frac{e^{s(X, y)}}{\sum_{\tilde{y}\in Y_{X}}e^{s(X, \tilde{y})} } $$

在训练期间, 将会最大化正确标签的对数概率, 即

$$ \log(p(y | X)) = s(X, y) - \log\left(\sum_{\tilde{y}\in Y_{X}}e^{s(X, \tilde{y})} \right) $$

其中 $Y_{X}$ 表示输入序列 X 对应的所有可能的标签序列. 通过最大化上式, 模型将会学习有效正确的标签顺序, 避免如(IOB) 这样的输出. 在解码时, 求解使得分数 s 最高的标签序列

$$ y^{*} = arg\max_{\tilde{y}\in Y_{X}} s(X, \tilde{y}) $$

因为这里的转移矩阵只考虑了 bigram 的相互作用. 所以在优化和解码时可以直接用 DP 计算.

不过论文(Segment-level sequence modeling using gated recursive semi-markov conditional random fields) 认为,  CRF 虽然强, 但是却不能充分利用段落(segment) 级别的信息, 这是因为词级别的编码表示不能完全表达段落的内在属性. 因此改论文提出 门递归半马尔科夫条件随机场(gated recursive semi-markov CRFs). 该模型直接对段落进行建模, 而不是 词, 并通过 门控递归卷积神经网络自动的学习段落级特征, 

近期, 论文(Hybrid semi-markov crf for neural sequence labeling) 提出混合 半马尔科夫 CRFs(hybrid semi-Markov CRFs) 用于序列标注. 该方法直接采用段落而不是词作为基本单元, 词级别特征被用于推导段落分数. 因此该方法能够同时使用 词和段落级别的信息.

### RNN

有一部分研究采用 RNN 来解码标注. 这里以论文(Deep active learning for named entity recognition)为例, 说一下解码流程. 如上图 C 所示, 图中的 $h_{i}^{Enc}$ 表示编码隐层向量,  $h_{i}^{Dec}$ 表示解码银城向量, 初始时, 给定 Go 标记(类似于 start), 当前时间点的编码向量和上一时间步的解码向量, 模型输出当前时刻的解码向量, 即

$$ h_{i+1}^{Dec} = f(y_{i}, h_{i}^{Dec}, h_{i+1}^{Enc}) $$

解码向量经过softmax 得到标签的概率分布, 取概率最大的作为最终标记. 如此循环直到解码完成.

### 指针网络

指针网络应用RNN来学习输出序列的条件概率，其中元素是与输入序列中的位置相对应的离散 token。 它通过使用softmax概率分布作为“指针”来表示可变长度词典。 

论文(Neural models for sequence chunking) 首先应用指针网络来产生序列标签。 如上图 （d）所示，指针网络首先识别块（或段），然后对其进行标记。 重复该操作，直到处理了输入序列中的所有单词。 在上图 d 中，给定起始标记“ <s>”，首先标识段“ Michael Jeffery Jordan”，然后将其标记为“ PERSON”。 分割和标记可以通过指针网络中的两个独立的神经网络来完成。 接下来，将“ Michael Jeffery Jordan”作为输入并输入到指针网络中。 结果，“was”被识别并标记为“ O”。

## 基于 DL 的NER 总结

下图给出了近期在 NER 方面的工作汇总

![](/img/in-post/kg_paper/ner_task_summary.png)

总结一下:

* 在语义编码截断, 用的最多的是 RNN(其中 LSTM最多, GRU 要少一点)    
* 解码用的最多的是 CRF. BiLSTM+CRF 组合的网络结构用的最多    
* 在向量嵌入方面, 词的话 Word2vec, Glove, SENNA 用的比较多    
* 字符级别的 LSTM 比 CNN 多一点.     
* 额外特征方面,  POS 用的更多一点, 但是，关于是否应该使用外部知识（例如，地名词典和 POS）或如何将其集成到基于DL的NER模型，尚未达成共识。 然而，对诸如新闻文章和网络文档之类的通用域文档进行了大量实验。 这些研究可能无法很好地反映特定领域资源的重要性，例如在特定领域中的地名词典。

# 应用深度学习技术 到 NER

论文介绍了了 多任务学习(multi-task learning), 深度迁移学习(deep transfer learning), 深度主动学习(deep active learning), 深度强化学习(deep reinforcement learning), 深度对抗学习(deep adversarial learning) 和 神经元注意力(neural attention) 用于 NER 的进展. 

## 深度多任务学习

多任务学习是一种可以一起学习一组相关任务的方法。 通过考虑不同任务之间的关系，与单独学习每个任务的算法相比，多任务学习算法有望获得更好的结果。 

论文(Natural language processing (almost) from scratch) 训练了一个窗口/句子(window/sentence) 网络来同时训练 POS，Chunk，NER和SRL任务。 在窗口网络中共享第一线性层的参数，在句子网络中共享第一卷积层的参数。 最后一层是特定于任务的。 通过最小化所有任务的平均损失来实现训练。 这种多任务机制使训练算法能够发现对所有感兴趣任务有用的内部表示形式。 

论文(Multi-task cross-lingual sequence tagging from scratch) 提出了一个多任务联合模型，以学习特定于语言的规律性，同时训练POS，Chunk和NER任务。 

论文(Semi-supervised multitask learning for sequence labeling) 发现，通过在训练过程中加入 无监督语言建模训练目标，序列标记模型可以实现性能改进。

除了将NER与其他序列标记任务一起考虑之外，多任务学习框架可以应用于实体和关系的联合提取, 如论文(Joint extraction of entities and relations based on a novel tagging scheme) 和 论文(Joint extraction of multiple relations and entities by using a hybrid neural network).

或将NER建模为两个相关的子任务：实体分割和实体类别预测, 如论文(A multitask approach for named entity recognition in social media data) 和 论文(Multi-task domain adaptation for sequence tagging).


