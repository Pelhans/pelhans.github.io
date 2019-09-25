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

上图是用 CNN 提取 字符级信息的示意图, 首先输入是一个一个的字符. 如 "Pad, P, l, a, y, i, n, g, Pad". 通过 lookup embedding 或者预训练得到的字向量, 将这些字符转化为向量表示"W = w0, w1, w2, w3, w4, w5, w6, w7, w8". 对 W 做卷积, 卷积核的深度是 wi 的维度, 宽度自己定, 一般可以取 3, 5 这种. 最终采用 均值池化或最大值池化 来获取固定维度的输出. 论文里卷积核宽度为3, 采用最大池化. 用了 30 个卷积核.

经过上面方法得到的字符级表示将会和词级别的表示连接在一起, 作为整个词的输入向量, 而后输入到 BLSTM 和CRF 做预测.  网络结构如下图所示

![](/img/in-post/kg_paper/ner_task_cnn_blstm_crf_total.jpg)

word embedding 用的是 Glove,在 60 亿 Wikipedia 和 网络文本上训练得到的 100 维词向量. char 的 embedding 采用的是在  \left(-\sqrt{\frac{3.0}{wordDim}}, \sqrt{\frac{3.0}{wordDim}}  \right)$$ 范围内的随机初始化. 其中 wordDim 论文里用的是 30.

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

###### word representation

模型采用固定窗口大小的 CNN 来获取词的字符级表示(和第一个论文里的一样),  CNN 窗口大小为3, 池化采用最大池化, 卷积核的数量为 50. 通过该方式得到的 word 表示和 SENNA 得到的 word 嵌入表示连接得到最终的 word 表示. 对于未登录词, 在 $$ \left(-\sqrt{\frac{3.0}{wordDim}}, \sqrt{\frac{3.0}{wordDim}}  \right) $$ 范围内进行随机初始化.

###### LSTM 特征 与 CNN 特征

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


