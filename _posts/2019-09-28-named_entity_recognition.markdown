---
layout:     post
title:      "命名实体识别论文笔记"
subtitle:   ""
date:       2019-09-28 03:27:18
author:     "Pelhans"
header-img: "img/ner.jpg"
header-mask: 0.3 
catalog:    true
tags:
    - Knowledge Graph
---


* TOC
{:toc}

# 概览

命名实体识别（NER）的任务是识别 mention 命名实体的文本范围，并将其分类为预定义的类别，例如人，位置，组织等。NER 是各种自然语言应用（例如问题解答，文本摘要和机器翻译) 的基础。

早起 NER 的研究是基于规则的，大家根据一些实体字典和词形等规则进行实体识别，比如 "199x-xx-xx"的大概率是时间。后来开始采用机器学习的方法，比如 SVM、HMM、CRF等，机器学习方法也很依赖人工构建的特征，但灵活性和可迁移性提升了很多，尤其是 CRF的出现，配合精良的特征的话，在中小数据上表现并不比深度学习差，速度还快，是一个很好的初期模型。随着深度学习的兴起，尤其是 CNN+BiLSTM + CRF 模型，可称为新一代 baseline 之王。在其基础上增加一些数据相关的特征或一些模型的改动就能解决大部分问题，简直棒极了。 2018年末 BERT 和 2019 年 各种预训练模型的出现，新一个时代来临了。但科研是个圈，BERT 在NER 中可以仅充当 embedding 层 也可以集合 embedding + 语义编码层，除此之外，加什么样的特征，或者关注NER 需要的全局啊、局部呀、上下文各种特征需不需要加，要加怎么加进去都是一个问题。而这些问题也是 NER 任务特性所带来的挑战，在各个时期都在被研究这，因此借古看今是十分有必要的。

本文对近几年的 NER 相关论文进行总结，给出 NER 任务的框架和大家感兴趣的研究点。

# 分类标准

根据论文 A Survey on Deep Learning for Named Entity Recognition，模型可以分为 3 层：


* 输入的分布式表示( Distributed representations for input  ): 基于 char 或者 word 嵌入的向量, 同时辅以 词性标签(POS), gazetter 等人工特征.    
* 语义编码(context encoder): 该层通过 CNN, RNN, LM, Transformer 等网络获取语义依赖.    
* 标签解码(tag decoder): 预测输入序列对应的标签, 常用的如 softmax, CRF, RNN, 指针网络(Point Network)

![](/img/in-post/kg_paper/ner_task_3tax.jpg)

这三层的每一层都包含了人们对于 NER 任务的先验输入。如

* 分布式表示部分：给模型尽可能丰富有效的额外知识。 word-embedding 是为了更好的词义表示， char-level 是赋予模型词内部信息，强迫模型关注局部依赖信息。POS 标签、 gazetter 、词形特征 是先验知识。    
* 语义编码层：通过 CNN 获取较强的局部依赖和多层 CNN 带来的大感受野获取长程依赖，通过 LSTM 获取长程依赖， BLSTM 获取长程依赖的同时还可以获取上下文语义信息，Transformer 获取长程依赖，递归 RNN 获取语言结构信息等等。每个模型都有对应的假设，应用哪些模型其实就是潜在地赋予模型对应的先验知识。    
* 解码层直接采用 MLP + softmax 就是最原始的，没有额外的先验知识，用了 CRF 的话，相当于赋予模型解码时观看全局的能力，对解码形成约束，让模型知道 O 到 I 的情况不应该出现。

我们对模型所做的改动也大体上的对目前模型在特定领域所缺少的先验进行增加或改进。因此我根据各论文的改动部分进行分块，方便阅读。


# 编码层改进

## End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF

2016, 该文章首先使用卷积神经网络将单词的字符级信息编码成其字符级表示形式, 之后将字符级和单词级表示形式进行组合, 并将它们输入到双向 LSTM 中, 以对每个单词的上下文信息进行建模. 在 BLSTM 上, 使用顺序 CRF 联合解码整个句子的标签. 这个论文很经典, 值得细看.

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

## CAN-NER Convolutional Attention Network for Chinese Named Entity Recognition

文章研究了一种卷积注意网络CAN，它由具有局部注意层的基于字符的卷积神经网络（CNN）和具有全局自注意层的门控递归单元（GRU）组成，用于从局部的相邻字符和全局的句子上下文中获取信息。

文章认为以往的用 char 嵌入的方式只能携带有限的信息，会丢失掉关于词和词的序列信息，如“拍”这个字，在“球拍” 和 “拍卖”中是完全不同的两个意思。因此文章指出，如何**更好的使用词内信息并发掘局部语义信息**是基于字符方法的关键。

文章结构如下图所示：

![](/img/in-post/kg_paper/ner_can.jpg)

首先输入的是一个一个的字符，卷积注意力层用来编码输入的字符序列并隐式地对局部语义相关的字符进行分组。输入用$$ x = [x_{ch}; x_{seg}]$$ 表示，其中 $$x_{ch}$$ 表示中文字符， $$x_{seg}$$ 表示分词信息，采用 BMES 编码。

之后对输入进行向量嵌入，包含 字向量、分词向量 和位置向量，$$ d_{e} = d_{ch} + d_{pos} + d_{seg} $$ 。得到输入向量后，采用局部 attention(local attention)来捕捉 窗口范围内 中心词和周围词的依赖，局部 attention 的输出被送到 CNN 中，最后采用 加和池化方案。具体来说，对于位置信息的嵌入，采用局部 one-hot 表示法，即在当前 local 范围内，除了所处位置的值为 1 ，其余都是 0。attention 是 self-attention。

得到局部特征后，进入到 BiGRU-CRF 中，而后采用全局的 attention来进一步捕捉句子级别的 全局信息。后面接 CRF，得到分类结果。self-attention 可以捕捉广义的上下文信息，减少无用中间词的干扰。

从直觉上将，att + cnn 有效的原因是 att 精准的捕获局部依赖关系，减少无用词的噪声，而后通过 CNN 来对精准的局部依赖建模，提升 CNN 的局部依赖效果。BiGRU + att 好的原因是 全局 attention 可以更好的捕捉长程依赖信息，而 GRU/LSTM 这种虽然也能对长期序列建模，但对靠前时间步会不友好，缺乏平等性。attention 可以在一定程度上弥补这点。与此同时， attention 完全没有管位置信息的缺点被 LSTM/GRU 很好的弥补了。更进一步，其他 CNN/LSTM 与 attention 结合的方式怎么样呢？从直觉上来看， CNN + Att 可以来代替 max / avg pooling，在编码降噪上效果也许会好。att + LSTM/GRU 可以理解是对进入 LSTM 的向量获得一个全局的表示，如对词与词关系的建模。

**思考： attnetion 和 CNN 或 RNN 结合的话，对于 NER 任务来讲，放在哪个位置比较好？**

上面虽然开了脑洞，但必要性还是值得考量的。不过好在 Transformer 在 NER 上的表现不太好，也许可以用一用。。。。

最终实验结果表明，该方法在 MSRA 数据集上比 CNN + BiGRU + CRF (92.34)效果好一些(92.97)，

![](/img/in-post/kg_paper/ner_can_att.jpg)

文章通过将 attention 可视化来证明方法的有效性。左图是local attention 的图，可以看出，模型学习到了短期的相关性，如 “美” 和 “国”这种。右图是全局的 attention，可以看出，模型也确实学习到了长期的依赖。

## Chinese NER Using Lattice LSTM

中文词内是包含大量的信息的，直接用基于词的方法会受到分词效果的影响，因此目前很多模型都是基于字的， 词级别信息则想办法通过好的向量嵌入和模型的上下文依赖来补充。改论文提出通过  Lattice LSTM 方法将词信息加入到基于字的 LSTM+CRF 模型中，减轻模型受到分词错误的影响，以此提升模型的效果。

![](/img/in-post/kg_paper/ner_lattice_arc.JPG)yy

模型输入是一个一个的字，词信息通过词典获得。输入的字符用 $c_{i}^{c}$ 表示，通过字符嵌入得到向量为 $ x_{i}^{c} = e^{c}(c_{i})$，LSTM 的输出为 $h_{i}^{C}$，词用 $ w_{be}^{w}$ 表示，嵌入后的向量为 $ x_{be}^{w}$，经过隐层后的输出时 $ c_{be}^{w}$。

词向量和字的隐层输出连接起来进入 LSTM 得到对应的输出 $ c_{be}^{w}$ ，LSTM 的内部计算公式如下所示

![](/img/in-post/kg_paper/ner_lattice_form1.JPG)

![](/img/in-post/kg_paper/ner_lattice_form2.JPG)

![](/img/in-post/kg_paper/ner_lattice_form3.JPG)

后面两个是控制词级信息流入的门，论文中对这两个门做了特殊处理使得他们的和加起来为 1.

![](/img/in-post/kg_paper/ner_lattice_form4.JPG)

模型在 MSRA 数据集上的表现显示，该模型的效果远超当时的其他模型，但该模型由于结构问题，github 的代码给出的 bath_size 为 1，训练太慢。

![](/img/in-post/kg_paper/ner_lattice_result.JPG)

看完有几个问题，1. 单个字的怎么办？比如 “是” 这种字；2. 词汇表太大的话怎么办？效率会不会很低。论文中没看到答案，但这个模型很有趣，有时间用的话好好研究。

## New Research on Transfer Learning Model of Named Entity Recognition

文章基于 BERT 模型，在后面加上了 BiLSTM + CRF 层，进行命名实体识别。在人民日报等语料库上进行训练和测试，最终表明，基于 BERT 强大的性能，该模型超过了以往的模型，同时相比于 BERT + MLP 做 NER ，也提升了一个点。算是意料之中的改进，就不详细介绍了。

## Tuning Multilingual Transformers for Named Entity Recognition on Slavic Languages

先用 BERT 先在 4 种 语言（Polish, Czech, Russian, Bulgarian）上进行无监督预训练，而后用 BERT + CRF 做 NER ，取得了很好的效果。论文的主要创新点在训练步骤上：文章认为从头训练 BERT 是极其昂贵的，因此文章用多语言模型去做初始化（文章认为多语言模型能够带来除目标语言外额外的通用模式），而后用 subword-nmt 构建字词字典。当单个斯拉夫语 token 由多个 subtoken 组成时， 采用 这些 subtoken 的平均作为 总 token 的向量表示。

模型结构如下图所示：

![](/img/in-post/kg_paper/ner_slav.JPG)

## FACTORIZED MULTIMODAL TRANSFORMER FOR MULTIMODAL SEQUENTIAL LEARNING

在模式内和模式间任意分布的时空动力学建模是多模态序贯学习研究的最大挑战。论文提出了一个新的变压器模型，称为因子化多模变压器（FMT）的多模顺序学习。FMT以因子分解的方式固有地在其多模态输入中对模式内和多式联运（涉及两个或多个模式）动力学建模。因子分解允许增加Self-attention 的数量，以便更好地模拟多模态现象。 论文研究了三种模态：语言、视觉、语言三种。

FMT 模型如下图所示

![](/img/in-post/kg_paper/ner_fmt.JPG)

输入首先经过一个嵌入层，然后是多个多模态 Transformer 层（MTL）。每个MTL由多因子多模态 slef-attention（FMS）组成。FMS在其多模输入中明确地考虑了多模和多模因素。S1和S2是两个总结网络。它们是FMT的必要组成部分，可以有效地增加 attention 的数量，而不会过度参数化FMT。

## Empower Sequence Labeling with Task-Aware Neural Language Model

文章要解决的问题是虽然神经网络很好用，但是比较吃人工标记的数据，因此该文章提出一个神经网络语言模型作为辅助的模型。除了正常的与训练单词嵌入包含的单词级知识外，还结合了字符神经网络语言模型来提取字符级知识。。更进一步采用迁移学习的方式，引导语言模型朝着目标方向学习。

文章提出的模型叫 LM-LSTM-CRF，结构如下图所示：

![](/img/in-post/kg_paper/ner_lm_lstm_crf_arc.JPG)

以往我们知道 char 级别的信息很重要，所以用 CNN 一类的去编码它，但实际上同一个char 在不同词的内部有不同的含义，因此有一个语言模型来提供char 级别更充分准确的信息是很必要的。为此文章 在最底层加了一个双向的 LSTM 来作为语言模型部分，其中正/逆向的 LSTM 预测下一个词。

char 得到的编码经过 highway 网络后和 word 向量输入到 BiLSTM 中，而后经过 CRF 得到 NER 结果。

![](/img/in-post/kg_paper/ner_lm_lstm_crf_result.JPG)

## Improving Clinical Named Entity Recognition with Global Neural Attention

临床命名实体识别（NER）是获取电子病历中的知识的基础技术。传统的临床神经网络方法存在着严重的特征工程问题。此外，这些方法将NER视为一个句子级的任务，忽略了语境的长期依赖性。文章提出了一种基于注意的神经网络结构来利用文档级的全局信息来缓解这一问题。全局信息是从具有神经注意的预训练双向语言模型（Bi-LM）表示的文档中获取的。利用未标记数据的预训练Bi-LM的参数可以传递到NER模型中，进一步提高性能。

Bi-LM 首先在无监督学习中，在无标记语料库上预训练一个单词嵌入模型和 BiLM，而后采用堆叠式 BiLSTM 对包含单词嵌入的输入句子进行编码，并将输入的句子所在文档中的所有句子表示与 attention 结合，最后使用 CRF 解码。模型结构如下图所示

![](/img/in-post/kg_paper/ner_cner_arc.JPG)

Bi-LM 采用 LSTM作为编码层，LM 输出的前向和后向向量连接在一起作为语言模型的编码结果。Bi-LM 在使用时对句子进行编码，每个句子被表示为一个向量，即上图右侧的 S1、S2等。

模型的整体流程为，词嵌入得到对应的向量表示，而后输入到第一个 BiLSTM 层，盖层的输出 $h_{1,i}$ 与 Bi-LM 的各部分做 attention 得到对应的全局表示 $g_{1,i}$，最终 $g_{1,i}$ 和 $h_{1,i}$ 联合输入到第二层 BiLSTM 中，最后一层是 CRF 来得到 NER 标签。模型的结果如下图所示

![](/img/in-post/kg_paper/ner_cner_result.JPG)

在充分利用文档级全局信息后，模型效果还是比 BiLSTM + CRF 要好一些的。

## Robust Lexical Features for Improved Neural Network Named-Entity Recognition

论文指出，在深度学习时代，之前流行的大量人工特征都被抛弃了，最常用的也就剩下 gazetteers 和 POS 特征等。但这是不公平的：词汇特征实际上是非常有用的。我们建议将单词和实体类型嵌入到一个低维向量空间中，我们利用Wikipedia从远程监控生成的带注释的数据进行训练。由此，我们离线计算一个代表每个单词的特征向量。当与普通的递归神经网络模型一起使用时，这种表示方法会产生实质性的改进。确实，在数据量不大或者专业领域中，人工特征真的是很有必要的。

WiFiNE 将单词和实体类型嵌入到一个联合向量空间中，WiFiNE是一个在维基百科中自动注释120个实体类型的资源。从这个向量空间，为每个单词计算一个120维向量，其中每个维编码单词与实体类型的相似性。我们称此向量为LS表示，用于词汇相似度。当包含在普通的LSTM-CRF净收益率模型中时，LS表示会带来显著的收益。

相比于 WiFiNE，传统的 gazetteers 用法文章认为有一些缺点：

* 二值化表示：一个实体可能有多种类型，one-hot 化的二值表示不能反映这种情况。    
* 复杂度高：之前需要一个一个去字典里比对，映射成向量后就不需要了。    
* 非实体词：传统方法没有充分利用非实体词，但向量法将词和实体类型都映射到同一空间，充分考虑当前标记受前面标记的影响的利用它们。

下图显示了这种方法的嵌入效果

![](/img/in-post/kg_paper/ner_wif1_embed.JPG)

可以看到，实体和对应的类型被嵌入到相近的范围内，当一个实体属于多个类型时，它会处于多个类型的中间，这就很合理。

网络的结构如下图左侧所示，右侧是 CNN 对 char 的嵌入表示。

![](/img/in-post/kg_paper/ner_wif1_arc.JPG)

除了这些还有文字特征，包含：全部大写、全部小写、开头大写、开头不是大写、数字或非字符。最后模型的效果如下所示，比单纯的 Bi-LSTM 要好一些。

![](/img/in-post/kg_paper/ner_wif1_result.JPG)

## Efficient Contextualized Representation Language Model Pruning for Sequence Labeling

文章针对预训练模型，对于特定的任务，预训练模型的信息只有一部分是有用的。这样的大规模LMs，即使是在推理阶段，也可能会导致繁重的计算工作量，使得它们对于大规模应用来说过于耗时。文章建议压缩庞大的LMs，同时保留有关特定任务的有用信息。具体想法是，由于模型的不同层次保留了不同的信息，文章提出了一种基于稀疏正则化的模型修剪层次选择方法。通过引入稠密的连通性，我们可以在不影响其他层的情况下分离任何层，并将浅层和宽层LMs拉伸成深层和窄层。

一个示意如下图所示

![](/img/in-post/kg_paper/ner_lms_sam.JPG)

左侧是原始的双层 RNN 结构，我们要用的话需要全部都用，现在将其在每一层内进行展开，利用稠密的连通性，通过分层选择来压缩模型，并将宽、浅RNN替换为深、窄RNN。虽然刚展开时，连接像中间图那样稠密，但是经过剪纸后，连接将变得稀疏，这样将会得到一个相当轻量级的模型。

# 语义编码层

## Leveraging linguistic structures for named entity recognition with bidirectional recursive neural networks
2017
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

## Neural Architectures for Named Entity Recognition

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

## Contextualized Non-Local Neural Networks for Sequence Learning

文章认为， Transformer 的成功可以归因于它的非局部结构偏差(no-local structure bias)，其中任何一对词之间的依赖关系都可以建模，这就允许模型学习句子的句法和语义结构。但缺点是局部的捕捉，相邻单词上存在的局部依赖性，限制了它的上下文学习能力。文章提出非局部语义神经网络(CN3),它基于 图神经网络，能够根据特定的任务动态的构造句子的结构并使用丰富的局部依赖信息。

属性的表示采用节点和边的特征编码，图本身的结构是任务相关的，可以动态的学习。这样就可以更好的从词的上下文表示句子复杂结构的序列化表示.文章这么做的目的就是希望通过图神经网络来学习到局部特征，以此改进 Transformer 的短期依赖不行的问题。事实上，这种假设潜在的认为，当前词对周围的词时存在依赖的，这是事实，有一些文章还认为 NER 任务或者其他任务存在整句的依赖和整个段落甚至文档的依赖，因此局部和非局部偏差是两种常见的偏差，隐含地存在于不同的序列学习模型中。

由于我没了解过GNN，因此文章的具体细节就跳过了。。。以后学习后回来补。

## GRN: Gated Relation Network to Enhance Convolutional Neural Network for Named Entity Recognition

基于CNN 的 NER 网络 GRN(gated ralation network)，相比于普通的 CNN， GRN 更能捕获上下文，在 GRN 中，首先使用 CNN 来探索每个单词的上下文特征，然后对词与词间的关系进行建模，并将它们作为门。最终将局部上下文特征融合到全局上下文特征中，用于预测标签。

网络结构分为四层：

* 表示层：word + char 向量嵌入，其中 word 采用 GloVe 嵌入，字符级特征采用 CNN 进行提取，然后用 max-pooling 弄到特定的维度。    
* 语义层：利用不同大小的卷积核 k = {1, 3, 5}，来提取 词级别特征，而后采用 最大池化操作 和 tanh 激活。    
* 关系层：分别建立一个词与其他词的关系，首先假设当前词为 $x_{i}$, 其他词为 $x_{j}$，则

$$ \alpha_{i j} = \sigma(W_{x}[x_{i};x_{j}] + b_{x} ) $$

$$ r_{ij} = W_{rx}[x_{i}; x_{j}] + b_{rx} $$

$$ r_[i] = \frac{1}{T}\sum_{j=1}^{T}\sigma(r_{ij}) \odot x_{j} $$

文字来看，就是分别计算当前词和其他词的 sigmoid 值，就像一个多维的门，而后这个门区控制 $x_{j}$ 的信息流入量。最终对所有时间步加和平均得到当前词的新表示。细看发现，怎么这么眼熟。。。。在 attention 里权重是对全局的归一化概率，而这里权重是每个词的 sigmoid 值除以时间步长度，同时 $$r_{ij}$$ 还是向量。因此论文中的方法更看重两个词之间的局部关系。论文认为这是一种 channel-wise 的 attention，与之对应的是 gating 机制的 attention，其中 

$$ r_{i} = \frac{1}{T}\sum_{j=1}^{T}\sigma(W_{x}[x_{i}; x_{j}] + b_{x}) * x_{j} $$

区别是，这的 $\sigma$ 得到的是一个标量。

* CRF 层：这个没什么好说的

总结下来，文章通过在 CNN 后面加上 关系层，对词与词之间的关系建模，可以更好地捕捉长程语义依赖信息。模型效果比 CNN +BiLSTM + CRF 和  CNN +BiLSTM + Att + CRF 要好。

** 这里不懂的地方是， 为什么它这种做法比 attention 要好，毕竟 attention 也对每个其他的词计算了。 论文里没有给解释。**

## Star-Transformer

论文认为 Transformer 计算复杂度是序列长度的平方，太高。同时 Transformer 也比较吃数据量，当数据比较少时，Transformer 表现得往往的没那么好。 那么 Transformer 为什么那么迟数据呢？文章认为，这是由于 Transformer 的设计缺乏先验知识导致的。当开始训练 Transformer 时就需要从零开始，从而增加了学习成本。因此在改动 Transformer 时加入一些任务需要的先验知识可以减轻这种情况。

基于以上原因，文章提出，通过将完全连接的拓扑结构移动到星形结构中来节省体系结构。改进的网络结构如下图右侧所示，对比于左侧的传统 attnetion ，内部的连接数量明显减少了。

![](/img/in-post/kg_paper/ner_star_trans.JPG)

具体来说，星形 Transformer 有两种连接方式。中间的节点叫根节点，周围的节点叫卫星节点，卫星节点实际就是一个一个 timestep 的输入，卫星节点到根节点的连接叫直接连接(Radical connections)，直接连接保留了全局的(non-local) 信息，消除冗余连接，全局信息可以通过根节点自由流通。卫星节点之间的相互连接叫环连接(Ring connections)，环连接提供了局部成分的先验性。这么设计的好处是1. 可以降低模型的复杂度从 $O(n^{2}d)$ 到 $O(6nd)$。2. 环连接可以减轻之前无偏学习负担，提高泛化能力。

Star-Transformer 的训练分为两个步骤：1：卫星节点更新；2：中继节点的更新；整个更新的流程图如下所示

![](/img/in-post/kg_paper/ner_star_update.JPG)

首先将输入文本序列嵌入得到 $$ E = [e_{1}, e_{2}, \dots,e_{n}]$$，用 E 去初始化卫星节点 $$ H = [h_{1}^{0}, \dots, h_{n}^{0}] $$。而后利用 E 得均值去初始化根节点 S。之后执行T轮更新，即卫星节点更新和根节点，更新公式图中所示。

模型在 CoNLL2003 与 CoNLL2012 数据集上相比于传统的 Transformer 有较大的提升。

![](/img/in-post/kg_paper/ner_star_result.JPG)

## TENER: Adapting Transformer Encoder for Named Entity Recognition

全连接 self-attention (Transformer) 的优势是并行性和对长程上下文语义建模。然而， Transformer 在 NER 中的表现却不那么如意。文章提出了 TENER，一种采用自适应 Transformer 的 NER 架构，对字符级特征和词级特征进行建模。通过结合方向感知、距离感知和 Un-scaled 的 attention，使得改进的 Transformer 在 NER 中获得了良好的表现。

具体来说，文章指出 Transformer 的两个不足：

* 原始 Transformer 使用的正弦位置嵌入只知道距离但不知道方向性，就这样，在训练过程中，距离性都保不住。但在 NER 中，距离可以帮助模型更好的关注上下文及边界。因此改进版的 Transformer 采用相对位置编码而不是绝对位置编码，可以获得更好的效果。    
* 原始 Transformer 的注意力分布被 scaled 和平滑了，但是对于 NER 来说，稀疏的注意力是必要的，毕竟不是所有的单词都要被关注，给定一个当前次，几个上下文单词就可以判断它的标签。平滑的注意力可能包含一些噪声信息，因此改进的 Transformer 抛弃了 scale 和 点乘注意力，改而采用 un-scaled 和 sharp 的 注意力。

![](/img/in-post/kg_paper/ner_tener_arc.JPG)

上图是网络的结构图，整体可以分为3个部分：嵌入表示层：字符级 embedding 用 Transformer 来提取特征，而后将字符级别嵌入和 词级别嵌入连接作为 Transformer 的输入。 经过 Transformer 后的输出到 CRF 层得到 NER 标签。

接下来重点说一下改进的这个 Transformer。首先证明原始 Transformer 不具有方向敏感性。

我们知道，原始 Transformer 的位置向量嵌入公式为：

$$ PE_{t, 2i} = sin(\frac{t}{10000^{2i/d}}) $$

$$ PE_{t, 2i+1} = cos(\frac{t}{10000^{2i/d}}) $$

因此可有

$$
\begin{aligned}
PE^{T}_{t}PE_{t+k} &= \sum_{j=0}^{\frac{d}{2}-1}[sin(c_{j}t)sin(c_{j}(t+k)) + cos(c_{j}t)cos(c_{j}(t+k))] \\
&= \sum_{j=0}^{\frac{d}{2}-1}cos(c_{j}(t-(t+k))) \\
&=\sum_{j=0}^{\frac{d}{2}-1}cos(c_{j}k)
\end{aligned}
$$

因为 cos 的函数性质，我们有

$$ PE_{t}^{T}PE_{t-k} = PE_{t}^{T}PE_{t+k} $$

因此 该种位置嵌入方式不具有方向敏感性，是对称的。如下图所示，横坐标是 k值，纵坐标是乘积结果， d 表示位置向量的维度。

![](/img/in-post/kg_paper/ner_tener_sys.JPG)

更进一步的，文章认为，原始的位置嵌入方式进行 self.attention 时，$$PE_{t}^{T}W^{T}_{q}W_{k}PE_{t+k}$$，连对称性和距离敏感性也会丧失，如下图所示

![](/img/in-post/kg_paper/ner_tener_unsys.JPG)

这里我比较困惑的是，上图中是加上随机 W 后的结果，但是当网络进行多轮学习后，能不能得到比较好的性质呢？其次，论文中提到的这种缺点，当采用随机嵌入的方式也会有么？毕竟随机嵌入的方式顶多算是没有先验，但不至于有这么严重的不对称性，随机嵌入的效果在 NER 上会比论文中的嵌入方式表现好么？

继续说论文，为了获得距离敏感性和方向敏感性，论文提出 Attention 的计算方式变成

$$ Q,K,V = HW_{q}, H_{d_{k}}, HW_{V} $$

$$ R_{t-j} = [\dots, sin(\frac{t-j}{10000^{2i}{d_{k}}})cos(\frac{t-j}{10000^{2i}{d_{k}}})、\dots]^{T} $$

$$ A_{tj}^{rel} = Q_{t}^{T}K_{j} + Q_{t}^{T}R_{t-j} u^{T}K_{j} + v^{T}R_{t-j} $$

$$ Att(Q,K,V) = softmax(A^{rel})V $$

首先变化一是 Key 没有乘权重，$R_{t-j}$ 是相对距离编码，该编码由于 sin 函数的性质，具有方向敏感性和距离敏感性。QK 表示注意力分数， QR 是相对距离偏置， uK 和 vR 表示额外的偏置。

我们还注意到，上式的 aatntion 分数计算时，没有进行 scale，这样得到的分布更 sharp，也许对 NER 任务更好。

论文在中文和英文数据集上的表现如下所示

![](/img/in-post/kg_paper/ner_tener_f11.JPG)

![](/img/in-post/kg_paper/ner_tener_f12.JPG)

# 解码层

## Neural Reranking for Named Entity Recognition

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

## A Multi-task Approach for Named Entity Recognition in Social Media Data

文章提出了一种新的多任务方法，该方法将命名实体（NE）分割（给定 token 是不是实体的二分类任务）作为次要任务，细粒度NE分类作为主要任务结合起来。

该模型使用卷积神经网络（CNN）在字符层捕获单词形状和一些正交特征。对于单词级的上下文和句法信息，如单词和词性（POS）嵌入，该模型实现了一个双向长短期记忆（BiLSTM）体系结构。最后，为了覆盖众所周知的实体，该模型使用了地名索引表示。一旦网络被训练，我们就用它作为一个特征抽取器来给条件随机场（CRF）分类器提供信息。CRF分类器联合预测最可能的标签序列，比网络本身给出更好的结果。

网络结构如下图所示：

![](/img/in-post/kg_paper/ner_mta.JPG)

模型分为五层：

* 输入层    
* 特征编码：特征包含字符级特征、词级别特征和字典特征。词级别包含 word embedding 和  POS 嵌入，POS 标签通过 CMU POS tagger,用均匀分布初始化。字典特征是在深度学习时代保留下来的人工特征之一，大体来说字典是通过各种方式像 Wiki 中的属性，重定向锚点这种收集到的实体和对应类型构成的字典。当作为模型的输入时， 我们查找输入的每个 token 在不在里面，如果在呢，就在对应的 one-hot 表示的对应位置记 1。当然这是一种方法，也可以用概率的方式，毕竟有时候一个实体可能有多种类型，像“希尔顿”，它可能是一个酒店名字，也可能是人名。因此，给出一个实体的每个类型的概率分布作为嵌入向量是更自然的一种选择。    
* 特征连接：上面的字符级特征用两个堆叠的 CNN 来提取，词级别特征用 BiLSTM 提取，再和字典特征连接得到综合的向量表示
* 全连接层
* 多任务输出层：三个输出，一个用 sigmoid 来输出二值分类，另一个用 softmax 输出NE 分类。 标记概率分布，右侧那个 CRF 则用来做 NER。一般来说，附加任务可以看做正则项，来提升模型的泛化能力，也可以迫使模型学习到想要的特征。

## Hybrid semi-Markov CRF for Neural Sequence Labeling

文章想解决的问题是以往的模型都忽略了词级别的转移关系，为此文章提出 LM-BLSTM-JNT 模型，模型结构如下图所示

![](/img/in-post/kg_paper/ner_jnt_arc.JPG)

首先输入是 字符级别和词级别的 embedding，而后进入 BiLSTM 得到语义表示 w。到输出这里就比较特殊了，首先定义 $$ s_{i} = (b_{i}, e_{i}, l_{i}) $$, 其中 $b_{i}$ 表示第 i 个词的开始位置， $e_{i}$ 表示第 i 个词的结束， $l_{i}$ 表示第 i 个词的 NER 标签。因此

$$ p(s|w) = \frac{score(s, w)}{\sum_{s^{'}score(s^{'},w)}} $$

$$ score(s, w) = \prod_{i=1}^{|s|}\psi(l_{i-1}, l_{i}, w, b_{i}, e_{i}) $$

$$ \psi(l_{i-1}, l_{i}, w, b_{i}, e_{i}) = exp{m_{i} + b_{i-1}, l_{i}} $$

$$ m_{i} = \phi(l_{i}, w, b_{i}, e_{i}) $$

其中 $m_{i}$ 是分词级别的转移分数，最终得到 HSCRF 的概率。模型在 CoNLL2003 数据集上的表现如下图所示

![](/img/in-post/kg_paper/ner_jnt_result.JPG)

比 CNN + BiLSTM + CRF 最好效果好一些，并且平均效果要好很多，更稳定。

## Improve Neural Entity Recognition via Multi-Task Data Selection and Constrained Decoding

论文提出了一个实体识别系统，用两种新的技术改进了传统 BiLSTM-CRF 模型。第一种技术是多任务数据选择，它确保源数据集和目标数据集之间的数据分布和标记准则的一致性。另一种是基于知识库的约束解码。模型的解码器在文档级别运行，并利用全局和外部信息源进一步提高性能。

文章提出的模型结构如下图所示

![](/img/in-post/kg_paper/ner_mtds_arc.JPG)

结构比较简单，左侧是源数据，右侧是目标数据的输出，两个任务共享 BiLSTM 层，而后通过 全连接层学习各自所需的特征。文章的重点是提出了一个多任务数据选取流程，该流程如下图所示

![](/img/in-post/kg_paper/ner_mtds_mts.JPG)

在每次迭代中，源域中的数据选择与模型参数更新交织在一起。训练数据是根据一致性(KL 散度)得分来选择的，一致性得分衡量目标和源数据分布之间的相似性。根据步骤4，从训练数据集中消除与目标不一致的数据。直到更新满足预设要求（没有额外要过滤掉的数据或者达到预设阈值）时停止更新。训练是交替迭代的。

至于解码的全局约束是指，首先通过各种渠道收集实体及其各种别名，如“微软”、“MS” 都是一个，那么当解码时遇到这种情况就要保证它们的 NER 标记一致。

## Bidirectional LSTM-CRF Models for Sequence Tagging

经典模型，率先提出使用 Bi-LSTM + CRF 进行序列标注，并证明 BiLSTM 是稳健的，新模型对单词嵌入的依赖更小。模型结构如下图所示

![](/img/in-post/kg_paper/ner_blstm_crf.JPG)

模型结构这里不多说，太经典了。这里说一下模型用的一些人工特征。特征分为拼写特征和语义特征。

* 拼写特征：开头是否大写、所有都大写、都小写、非大写开头、数字字母混合、有标点符号、前缀和后缀、撇号结尾、只有字母、不止字母、词的模式特征等。    
* 语义特征：一元和二元特征，POS 的话用 了三元特征。

上面得到的特征直接输入到解码层来提高效率的同时又不会减少精度。

# 人工特征大集结

从两篇论文中找到了很全的人工特征集合，如下所示
## 实体词典特征-基于实体词典与机器学习的基因命名实体识别

* 单词特征：单词是文本自动分析和实体标注的基本单位，单词特征能够反映命名实体的语言信息，是最核心、最重要的特征。    
* 构词特征：本文根据当前次是否由大小写字母、数字、连字符、希腊字母、罗马数字、引号、括号等字符组成构建了构词特征，共包含 18 个子特征。

![构词特征](/img/in-post/kg_paper/ner_feature_struc.JPG)

* 关键词特征：关键词是指在命名实体中出现频率较高的子/词，通过判断当前词是否为关键词，可以识别出可能出现在当前词附近的命名实体。    
* 词缀特征：词缀是一种附着在词根或词干的语素，为了规范词素，不能单独成字。粘附在词根前面的称为前缀，后面的是后缀。    
* 词形特征：同一类型实体往往具有相似的词形，英文中，通常将大写字母都替换成A，小写字母都替换为a，数字替换为0，其他字符替换为x。    
* 边界词特征：边界词是指命名实体的第一个和最后一个单词，利用边界词信息可以提高边界词识别能力，减少符合性实体的识别错误率。    
* 一元词特征：英文中，存在大量只有一个单词构成的实体，以一元词是否出现作为特征，可以为当前词是否是命名实体提供有效信息。    
* 嵌套词特征：词和语素按照一定规则组合起来构成的合成词叫复合词。    
* 停用词特征：判断是否为停用词，减少识别过程中无用信息的干扰。    
* 通用词特征：通用词是指使用频率比较高、单词本身也具有实际意义，但在各个领域都通用的单词。    
* 上下文特征：上下文信息是指实体的前一个词和后一个词的信息，利用上下文信息可以提高实体的边界识别效果。    
* 词行特征    
* 词典特征：传统基于词典的命名实体识别是在识别过程中完全依赖词典，一般使用不同的词典匹配方式在所构建的词典中查找字符串。实际使用过程中可以采用one-hot 或者概率分布、向量嵌入等方式作为特征输入。

## A survey of named entity recognition and classification

特征分为 Word-level、List lookup 、文档级别特征三大类。如下图所示

![](/img/in-post/kg_paper/ner_feature_word.JPG)

![](/img/in-post/kg_paper/ner_feature_list.JPG)

![](/img/in-post/kg_paper/ner_feature_doc.JPG)

# 总结

收获颇丰，对机器学习的认识又深入了一步，了解每个基本模块的使用优劣，并利用它们去改进现有模型的缺点是核心思路。不同的基本模块代表了不同的先验假设，现在回头看像 highway 、LSTM 等神来之笔就没那么突兀了。

接下来是根据最近看论文的过拟合总结

## 输入部分

* Word embedding: 英文直接就是单词， "hello" 这种，中文的话在 NER 中还是字符的好一点，这是因为基于词的 中文 NER 会受分词效果的影响，切词的错误会传递到 NER 里来。但也有例外，比如可以作为辅助先验输入到网络中， Lattice LSTM 就利用了中文 词级别信息，通过门机制使其与char level 的进行结合。嵌入方式就是 Word2vec、Glove等。    
* Char-level：英文的话，"h ~e~ l~ l~ o" 这样，可以大大减少 OOV 的问题。对于中文来说，字符级是主要的输入方式，因此花样比较多。最简单的是用字的 embedding 输入到网络中。但中文的字往往有多重含义，像“球拍”、“拍卖”里的“拍”字有不同的含义，因此想要基于字符输入的话，需要对上下文语义进行建模来充分表达字的含义。最典型的就是用 CNN 来做，LSTM 也可以。    
* POS tag： 词性标签，用额外的工具对输入文本进行词性标注，得到每个输入对应的词性标签，而后使用 One-hot 或者 随机 embedding 的方式作为特征输入到网络中。比较流行的特征。    
* Gazetteer: 额外词典，存放着实体和对应的类型。对输入进行处理得到对应的实体类型标签，然后进行编码作为特征输入。很有用的特征，尤其是领域性较强的数据上。    
* 语言模型：外挂语言模型来获得更好的表示和先验知识    
* Word-shape：词形特征，英文的相对丰富一些，如开头是否是大写，是否全部都是大写。。。详见上面最后一节    
* Segment：用工具对输入进行分词得到每个字的 BMES 标签，embedding 后作为特征向量输入。

## 语义编码层

* CNN：CNN 部分主要用来捕捉局部依赖特征，整体上就像一个参数版的 N-Gram。虽然可以通过多层 CNN 的叠加来获得更大的感受野得到全局的特征，但对于 NER 来说，这种长程依赖性还是不够，因此可以作为 先锋部队，后面加 LSTM 或者 attention 来强化长程依赖能力。    
* RNN：杰出代表 LSTM /GRU ，可以较好的获得长期依赖能力，同时由于门机制的存在，局部依赖性也可以有，在加上 BiLSTM 就有了上下文的特征。因此 BiLSTM + CRF 在很多时候都是一个不错的 baseline 。    
* Transformer： Transformer 可以说是这两年的研究热点，凭借着优异的表现，已经逐步在各任务中替代 LSTM/ CNN 了。 但 在 NER 任务中却表现不好，甚至不如 BiLSTM + CRF ，究其原因是 NER 要求模型具有后：    
    * 1、局部依赖的捕捉。    
    * 2、位置信息。    
    * 3、更 sharp 的attention 分布，毕竟决定 NER 类型不需要太多的全局信息。    
    * 4、NER 的数据集貌似都不大，Transformer 又比较吃数据，缺乏先验    
    幸运的是，已经有论文研究表明，针对以上缺点的改进后可以获得比 CNN + BiLSTM + CRF 更好的效果

* CNN/LSTM 与 attention 的结合：从直觉上将，att + cnn 有效的原因是 att 精准的捕获局部依赖关系，减少无用词的噪声，而后通过 CNN 来对精准的局部依赖建模，提升 CNN 的局部依赖效果。BiGRU + att 好的原因是 全局 attention 可以更好的捕捉长程依赖信息，而 GRU/LSTM 这种虽然也能对长期序列建模，但对靠前时间步会不友好，缺乏平等性。attention 可以在一定程度上弥补这点。与此同时， attention 完全没有管位置信息的缺点被 LSTM/GRU 很好的弥补了。更进一步，其他 CNN/LSTM 与 attention 结合的方式怎么样呢？从直觉上来看， CNN + Att 可以来代替 max / avg pooling，在编码降噪上效果也许会好。att + LSTM/GRU 可以理解是对进入 LSTM 的向量获得一个全局的表示，如对词与词关系的建模。   上面虽然开了脑洞，但必要性还是值得考量的。不过好在 Transformer 在 NER 上的表现不太好，也许可以用一用。。。。

## 解码

* CRF：一般情况下够了。。。。    
* 重排序：CRF 解码会得到一个概率最高的标签序列，而重排序模型采用 N-best 输出进行再排序，最终结合重排序和 CRF 的概率输出得到最终序列标记    
* 多任务学习：通过多任务学习来提升泛化能力并引导模型学习预期的特征。
* Semi-markov CRF：

以上就是个人通过观看论文得到的的过拟合总结。等有更深的理解时再进行更新。
