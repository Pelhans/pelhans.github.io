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

对于正向 LSTM, 它从前向后读取每个时间步的输入, 我们取得每个时间步的输出 $H^{正} = ( h_{1}^{正}, h_{2}^{正}, \dots, h_{n}^{正} )$ 作为正向 LSTM 的输出. 方向 LSTM 以相反的时间步获取文本输入而后得到每个时间步的输出 $H^{反} = ( h_{1}^{反}, h_{2}^{反}, \dots, h_{n}^{反}  )$. 最终我们把对应时间步的正向和反向的 LSTM 输出拼接得到 BLSTM 层的输出. $H = {(H_{1}^{正}, H_{1}^{反}), (H_{2}^{正}, H_{2}^{反}), \dots, (H_{n}^{正}, H_{n}^{反}) } $.

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

![](/img/in-post/kg_paper/ner_lattice_arc.JPG)

模型输入是一个一个的字，词信息通过词典获得。输入的字符用 $c_{j}$ 表示，通过字符嵌入得到向量为 $ x_{j}^{c} = e^{c}(c_{j})$，经过隐层的输出为 $h_{j}^{c}$。正常情况下，用基于 char 的模型，LSTM 的计算公式如下所示：

![](/img/in-post/kg_paper/ner_lattice_lstm.png)

比较经典的做法，有三个门：输入、输出、遗忘。黑体的 $$ c_{j}^{c}$$ 表示 cell 状态。隐层输出时结合了上一时间步的 cell 状态和当前的输入。但对于 Lattice LSTM 来说，情况发生了变化，因为我们多了来自 gazetteers 的信心，词用 $ w_{be}^{w}$ 表示，嵌入后的向量为 $ x_{be}^{w}$。为此我们先让词向量经过一个没有输出门的 LSTM(因为模型只允许在字符节点进行预测)，公式如下所示：

![](/img/in-post/kg_paper/ner_lattice_form1.JPG)

需要注意的是，此时的输入是词向量 $$x_{b,e}^{w}$$ 和开始字符的隐层输出 $$ h_{b}^{c}$$。经过计算我们得到词的 cell 输出 $$ c_{b,e}^{w}$$。

因为一个结束字符可能对应多个匹配到的词，因此采用 sigmoid 函数作为一个门来控制每个词的贡献。

![](/img/in-post/kg_paper/ner_lattice_form2.JPG)

最终利用所有词的贡献和当前输入的贡献进行加权求和得到得到当前的 cell 状态。(这个公式里没有看到上一时刻的 cell 状态？？)：

![](/img/in-post/kg_paper/ner_lattice_form3.JPG)

后面两个是控制词级信息流入的门，论文中对这两个门做了归一化处理.

![](/img/in-post/kg_paper/ner_lattice_form4.JPG)

模型在 MSRA 数据集上的表现显示，该模型的效果远超当时的其他模型，但该模型由于结构问题，github 的代码给出的 bath_size 为 1，训练太慢。

![](/img/in-post/kg_paper/ner_lattice_result.JPG)

上图结果一个比较有意思的点是 bichar + softword 竟然这么强，有机会用一下可以。

char bigram embedding：$$ e(c_{i}, c_{i+1})$$


## An Encoding Strategy BasedWord-Character LSTM for Chinese NER

在论文 Lattice LSTM 中，单词信息被集成到单词的开始字符和结束字符之间的一个快捷路径中。然而，捷径的存在可能导致模型退化为部分基于词的模型，从而产生分词错误。此外，由于格点模型的 DAG结构，它不能进行批量训练，速度很慢。

下图是 Lattice LSTM 退化成基于词模型的极端情况，此时词的信息会主导模型，导致边界错误的情况。

![](/img/in-post/kg_paper/wc-lstm_bad.PNG)

另外，由于字长可变，整个路径的长度不固定。此外，每个字符都有一个可变大小的候选词集，这意味着输入和输出路径的数量也不是固定的。在这种情况下，Lattice LSTM模型被剥夺了批量训练的能力，因而效率很低。

为了防止模型退化为部分基于单词的模型，论文将单词信息分配给单个字符，并确保字符之间没有快捷路径。具体地说，在前向WCLSTM和后向WCLSTM中，单词信息分别分配给其结束字符和开始字符。同时引入四种策略从不同的单词中提取固定大小的有用信息，保证了模型能够在不丢失单词信息的情况下进行批量训练。 

这其实是一个很好的思路，后面很多论文对 Lattice LSTM 的改进也都是这样，即尽可能的将外挂词信息加到字符信息上，简化输入，实现并行。

模型结构如下图所示

![](/img/in-post/kg_paper/wc-lstm_arc.PNG)

假设输入序列为 $${c_{1}, c_{2}, \dots, c_{n} } $$，通过与 gazetteers 匹配可以得到一系列的词，用 $$ w_{s_{i}}$$ 表示以第 i 个字符结尾对应的实体集合，则我们得到模型的正向输入：

$$ \overrightarrow{rs} = {(c_{1}, \overrightarrow{ws_{1}}), (c_{2},\overrightarrow{ws_{2}}), \dots, (c_{n},\overrightarrow{ws_{n}})}$$

将字符进行 embedding 得到 $$x_{i}^{c}$$，词 embed 得到 $$ x_{il}^{\overrightarrow{w}} $$，其中 l 表示 字符 i 对应的第 l 个匹配到的词。

那么问题来了，每个字符有好多个匹配到的词，用哪一个呢？为此论文提出四个策略：

* 用最长的    
* 用最短的    
* 取集合内词的平均作为表示    
* 用 attention 加权表示，query 是随机初始化的矩阵

最终得到每个字符对应的词级表示 $$ x_{i}^{\overrightarrow{ws}} $$，反向的也同理。论文将字符级表示和词级表示通过 concat 的方式进行连接，而后通过 双向 WC-LSTM 得到每个时间步的输出。最后通过 CRF 进行序列标注。

WC-LSTM 的计算公式如下所示：

![](/img/in-post/kg_paper/wc-lstm_for.PNG)

论文在 OntoNotes 、MSRA、Weibo 数据集上的表现如下图所示：

![](/img/in-post/kg_paper/wc-lstm_res1.PNG)

![](/img/in-post/kg_paper/wc-lstm_res2.PNG)

![](/img/in-post/kg_paper/wc-lstm_res3.PNG)

可以看到，最长和最短的效果比较稳定，都还可以。具体用最长还是最短的策略还是要根据自身的需要长的实体还是短的去决定，比如嵌套NER 就喜欢短的。平均的和 attention 的在 Weibo 数据集上表现不好，论文认为这是数据集小，模型参数多(attention)，导致泛化能力差。

下图是各方案的 case study，可以看到各策略的效果对比：

![](/img/in-post/kg_paper/wc-lstm_case.PNG)

为了检验模型的速度，论文将其与 Lattice LSTM 、char-BbiLSTM 做了对比，结果如下图所示：

![](/img/in-post/kg_paper/wc-lstm_speed.PNG)

下图展示了模型的收敛速度，显示收敛速度和 Lattice LSTM 一致：

![](/img/in-post/kg_paper/wc-lstm_shoulian.PNG)

## Simplify the Usage of Lexicon in Chinese NER

同样是针对 Lattice LSTM 的，也是嫌弃它慢，但 Lattice LSTM 能充分利用已被证明有用的实体信息这个优点又不能抛弃，为此论文提出使用 BMES 标签的 ExSoftword 技术，来保存 Lattice LSTM 的优点，并提高速度。

传统的 softword 方法是先通过分词等工具，得到每个字符对应的 BMES 标记，而后通过 embedding 进行输入。对应到 gazetteers 的情况，对于每个字符，往往会存在多个标签，因此 ExSoftword 允许多个分割标签的存在。

举例来说，假设输入为 $$ s = {c_{1}, c_{2},\dots, c_{5} }$$，匹配 gazetteers 得到实体 $$ \{c_{1}, c_{2}, c_{3}, c_{4} \}$$ 和 $$\{c_{3}, c_{4} \}$$ ，则每个字符对应的分词标记为：$$ seg(s) = \{ \{B\}, \{M\}, \{B,M\}, \{E\}, \{O\} \} $$。每个字符都用一个 5 维的二值表示，有哪个标记记为 1，否则为0.

这么做虽然引入单词的边界信息了，但它有两个致命缺点：

* 没有引入词向量信息，只用了边界信息    
* 尽管它试图通过允许一个字符有多个分割标签来保持所有的词典匹配结果，但仍然丢失了大量的信息。在很多情况下，我们无法从分割标签序列中恢复匹配结果

句子 s 的每个字符c对应于由四个分段标签“BMES”标记的四个单词集。词集B（c）由在句子s上以c开头的所有词库匹配词组成。同样，M（c）由c出现在句子s中间的所有词库匹配词组成，E（c）由以c结尾的所有词库匹配词组成，S（c）是由c组成的单个字符词。如果一个词集是空的，我们将在其中添加一个特殊单词“None”来表示这种情况。

对于每个词集，都压缩成一个固定维度的向量，最后将四个词集的表示(BMES)连接起来表示成一个整体，并将其添加到字符表示中。词集的压缩用加权求和解决，权值是词频。基本思想是，字符序列在数据中出现的次数越多，它就越可能是一个单词。注意，单词的频率是静态值，可以脱机获取。这可以大大加快计算每个单词的权重（例如，使用查找表）。这个其实和上面的四个策略想法是一致的。

为了防止稀有词出现 0 的问题，论文对每个词的权重做了平滑处理(加一个常数)。至此，模型整体流程比较明确：首先，我们用词库扫描每个输入句子，得到句子中每个字符的四个“BMES”词组。其次，我们在统计数据集上查找每个单词的频率。第三，我们得到每个字符的四个词集的矢量表示，并根将其添加到字符表示中。最后，在增广字符表示的基础上，我们使用任何合适的神经序列标记模型来进行序列标记，如基于LSTM的序列建模层+CRF标记推理层。

最终模型的效果如下图所示：

![](/img/in-post/kg_paper/sner_res1.PNG)

![](/img/in-post/kg_paper/sner_res2.PNG)

![](/img/in-post/kg_paper/sner_res3.PNG)

![](/img/in-post/kg_paper/sner_res4.PNG)

为了验证模型的速度，论文做了对比试验，发现要比 Lattice 快 7 倍。

![](/img/in-post/kg_paper/sner_speed.PNG)

为了验证论文提出方法的通用性，论文在不同模型结构上使用该方法：CNN/LSTM/Transformer ，实验结果如下图所示，在各种架构上都可以提升效果。

![](/img/in-post/kg_paper/sner_qianyi.PNG)

论文还对比了不同压缩词集的方法，包含：直接取平均、频率加权平均、加平滑的加权平均。实验结果如下图所示，看起来没平滑的加权平均效果最好。

![](/img/in-post/kg_paper/sner_jiaquan.PNG)

## Leverage Lexical Knowledge for Chinese Named Entity Recognition via Collaborative Graph Network

依旧是针对 Lattice LSTM 的。。。论文认为 Lattice LSTM 有两个缺点：

* Lattice LSTM 只能利用部分匹配结果，如 “机”这个字在 LATTICE LSTM 里只能匹配到 “机场”，但其实 “北京机场”也是有帮助的。    
* 没有用到语义相关词特征(语义相关词是指有语义关联的其他词)

为此论文提出了用图神经网络来利用这些特征的想法。模型的网络结构为四层：编码层、图神经网络层、融合层、解码层。网络结构如下图所示：

![](/img/in-post/kg_paper/threegat_arc.PNG)

编码层是用 预训练的 词向量 对 char 和 word 进行编码。图神经网络层分为三部分：

* 包含图(C-graph)，为了是用自匹配词典的词而设计，它提供了 char 和 自匹配词典词的连接。它先将字符进行连接，而后将匹配词与每个相关字符添加连接。    
* 转移图(T-graph)：表示 char 和最近语义相关词间的连接，和 word cutting graph 一样    
* 移除 LSTM 的格点图(L-graph)

将三种图用 GAT 处理，而后通过映射矩阵映射后加和进行融合。得到融合的字符级表示，最终用 CRF 进行解码。


## A Neural Multi-digraph Model for Chinese NER with Gazetteers 

还是针对 Lattice LSTM 的，主要为了解决在使用 gazetteers 时，由于匹配策略导致出现多个匹配结果的情况，如 “张三在吃饭”，可能匹配出“张三”和“张三在”两个人名。论文延续前面的思路，将 DAG 转化为字符的链式信息。论文提出 有向图 + GGNN + BiLSTM + CRF 的模型来尝试解决该问题。

虽然这些背景知识可能会有所帮助，但在实践中，gazetteers 也可能包含不相关甚至错误的信息，这些信息会损害系统的性能。中文 NER 里情况更严重，因为错误匹配的实体可能导致巨大的错误。汉语本身就是模棱两可的，因为单词的粒度/边界比其他语言（如英语）定义得不太清楚。因此，大量错误匹配的实体可以通过使用 gazetteers 生成。从下图所示的例子可以看出，将一个简单的9个字符的句子与4个地名索引匹配可能会导致6个匹配的实体，其中2个是不正确的：

![](/img/in-post/kg_paper/ggnn_exam.PNG)

为了消除错误，现有的方法通常依赖手工制作的末班或预定义的选择策略。可以参考之前的论文提出几种解决冲突匹配的方法，如最大化匹配。这些规则都有一些倾向性，没有充分利用词的语义信息。该论文提出我们也许可以通过数据驱动的方法来做。为此论文提出 Gated Graph Sequence Neural Networks(GGNN) + LSTM + CRF 的方法。其中改进的 GGNN 是一个新的多重有向图结构，可以显式的模拟文字和 gazetteers 的交互。

这里先介绍一下 GGNN，这里引用[图网络学习算法之——GGNN (Gated Graph Neural Network)](https://zhuanlan.zhihu.com/p/83410937) 的介绍。首先构建一个图 $$ G = （V， E） $$，V 是节点， E 是边的集合，通过利用 GGNN 来多次迭代的学习节点 V 的 embedding，并最终由所有节点的 embedding 得到全图的表达。 GGNN 是一种基于 GRU 的空间域信息传递模型。 message passing 的通用框架共包含三部分操作：信息传递(M)、更新操作(U)、读取操作(R)。节点的更新公式如下所示：

首先用节点的输入表示 $$ x_{v}$$ 初始化节点 v 的表示，维度不够的用 0 填充，最终每个节点的维度为 D。

$$ h_{v}^{(1)} = [x_{v}^{T}, 0]^{T} $$

而后构建邻接矩阵 A， A 包含两部分: 入度和出度，每部分维度都是 D，因此一个节点的邻接矩阵维度是 D * 2D。通过邻接矩阵 A，当前节点通过边与周围节点进行相互作用：

$$ a_{v}^{(t)} = A_{v}^{T}[h_{1}^{(t-1)}, \dots, h_{|V|}^{(t-1)T}]^{T} + b $$

得到当前节点的交互表示后，通过 GRU 对节点的状态进行更新，如下公式所示，其中 z 是遗忘门， r 是更新门：

$$ z_{v}^{t} = \sigma(W^{z}a_{v}^{(t)} + U^{z}h_{v}^{(t-1)}) $$

$$ r_{v}^{t} = \sigma(W^{r}a_{v}^{(t)} + U^{r}h_{v}^{(t-1)}) $$

$$ \tilde{h}_{v}^{(t)} = tanh(Wa_{v}^{(t)} + U(r_{v}^{t} \odot h_{v}^{(t-1)})) $$

$$ h_{v}^{(t)} = (1-z_{v}^{t})\odot h_{v}^{(t-1)} + z_{v}^{t} \odot h_{v}^{t}  $$

最终，模型可以输出两种粒度的信息：节点级别的，可以解决节点的分类问题，图级别的，可以做图的分类或嵌入，由节点级别到图级别的可以通过  self-attention 来做。

介绍完 GGNN 后，来看看怎么用在 NER 中的。首先网络结构如下图所示：

![](/img/in-post/kg_paper/ggnn_arc.JPG)

假如输入是 “张三在北京人民公园”，经过和 gazetteers 匹配我们得到 “PER1”、“PER2”、“LOC1”、“LOC2” 4 个词。为了构建图，首先我们对应每个输入字符建立对应的节点，这里是 9 个。然后再为匹配到的每个词建立 2 个节点--开始 s 和结束e。节点构建完了，构建有向边。首先添加输入系列字符间的有向连接，而后构建每个匹配词到对应的开始和结束字符间的有向连接。

作者认为，原始的 GGNN 不能直接用，因为不能区分带不同标签权重的边。为此提出两个扩展：

* 使得邻接矩阵 A 包含不同标签的边，包含相同标签的边共享权重    
* 为每个边定义了一系列的可训练贡献系数，它代表每个结构信息类型(gazetteers 和  character sequence)的贡献。

模型细节上，节点的初始化中，字符节点采用 char 和 bigram embedding concat 的方式做，开始和结束节点用 gazetteers embedding 得到。

![](/img/in-post/kg_paper/ggnn_init.JPG)

因为 邻接矩阵 A 要包含不同的边标签，因此：

$$ A [A_{1}, \dots, A_{||L||}] $$

贡献系数被定义为：

$$ [w_{c}, w_{g1}, \dots, w_{gm}] = \sigma ([\alpha_{c}, \alpha_{g1}, \dots, \alpha_{gm}]) $$

相同标签的边共享权重。接下来就利用 GRU 进行更新，其实除了上面那俩，剩下的和 GGNN 差不多：

![](/img/in-post/kg_paper/ggnn_gru.JPG)

最终论文除了在 OntoNotes 4.0 、MSRA、Weibo-NER 数据集上做了实验，还构建了一个电商领域的 NER数据集，最终结果如下图所示：

![](/img/in-post/kg_paper/ggnn_res.JPG)

可以看出，论文提出的方法在在加上 gazetteers 后表现达到了最优，并且比之前的 Lattice LSTM 要好不少。其实 Lattice LSTM 就是一个 DAG，用图来解决这件事很自然，感觉是个很好的探索。


## Neural Chinese Named Entity Recognition via CNN-LSTM-CRF and Joint Training withWord Segmentation

论文思想比较简单，就是一个联合 NER + CWS联合训练模型和通过实体替换的数据扩增方法。模型结构如下图所示：

![](/img/in-post/kg_paper/ncner_arc.PNG)

先 embedding 而后利用 CNN 提取局部特征，得到结合前后文的字符集表示。用该字符级表示做 CWS 任务。除此之外，该字符集表示还被输入到 BiLSTM + CRF 中做 NER 任务。 整体模型还是很简单的。最后损失函数由 NER 和 CWS 任务的损失联合得到，具体的贡献比例通过超参 $$\lambda$$ 控制，一般 0.4 - 0.5 之间效果较好：

$$ \lambda = (1-\lambda)L_{NER} + \lambda L_{CWS} $$

至于数据增广方法是指将原数据中的实体替换为相同类型的其他实体，该增广方法可有效提升模型效果, 6% 左右，如下图所示：

![](/img/in-post/kg_paper/ncner_pesdo.PNG)


## New Research on Transfer Learning Model of Named Entity Recognition

文章基于 BERT 模型，在后面加上了 BiLSTM + CRF 层，进行命名实体识别。在人民日报等语料库上进行训练和测试，最终表明，基于 BERT 强大的性能，该模型超过了以往的模型，同时相比于 BERT + MLP 做 NER ，也提升了一个点。算是意料之中的改进，就不详细介绍了。

## Multilingual Named Entity Recognition Using Pretrained Embeddings, Attention Mechanism and NCRF

基于 BERT + BiRNN + attention + NCRF ++ 模型，其中在 BiRNN 后加了 attention 来弥补 BiRNN 的长序列问题。解码层用神经CRF++层而不是普通CRF。它通过在相邻标签之间添加转换分数来捕获标签依赖项。NCRF++支持用句子级最大似然损失训练的CRF。在解码过程中，采用Viterbi算法搜索概率最大的标签序列。此外，NCRF++在n-best输出的支持下扩展了解码算法。最终选择 nbest 参数等于11，因为有11个有意义的标签。

除了上述改动外，论文是在多任务上做的尝试，因此还引入了一个辅助任务--输入语言种类判定。它通过获取 BiLSTM token 级别输出的 max 和 avg 进行分类，看看是输入是  保加利亚语、捷克语、波兰语和俄语 中的哪一个。

最终模型结构如下图所示：

![](/img/in-post/kg_paper/mner_arc.PNG)

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


## Fast and Accurate Entity Recognition with Iterated Dilated Convolutions

传说中的IDCNN，通过膨胀的方式，使得CNN用较浅的网络快速获得全局的感受野。它比传统CNNs具有更好的大上下文和结构化预测能力。与LSTMs的长度N的句子的顺序处理需要O（N）时间（即使面对并行性）不同，IDCNNs允许固定深度卷积在整个文档中并行运行，在保持与BiLSTMCRF相当的精度的同时，显著提高了14-20倍的速度。

传统的 CNN 想要大感受野就要堆层数， 第 l 层可见的文本范围是 $$ r = l(w-1) + 1 $$。像 NLP 中文本输入几百的话，这层数就太深了。也可以用池化，但序列标注任务中做池化的话，就要丢失序列中某些输入的信息。

为此，论文提出膨胀 CNN，对于膨胀卷积，有效输入宽度可以随着深度呈指数增长，为 $$ 2^{l+1} - 1 $$，在每一层上没有token损失，并且需要估计的参数数量适中。与典型的CNN层一样，扩张卷积在序列上的上下文滑动窗口上操作，但与传统卷积不同，上下文不需要连续；扩张窗口跳过每个扩张宽度d输入。通过对扩张宽度指数增加的扩张卷积层进行叠加，我们可以仅使用几层来扩展有效输入宽度的大小，以覆盖大多数序列的整个长度。 IDCNN 结构如下图所示：

![](/img/in-post/kg_paper/idcnn_arc.PNG)

应用到 NER 中时，每个 token 的输出可以加上 CRF 或者直接用线性输出做序列标注。最终模型在 CoNLL2003 上的表现如下图所示，看起来和基于 BiLSTM 的精度差不多(误差范围内)，强一点点：

![](/img/in-post/kg_paper/idcnn_res.PNG)

论文还测试 IDCNN 的速度，如下图所示，基本上稳定快 1.3 - 1.5 倍之间。

![](/img/in-post/kg_paper/idcnn_speed.PNG)

论文还尝试了带 linear dropout 和不带的情况对比，如下图所示：

![](/img/in-post/kg_paper/idcnn_dr.PNG)

## CNN-Based Chinese NER with Lexicon Rethinking

前面像 Lattice LSTM 模型都想办法利用 Gazetteer 信息，但是这些方法仍然存在两个问题：

* RNN 这种不能并行    
* 他们很难处理潜在词汇之间的冲突：一个字符可能对应于词汇中的潜在词汇对，这种冲突可能误导模型，使其预测不同的标签

针对第一个问题，论文使用CNN处理整个句子以及所有潜在单词的并行处理。直觉是，当卷积运算的窗口大小设置为2时，所有潜在的单词都可以很容易地融合到相应的位置。

针对第二个问题，通过使用反思(rethinking)机制来解决，通过添加反馈层并反馈高层次特征，这种重新思考机制可以利用高层次语义来细化嵌入单词的权重，并解决潜在单词之间的冲突。在四个数据集上的实验结果表明，该方法比字级和字符级基线方法都能获得更好的性能。

假设输入序列为 $$ C = {c_{1}, c_{2, \dots, c_{M}}} $$ ，其中 $$ c_{m} $$ 表示第 m 个字符。第 m 个字符匹配到的词为 $$ w_{m}^{l} = {c_{m}, \dots, w_{m+l-1}} $$ ，其中 l 表示词的长度为 l。

接下来用 CNN 对 char 级别输入进行特征抽取，如窗口大小为 2， 则得到 2-gram 特征，窗口大小 l 则得到 l-gram 特征：

$$ C_{m}^{l} = tanh(<C^{l-1} [ *, m:m+1], H_{l-1}> + b_{l-1}) $$

其中 $$ C^{l}_{m}$$ 表示第 m 个字的 l-gram 特征。

论文使用 vector-based 的 attention 来结合 l-gram 特征和词特征：

$$ i_{1} = \sigma(W_{i}C_{m}^{l} + U_{i}w_{m}^{l} + b_{i}) $$

$$ f_{1} = \sigma(W_{f}C_{m}^{l} + U_{f}w_{m}^{l} + b_{f}) $$

$$ u_{1} = tanh(W_{u}C_{m}^{l} + U_{u}w_{m}^{l} + b_{u}) $$

$$ i^{'}_{1}, f^{'}_{1} = softmax(i_{1}, f_{1}) $$ 

$$ X_{m}^{l} = i^{'}_{1} \odot u_{1} + f^{'}_{1} \odot C_{m}^{l} $$

接下来就是 Rethinking 机制了，通过我们刚刚得到的表示 $$ X_{m}^{l} $$，根据下面的公式重新计算一遍权重来对之前的权重进行调整。

![](/img/in-post/kg_paper/rethink_for.PNG)

最终得到修正之后的 token 表示后，加上 CRF 进行序列标注。最终实验结果如下图所示：

![](/img/in-post/kg_paper/rethink_res.PNG)

可以看出，比 Lattice LSTM 要强不少。

消融实验如下图所示，可以看出，Lexicon 是最重要的， Rethink 提升貌似不是很大。。。。

![](/img/in-post/kg_paper/rethink_xiaorong.PNG)

## Joint Learning of Named Entity Recognition and Entity Linking

NER 和 EL 相辅相成，是一个较为连续的任务流。现有任务经常分开做，这回造成错误的传递。为此论文提出一种联合 NER 和 EL 的模型方案，最终两个任务上的表现都和对应的  SOAT 差不多。

这个论文我的思路比较陌生，近期也没有相关方向跟进的想法，因此这里只是简单介绍。

模型的结构如下图所示，整体结构的基础时 stack-LSTM, stack - LSTM对应于一个基于 action 的系统，该系统由LSTMs和堆栈指针组成。与最常见的检测整个序列的实体 mention 的方法不同，使用堆栈LSTMs，实体 mention 是动态检测和分类的。这是我们模型的一个基本属性，因为我们在检测到提及时执行EL。此模型由四个堆栈组成：包含正在处理的单词的 Stack 、包含已完成块的 output 、包含当前文档处理过程中以前执行的操作的 Action 堆栈和包含要处理的单词的 Buffer。

除此之外，还有三个 action：

* SHIFT: 从缓冲区弹出一个单词并将其推入堆栈。这意味着缓冲区的最后一个字是命名实体的一部分。    
* OUT: 从缓冲区弹出一个单词并将其插入输出。这意味着缓冲区的最后一个字不是命名实体的一部分。    
* REDUCE: 将堆栈中的所有单词弹出并将其 push 到 OUTPUT。对于每种可能的命名实体类型，都有一个action Reduce，例如Reduce PER和Reduce LOC。

此外，可以在每个步骤执行的操作都是受控制的：只有堆栈为空时，才能执行操作Out；只有堆栈不为空时，才能使用操作Reduce。

![](/img/in-post/kg_paper/joint_arc.PNG)

剩下的部分再说吧。。。。

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
