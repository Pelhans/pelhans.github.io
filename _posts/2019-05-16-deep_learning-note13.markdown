---
layout:     post
title:      "深度学习笔记（十三）"
subtitle:   "BERT"
date:       2019-05-16 00:15:18
author:     "Pelhans"
header-img: "img/dl_background.jpg"
header-mask: 0.3 
catalog:    true
tags:
    - Deep Learning
---

> 眼睛看过只是别人的，整理出来并反复学习才是自己的。

* TOC
{:toc}

# 概述

最近开始做关系抽取相关的任务，因为BERT 现在势头很猛所以一直想试一下，同时还可以学学 Transformer 的用法细节。这篇主要记录 BERT 的原理等，下一篇介绍下用 BERT + CNN 做关系抽取。如有错误，还请大佬指出。

BERT 是 Bidirectional Encoder Representations from Transformers 的简称。其实从名字也可以看出来，它的主要特点是采用了双向的 Transformer 做 Encoder。而 GPT 用的是单向的 Transformer， ELMO 用的是双向的 LSTM。这里的单向是指网络只可以利用当前词之前的输入，双向是网络可以利用前后文，在 token-level 时，如 SQuAD，双向明显效果会更好一点。

另一方面，BERT 还采用了两种新的自监督任务来训练模型--MLM(Masked Language Model) 和 NSP(Next Sentence Prediction)。MLM 任务类似于完形填空，就是给定前后文预测中间的词。NSP就是给两个句子，判断它们是不是上下文关系。两个任务可以分别捕捉词语级别和句子级别的表示。

另外谷歌还提供了$BERT_{BASE}$ 和 $BERT_{LARGE}$ 两个版本，小的版本是为了和GPT做比对。实验表明， BERT 在多项任务上都超过了之前的最优模型。

# 模型结构

![](/img/in-post/tensorflow/bert_struct.png)

BERT 的模型结构如上图所示，其中最左侧的是 BERT 的网络结构图，最底层的是 embedding 的输入，包含 position embedding、segment embedding、Token embedding 三部分：

* Position embedding: 位置向量，就是位置的 id，而后查表得到，作者没用Transformer 中的那个公式来得到位置向量    
* Segment embedding: 句子向量，如果当前句是第一句，用那么 EA 就是 0，第二句 EB 就是1，再查表    
* Token embedding: 字向量

最终的输入是它们三个相加。如下图所示。其中需要注意的是，每个序列的开头都是 CLS，两句话之间用 SEP 分隔。

![](/img/in-post/tensorflow/bert_embed.png)

组合之后的向量作为网络的输入。隐藏单元由 Transformer block 构成，对于 BASE 版本，包含L 12 层、隐藏单元H大小为 768，multi-head self-attention 的 head 数量为A 为 12，总参数 110M。LARGE 版本层数不变，H隐藏单元大小变为1024，head 数变为 16，总参数为 340M。

横向比对一下 BERT 和 GPT 与  ELMO的网络结构。可以看到相比于 GPT，BERT 的个隐藏单元都结合了前后文的信息，而GPT 只利用了当前时间步之前的信息。ELMO 倒是使用了双向的LSTM来结合前后文信息，但只是在输出的时候结合了一下，并不是像 BERT 那样每一层都结合。因此文章反复强调，BERT 的这种深度双向表示模型很重要。这个很好理解，毕竟有很多任务都比较依赖于前后文，在预训练时只关注前文效果一定会差一点。

# 预训练任务
## MLM

MLM 受启发于完形填空，它的核心思想是随机 mask 15%的词作为训练样本然后预测被 mask 掉的词。像下面这样：

* my dog is hairy --> my dog is [MASK]

但作者又考虑到训练时句子中包含很多的 MASK，这与实际使用中不符，因此：

* 80% 的概率将目标词像之前一样用 [MASK] 替换    
* 10% 的概率用一个随机的词替换目标词： my dog is hairy --> my dog is apple    
* 10% 的概率保持不变：my dog is hairy -->my dog is hairy

Transformer encoder 不知道那个词是要预测的或者被随机替换的，因此模型将强迫模型去关注每一个词。不过 MLM 的收敛速度比left-to-right(预测每一个词的) 要慢一点，不过效果却好很多。

## NSP

这个任务用来关于句与句之间的联系，如 QA或者 NLI 这种。因此 NSP 任务就是预测两句话是不是上下句关系。语料建议从 文档级别的里面抽取句子对，这样可以更好的获取连续长特征的能力。训练语料中，50%的概率是原文中的下下句，50%的概率是随机选取的句子。最终模型在这项任务上得到的准确率是 97%-98%，实验证明该项任务可以明显给QA 和 NLI 类任务带来提升。

关于预训练的一大堆参数就不列了。其中感觉需要注意的是激活函数用的 GELU。

# Fine-tuning Procedure

针对 NLP 中常见的任务分类，BERT 提供了对应的解决方案：

* 句对关系判断，加上一个起始和终止福海，句子之间加分隔符，输出时第一个起始符号[CLS]对应的Transformer编码器后，增加简单的Softmax层，即可用于分类；
* 分类任务：文本分类/情感计算...，增加起始和终结符号，输出部分和句子关系判断任务类似改造；
* 序列标注：分词/POS/NER/语义标注...,输入部分和单句分类是一样的，只需要输出部分Transformer最后一层每个单词对应位置都进行分类即可。
* 生成式任务：机器翻译/文本摘要：问答系统输入文本序列的question和Paragraph，中间加好分隔符，BERT会在输出部分给出答案的开始和结束 span。

在参数的设置上，作者建议大部分超参不用变，至修改 batch size 、learning rate、number of epochs 这仨就够了。：

* batch size: 16，32    
* Learning rate(Adam): 5e-5, 3e-5, 2e-5    
* number of epochs: 3, 4

作者还说，训练数据集越大时，对超参就越不敏感，而且 fine-tuning 一般来说收敛的很快。

# 总结

看完论文感觉 BERT 就像一个偷学百家的武林高手，看起来没有什么特别大的创新，但玩法确实高级，也很暴力。我觉得 BERT 的优点为：

* 用大量非标注语料得到训练数据，因此可以充分利用互联网上海量的文本数据    
* 作者根据大量的文本数据，量身打造了两个任务--MLM 和 NSP，分别应对 词级别和 句子级别的任务，效果卓群    
* 搭建了深度双向网络，充分利用前后文信息    
* 只用 Transformer ，完全抛弃了RNN 和 CNN，告诉大家原来 Transformer 可以这样用

那么缺点呢？作者认为：

* [MASK]标记在实际预测中不会出现，训练时用过多[MASK]影响模型表现    
* 每个batch只有15%的token被预测，所以BERT收敛得比left-to-right模型要慢（它们会预测每个token）

不过我把它用到关系抽取任务中，发现 BERT 对中文提供的模型是基于字的，最终 用 BERT + CNN 做的关系抽取模型在中文语料上效果相比于 基于词的 word2vec + CNN 的还是差了一点，而且 word2vec 得到的词向量效果越好，这个差距就越大， word2vec 得到的向量较差时，甚至还能超过。因此 BERT 如果应用到 英文类的关系抽取任务中有较大的改进，但中文任务上就得动动脑子了，毕竟自己也没钱训练一个基于词的。。。。

头一阵看到百度的一个对BERT 的改进 [ERNIE](https://zhuanlan.zhihu.com/p/59436589)，它的想法比较直接：

> ERNIE 模型通过建模海量数据中的实体概念等先验语义知识，学习真实世界的语义关系。具体来说，ERNIE 模型通过对词、实体等语义单元的掩码，使得模型学习完整概念的语义表示。相较于 BERT 学习原始语言信号，ERNIE 直接对先验语义知识单元进行建模，增强了模型语义表示能力。例子如下图：

![](/img/in-post/tensorflow/ernie_exam.png)

微软也对 BERT 做了一些改进，叫 MASS，大体上就是你 BERT 盖住的不是一个字么，这个在生成类任务中不太够用，我现在随机盖住50%连续的字，然后生成你，这样在生成类任务中会有更好的表现。

说远了，不过想说的是， BERT 只是一个较为通用的预训练模型，要想落实应用，需要针对具体的任务类型做一些改动才能达到更好的效果。
