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
* 基于特征的机器学习方法: 需要标注数据, 同时一般结合精心设计的特征.    
* 基于深度学习的方法: 需要标注数据, 自动学习特征, 可以端到端的搞

现在一般领域性比较强, 数据量特别少的会用规则, 其余基本上都是机器学习或者深度学习. 尤其是在数据量比较充足的时候, 深度学习一般都可以获得比较不错的指标.

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
