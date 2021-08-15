---
layout:     post
title:      "对比学习笔记"
subtitle:   ""
date:       2021-06-29 00:15:18
author:     "Pelhans"
header-img: "img/attention.jpg"
header-mask: 0.3 
catalog:    true
tags:
    - NLP
---


* TOC
{:toc}

# 概览

# SimCLR：A Simple Framework for Contrastive Learning of Visual Representations

SIMCLR 是一个用在 CV 上 的对比学习框架，用来生成下游任务可用的通用表征。该论文的主要贡献为：

* 证明了数据扩充的重要性，并提出采用了两种数据扩充方式：
    * 空间/几何转换，包含裁剪、调整水平大小（水平翻转）、旋转合裁剪
    * 外观变换：如色彩失真（包含色彩下降、亮度、对比度、饱和度、色调）、高斯模糊合 Sobel 滤波
    * 同时还验证了数据扩充方式组合的重要性， 无监督对比学习比监督学习更可以从数据扩充中获益。
* 在特征表示数据和对比损失之间引入非线性转换，可以提高特征表示质量
* 与监督学习对比，更大的 bach size 和更多的训练步骤有利于对比学习
* 对比交叉熵损失的表示学习受益于 norm embeddig 和适当的 temperature 参数。一个适当的temperature可以帮助模型学习困难负例

整个模型的网络结构还是比较经典的，如下图所示。它包含四个组成部分：1）一个随机数据增强模块，用于产生同一个示例的两个相关图片，这两个相关图片可以被认为是正例。数据增强方式就是前面提到的（随机裁剪而后调整到与原图一样大小，随机颜色扭曲、随机高斯模糊）。2）一个神经网络编码层f()，用于提取表示向量，这部分对网络结构没有限制，论文里用的是 ResNet。3）映射层，用于将表示层输出映射到对比损失空间。4）对比学习loss。

![](/img/in-post/duibixuexi/simclr_fig2.PNG)

对比学习 loss 的计算公式为:

$$ l_{i,j} = -\log \frac{exp(sim(z_{i}, z_{j})/\tau)}{\sum_{k=1}^{2N}1_{k\neq i}exp(sim(z_{i}, z_{k})/\tau) }  $$

# 引用

## 对比学习发展
* CV 开端：SimCLR
* CPC
* AMDIM（https://zhuanlan.zhihu.com/p/228255344）
* SimCSE: Simple Contrastive Learning of Sentence Embeddings
* ConSERT: A Contrastive Framework for Self-Supervised Sentence Representation Transfer

## 对比学习原理探索
* Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere
* Debiased Contrastive Learning
* ADACLR: ADAPTIVE CONTRASTIVE LEARNING OF REPRESENTATION BY NEAREST POSITIVE EXPANSION
* NCE 到 infoNCE 的推导：https://zhuanlan.zhihu.com/p/334772391
* 理解对比损失的性质以及温度系数的作用：Understanding the Behaviour of Contrastive Loss
* Can contrastive learning avoid shortcut solutions?

## 对比学习在 NLP 中的应用
* CIL: Contrastive Instance Learning Framework for Distantly Supervised Relation Extraction
* Learning to Rank Question Answer Pairs with Bilateral Contrastive Data Augmentation
* SimCLS: A Simple Framework for Contrastive Learning of Abstractive Summarization
* Sentence Embeddings using Supervised Contrastive Learning
* Hybrid Generative-Contrastive Representation Learning
* Self-Guided Contrastive Learning for BERT Sentence Representations
* Biomedical Entity Linking with Contrastive Context Matching
* Investigating the Role of Negatives in Contrastive Representation Learning
* Semi-supervised Contrastive Learning with Similarity Co-calibration
* Improving BERT Model Using Contrastive Learning for Biomedical Relation Extraction
* Constructing Contrastive samples via Summarization for Text Classification with limited annotations
* Unsupervised Document Embedding via Contrastive Augmentation

