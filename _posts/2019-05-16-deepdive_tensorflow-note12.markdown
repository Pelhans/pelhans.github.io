---
layout:     post
title:      "深度学习笔记（十二）"
subtitle:   "BERT"
date:       2019-05-16 00:15:18
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

最近开始做关系抽取相关的任务，因为BERT 现在势头很猛所以一直想试一下，同时还可以学学 Transformer 的用法细节。这篇主要记录 BERT 的原理等，下一篇介绍下用 BERT + CNN 做关系抽取。

BERT 是 Bidirectional Encoder Representations from Transformers 的简称。其实从名字也可以看出来，它的主要特点是采用了双向的 Transformer 做 Encoder。而 GPT 用的是单向的 Transformer， ELMO 用的是双向的 LSTM。这里的单向是指网络只可以利用当前词之前的输入，双向是网络可以利用前后文，在 token-level 时，如 SQuAD，双向明显效果会更好一点。

另一方面，BERT 还采用了两种新的自监督任务来训练模型--MLM(Masked Language Model) 和 NSP(Next Sentence Prediction)。MLM 任务类似于完形填空，就是给定前后文预测中间的词。NSP就是给两个句子，判断它们是不是上下文关系。两个任务可以分别捕捉词语级别和句子级别的表示。

另外谷歌还提供了$BERT_{BASE}$ 和 $BERT_{LARGE}$ 两个版本，小的版本是为了和GPT做比对。实验表明， BERT 在多项任务上都超过了之前的最优模型。

# 
