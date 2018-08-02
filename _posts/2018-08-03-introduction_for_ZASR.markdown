---
layout:     post
title:      "基于tensorflow 的中文语音识别"
subtitle:   "基于DeepSpeech2 论文"
date:       2018-08-03 00:15:18
author:     "Pelhans"
header-img: "img/speech_process.jpg"
header-mask: 0.3 
catalog:    true
tags:
    - speech process
---


> 目前网上关于tensorflow  的中文语音识别实现较少，而且结构功能较为简单。而百度在PaddlePaddle上的 Deepspeech2 实现功能却很强大，因此就做了一次大自然的搬运工把框架转为tensorflow....

* TOC
{:toc}

# 简介

百度开源的基于PaddlePaddle的Deepspeech2实现功能强大，简单易用，但新框架上手有难度而且使用过程中遇到了很多bug，因此萌生了转成tensorflow的想法。网上看了一圈，发现基于tensorflow的中文语音识别开源项目很少，而且功能较为简单。英语的项目倒是很多，但奈何写代码的人功力太深厚，想转成中文的很麻烦。因此本项目的目标是做一个简单易理解，方便新手入门的基于神经网络的ASR模型，同时把常用的功能加上方便参考。(实际上是代码功力太差...)

# 识别流程

