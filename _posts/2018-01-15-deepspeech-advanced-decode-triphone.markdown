---
layout:     post
title:      "语音识别笔记 (六)" 
subtitle:   "多遍解码、三音子模型"
date:       2018-01-15 20:39:28
author:     "Pelhans"
header-img: "img/post_deepspeech_ch1_ch2.jpg"
header-mask: 0.3 
catalog:    true
tags:
    -  speech process
---


> 本讲我们来简要讨论一些语音识别的高级话题，包含多遍解码和三音子模型。 

* TOC
{:toc}

# 第六讲

## 多通道解码(Multi-pass Decoding)

在上一讲中，我们介绍了基于Viterbi算法的解码方案。但在实际中的应用，它有两个主要的限制。第一个是因为Viterbi算法算法返回的实际上不是最大概率的单词序列，而是计算与这样的单词序列的近似。这就带来了与实际最大值间的差别，在大多数情况下这种差别并不重要，但对于解码问题来说，有时概率醉倒的音子序列并不对应于概率最大的单词序列。如一个单词具有多个发音时，由于概率归一的影响，通过它的分支概率就会较小，这样算法就会倾向于选择概率较大的分支较少的词，从而带来错误。

另一个限制是它不能用于所有的语言模型。这个限制来自于Viterbi算法的既有事实，即破坏了动态规划恒定的假定。简单来说就是该假定要求一个最佳路径一定包含一直到状态$$q_{i}$$之前和$$q_{i}$$本身在内的最佳路径。但三元语法显然有可能会破坏这点。

因为这两个限制的存在，人们给出两种解决方案：

### 修改Viterbi算法

修改Viterbi解码算法，让它返回多个潜在的语段。然后再使用其他更复杂的语言模型或发音模型算法，重新给多个输出排序。一般来说，这种多遍解码方法在计算上是有效的，但若先使用二元语法这种不太复杂的模型来进行第一遍粗解码，然后在使用更复杂但速度较慢的解码算法继续工作就可以减少搜索空间。

例如Wchwartz提出的一种类似于Viterbi的算法，称为N-best Viterbi算法。对于给定的语音输入，这种算法返回N个最佳的句子，每个句子带有它们的似然度打分。然后使用三元语法给每个句子指派一个新语言模型的先验概率。这些先验概率与每个句子的声学模型似然度结合，生成每个句子的后验概率。然后使用这种更复杂的概率重新给句子打分。下图给出该方法的示例图：

![](/img/in-post/deepspeech_ch6/deepspeech_ch6_1.jpg)

还有另一种方案也是用N-best的办法来提升Viterbi算法，但返回的不是一个句子表，而是一个单词格。单词格是单词的有向图，单词之间用单词格连接之后就可以对大量的句子进行紧致的编码。在格中的每个单词使用它们的观察似然度来扩充，这样通过格的任何路径肚皮可以与更复杂的语言模型中推到的先验概率结合起来进行改进。

![](/img/in-post/deepspeech_ch6/deepspeech_ch6_2.jpg)

### 使用$$A^{*}$$算法

Placeholder

## 三音子模型

之前我们讨论的都是单音素模型，但我们知道发声是会收到前后文影响的。因此提出三音子模型(Triphone Model).一个三音子模型表示在左右文本限定情况下的音素模型。举例来说，一个三音子$$[y-eh+l]$$表示$$[eh]$$的前面是$$[y]$$，后面跟着$$[l]$$。当凑不齐三个时，也可以使用其中的两个来表示，如$$[y-eh]$$表示$$[eh]$$前面是$$[y]$$，$$[eh+l]$$表示$$[eh]$$后面是$$[l]$$。

虽然三音子的引入能够帮助我们捕捉声音中的变化，但也同时带来了稀疏性问题。假设我们有50个单音素，那组合起来就是$$50^{3} = 125, 000$$个三音子，而且其中很大部分是不常见甚至不存在的。为了减少三音子的数量，Young提出了子音素绑定的方法。其主要思想为将那些相似的音素归为一类(Cluster)。如$$[m-eh+d]$$和$$[n-eh+d]$$这两个三音子，将它们归为一类后就可以采用一个高斯模型来训练它们。

那怎么判断哪些音素该被归为一类呢？最常用的方法是决策树。从根节点开始，如$$/ih/$$，在每个节点都问一些问题并对其分类，直到最终类别为止。下图给出一个决策树分类的例子。这个决策树的训练也和正常的决策树训练类似，对于每个节点，它会考虑新分支将会给训练数据的声学模型似然度带来的影响并选择似然度最大那个节点和问题。如此反复进行迭代制止到达叶子节点。

![](/img/in-post/deepspeech_ch6/deepspeech_ch6_3.jpg)

下图给出一个完整的给予上下文的GMM三音子模型的建立，其中采用的是 two cloning-and-retraining 流程，具体这里就不展开介绍了。

![](/img/in-post/deepspeech_ch6/deepspeech_ch6_4.jpg)

# Ref

[1] Speech and Language Processing 2nd; ch10
