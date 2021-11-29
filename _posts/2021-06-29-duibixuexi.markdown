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

> 五月份就想写的笔记，一直拖到八月份，也是没谁了。。。

# 概览
对比学习是自监督学习的一种，该任务目的是在无标注的数据中，通过数据扩充构造相似正例和不相似负例，学习通用特征表示用在下游任务中，最终使得相似的实例在投影空间比较接近，而不相似的实例在投影空间离得比较远。

本篇笔记先整理对比学习的发展，再配合一些对比学习原理的讨论，最终看下对比学习在 NLP 领域的应用。

# 论文

## SimCLR：A Simple Framework for Contrastive Learning of Visual Representations

SIMCLR 是一个用在 CV 上 的对比学习框架，用来生成下游任务可用的通用表征。该论文的主要贡献为：

* 证明了数据扩充的重要性，并提出采用了两种数据扩充方式：
    * 空间/几何转换，包含裁剪、调整水平大小（水平翻转）、旋转合裁剪
    * 外观变换：如色彩失真（包含色彩下降、亮度、对比度、饱和度、色调）、高斯模糊合 Sobel 滤波
    * 同时还验证了数据扩充方式组合的重要性， 无监督对比学习比监督学习更可以从数据扩充中获益。
* 在特征表示数据和对比损失之间引入非线性转换，可以提高特征表示质量
* 与监督学习对比，更大的 bach size 和更多的训练步骤有利于对比学习
* 对比交叉熵损失的表示学习受益于 norm embeddig 和适当的 temperature 参数。一个适当的temperature可以帮助模型学习困难负例

整个模型的网络结构还是比较经典的，如下图所示。它包含四个组成部分：1）一个随机数据增强模块，用于产生同一个示例的两个相关图片，这两个相关图片可以被认为是正例。数据增强方式就是前面提到的（随机裁剪而后调整到与原图一样大小，随机颜色扭曲、随机高斯模糊）。2）一个神经网络编码层f()，用于提取表示向量，这部分对网络结构没有限制，论文里用的是 ResNet。3）映射层，用于将表示层输出映射到对比损失空间。4）对比学习loss。

![](/img/in-post/duibixuexi/simclr_fig2.png)

对比学习 loss 的计算公式为:

$$ l_{i,j} = -\log \frac{exp(sim(z_{i}, z_{j})/\tau)}{\sum_{k=1}^{2N}1_{k\neq i}exp(sim(z_{i}, z_{k})/\tau) }  $$

其中 sim 使用 L2 标准化的表征上计算 cosine 相似度。公式的含义是正例的相似度与其他所有负例的相似度在除以 $$\tau$$ 后算一下 softmax loss。也就是尽肯能的让正例在样本空间与原图片更相近，负例推得更远。

论文还论证了数据扩充方式组合的重要性。当组合增加时，对比预测任务变的更加困难，但表示质量显著提高。

![](/img/in-post/duibixuexi/simclr_fig5.png)

还有一些其他的结论：
* 有监督模型和在无监督模型上训练的线性分类器之间的差距随着模型规模的增大而减小，这表明无监督学习比监督模型从更大的模型中受益更多
* 使用表示层和loss 层间，加一个非线性投影可以显著改善特征表示质量。这个比较好理解，不映射的话高层特征表示会被任务污染，损失一些信息
* 相比于NT-logistic 和 triplet 损失函数，一方面这俩loss 需要使用 semi-hard 负例挖掘，另一方面这俩loss 的表现即使用负例挖掘表现也不如 NT-Xent loss。
* 更大的 batch size 和更长的训练，更有利于对比学习。随着 epoch 变多，不同 batch size 之间的差距会减少或消失

SimCLR 后续还有一些改进-SimCLR v2，主要改进点为：
* 编码层：将 ResNet50 换成了带有 SK 的 ResNet152
* 增大了非线性层的深度，变成了 3 层，并且在进行下游任务时保留了第一层
* 借鉴 Moco 中的记忆机制

    
## YADIM: A Framework For Contrastive Self-Supervised Learning And Designing A New Approach

2020 年对比学习有比较多的进展，包含SimCLR、AMDIM、Moco、BYOL 等。该论文为描述对比学习自监督学习方法定义了一个概念框架。并使用这个框架分析了三种对比学习示例：SimCLR、CPC、AMDIM。结论为：尽管这些方法似乎表面上看起来各不相同，但事实上它们都只是对彼此做出了细微的调整。

该框架通过五种视角来对比 CLS 的各种方法，包括：
* 数据增强 
* 编码器选择
* 表示向量获取
* 相似度计算
* 损失函数

先从数据增强角度来看：
* CPC 在抖动、灰度、翻转之外，还将原来的图像分割成 Patches 小块来生成更多正负样本对。
* AMDIM 是在对图像进行基本增强后，对同一个图像使用两次数据增强，生成两个版本的图像。
* SimCLR 就基本的几种方法

从编码器的角度看差别不大，大多都使用了不同宽度和深度的 ResNet 结构。消融实验表明，CPC 则受编码器改变的影响较小。

从表示向量获取角度来看区别就比较大了。表示向量是独特特性的集合，它使一个系统（以及人类）可以理解某事物与其他事物的区别。
* CPC 引入了通过预测潜在空间中的未来情况来学习表示向量的思想
* AMDIM 是比较从 CNN 不同层提取的各个特征图，对比不同视图的表示
* SimCLR 使用了 AMDIM 想同的思想，区别是仅仅使用最后的特征表示和利用一个映射处理该表示。
* Moco 与AMDIM 相似，但它保留了处理过的所有 batch 历史记录，并以此增加负样本的数量

根据本论文的实验结果发现，CPC 和 AMDIM 策略对结果的影响可以忽略不计，反而增加了计算复杂度。使这些方法生效的主要动力是数据增强过程。

从相似性度量的角度看，本论文的实验结果表明，对于相似度的选择在很大程度上是无关紧要的。至于在 loss 上的评测，除了 BYOL 外都用了噪声对比（NCE）loss。NCE loss 包含两个部分，其中分子鼓励相似的向量靠近，分母推动所有其他的的向量远离。

最终论文综合了上面的各个部分，得到 YADIM 框架：数据增强过程融合了 CPC 和 AMDIM。编码层用了 AMDIM 的编码器。表示向量获取 层用的是对图像的多个版本进行编码，并使用最后的特征进行对比。相似性度量用的是点积。损失函数是 NCE loss。

## SimCSE: Simple Contrastive Learning of Sentence Embeddings

简单的介绍了一下对比学习在 CV 那面的应用，接下来就开始看对比学习在 NLP 领域的应用。首先开头的是大佬的 SimCSE。前面我们看到，CV 领域中的数据增强还是有一些好用的方法的，但是在 NLP 领域是真的难受，以往我们常用到的方式大体为翻译、回传、删除、插入、替换、交换顺序、模板扩充等。但基于规则得到的扩充文本模式太明显，学习不好。模板放的太开还容易带来数据噪声。因此当我在开始做对比学习的时候用传统方法得到的提升很有限。

本论文提出了一种简单高效的数据扩充方法：将一个输入 dropout 两次作为对比学习的正例。对应无监督任务的话，这对正例可以通过分别输入到 BERT 中得到的两个向量组成。论文通过实验发现默认的 dropout rate 是最好的。

除了在下游任务中进行评测外，作者还采用了 alignment 和 unniformity 两个指标。其中 alignment 用于衡量相似样本的特征向量在空间上的分布是否足够接近。

$$ l_{align} = E_{x, x^{+}\~ p_{pos}} ||f(x) - f(x^{+}) ||^{2}  $$

Uniformity 用于衡量系统保留信息的多样性，评价指标是映射到单位球面的特征，尽可能的均匀分布在球面上，分布的越均匀，意味着保留的信息越充分。

$$ l_{uniform} = log E_{x,y\~ p_{data}} e^{-2||f(x) - f(y) ||^{2}}  $$

作者将不同方法得到的表示用这两个指标对比，发现 SimCSE 大幅度的提升了 Uniformity 和 改善了 Alignment，在二者间找到了一个较好的平衡。

很好的想法，赞叹，过一阵细读下。

## ConSERT: A Contrastive Framework for Self-Supervised Sentence Representation Transfer

美团的论文，也是针对 BERT 本身表示存在的坍缩问题，人话来讲就是当直接用 BERT 的输出计算相似度时整体的相似度都很高，之前自己做的时候是通过在取中间层和在末层加结构去绕过它。但 ConSERT 和前面的 SimCSE 都发现对比学习的特性可以改善这点。相比于 SimCSE，ConSERT 在做数据增强时探索了以下方案：
* shuffle：更换 pos id 的顺序
* token cutoff：在某个token维度把embedding置为0 
* feature cutoff：在embedding矩阵中，有768个维度，把某个维度的feature置为0 
* dropout

最终实验结果表明 feature cutoff + shuffle 组合的效果最好。最终得到的模型相比于 BERT 有一定的改善(大数据  VS 小数据？)。

## R-Drop: Regularized Dropout for Neural Networks

和 SimCSE dropout 两次的思想一样，只是加了一些 KL loss 在里面。具体来说在给定输入 x 后，经过两次有 dropout 的网络后，会得到两个预测输出：
$$p_{1}(y_{i}|x_{i})$$ 
和 
$$p_{2}(y_{i}|x_{i})$$ 
。因为 dropout 是随机的，所以这俩概率分布是不一样的。但这俩又是同一个 input 的表示，因此我们希望这两个表示别差别太大，因此加一个 KL loss 来限制这两个分布：

$$ L_{KL}^{i} = \frac{1}{2}(D_{KL}(P_{1}(y_{i}|x_{i})||P_{2}(y_{i}|x_{i}) ) + D_{KL}(P_{2}(y_{i}|x_{i})||P_{1}(y_{i}|x_{i}) )  )   $$

除此之外再加上传统的最大似然 loss：

$$ L_{NLL}^{i} = -\log~~P_{1}(y_{i}|x_{i}) - \log~~P_{2}(y_{i}|x_{i}) $$

总的 loss 就是这俩的和：

$$ L_{i} = L_{NNN}^{i} + \alpha* L_{KL}^{i}  $$

直观地说，在训练时，Dropout 希望每一个子模型的输出都接近真实的分布，然而在测试时，Dropout 关闭使得模型仅在参数空间上进行了平均，因此训练和测试存在不一致性。而 R-Drop 则在训练过程中通过刻意对于子模型之间的输出进行约束，来约束参数空间，让不同的输出都能一致，从而降低了训练和测试的不一致性。（dropout 当时大家说有用的愿意是相当于多个模型集成。现在又削弱这种多个模型的多样性 = =。。。）

# 对比学习原理探索
## Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere

该论文想发掘对比学习为啥能获得这么好的效果。为此论文指出了对比学习的两个重要属性：Alignment 和 Uniformity。这俩的定义前面提到过，这里在给出一下：

* Alignment：两个正例对应在样本空间应该是很近的，用来衡量正例对样本间的近似程度
* Uniformity：特征向量应该大致均匀地分布在单位超球面上，尽可能多的保留数据信息，用来衡量norm 后特征在单位超求面上分布的均匀性

当模型可以有效的优化这两个指标时，学到的特征在下游任务上可以表现的更好。对比学习就是，从loss 上看：

$$ L_{contrastive}(f,\tau,N) = E_{x,y\~p_{pos}}[-f(x)^{T}f(y)/\tau ] + E_{x,y\~p_{pos}}[\log (e^{f(x)^{T} f(y)/\tau} + \sum_{i}e^{f(x^{-}_{i})^{T}f(y)/\tau}  ) ]  $$

最小化第一项意味着要求正例尽可能的接近。第二项中，第一个 log 永远是正的，我们假设极端情况 $$ p[f(x) = f(y)] = 1 $$ ,那么第一部分就变成了常数，而优化第二项就意味着使得样本中的数据尽可能的分散。

用公式量化表达这两个量(前面定义过) Alignment 和 Uniformity：

$$ l_{align} = E_{x, x^{+}\~ p_{pos}} ||f(x) - f(x^{+}) ||^{2}  $$

$$ l_{uniform} = log E_{x,y\~ p_{data}} e^{-2||f(x) - f(y) ||^{2}}  $$

经公式定义后，就可以通过定义 loss 来直接优化这两个量，并且优化这两个指标的训练学到的特征在下游任务上表现更好。

## Debiased Contrastive Learning 
CV 的那几个论文里，负样本都是随机从同一 batch 中获得的。一般来说这样的 batch 都会比较大来获得较好的采样效果。但尽管如此，也还是有一定的概率采样到和正样本很相似的图片的（称这种图片为 false negative samples）。本论文基于这种情况作出改进。

论文基于理论推到，假设数据分布为 p(x)，其中正样本的分布概率为 $$ p(c) = \tau^{+} $$ ，它是均匀分布的，也就是在 batch 中的负样本也是这个概率为正样本。那么负样本的概率为  $$ 1 - \tau&{+} $$ 。但实际情况中这俩概率我们都是没有的，论文采用一种方法来做近似。最终通过重写损失函数证明通过增加正样本的数目(在负例中减去其中可能包含的正例部分贡献)，我们从 p(x) 采样得到的负样本得到逐渐强烈的修正。debiased 将会和 unbiased 的结果越来越接近。





# 引用

## 对比学习发展
* SimCLR
* YMDIM
* SimCSE: Simple Contrastive Learning of Sentence Embeddings
* ConSERT: A Contrastive Framework for Self-Supervised Sentence Representation Transfer
* R-Drop: Regularized Dropout for Neural Networks

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

