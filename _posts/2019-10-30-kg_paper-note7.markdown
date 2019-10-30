---
layout:     post
title:      "实体链接(三)"
subtitle:   "文本匹配"
date:       2019-10-30 00:15:18
author:     "Pelhans"
header-img: "img/tensorflow.jpg"
header-mask: 0.3 
catalog:    true
tags:
    - Knowledge Graph
---


* TOC
{:toc}

# 文本匹配任务概览

文本匹配是 NLP 领域最基础的任务之一，目标是找到与目标文本 Q 最相似的文本 D。 像信息检索的重排序(Information retrieval rank)， 问答系统中(Quextion Answering)的候选答案的选取， 文本蕴含任务(textual entailment )，自然语言推理(Natural Language Inference) 等都可以看做文本匹配任务在各个领域的具体应用。

传统的模型如 TF-IDF， BM25 等通常会从字符串层面上计算 token 的匹配程度。 可是该方法无法处理 目标文本 Q 和 相似文本 D 间存在语义相关但文本并不匹配的情况 。为此研究人员先后提出了 LSA， LDA 等语义模型。随着深度学习的兴起，通过一系列非线性映射并结合 CNN，LSTM等结构提取文本深层次的语义特征低维向量表示取得了很好的效果。下面我们对近年来深度学习在文本匹配任务中的研究做一个概览。

从模型结构上可分为两类：

* 一类是纯基于表示的模型， 该类方法采用双塔式结构， 通过一些列深度网络将 Query 和 Doc 用低维向量表示出来， 而后采用 cosin , 欧氏距离等方式计算 Q 与 D 间的相似度。 典型的结构代表是 SiameseNet, DSSM, CDSSM, LSTM-DSSM, MV-DSSM 等。    
* 另一类在获取Q 和 D 的表示后， 还加入了了二者的交互匹配策略， 该类方法认为纯基于表示的模型没有有效利用文本间的信息， 忽略了 Q 与 D 间的特征匹配，通过构建不同的特征匹配策略，模型可以获得更好的表现， 这里比较有代表性的就比较多，如 DecAtt, ESIM, BiMPM, ARC-I, MatchPyramid 等。

## 基于表示的模型

### SiameseNet -- Signature Verification using a "Siamese" Time Delay Neural Network

深度学习大佬 Yann LeCun 的文章， 论文目的是手写签字验证。网络结构如下图所示：

![](/img/in-post/kg_paper/text_match_siamese.jpg)

论文比较久远，图片比较糊。。。。我用文字表述一下， 其中我们只关注左右两个网络中的一个， 因为它们的参数是共享的。

* 首先输入是一个手写签名， 通过一系列预处理程序后得到对应的向量表示，就是标记 200 的那个(input 阶段)    
* 该表示通过多层 CNN 提取特征， 采用均值池化（ 特征表示阶段 ）    
* 通过下采样和均值操作得到低维表示（聚合阶段）    
* 通过计算余弦得到两个向量间的相似度（相似度计算阶段）    

这篇论文是很值得学习的， 虽然结构比较简单， 但很经典， 后面的大部分研究都是在该结构的基础上进行改进的。

至于损失函数， 常用的是对比损失函数(contrastive loss)、Hinge loss, 交叉熵损失函数等。对比损失函数的公式为

$$ L = \frac{1}{2N}\sum_{n=1}^{N}\{yd^{2} + (1-y)\max(margin - d, 0)^{2}  \} $$

其中 

$$ d = || a_{n} - b_{n}||_{2}$$ 
是两个样本特征的欧氏距离。y 是两个样本是否匹配的标签。 margin 是人为设定的阈值。

### DSSM -- Learning Deep Structured Semantic Models for Web Search using Clickthrough Data

DSSM 是工业界最常用的模型之一，来自微软，该模型结构简单，效果较好。后续也有很多跟进的改进， 如 CDSSM， LSTM-DSSM 等。

模型的流程如下图所示：

![](/img/in-post/kg_paper/text_match_dssm.jpg)

* DSSM 首先将输入通过 word hashing 进行编码(input)    
* 执行非线性投影将查询 Q 和文档 D 映射到公共语义空间， 中间的 multi-layer 采用的是线性连接+ tanh 激活函数(特征表示+聚合)    
* 计算 Q 和 D  对应语义表示之间的余弦相似度（相似度计算）    
* 对相似度执行  softmax，得到归一化概率分布来表示Q 和 每个文档 D 的匹配程度    
* 用 Clickthrough 数据训练(用户的检索 Q 和 4 个点击的文档 D 构成的数据集)，训练目标是条件似然最大化

可以看出该模型遵循 SiameseNet 的四段式结构。这里说一下比较特殊的地方。 首先是 word hashing， 它的目的是解决大规模搜索中词汇表过大的问题。经统计，未采用 word hashing 前，词汇表的大小达到 500k。而经过 word hashing 后， 词汇表大小就变成 30 k 了。

word hashing 是基于 "字符 n-gram "的，以例子说明， 对于 输入 "word", 我们对其加上开始和终止符号 "#word#"， 如果采用 3-gram 的话， 那么我们就会得到 "#wo"、"wor"、"ord"、"rd#" 四个部分。对于所有的输入都进行这样的拆分并统计就得到拆分后的词汇表，也就是 30k 的那个。 当我们输入一个词时， 首先我们对其进行这样的切分， 而后类似于 one-hot 那样， 在 30k 的词汇表内出现的位置标记为 1， 其余为 0. 这样我们就得到了该词的 multi-hot 表示。该方法的缺点是比较怕碰撞，这样效率就会降低， 不过作者统计了一下， 在 3-gram 的情况下碰撞概率只有 0.0044%(22/500,000)。几乎没有影响。

第二个特殊的地方是， DSSM 中 Q 和 不同的 D 对是共享参数的， 即 W1, W2, W3, W4 即用来处理Q，也用来处理D。

第三个点是优化目标， 对应的公式为

$$ L(\theta) = -\log\prod_{Q, D^{+}}p(D^{+}|Q) $$

即只考虑了正例带来的损失， 负例不参与反向传播。

### CDSSM -- A Latent Semantic Model with Convolutional-Pooling Structure for Information Retrieval

CDSSM(CLSM) 是对 DSSM 的改进版， 它将 DNN 换成 CNN， 更充分考虑语义上下文信息。模型整体结构和 DSSM 一样， 只不过是将 DNN 部分换成了 CNN， CNN 部分如下图所示：

![](/img/in-post/kg_paper/text_match_cdssm.jpg)

整体流程为：

* 输入的 Query 和 Document 是一系列词    
* 采用窗口大小为3 , 步长为 1 的滑动窗口获得一系列的 tri-gram。    
* 对于 tri-gram 内的每个词， 采用 word hashing 的方法得到对应的 multi-hot 表示， 而后将 3 个词对应的表示首尾连接在一起, 如 3 个 (1,30k) 的组成 (1, 90k) 的(input 层)    
* 对该向量进行卷积、最大池化、tanh 激活等操作得到定长特征表示(特征表示)    
* 利用线性映射， 得到低维语义表示(聚合操作)    
* 通过  cosine 计算 Q 和 D 间的相似度， 后面的和 DSSM 一样了

### LSTM-DSSM  -- SEMANTICMODELLING    WITHLONG-SHORT-TERMMEMORY  FORINFORMATIONRETRIEVAL

DSSM 的另一种改进， 第一次尝试在 信息检索（IR）中使用 LSTM，模型的创新点是将 DNN 替换成了 LSTM，用 LSTM 最终时间步的输出作为语义编码结果。模型结构如下图所示

![](/img/in-post/kg_paper/text_match_lstm-dssm.jpg)

模型整体结构没什么细说的，需要注意的一个是这里的 LSTM 不是标准的， 而是添加了 C_{t-1} 的 peephole 连接去辅助计算各个门。如下图所示， 更进一步可以看[LSTM 详解](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

![](/img/in-post/kg_paper/text_match_lstm-dssm-lstm.png)

用 LSTM 的好处是可以进一步考虑长时序相关的信息， 还可以减轻梯度消失等问题。

### MV-DSSM -- A Multi-View Deep Learning Approach for Cross Domain User Modeling in Recommendation Systems

该改进是为了更好的在推荐系统领域应用语义匹配方法，所谓 Multi-view 是指采用了多类数据，而非原始中只有 Q 和 D，是单一领域的， 可以看做 single-view，通过该改进模型就具备了多领域匹配能力。 另一个改进是 MV-DSSM 中各 view 的 DNN 的参数是独立的， 而原始 DSSM 模型的 DNN 参数是共享的， 独立参数可以有更好的效果。

### Learning to Rank Short Text Pairs with Convolutional Deep Neural Networks

用 textCNN 提取语义特征， 而后采用相似度矩阵计算特征表示间的相似度。根据该相似度，再结合Q 和 D 的语义特征向量连接起来作为聚合表示。 最后通过 线性层和 softmax 进行分类。模型流程如下图所示

![](/img/in-post/kg_paper/text_match_textcnn_sim.jpg)

模型流程为：

* 输入一个文档 Q/D, 以 Q 为例， 它的长度为 $l_{q}$，其中每个词采用 word2vec 的嵌入向量表示，维度为 d。(input)    
* 对输入进行卷积操作， 卷积核大小为(3, d)，步长为1， 对于 pad 的问题， 作者认为， pad 后卷积核可以平等的对待边界部分的单词，同时对于长度小于卷积核大小的情况也能处理， 因此建议 pad。池化部分采用最大池化，作者认为平均池化时每部分都有贡献，正向信号会被负向信号抹平。而最大池化会突出正向信号， 但缺点是容易过拟合。该部分的输出为 $x_{q}$/$x_{d}$(特征表示)     
* 使用 $x_{q}$、$x_{d}$ 计算相似度 $$ sim(x_{q}, x_{d}) = x_{q}^{T}Mx_{d} $$， 该相似度得分是一个标量    
* 将相似度得分和 $x_{q}$、$x_{d}$连接起来得到聚合的向量表示， 在该步骤还可以添加额外的特征 $x_{feat}$    
* 通过线性层和 softmax 部分进行分类(相似度计算)

改论文相比于 CDSSM 有几个比较明显的改变。 首先改论文发表于 2015 年， 此时已经有 word2vec 等一系列较为成熟的 word embedding 方法， 因此没有采用 word hashing 方法。

另一方面， 卷积操作也采用我们现在比较熟悉的方式， 即卷积核是沿着时间序列扫描的， 深度为词向量维度 d。而 CDSSM 里是在 (1,90k) 的向量上扫描的， 个人认为这得益于 word2vec 的稠密向量表示。

还有该模型在一定程度上考虑了 Q 和 D 间的交互匹配， 也就是 $$x_{q}^{T}Mx_{d} $$ 这块，后续交互改进的一大重点就是挖掘Q 和 D 间的交互匹配。

最后一个变化是训练优化目标变成了交叉熵损失函数， L2 正则化。

## 基于交互的模型

除了考虑输入的表示外， 还利用各种 Q 和 D 间的交互匹配特征，尽可能的匹配二者间的不同粒度信息。

具体来说：

* 词级别匹配：两个文本间的单词匹配，包含字符相等匹配(down-down)和语义相等匹配(famous-popular)    
* 短语级别匹配： 包含 n-gram 匹配和  n-term 匹配，n-gram指像 (down the ages 和 down the ages)的匹配， n-term 匹配是指像 (noodles and dumplings 和 dumplings and noodles) 的匹配。    
* 句子级别的匹配: 由多个低层次的匹配构成

除此之外，还有词和短语间的匹配等等模式， 如何很好的发掘这种模式并利用起来是研究的重点之一。

### MatchPyramid -- Text Matching as Image Recognition

将Q 和 D 进行匹配得到 匹配矩阵(matching matrix)，矩阵的每一个点表示 Qi 和 Dj 的相似度，相似度计算分别尝试用 indicator(0, 1)， cosine 和 dot 三种。 而后将该矩阵看做一张图像，采用图像那面的卷积方法来处理它。通过卷积操作，模型可以捕获短语级别的交互并通过深层网络进行深层次的组合。  模型流程如下图所示

![](/img/in-post/kg_paper/text_match_mp.jpg)

这里重点说一下从文本到 matching matrix 这步，给定 $Q = (w_{1}, w_{2}, \dots, w_{lq})$ 和文档 $D = (v_{1}, v_{2}, \dots, v_{ld})$，我们将会得到一个形状为 lq * ld 的矩阵M， 其中的每个元素 $M_{ij}$ 表示 $w_{i}$ 和 $v_{j}$ 计算相似度得到的值。

* Indicator 函数， 则当 $w_{i}$ 和 $v_{j}$ 一样时 为 1， 否则为 0.    
* Cosine 函数，
$$ M_{ij} = \frac{\overrightarrow{\alpha_{i}}^{T}}{\overrightarrow{\beta_{j}}}{||\overrightarrow{\alpha_{i}}^{T} ||* ||\overrightarrow{\beta_{j}}||}$$     
* Dot product: $$ M_{ij} = \overrightarrow{\alpha_{i}}^{T}* \overrightarrow{\beta_{i}} $$

实验结果表明 点乘的效果最好。

###   Pairwise Word Interaction Modeling with Deep Neural Networksfor Semantic Similarity Measurement

该论文充分利用了 Q 和 D 间的相似匹配特征，构建了一个深度为 13 的匹配矩阵，而后加一层 mask， 该mask 会对重要的信号进行放大， 弱相似的缩小。通过上述方式得到 matching matrix, 而后采用 19 层的深度 CNN 进行特征提取得到特征表示，最终模型通过 2 个全连接层和softmax操作进行分类。改论文充分利用了两个文本之间的多种相似性模式，并以此作为后续 CNN 的输入， 不过个人感觉，这种完全采用相似矩阵作为后续输入的方式是否真的通用有效， 毕竟从文本到相似矩阵这步丢失了很多语义信息。

模型流程如下图所示

![](/img/in-post/kg_paper/text_match_multi_match_cnn.jpg)

具体来说， 首先采用 BiLSTM 对输入的文本进行语义编码， 得到 t 对应的输出 $h_{1t}^{for}$ 和 $h_{1t}^{back}$。对弈 s 得到 $h_{1s}^{for}$ 和 $h_{1s}^{back}$ . 而后通过以下方式构造深度为 13 的 matching matrix .

* 1 - 3 表示 $$ coU(h_{1t}^{b_{i}}, h_{2s}^{b_{i}}) $$， 其中 $$(h_{1t}^{b_{i}}) = [h_{t}^{for}, h_{t}^{back}] $$, 其中 $$ coU(h_{1}, h_{2}) = \{cos(h_{1, h_{2}}), L2Euclid(h_{1}, h_{2}), DotProduct(h_{1}, h_{2}) \} $$    
* 4 - 6 表示 $$ coU(h_{1t}^{for}, h_{2s}^{for})$$    
* 7-9 表示 $$ coU(h_{1t}^{back}, h_{2s}^{back})$$    
* 10 - 12 表示 $$ coU(h_{1t}^{add}, h_{2s}^{add}) $$， 其中 $$h^{add} = h^{for} + h^{back}$$    
* 13 表示 indicator

构造过程用伪代码表示为

![](/img/in-post/kg_paper/text_match_multi_match_cnn_matching.jpg)

接下来构造 mask 矩阵， 该矩阵会在匹配程度较高的位置设置值 1， 其余位置 设为 0.1 。这样通过将 mask 乘以上面那个 matching matrix 对相应的信号进行放大缩小。用伪代码表示为 

![](/img/in-post/kg_paper/text_match_multi_match_cnn_mask.jpg)

其中 calcPos 函数返回相对位置。 再往后就是深度卷积网络了， 对应的配置如下所示

![](/img/in-post/kg_paper/text_match_multi_match_cnn_config.jpg)

### Sentence Similarity Learning by Lexical Decomposition and Composition

前面的模型都是尽可能的去匹配 Q 与 D 间相似的部分， 但该论文认为， 不相似的部分也同样重要。论文将输入分解为相似部分和不相似部分，将二者加权求和送入 CNN 进行聚合， 最通过 线性层和  sigmoid 函数终计算相似程度. 论文的完整流程如下图所示

![](/img/in-post/kg_paper/text_match_cnn_unsim_over.jpg)

假设输入为 $ S = (s_{1}, s_{2}, \dots, s_{m})$ 和 $T = (t_{1},. t_{2}, \dots, t_{n})$。通过 T 和 S 我们可以构建相似匹配矩阵 $A_{mn}$。 A 中每个元素通过  
$$ a_{ij} = \frac{s_{i}^{T}t_{j}}{||s_{i}||*||t_{j||}} $$ 得到。

接下来计算 $\hat{s}_{i} = f_{match}(s_{i}, T)$, 其中 $f_{match}$ 被定义为

![](/img/in-post/kg_paper/text_match_match_cnn_fmatch.jpg)

其中 global 考虑了整个 T 序列， max 部分只考虑了单独一个 t， 而 local-w 操作综合了以上两个， 其T 中的一部分(以 k为中心的前后 w 个).从另一个角度看，这其实就是从不从颗粒度上的匹配， 即 句子、短语、词级别的匹配。实验结果表明 max 操作效果最好。

有了 $\hat{s}$ 和 $\hat{t}$ 后我们将对 $s_{i}$ 进行分解得到相关部分和不想管部分。具体来说有 3 种分解策略 -- rigid、linear、orthogonal。

* rigid 分解和 indicator 一样， 当 $$s_{i}$$ 和 $$\hat{s}_{i}$$ 相等时， $$s^{+} = s_{i}, s^{-} = 0$$， 其中 $$s^{+}$$ 表示相关部分， $$s^{-}$$ 表示不相关部分。反之则 $$s^{-} = s_{i}, s^{+} = 0 $$    
* linear 分解，该方法计算 $$s_{i}$$ 和 $$\hat{s}_{i}$$ 间的相似度
$$\alpha = \frac{s^{T}_{i}\hat{s}_{i}}{||s_{i}||*||\hat{s}_{i}||}$$，而后有 $$ s^{+}_{i} = \alpha s_{i} $$， $$s^{-}_{i} = (1-\alpha)s_{i}$$    
* orthogonal 分解，正交分解， 将平行的组分分给 $$s^{+}_{i} = \frac{s_{i}*\hat{s}_{i}}{\hat{s}_{i}*\hat{s}_{i}}$$， 垂直的给$$s_{i}^{-} = s_{i} - s^{+}_{i}$$

实验结果显示， rigid 的效果最差，因为它只考虑严格的相等，这个很好理解，至于 linear 和  orthogonal， 结果显示MAP中这两个效果差不多，单 MRR 上 orthogonal好一些，因此论文里最后用的是 orthogonal。

分解完后，将采用深层 CNN 来进行聚合操作得到低维向量表示， 最终通过线性映射和 sigmoid 给出 0-1 之间的相似值。

该论文的亮点是分解操作， 得到相似和不相似的部分，不过这个分解到底能从哪种程度上进行分解论文里没写，有时间研究一下。


