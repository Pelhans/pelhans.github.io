---
layout:     post
title:      "实体链接(三)"
subtitle:   "文本匹配 大礼包"
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

接下来计算 $$\hat{s}_{i} = f_{match}(s_{i}, T)$$, 其中 $$f_{match}$$ 被定义为

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

### BiMPM -- Bilateral Multi-Perspective Matching for Natural Language Sentences

比较经典的双塔式结构， 在使用 BiLSTM 得到语义表示后通过四种匹配策略(Full matching, maxpooling matching, attentive matching, max attentive matching) 得到对应的交互匹配表示，最终再次采用 BiLSTM 的最终时间步输出进行聚合， 通过两个线性层和 softmax 计算相似概率。 模型结构如下图所示

![](/img/in-post/kg_paper/text_match_blstm_bimpm.jpg)

首先 word representation layer 对输入的词进行嵌入，可以用 word2vec 或者 glove。 context 表示层采用 BiLSTM 对句子进行语义编码，其中 对于文章 p 的第 i 个时间步的前后向输出为 $$ h_{i}^{p, for}, ~h_{i}^{p, back} $$, 查询 q 的第 j 个时间步的前后向输出为 $$ h_{j}^{q, for}, ~h_{j}^{q, back} $$.

matching layer 这细说一下， 以 P --> Q 的匹配为例(即 P 中的某一个 $$h_{i}^{p}$$ 对整个 Q 序列计算余弦相似度， Q --> P 同理)。四个匹配方式如下图所示

![](/img/in-post/kg_paper/text_match_blstm_bimpm_matchlayer.jpg)

第一个 Full-matching, 用 P 中 $$h_{i}^{p}$$ 和 Q 的最终时间步输出 $$h_{N}^{q}$$/$$h_{1}^{q}$$ 计算余弦匹配函数。即

$$ m_{i}^{full, for} = f_{m}(h_{i}^{p. for}, h_{N}^{q, for}, W^{1}) $$

$$ m_{i}^{full, back} = f_{m}(h_{i}^{p. back}, h_{1}^{q, back}, W^{2}) $$

$$ f_{m}(v1, v2, W) = cosine(W_{k}*v1, W_{k}*v2) $$

第二个是 maxpooling 匹配，它首先正常计算 $$h_{i}^{p}$$ 与 Q 中每个元素的余弦匹配函数$$f_{m}$$， 最终进行 element-wise maximum 选取每一行最大的元素作为最终向量表示的一部分。

第三个是 attentive 匹配，和注意力机制类似， 先分别计算 $$h_{i}^{p}$$ 与 Q 中每个元素的余弦相似度，并归一化得到对应的权重， 并加权求和得到 Q 的聚合表示 $$\hat{Q} $$, 最终用 $$h_{i}^{p}$$ 与 $$\hat{Q}$$ 计算余弦相似度匹配。

最后一个是 max-attentive 匹配， 上一个不是计算完权重后进行加权求和嘛， 这里变了， 只要权值最大的那个作为 Q 的聚合表示， 之后用该表示和 $$h_{i}^{p}$$ 计算余弦相似度匹配。

对于这四个匹配函数， 作者做了消融实验， 结果表明每个都很有用， 移除后模型下降程度差不多， 因此都保留了。最终四个匹配函数加上前后时间步， 一共 8个向量连接在一起作为该时间步的新表示。

经过上一步 matching 层后， P 和 Q 都有了新的表示， 之后将新的表示输入到 BiLSTM 里进行聚合并计算 softmax 操作就可以了。

除此之外作者还对比了只用 P --> Q 和 只用 Q --> P 的模型， 结果显示模型效果下降了很多， 并且下降效果在 Quora 上是对称的， 但是在 SNLI 上是不对称的， Q --> P 更重要一点， 因此是否采用双向语义匹配是需要根据数据集来定的， 但解释论文里没写，不过如果复杂度可以接受的话还是都用了好， 毕竟效果至少不会下降 = =。

### DecAtt -- A Decomposable Attention Model for Natural Language Inference

很轻量级的一个模型， 作者利用 attention 获得句子的交互表示， 而后利用全连接层和加和池化进行聚合。持此之外， 作者还提出  intra-sentence attention， 即将输入的向量表示进行自对齐(self-aligned)作为新的输入表示，实验结果表明 intra-sentence attention 可以有效提升模型性能。

模型结构如下图所示

![](/img/in-post/kg_paper/text_match_att_decatt.jpg)

具体流程为：

* 首先 输入采用 Glove 得到词向量嵌入 $p_{i}, ~q_{j} $    
* 计算 注意力权重(attention weights) $$e_{ij} = F^{'}(p_{i}, q_{j}) = F(p_{i})^{T}F(q_{j}) $$， 其中 F 表示全连接 + ReLU 做线性和非线性变换    
* 对权值矩阵归一化并加和平均得到新的软对齐表示 

$$ \beta_{i} = \sum_{j=1}^{l_{q}}\frac{exp(e_{ij})}{\sum_{k=1}^{l_{p}}exp(e_{ik})}p_{j} $$

$$ \alpha_{j} = \sum_{i=1}^{l_{p}}\frac{exp(e_{ij})}{\sum_{k=1}^{l_{q}}exp(e_{ik})}q_{i} $$

将原表示和软对齐表示进行连接输入到全连接层进行比较

$$ v_{1, i} = FN([p_{i}, \beta_{i}]) $$

$$ v_{2, j} = FN([q_{j}, \alpha_{j}]) $$

最终通过加和方法进行聚合

$$ v_{1} = \sum_{i=1}^{l_{a}} v_{1, i} $$

$$ v_{2} = \sum_{j=1}^{l_{b}} v_{2, j} $$

剩下的就是用全连接层进行预测了。 除了上面的组块外， 作者提出可以用 Cheng 等人 提出的句内注意力(self-attention) 机制来捕获句子中单词之间的依赖语义表示：

$$ f_{ij} = FN(p_{i})^{T}FN(p_{j}) $$

$$ p_{i}^{'} = \sum_{j=1}^{l_{a}}\frac{exp(f_{ij} + d_{i-j})}{\sum_{k=1}^{l_{a}}exp(f_{ik} + d_{i-k})} $$

距离偏置项 d 为模型提供了最小化的序列信息(这句话没太懂， 感觉是对于那些里的特别远的词， 该项会变大，使得模型能够考虑较远的依赖?)  同时保留了可并行化的特点。 对于所有距离大于 10 的词共享偏置， 也就是距离大于10 的话就都按照 10 算。 实验结果表明， 在embedding 后， 加上句内注意力可以有效提升模型效果。

### ESIM -- Enhanced LSTM for Natural Language Inference

很火的一个模型， 基于Parikh 的 DecAtt 改造得到的模型，作者认为 DecAtt 模型虽然考虑了句子内的对其匹配， 但没有考虑词序和上下文语义信息， 因此作者在匹配前后添加了 BiLSTM 层来获取更好的语义编码，充分利用时序和上下文语义信息。 最终该模型在NLI 任务上取得了很好的效果。不过这里其实带来一个小问题就是原论文 DecAtt 重点打的是快和参数少， 因为 DecAtt 只用了 attention 和 全连接层， 可以并行化处理， 用上 BiLSTM 的话并行化就会麻烦很多， 所以实际使用时可以权衡一下。

模型的结构如下图(左面那个)所示：

![](/img/in-post/kg_paper/text_match_blstm_esim.jpg)

上图左侧可以分为四个部分: 输入编码(input encoding)，局部推断模型(local inference modeling)， 推断组件(inference composition) 以及聚合与相似度计算部分。重点是中间两个， 输入编码采用的是 BiLSTM，前向输出和后向输出连接在一起作为新的语义表示 $$h_{i,p}, ~~ h_{i,q}$$。

局部推断模型首先根据 $$h_{ip}, ~~ h_{iq}$$ 计算匹配矩阵 E， E 中每个元素 $$ e_{ij} = h_{i,p}^{T} + h_{j,q} $$。接下来对相似度矩阵的进行第 j 列相似度进行归一化得到对应的权重， 而后利用这些权重与对应的 $$h_{j, q}$$ 加权求和得到 $$h_{i, p}$$ 的交互表示。是不是很眼熟。。。其实就是 attention 计算， 只不过用矩阵的方法来做， 按照 DecAtt 里的说法这么做时间复杂度会变为线性的(没想懂)。即

$$ \hat{h}_{i, p} = \sum_{j=1}^{l_{q}}\frac{exp(e_{ij})}{\sum_{k=1}^{l_{q}}exp(e_{kj})}h_{j,q} $$

$$ \hat{h}_{j, q} = \sum_{i=1}^{l_{p}}\frac{exp(e_{ij})}{\sum_{k=1}^{l_{p}}exp(e_{kj})}h_{i,p} $$

其中 $l_{p}$ 和 $l_{q}$ 是 p 与 q 的长度 有了交互匹配的表示后， 接下来进一步利用这些表示， 得到  Enhancement of local inference information，也就是构建下面这个新的向量：

$$ m_{p} = [h_{p}, \hat{h}_{p}, h_{p} - \hat{h}_{p}, h_{p}*\hat{h}_{p}] $$

$$ m_{q} = [h_{q}, \hat{h}_{q}, h_{q} - \hat{h}_{q}, h_{q}*\hat{h}_{q}] $$

这样模型就可以利用更高层次(high-order)的交互信息。 再往后这两个 m将会被送入 BiLSTM 再次进行语义编码， 利用 最大或均值池化进行聚合， 利用全连接层， softmax 进行分类。

### A COMPARE-AGGREGATE MODEL FOR MATCHING TEXT SEQUENCES

作者对之前的研究进行了总结， 提出通用的 "比较-聚合框架"(Compare-Aggregate)，明确将模型分为预处理、attention表示、比较、聚合 四个步骤，并且对比了 5 个比较函数， 得出两个比较有效的比较函数。模型结构如下图所示

![](/img/in-post/kg_paper/text_match_att_ca.jpg)

接下来看图说话， 以 Query 为例， 输入 $$ Q = (q_{1}, q_{2}, \dots, q_{Q}) $$, 经过预处理层， 采用只保留输入门的 LSTM 对其进行处理得到新的语义表示，保留输入门可以让网络记住有意义的词的信息。

$$ \hat{Q} = \sigma(W^{i}Q + b^{i}) * tanh(W^{u}Q + b^{u}) $$

$$ \hat{A} = \sigma(W^{i}A + b^{i}) * tanh(W^{u}A + b^{u}) $$

其中 $$W^{i}, ~~W^{u} \in R^{l\times d} $$, $$ b^{i},~b^{u}\in R^{l} $$.  

之后 $$\hat{A}$$ 对 $$\hat{Q}$$ 做 attention，得到 $$\hat{Q}$$ 的新表示 H

$$ H = \hat{Q} * softmax((W^{g}\hat{Q} + b^{g})^{T}\hat{A}) $$

其中 $$H\in R^{l\times A} $$， 接下来进入比较层。作者对比了 5 个比较函数， 其中

* NEIRALNET(NN)：将两个向量连接起来， 用线性层+非线性激活的方式， $$ t_{j} = \hat{a_{j}, h_{j}} ReLU(W[\hat{a}_{j}, h_{j}]^{T + b}) $$    
* NEURALTENSORNET(NTN):  将两个向量看做矩阵， 用矩阵乘法的方式做， $$ t_{j} = f(\hat{a}_{j}, h_{j}) = ReLU(\hat{a}_{j}^{T}T^{[1,\dots,l]}h_{j} + b  ) $$， NN 和 NTN 都没有考虑到语义相似度，因此接下来用了一些相似度度量函数来做这件事     
* EUCLIDEAN+COSINE(EucCos)：将两个向量的余弦相似度和欧几里得距离连接起来， 
$$t_{j} = f(\hat{a}, h_{j}) = [||\hat{a}_{j} - h_{j}||_{2}, cos(\hat{a}_{j}, h_{j})]^{T} $$, 但这又有问题了， 就是一下子都变成标量了， 丢失了很多语义信息    
* SUB、MULT：既然想保留语义信息， 那就用元素级的点乘呗，这样得到的还是一个向量， $$SUB:~~t_{j} = (\hat{a}_{j} - h_{j})*((\hat{a}_{j} - h_{j}))$$,  $$ MULT~~~t_{j} = \hat{a}_{j}*h_{j} $$    

作者总结了一下， 认为 SUB 可以在一定程度上代替 欧几里得距离， 而 MULT 呢和 cos 很像， 因此作者就把 SUB 和 MULT 用 NN 的方式结合在了一起。。。实验结果也表明这个效果最好(其实从结果来看， 单纯的 NN/MULT 效果也没差太多, MULT 在一些任务中还超过了一点， 因此根据情况定用哪个吧)

$$ t_{j} = ReLU(W[(\hat{a}_{j} - h_{j}) * (\hat{a}_{j} - h_{j}), \hat{a}_{j}*h_{j}]^{T} + b ) $$

聚合部分采用 CNN 来做。

### DAM -- Multi-Turn Response Selection for Chatbots with Deep Attention Matching Network

百度 2018 年提出的模型， 作者认为虽然 BiLSTM 能够捕捉序列的前后语义信息， 但代价比较大， 受到 Transformer 的启发， 作者提出了使用两种注意力机制来获取表示和匹配信息的模型。模型的结构模型如下图所以

![](/img/in-post/kg_paper/text_match_dam_over.jpg)

可以看到模型被分为 4 部分， input, representation, matching, Aggregate 这四部分， 还是比较经典的结构的。对于input, embedding 采用 word2vec 得到。

对于 表示部分和匹配部分用到的 attention 组件， 作者使用 attentive module， 它是根据 Transformer 改变得到的， attentive module 的结构如下图所示

![](/img/in-post/kg_paper/text_match_dam_att.jpg)

输入为 query -->  $$ Q = [e_{i}]_{i=0}^{n_{Q}-1}$$,   key --> $$K = [e_{i}]_{i=0}^{n_{K} - 1}$$, value --> $$ V = [e_{i}]_{i=0}^{n_{V}-1}$$ 。

首先根据 Q 和 K 计算 scaled dot-product attention, 之后将其应用到 V 上， 即

$$ Att(Q, K) = [softmax(\frac{Q[i]K^{T}}{\sqrt{d}})]_{i=1}^{n_{Q} - 1} $$

$$ V_{att} = Att(Q, K) * V $$

$$ x_{i} = V_{att}[i] + Q[i] $$

$$ FFN(x) = max(0, xW_{1} + b_{1})W_{2} + b_{2} $$

模块中的 Norm 用的是 layer Norm. 激活函数用的是 ReLU。以上模块被记为 AttentiveModule(Q， K， V)

定义了 AttentiveModule 后， 表示层就是对输入的相应 R 和 多轮句子 U 分别用 该模型， 实验表明 5 层 self-attention 效果最好。

$$ U^{l+1} = AttentiveModule(U_{i}^{l}, U_{i}^{l}, U_{i}^{l}) $$

$$ R^{l+1} = AttentiveModule(R^{l}, R^{l}, R^{l}) $$

matching 部分可以匹配到段落与段落间的关系，包含 self-attention-match 和  cross-attention-match两种

$$ M_{self}^{u_{i}, r, l} = \{U_{i}^{l}[k]^{T} * R^{l}[t] \}_{n_{u_{i}}\times n_{r}} $$

其中 $$U_{i}^{l}[k]$$ 和 $$R^{l}[t]$$ 根据 self AttentiveModule 得到。对于 cross-match 

$$ \hat{U}_{i}^{l} = AttentiveModule(U_{i}^{l}, R^{l}, R^{l}) $$

$$ \hat{R}^{l} = AttentiveModule(R^{l}, U_{i}^{l}, U_{i}^{l}) $$

$$ M_{corss}^{u_{i}, r, l} = \{\hat{U}_{i}^{l}[k]^{T} * \hat{R}^{l}[t]  \}_{n_{u_{i}}\times n_{r}} $$

最终将每个utterance和response中的所有分段匹配度聚合成一个3D的匹配图像Q。 Q再经过一个带有最大池化层的两层3D卷积网络，得到fmatch(c,r)，最后经过一个单层感知机得到匹配分数。

为了证明模型的有效性和必要性， 作者设计了一系列实验， 如 DAM-first 和 DAM-last 是只考虑第一层和最后一层 self-attention，但效果都不如 DAM 整体， 因此证明了使用多颗粒表示的好处。 还有 DAM-self 和 DAM-cross 是只用 self Attention match 和  cross attention match，效果也下降了， 表明选择响应时必须共同考虑文本相关性和相关信息。

对于具有不同平均话语文本长度的上下文， 堆叠式自注意力可以持续提高匹配性能， 这意味着使用多粒度语义表示具有稳定的优势。还有对于 0-10 个单词的部分效果明显不如长的， 这是因为文本越短， 包含的信息就越少。 对于长度超过 30 的长话语， 堆叠式自注意力可以持续提高匹配性能。但所需要的堆叠层数要越多， 以此来捕获内部深层的语义结构。

![](/img/in-post/kg_paper/text_match_dam.jpg)

### HCAN - Bridging the Gap Between Relevance Matching and Semantic Matching for Short Text Similarity Modeling

作者认为文本匹配大体上可分为两种：关联匹配和语义匹配。其中关联匹配更看重字符上的匹配，而语义匹配则更看重实际含义上的匹配。通常来说，针对这两种匹配任务所设计的模型不是通用的，为此作者提出了一个可以同时在两种任务上表现都很好的网络 HCAN(Hybrid Co-Attention Network)。

该网络包含三个部分：混合编码模块，相关性匹配模块和 co-attention 的语义匹配模块。整体的模型结构如下所示：

![](/img/in-post/kg_paper/hcan-global.JPG)

第一层是一个混合编码模块，作者分别尝试了 Deep(堆叠 CNN)、Wide(同一层不同大小卷积核)和 BiLSTM 三种编码方式。这三种编码器代表了不同的权衡，基于 CNN 的更容易并行化处理，同时也允许我们显示的通过控制窗口大小获得不同粒度的短语特征，这在相关性匹配中很重要。同时更深的 CNN 可以通过组合获得更大的感受野得到更高层次和更整体化的特征。而 BiLSTM 的上下文语义编码则更看重整体的语义信息和位置相关信息。

第二部分相关性匹配，首先计算混合编码层输出($$U_{q}$$, $$U_{c}$$)的相关性匹配矩阵

$$ S = U_{q}U_{c}^{T},~~~ S \in R^{n\times m} $$

而后在 context columns 上做 softmax 将其转化为 0-1 之间的相似性分数 $$\tilde{S}$$。接下来对于每个 query 短语 i，分别采用 max 和 avg 池化来获得更显著的特征表示。

$$ Max(S) = [max(\tilde{S}_{1,;}), \dots, max(\tilde{S}_{n,;})] $$

$$ Mean(S) = [mean(\tilde{S}_{1,;}), \dots, mean(\tilde{S}_{n,;})] $$

$$ Max(S),~~Mean(S)\in R^{n} $$

Max 池化可以得到最显著的匹配特征，Avg 特征可以从多个匹配信号中获益，但可能会受到负面信号的干扰。到这里我们就乐意将它们两个连接起来用了。但论文有一个更好的想法，就是针对 IR 等任务来说，我们可以赋予各个 Term 不同的权重，这个权重可以是 IDF 或者其他的什么权重。即

$$ o_{RM} = {wgt(q) \odot Max(S), wgt(q) \odot Mean(S)} $$

$$ o_{RM} \in 2\dot R^{n} $$

第三个是语义匹配，论文里是用 co-attention 来做，即堆叠的 Query-context 和 context-query 的 attention。需要注意的是，语义匹配和第二个是并列的。第一个是 bilinear attention

$$ A = REP(U_{q}W_{q}) + REP(U_{c}W_{c}) + U_{q}W_{b}U_{c}^{T} $$

$$ A = softmax_{col}(A)~$$

$$U_{q}\in R^{n\times F}~~,~~U_{c}\in R^{m\times F}~~,W_{q},W_{c}\in R^{F}~~,~~W_{b}\in R^{F\times F}~~,~~A\in R^{n\times m}~~,~~ $$

REP 是将输入向量扩展到 $$n\times m$$ 的维度。有了相似矩阵后，则有

$$ \tilde{U}_{q} = A^{T}U_{q} $$

$$ \tilde{U}_{c} = REP(max_{col}(A)U_{c}) $$

$$\tilde{U}_{q} \in R^{m\times F}~~,~~ \tilde{U}_{c}\in R^{m\times F} $$

我们就得到了query 和 context  的交互语义表示。接下来用 BiLSTM 对它们的组合进行语义编码：

$$ H = [U_{c}; \tilde{U}_{q}; U_{c}\otimes \tilde{U}_{q}; \tilde{U}_{c}\otimes \tilde{U}_{q}] $$

$$ o_{SM} = BiLSTM(H) $$

$$ H = R^{m\times 4F},~~~o_{SM} \in R^{d} $$

最终结合第二部和第三部的输出放进 MLP + softmax 进行分类

$$ o = softmax(MLP([o_{RM}^{l}; o_{SM}^{l}])) $$

$$ l = 1, 2, \dots, N~~~,o\in R^{num_class} $$

下图是HCAN 与各个模型的对比结果，其中 RM 是只用关联匹配(第二个)，SM 是只用语义匹配(第三个)部分。我们发现，在这三个数据集上，关联匹配（RM）比语义匹配（SM）具有更高的效率。它在TrecQA数据集上以较大的优势击败了其他竞争性基线（InferSent、DecAtt和ESIM），并且仍然可以与TwitterURL和Quora上的基线相媲美。这一发现表明，**对于许多文本相似性建模任务，单靠软项匹配信号是相当有效的**。然而，SM在TrecQA和TwitterURL上的性能要差得多，而在Quora上，SM和RM之间的差距减小了。通过结合SM和RM信号，我们观察到在所有三个数据集中HCAN的一致有效性增益。

![](/img/in-post/kg_paper/hcan-res1.JPG)

下图是比较不同语义编码层的区别，整体来说，当关键字匹配更重要时， CNN 可能获得更好的结果，更看重语义和长距离依赖时， 上下文编码更好。

至于 RM 和 SM，SM 往往需要更大的数据集才能获得较好的表现，因为它的参数空间更大。对于所有任务来说，将二者进行结合都可以得到一定的增强。

![](/img/in-post/kg_paper/hcan-res2.JPG)

### QANet - Extending Neural Question Answering with Linguistic Input Features

论文提出，对于专业领域QA，一个有效的方法是先学习通用领域的，再在专业领域上做适应性训练。而学习通用领域普遍的知识一个很好的途径是利用句法、语义抽象的高层上丰富的语言知识表示。

作者分别利用了三个层次的语言学特征：POS 词性标记、依存句法、语义角色关系三种，其中 

* POS 可以减少特定类型候选答案的数量。用 Spacy 工具获得    
* 依存句法可以精准预测 Span 的边界。也是用 Spacy 工具获得。    
* 语义角色标记对回答类似于“谁”对“谁”、哪里、何时、做了什么这类问题有帮助。

有了它们后，将对应标记做 embedding 而后 concat 起来。具体流程如下图所示

![](/img/in-post/kg_paper/qanet_embed.JPG)


# 总结

看完上面的论文， 大体总结出一个框架， 分为 输入、再编码、交互匹配、聚合、输出 四个部分。

* 输入：除了 DSSM 等一开始用 word hashing 外， 基本上用的都是 word2vec 和 Glove 的各种向量嵌入方法,不过对于 IR 领域词典大的问题， 可以选择适合的方法    
* 再编码：可选的一部分， 前期像DSSM 一类的没有用， 但后面很多工作用了，可选的方式为 self-attention(以及其他自对齐的方法)、BiLSTM, CNN，目的是获取句子内部的前后文语义信息    
* 交互匹配: 基于表示的模型没用这步， 但这基本上都是早期的工作， 今年的工作基本上都考虑了交互匹配这步， 可选的方式如： cros-attention, 通过 cosine/ dot/ indicator/ 欧几里得距离 等构造的匹配矩阵、元素级别的点乘和相减等。    
* 聚合： 交互匹配得到的通常是维度比较高的矩阵， 因此需要采用一定的方式进行降维，得到定长的低维向量表示， 可选方式为： 池化(最大池化，均值池化， 加和等)、线性+激活函数、CNN、BiLSTM 等方式    
* 输出： 有了定长的向量表示后， 可以采用 cosine 和 FN + softmax 等方式计算相似度

以上就是个人总结出来的框架。需要指出的是， 模型这种东西在论文里的效果和在实际使用中的效果是两回事， 生产中不是 stat-of-the-art 就好， 好需要考虑模型的复杂性， 与当前数据的匹配程度，模型的容量等等。大部分时候都需要自己根据数据去做相应的修改，不要迷信模型， 理解数据很重要。
