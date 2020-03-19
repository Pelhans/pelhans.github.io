---
layout:     post
title:      "实体链接论文大礼包"
subtitle:   ""
date:       2019-02-22 00:15:18
author:     "Pelhans"
header-img: "img/kg_bg.jpg"
header-mask: 0.3 
catalog:    true
tags:
    - knowledge graph
---


* TOC
{:toc}

# 概览

实体链接是将文本中提到的实体与其知识库中相应的实体链接起来的任务。潜在的应用包括信息提取、信息检索和知识库填充。但是，同一个实体通常有很多别称，同一名称也可能指向多个实体，此任务具有挑战性。

从方法上看，可分为两段式方法和端到端的方法，其中两段式方法先进行 NER 将文本中的实体识别出来，而后根据该 mention 找到候选实体集，利用实体消岐技术找到对应的候选实体，在两段式方法中，我们主要关注实体消岐部分。在端到端的方法中，则采用一体化模型同时进行实体识别和消岐。

目前 EL 最新的排行榜可以参考: [entity_linking](https://nlpprogress.com/english/entity_linking.html)

![](/img/in-post/entity_linking/el_disambi.PNG)

![](/img/in-post/entity_linking/el_end2end.PNG)

# 论文解读
## 仅实体消岐
### DeepType Multilingual Entity Linking by Neural Type System Evolution

DeepType通过显式地将符号信息集成到具有类型系统的神经网络的推理过程中，该论文认为，如果我们可以确定 mention 的类型，那就可以大大减少候选实体的数量，离最终完成实体消岐也就不远了。

下图是一个实例，比如第一句话的 jaguar，在森林里跑，当你预测出是动物是，那就基本上确定是美洲豹而不是捷豹汽车。

![](/img/in-post/entity_linking/deeptype_example.JPG)

论文为此设计了一个三段式得实体消岐框架：

* 通过启发式和随机优化方法确定类型系统(选定得类型集合)    
* 通过梯度下降训练类型分类器    
* 推断：结合 LinkCount 进行实体消岐

因此该模型的核心是类型系统，首先定义类型系统：

* 关系 Relation：成员间的继承关系，比如 "北京 Instance of City"，其中 Instance of就是 一个关系。    
* 类型 Type：由关系定义的标签， 比如 "张三 isInstance of Human", IsHuman 就是一个类型    
* 类型轴 Type Axis：一堆互斥的类型的集合，如 IsHuman ^ IsPlant = {}    
* 类型系统 Type System：一组类型轴的集合加类型标注函数，比如有一个两轴的类型系统 {IsA, Topic}，用到 Geore Washington 就是 {Person, Politics}    

假设类型系统的离散参数为 A，类型分类器的连续参数 $\theta$。给定文本和对应的 mention集合：

$$ M = (m_{0}, e_{0}^{GT}, \epsilon_{m0}), \dots, (m_{n}, e_{n}^{GT}, \epsilon_{mn}) $$

其中 $$e_{i}^{GT} $$ 是 Ground Truth 的实体， $$\epsilon_{mi}$$ 是候选实体集，令 $$S_{model}(A,\theta)$$ 表示 EL 系统的消岐准确率 。设 $$EntityScore(e,m,D,A,\theta)$$ 表示给定mention m 和文档 D 时模型的得分。因此在给定 mention时，EL 系统给出得分最高的实体为预测结果：

$$ e^{*} = \arg\max_{e\in\epsilon_{m}}EntityScore(e, m, D, A, \theta) $$

如果 $$e^{*} = e^{GT}$$，则实体消岐成功。综合下来，消岐问题可以定义为：

$$ \max_{A}\max_{\theta}S_{model}(A,\theta) = \frac{\sum_{m,e_{GT},\epsilon_{m}}I_{GT}^{(e^{*})}}{|M|} $$

一起优化的话，时间复杂度会很高，因此文章将类型系统和类型分类器分成两个独立的部分进行优化。

对于类型系统，文章采用损失 J(A) 来近似 $$S_{model}(A,\theta) $$。文章假设分类器有两种极端：

* 分类器的上届 Oracle：类型分类器一看就找到 mention 对应的正确类型，得到候选实体集 $$\epsilon_{m, Oracle}$$，而后 LinkCount(m,e) 消岐算法从候选集中选出得分最高的，令：

$$ P(e|m) = \frac{LinkCount(m,e)}{\sum_{j\in\epsilon_{m}}LinkCount(m,j)} $$

表示 LinkCount 的给定 mention 预测实体的概率。则 Oracle 预测的实体为：

$$ Oracle(m) = \arg\max_{e\in\epsilon_{m,Oracle}}P_{entity}(e|m,types(x)) $$

Oracle 的准确率用 $$S_{Oracle}$$ 表示，则

$$ S_{Oracle}  = \frac{\sum_{(m,e_{GT},\epsilon_{m})\in M}I_{eGT}(Oracle(m))}{|M|} $$

* 分类算法的下届 Greedy：不预测type， 直接从候选实体集中选 LinkCount 最高的：

$$ Greedy(m) = \arg\max_{e\in\epsilon_{m}}P(e|m) $$

$$ S_{Greedy} = \frac{\sum_{(m,e_{GT},\epsilon_{m})\in M}I_{eGT}(Greedy(m))}{|M|} $$

有了最好的和最坏的情况，那我们就可以定义近似的目标函数 J：

$$ J(A) = (S_{Oracle} - S_{Greedy})*l(A) + S_{Greedy} - \lambda|A| $$

其中 l(A) 表示分类器的学习能力，用最好的 Oracle 减去 Greedy 表示整个学习的进步空间，再乘以学习能力l(A)表示进步的大小。之后再加上$$S_{Greedy}$$ 表示模型优化后的效果。

除此之外，过多的类型轴可以用来提高表现，但是会更难训练，推断更慢，因此要加一个对 类型轴数量的正则项。于是就有 J 的定义。

从上面公式我们发现这个学习能力很玄学。。。怎么得到呢？作者提出用二元分类器代替多元分类器，通过计算AUC 得到：

$$ l(A) = \frac{\sum_{t\in A}AUC(t)}{|A|} $$

更细节来说，作者用 MLP 进行训练，输入是一个句子，我们取mention 前后 10 个词大小的窗口，将这些词进行连接而后输入到全连接层进行二分类。

经过上面的步骤，A 就得到了，接下来进行类型分类器的优化得到 $\theta$。类型分类器的训练则采用正常的多元分类，网络结构如下图所示(左侧是二元分类，右侧是多元分类)：

![](/img/in-post/entity_linking/deeptype_arc.JPG)

有了类型系统和类型分类器后，实体链接用 LinkCount(通过连接数确定候选实体的概率)进行消岐，因此我们可以得到前面说的 EntityScore：

$$ s_{e,m,D,A,\theta} = P(e|m)(1-\beta + \beta(\prod_{i=1}^{k}(1-\alpha_{i} + \alpha_{i}P(t_{i}|m_{i},D)))) $$

其中 $$\alpha$$ 和 $$\beta$$ 是 0-1 间的平滑系数，$t_{i}$ 是 A 的第 i 个轴。

下图是实体链接在各个数据集上的结果，在ADID CONLL 数据及上，获得了 94.88 的准确率。

![](/img/in-post/entity_linking/deeptype_res.JPG)

### Deep Joint Entity Disambiguation with Local Neural Attention
#### 概览
该模型提出了一个新的深度学习 ED 模型，抛弃了之前基于人工特征的方法，采用神经网络来自动学习特征。模型的关键组件为：

* 实体嵌入 Entity Embedding：知识库的实体 Embedding，从它们的规范实体页面和超链接注释的本地上下文进行实体嵌入。该向量代表是实体的语义，可大大减少人工特征的需求。    
* 上下文注意力 Context Attention：用 attention 机制来获得 context 表征，学习后的基于上下文的实体分数和 mention–entity 先验组合产生最终的本地分数。    
* 全局 消岐：考虑实体间的共现，联合消岐。文档中的 mention 使用带有参数化的势函数的 CRF，将循环信念传播 LBP 作为一个扩展层来学习它。

整个模型的结构如下图所示

![](/img/in-post/entity_linking/deep_joint_ed.png)

#### 实体嵌入

对于实体嵌入部分，为了保证 entity 和 word 处于同一向量空间中(这样计算的向量相似度才有意义)，论文提出一种实体嵌入方法。论文假设生成模型 
$$ p(w|e) $$ 表示实体e 和词 w 的共现概率。为了得到该概率，论文从以下来源进行统计：

* 实体的规范描述页面(如 wiki 页面)    
* 标注语料中，mention-entity 周围窗口内的词(wiki 中的超链接)

对 实体 e 和 词 w 的共现进行统计 $$ #(w,e)$$作为条件概率
$$ p(w|e) $$ 的近似。称这种有共现的词为“正例”词，其他的词为 “负例”词 ，它的概率为 $$q(w)$$。我们优化的目标是使得正例词的向量比负例词的向量离实体向量更近。为此使用 max-margin 作为优化目标，令 
$$w^{+}\~p(w|e)$$， $$w^{-}\~q(w)$$：

$$ h(z,w,v) = [\gamma - <z, x_{w}-x_{v}>]_{+} $$

$$ J(z;e) = E_{w^{+}|e}E_{w^{-}}[h(z;w^{+},w^{-})] $$

$$ x_{e} = \arg\min_{z:||z||=1}J(z;e) $$

其中 $$\gamma > 0$$表示 margin，$$[\dot]_{+}$$ 表示 ReLU 激活函数。

#### 局部消岐

论文认为，**在消岐阶段，只有少数的词对消除歧义有影响,只关注这些单词有助于减少噪音和提高消歧能力**。因此采用 attention 来完成这件事。

设取 mention m周围的 k 个词作为上下文 $$ c = {w_{1},\dots, w_{k}} $$，m 对应的候选实体集为 $$ e\in \Gamma(m)
$$ 。经过 前面的 Embedding 后得到对应的向量 $$ x_{e}$$ 和 $$ x_{w} $$。这样我们就可以得到未归一化得相关性分数：

$$ u(w) = \max_{e\in \Gamma}x_{e}^{T}A x_{w} $$

A 是可学习的参数矩阵，需要注意的是这个分数是对所有候选实体取最大的。论文发现，在计算 attention 时，发现非规范词(停用词那种)带来的噪声，为此论文提出 hard 剪枝，移除 $$ R<= K $$ 排名外的词(相当于 hard attention， 取 top k)。对剩余的 K 个词计算 softmax ，剩余的被剪枝掉的概率赋值 0。因此可以得到 attention 的权值

$$ \beta(w) = \left\{
\begin{aligned}
\frac{exp[u(w)]}{\sum_{v\in \hat{c}}exp(u(v))},~~~ if w\in \hat{c},~~\hat{c} 表示没被剪掉的词 \\
0,~~~ otherwise
\end{aligned}
\right.
$$

} 最终，得到上下文得分

$$ \Psi(e, c) = \sum_{w\in \hat{c}}\beta(w)x_{e}^{T}Bx_{w} $$

将上下文得分和LinkCount 分数
$$ \hat{p}(e|m)$$ 组合经过两层 FFNN +ReLU 得到最终局部消岐模型的候选实体分数：

$$ \Psi(e,m ,c) = F(\Psi(e,c), \log\hat{p}(e,|m)) $$

综上所示，模型的参数只有 矩阵A、B和 FFN 层，为了优化它们，损失函数采用 max-margin loss 来使得正例的得分比负例高。即

$$ \theta^{*} = \arg\min_{\theta}\sum_{D\in \Docs}\sum_{m\in D}\sum_{e\in \Gamma(m)}g(e, m) $$

$$ g(e, m) = [\gamma - \Psi(e^{*}, m, c) + \Psi(e, m ,c)] $$

其中 $$\gamma > 0 $$ 是 margin 参数，$$\Docs$$ 时训练集中的所有文档，我们的目标是优化参数使得正例实体$$e^{*}$$ 比负例实体 e 的得分至少高 $$\gamma$$。

一堆流水账，看的迷迷糊糊的。停下来想想为什么要你那个这一大堆，尤其是弄了 矩阵 A 和 B 那两个式子？整个模型的目的是什么？首先我们要明确论文的核心观点：不是所有的词对消岐都是有用的，剩余的部分就都是噪音，因此 $$\beta$$ 的目的就是一个过滤标准，通过计算句子和所有实体的 attention 分数，并 取top K 来过滤掉得分低的噪声词。过滤方式就是矩阵 B 那步用 $$\beta$$ 乘以 $$x_{w}$$，权重被设为 0 的词向量就被干掉了。我们设乘完的矩阵的 $$ x_{c}$$ ，这样再与所有的实体向量计算相似性矩阵 $$ x_{c}Bx_{e}$$ 就可以在无噪音下筛选真正的实体了。嗯。。。这么一想，妙啊。。。。



#### 全局消岐

这里没看懂。。。。之后再加上再补

最终结果如下图所示，在 AIDA-B 上 F1 为 92.22%

![](/img/in-post/entity_linking/deep_joint_ed_res.png)

### Neural Cross-Lingual Entity Linking

为了比较跨语言的文本线索，我们需要计算跨语言文本片段之间的相似度。本文结合卷积和张量网络，提出了一种从多个角度训练** Query 和候选文档之间细粒度相似性和不相似性**的神经EL模型。

再 DeepType 论文中，我们知道确定 mention 的类型后，很大大降低EL 难度。但万一候选实体中存在类型一致的呢？纯靠 LinkCount ？本论文提出，在这种情况下，考虑 query mention 和 候选实体标题页面的细粒度相似性是很有用的。

#### 向量嵌入

模型用到了很多特征，因此单开一节介绍。

词向量嵌入，用 Word2vec 得到。

Wiki 百科的实体页面嵌入，之前有工作用 CNN 提取整个页面得到固定的向量表示，作者认为这种方法成本太高了，因此本文中，作者提出采用页面中每个词向量和对应的 IDF 加权平均得到页面表示：

$$ e_{page} = \frac{\sum_{w\in p}e_{w}IDF_{w}}{\sum_{i} IDF_{i}} $$

Mention m 的上下文表示(进行指代消解后，所有相同 m 的上下文)：分为细粒度和粗粒度的：

* 句子级别粗粒度表示：用 CNN 将一系列包含 mention 的句子来产生固定大小的向量。

* 细粒度上下文表示：前后窗口大小为 n，实验中为 4，则细粒度的表示为    
$$ NTN(l, r;W) = f([l;r]W[l;r]^{T}) $$   

网络结构如下图所示：

![](/img/in-post/entity_linking/ncle_sen.png)

#### 网络结构

整体的网络结构如下图所示：

![](/img/in-post/entity_linking/ncle_arc.png)

网络结构很简单，就两层神经网络来预测是否可连接。麻烦的是这些个输入，下面挨个介绍：

* A: Similarity Features by comparing Context Representations：    
    * “Sentence context - Wiki Link” Similarity 计算mention的句子上下文和wiki link的相似度。    
    * “Sentence context - Wiki First Paragraph” Similarity 计算mention的句子上下文和wiki 第一段文本的相似度。    
    * “Fine-grained context - Wiki Link” Similarity 计算mention的细粒度的上下文和Wiki link的相似度。    
    * Within-language Features 利用anchor-title计算后验概率
    P（li | m）,即mention m 链接到候选实体li的概率    
* B: Semantic Similarities and Dissimilarities：mention 和 候选实体的相似度     
    * Lexical Decomposition and Composition (LDC)：计算 source context S 和 wiki 页面 T 间的相似性。论文就是那个将输入分解为相似和不相似部分的文本匹配论文。    
    * Multi-perspective Context Matching (MPCM)：网络见下图

![LDC 网络结构](/img/in-post/entity_linking/ncle_ldc.jpg)

![MPCM 网络结构](/img/in-post/entity_linking/ncle_mpm.jpg)

损失函数就是交叉熵损失函数。最终表现如下图所示

![](/img/in-post/entity_linking/ncle_res.png)

## 端到端方法
### End-to-End Neural Entity Linking
第一个神经网络端到端 EL 系统，主要思想是考虑所有可能的 spans 作为潜在的 mention，并学习其候选实体的上下文相似性分数，mention 检测( MD) 任务发掘得发亮信息链接跨度可以为 ED 提供更多的上下文信息。而 ED 也可以为 MD 提供知识库中实体的相关信息，减轻 MD 的边界问题。

关键组件是**上下文感知的 mention 嵌入**、**实体嵌入**和 **mention-entity 的概率映射**。大体流程为：首先生成所有可能的 spans(mention)，其中至少有一个可能的候选实体，而后每一对 mention - candicate 都会收到一个基于单词和实体嵌入的上下文语义感知兼容分数(context-aware compatibility)，再加上一个神经注意力机制和全局投票机制。在训练过程中，我们要求 gold 实体-mention 对的得分高于所有可能的错误候选或无效 mention，从而共同作出 ED 和 MD 决定。

模型结构如下图所示：

![](/img/in-post/entity_linking/endtoend_arc.jpg)

按照顺序从下往上一层层得说，首先是 word 和 char 嵌入，word 嵌入用 word2vec，char 用 BiLSTM 从正向和反向处理 word得到 $$ [h_{L}^{f},h_{1}^{b} ] $$，最终 char 和 word 嵌入向量得到 $$ {v_{k}} $$ 作为整个词的表示。

对于每个可能的 mention，用向量 $$ g^{m} = [x_{q}, x_{r}, \hat{x}^{m}] $$ 表示，其中头两个为该 mention 的起始和末尾向量为 $$x_{q}$$ 和 $$x_{r}$$，$$ \hat{x}^{m} $$ 通过对 mention 内各词做 soft attention 得到，其中 query 是可优化参数 $$w_{\alpha} $$：

$$ \alpha_{k} = < w_{\alpha}, x_{k} $$ 

$$ \alpha_{k}^{m} = \frac{exp(\alpha_{k})}{\sum_{t=q}^{r}exp(\alpha_{t})} $$

$$ \hat{x}^{m} = \sum_{k=q}^{r}\alpha_{k}^{m}*v_{k} $$

$$ x^{m} = FFNN_{1}(g^{m}) $$

不过作者表示 soft-attention 在 mention 较短时作用很小，因此使用需要看情况。最后再加一个非线性全连接层得到 $$ x^{m} $$ 。至此 mention 的表示就得到了。对于实体(知识库中的实体)的嵌入，用条目的描述来训练得到，其中词汇初始化用 word2vec 。(此处可以用 TransE 等更好的方法)。

候选实体的选取通过 mention-entity 映射概率得到，比如 LinkCount，用 
$$ p(e_{j}|m) $$ 表示。将该概率和 mention-实体 间的相似性 $$ <x^{m}, y_{j} $$ 连接起来经过全连接层得到最终的局部分数：

$$ \Phi(e_{j}, m) = FFNN_{2}([log p(e_{j}|m); <x^{m}, y_{j}]) $$

作者提到，有时候长程的上下文 attention 会对消岐有用，因此用 mention 上下文的 attention 向量和 候选实体向量做点积得到相似分数，和上面那俩一起输入到 FFNN2 中。

此时我们已经可以用 $$\Phi$$ 进行消岐了，但当给定的文档较长时，全局消岐的一致性是需要考虑的，为此作者又加了一个全局消岐部分，定义

$$ V_{G}^{m} = {e | (m^{'},e)\in V_{G} and m^{'}\neq m} $$

表示全文中，除了当前 mention 的其他所有满足条件的 mention - entity 对(这里的 entity 是经过$$\Phi$$ 筛选的，因此数量少且比较靠谱)，将所有的 实体向量进行加和后和当前 mention 正在消岐的 entity 向量进行 cos 计算得到相似度：$$ G(e_{j}, m) = cos(y_{e_{j}}, y_{G}^{m}) $$。这一步的意义是联合其他实体，对当前实体进行约束，毕竟同一个文档里大家的圈子都类似。

最终将 $$ \Phi $$ 和 G 连接经过全连接层得到

$$ \Psi(e_{j}, m) = FFNN_{3}([\Phi(e_{j},m); G(e_{j}, m)]) $$

最终我们的优化目标是

$$ \theta^{*} = arg\min_{\theta}\sum_{d\in D}\sum_{m\in M}\sum_{e\in C(m)}V(\Phi_{\theta}(e, m)) + V(\Psi_{\theta}(e,m)) $$

其中 V 要求 正例和负例的线性分离：

$$ V(\Psi(e,m)) = \left\{
    \begin{aligned}
\max(0, \gamma - \Psi(e,m)),~~~if(e, m)\in G \\
\max(0, \Psi(e,m)), ~~~ otherwise
    \end{aligned}
    \right.
$$

}下图是实验结果，我们关注 AIDA A 数据集，可以看到该论文的方法比之前的 baseline 好很多(~17%)，同时还证明附加的 attention 和 全局消岐的有效性。

![](/img/in-post/entity_linking/endtoend_res.jpg)

## Robust Disambiguation of Named Entities in Text

论文提出一种集体消岐方法，该方法充分利用知识库中的各种信息，通过 coherence graph 得到稠密子图，从而消除所有 Mention 的歧义。论文用到了三种特征：

* LinkCount 概率    
* mention 的上下文和实体间的相似性    
* 所有 Mention 的所有候选实体间的一致性

作者认为在短文本中，上细纹信息可能不足以确定链接实体，解决短文本 EL 问题的核心思路是考虑输入文本中其他 Mention 的候选实体的语义一致性。但当文本太短或者这些实体之间的相关性很弱或者每个实体的指向都不明确时，现有的集体消岐法可能会出现一错错一串的情况。

大体流程为，先提取以上三种特征，而后将mention和候选实体作为节点，将权重作为边。mention和entity间的权重来衡量语义相似性。entity间的权重(边)用来衡量共现概率。这种图问题是 NP-hard 的，因此该论文提供了一个贪心算法，提供了高质量的近似。

### 模型框架

首先用斯坦福 NER Tagger 进行分割和实体识别。而后利用维基百科的消岐页面、重定向和连接等生成候选实体。

对于给定的 mention-entity 对，计算上述的三种特征：

* LinkCount：根据链接锚文本生成    
* Mention 的上下文和实体的相似度：    
    * 将 mention 的上下文的词语(停用词和本身外)，作为mention 侧的关键词    
    * entity 侧的实体关键词特征：维基百科文章的链接锚文本，包括类别名称、引文标题和外部引用。所有的这些特征用 KP(e) 表示。对于每个关键词，赋予它们一个权重，权重是实体e和关键词w间的互信息(MI)。MI反映w是否包含在e的关键字组或链接到e的实体的任何关键字组中。N表示实体总数。$$ p(e, w) \frac{w\in (KP(e)\cup U_{e^{'}\in IN_{e}}KP(e^{'}))}{N} $$，其中IN(e)反映w是否包含在e的关键字组或链接到e的实体的任何关键字组中     
    * 由于 entity 内的关键词可能只有一部分出现在 Mention 句子中，因此定义一个新的相似度度量： 实体e的 mention m 的相似性的算法计算文本中e的关键短语的部分匹配匹配单个单词并奖励适当的分数，以此计算相似度。具体来说，计算每个关键字短语的最短单词窗口，该窗口包含关键字短语的最大单词数(cover)。例如，匹配文本“winner of many prices including the Grammy”将导致关键字短语“Grammy award winner”的封面长度为7。    
    * 对于mention m 与候选实体e的相似性，该分数在e的所有关键短语及其在文本中的所有部分匹配上聚合，从而得出相似性分数。    
* 基于语法的相似度：    
    * 除了单词和短语的表面特征外，我们还利用实体 mention 的直接句法上下文的信息。    
    * 利用一个大的文本语料库进行训练，我们收集了哪些类型的实体倾向于作为“游戏”的主题出现的统计数据，然后根据它们与动词的相容性对候选实体进行排序。    
    * 使用（Thater10）框架，它允许我们在句法上下文中导出单词的矢量表示（例如作为特定动词的主语）。    
    * 使用与实体的YAGO类型及其所有超名称关联的WordNet语法集。对于每个替换，我们根据（Thater10）计算一个标准分布向量和一个上下文化向量。然后，基于语法的cxt（e）和上下文cxt（m）之间的相似度被定义为这两个向量之间的标量积相似度之和。如果句法语境化只导致向量的微小变化，反映了实体替代物的相容性，则会导致高度相似。    
* 实体和实体的一致性：利用YAGO的语义类型系统，衡量两个实体间的类型和sibclassOf边距离。通过维基百科文章共享的传入链接数量来量化两个实体之间的一致性。

### 图模型算法

根据流行度、相似性和一致性度量，构造了一个以mention和候选实体为节点的加权无向图。使用相似性度量或流行度和相似性度量的组合对提及实体的边进行加权。权值从held数据习得。实体实体的边是基 于Wikipedia链接重 叠、类型距离或这 些行的某些组合进 行加权的。

给定一个mention实体图，我们的目标是计算一个稠密的子图，该子图理想地包含所有mention节点，并且每个mention只包含一个mention-entity边，从而消除所有mention的歧义。mention-entity 图的例子如下所示

![](/img/in-post/entity_linking/rdne_arc.JPG)

下图是图消岐算法流程：

![](/img/in-post/entity_linking/rdne_alg.JPG)

最终模型在 CONLL 数据集上的表现如下所示，由于论文比较老(2011 年)，因此效果一般。但整篇论文对于特征的应用还是很值得借鉴的。

![](/img/in-post/entity_linking/rdne_res.JPG)
