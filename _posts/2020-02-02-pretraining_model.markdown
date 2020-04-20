---
layout:     post
title:      "预训练模型总结"
subtitle:   ""
date:       2020-02-02 00:15:18
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

想要让计算机处理自然语言，就要把输入的文字用向量进行表示。最开始人们用 one-hot 表示法，但 one-hot 表示法会因为词典过大导致稀疏性问题，同时各个输入之间互相正交，没有办法表示它们之间的语义关系。 对于稀疏性问题，一种方案是像 微软在 DSSM 里那种做 word-hashing 得到输入的 multi-hot 表示，将词典大小由 500k 缩小到 30k。

但 multi-hot 其实也还蛮稀疏的，同时输入间的语义关系也没有解决。直到 word2vec 和 Glove 等词向量表示方法的提出，在一定程度上解决了这两个问题。它们的核心思想具有相似上下文的词与具有相似的语义。具体实现是通过一个简单的神经网络做给定上下文预测当前词和给定当前词预测上下文的任务，以此学习得到输入文字的中间隐层向量表示并作为该词的对应向量表示。这种方法取得了很好的效果，不仅降低了维度(128/256维一般就可以)，还可以获得 “中国 - 北京 = 美国 - 华盛顿” 这种相似关系。因此 word2vec 及其它类似方法 几乎是所有深度模型必备的方法。

但科学的研究显然不满足于此的，word2vec 有什么缺点呢？最明显的一个就是多义词问题，像 play，除了 play game 的 “玩”之外，还有 play music 的“演奏”含义等等。在 word2vec 中却只用一个向量表示了，这就注定会对原始的多语义造成损失。那么解决方法是什么呢？一个目前大家都在走的路就是尽可能的把上下文(单纯的上文或下文也可以)信息包含进来，以此获得当前词更精准的表示，毕竟 play 和 music 放在一起我们就一目了然它的含义了。这条路还需要大量的数据和能吃下这么多数据的大容量网络，这样我们的模型“见多识广”，自然能学习到更精准的语义表示。

这条路上个人认为做的比较有代表性的有 ELMO、GPT1.0、BERT、XLNET 等，除此之外还有像 GPT2.0、ULMFiT、SiATL、MASS、ULINM、百度的 ENRIE1.0、ENRIE2.0、清华的 ENRIE、MTDNN、SpanBERT、RoBERTa、TinyBert、ALBERT、K-BERT 等等。。。目前这个领域还处于飞速发展之中，但整体脉络还是很清晰的。借用一张清华的图:

![](/img/in-post/pretrain_model/PLMfamily.jpg)

# 模型概览
## ELMO - Deep contextualized word representations

ELMO 这个名字来源于芝麻街中的一个角色名，后续的很多模型都顺着这个叫，比如 BERT， ENRIE。

论文认为，在理想情况下，一个好的语言模型 应该同时模拟：

* word 使用的复杂特征（如语法和语义）     
* 这些用法在不同的语言上下文中的变化（即，模拟多义词）

为此，论文引入一个双层的 BiLSTM 模型作为基本模型，令其做自回归生成任务。最终获得的模型中，其底层 LSTM 更看重语法信息(POS 那种)，高层 LSTM 捕获上下文相关的语义信息(适合词义消歧任务)。

ELMO 采用两阶段模式，第一阶段根据自回归任务和大量数据进行训练，得到模型。第二阶段即使用阶段通过对隐层表示和输入进行线性组合得到新的更好的表示作为目标模型的输入。

预训练阶段采用正向 + 反向 LSTM 组合的 biLM 形式，分别获取当前词的上文和下文信息。其中正向 LSTM 通过给定上文预测当前词，反向 LSTM 则是给定下文预测当前词。

给定一个长度为 N 的序列 $$ t_{1}, t_{2}, \dots, t_{N}$$, 用公式表达就是

$$ p(t_{1}, t_{2}, \dots, t_{N}) = \prod_{k=1}^{N}p(t_{k} | t_{1}, t_{2},\dots, t_{k-1}) $$

$$ p(t_{1}, t_{2}, \dots, t_{N}) = \prod_{k=1}^{N}p(t_{k} | t_{k+1}, t_{k+2}, \dots, t_{N}) $$

预训练目标是最大化正向反向似然函数的和

$$ \sum_{k=1}^{N}(log p(t_{k}| t_{1}, \dots, t_{k-1}; \Theta_{x}, \overrightarrow{\Theta}_{LSTM}, \Theta_{s})  + log p(t_{k}| t_{k+1}, \dots, t_{N}; \Theta_{x}, \overleftarrow{\Theta}_{LSTM}, \Theta_{s}) )$$ 

其中 $$\Theta_{x}$$ 是输入的表示， $$\Theta_{s}$$ 是 softmax layer 的表示。

在使用阶段，对于每个输入 $$t_{k}$$，一个 L 层的 biLM 模型能够给出 2L + 1 个表示(正向L + 反向 L + 输入 1)：

$$ R_{k} = \{x_{k}^{LM}, \overrightarrow{h}_{k,j}^{LM}, \overleftarrow{h}_{k,j}^{LM} | j=1, \dots, L  \} = \{h_{k,j}^{LM} | j=0,\dots, L  \} $$

ELMO 通过计算它们的线性组合得到单一的向量表示作为下游任务模型的输入：

$$ ELMo_{k}^{task} = E(R_{k};\Theta^{task}) = \gamma^{task}\sum_{j=0}^{L}s_{j}^{task}h_{k,j}^{LM} $$

其中 $$s^{task}$$ 是经过 softmax 归一化(softmax-normalized)的 权重，$$\gamma^{task}$$ 是标量权重，用来调整 ELMo 最终向量的大小。$$\gamma$$ 是用来帮助优化的经验性的参数，$$ s$$ 则是要学出来的。因为 ELMO给下游提供的是每个单词的特征形式，所以这一类预训练的方法被称为“Feature-based Pre-Training”

一定程度的 dropout 可以提升模型的泛化能力。除此之外，还可以采用 L2 正则项 
$$ \lambda ||w||_{2}^{2} $$。大的 $$\lambda$$ 将使得权重趋同，小的 $$\lambda$$ 使得 同一层的权值多样。实验中发现，对于大多数任务，小的 $$\lambda$$ 效果较好，如下图所示：

![](/img/in-post/pretrain_model/elmo_lambda.JPG)

更细节来说， biLM 是层数为2的 biLSTM，隐藏单元大小为 4096，projections 维度 512， 第一层和第二层之间有残差连接。如果要用 char level 的信息，可以加 CNN 来捕获 char 的信息，即 2048 character n-gram 卷积核 + 2层 Highway 网络。下图是 elmo 的网络结构

![](/img/in-post/pretrain_model/elmo_total.JPG)

网络介绍完了，接下来思考一下细节。 

为什么要用 biLSTM 的输出和输入向量特征表示的线性组合形式作为下游任务的输入？个人理解是，对于一个深度的 lstm 模型，底层网络学习到的是句法方面的低级特征，而高层网络则更倾向于学习词义上的上下文相关特征。具体体现为用第一层 LSTM 做 POS 标记任务效果更好。原始输入的向量表示作为基础词义信息跑不掉的。这样通过不同层的线性组合我们就得到了词义信息 + 句法信息 + 上下文相关的语义信息 的综合体。再通过学习权重对不同任务偏向的需求进行适配，简直完美。

论文通过设计 WSD 和 POS 实验，如下图所示。通过使用 biLM 的不同层来完成 WSD 和 POS 任务，发现底层 LSTM 更适合 POS 类语法相关任务，而高层 LSTM 则更适合语义消岐这种需要上下文的词义表示任务。这两个实验表明，不同层代表不同类型的信息，并解释了为什么包含要用上所有层信息。

![](/img/in-post/pretrain_model/elmo_pos_wsd.JPG)

论文还试验了，使用各层的平均比只是用最后一层强很多，使用各层的线性加权求和比单纯的平均还要好一点。

什么时候适合用 ELMO 呢？ELMo 适合短文本任务，ELMo迁移到下游网络中， 一个是答案较短的数据集，提升有3-4个点， 一个答案较长的数据集，提升只有0.5 左右。

在使用 ELMO 的时候，权重是fixd 的还是 fine-tune 的呢？是固定的，而后通过训练线性组合参数来适应不同下游任务。

ELMO 模型怎么使用呢？文本除了正常的 word embedding 外，还输入到 ELMO 中，这样我们就可以获取各层的输出并进行线性加权求和，并学习更新这些权重。将加权求和得到的向量与 word2vec 得到的向量进行 concat 得到最终的向量表示。

ELMO 就上面一种用法么？就不能加到模型其他地方么？论文也做了尝试，如下图所示，论文将 ELMO 得到的向量除了放到输入上外，还放到像 ESIM 模型的第一层 BiLSTM 层后，实验表明，在特定于任务的体系结构中，在biRNN的输出中包含ELMo可以提高某些任务的总体结果。在SNLI和SQuAD的输入和输出层包括ELMo比仅在输入层改进，但对于SRL（和coreference resolution，未显 示）而言，仅在输入层包括ELMo时性能最高。在SNLI和SQuAD的输入和输出层包括ELMo比仅在输入层改进，但对于SRL（和coreference resolution，未显示）而言，仅在输入层包括ELMo时性能最高。

![](/img/in-post/pretrain_model/elmo_pos.JPG)

最终 ELMO 在各任务上的表现如下图所示：

![](/img/in-post/pretrain_model/elmo_res.JPG)

其实 EMLo 的缺点主要有两个：

* 用 LSTM 而不是 Transformer，Transformer 模型容量大，能吃下更多的数据，非常适合做预训练模型。    
* 虽然 ELMo 意识到双向的重要性，采用正向和反向组合的形式进行弥补，但两个方向的表示间没有充分的交互，还差了点意思。

因此，如何做下一步改进已经呼之欲出了，那就是用双向的 Transformer 来做大数据量下的无监督预训练任务，这就是 BERT 做的，但简单的用 Transformer 做自回归语言模型会有泄密的问题，因此 BERT 把任务换成预测被 MASK 掉的单词，也就是 autoencoding(AE) 任务。

那有没有即用自回归语言模型还用 Transformer 模块的呢？有，那就是 GPT 模型。

## GPT - Improving Language Understanding by Generative Pre-Training

GPT 模型顺着 ELMo 的思路，将 LSTM 替换成了 Transformer 这么做有三个好处

* 可以更好处理长期依赖关系    
* 速度更快    
* 模型容量大

下图左侧是 GPT 模型的整体架构

![](/img/in-post/pretrain_model/gpt_finetune.JPG)

下图右侧是对各种下游任务的使用方式:

![](/img/in-post/pretrain_model/gpt_total.JPG)

模型的整体结构比较简单，就是多层的单向 Transformer，只不过这个 Transformer 是带 mask 的。相当于 《Attention is all you need》 中 Transformer 中的 decoder 层。 加 mask 的目的是可以让模型在做 self-attention 时只关注上文信息。

任务的话， GPT 还是采用自回归任务，优化目标还是最大化对数似然函数

$$ L_{1}(U) = \sum_{i}\log P(u_{i} | u_{i-k}, \dots, u_{i-1}; \Theta) $$

其中 k 是上文窗口大小, P 是给定上文时当前词的条件概率。优化的话用 SGD。

当应用到下游任务中时，假设下游任务 C 的输入是$$ x^{1}, \dots, x^{m} $$，对应的标签是 y，则下游任务的优化目标为

$$ L_{2}(C) = \sum_{x,y}\log P(y | x^{1}, \dots, x^{m}) $$

论文额外指出，在训练下游任务时可以把预训练的任务作为辅助任务做 fine-tuning。这样做有两个好处

* 增强模型的泛化性    
* 加速收敛

即这样下游任务的最终优化目标为

$$ L_{3}(C) = L_{2}(C) + \lambda * L_{1}(C) $$

对下游任务的具体适配的话，如上图右侧所示：

* 分类任务：在文本开始和结尾添加指定符号 "Start" 和 "Extract"，输入 Transformer + linear 即可    
* 文本蕴含：文本蕴含任务包含两个文本，因此用 "Start" + "文本一" + "Delim" + “文本二” + "Extract" 的方式组合，输入 Transformer + linear 得到输出    
* 文本相似性：同样输入包含两个文本，不同的是这两个文本的相似性应该是关于文本对称的，因此通过和文本蕴含相似的组合后，再交换一下，得到两个表示，组合输入 Linear    
* 多选问题： 一个context文本对应多个候选文本情况，context文本和每一个候选文本进行组合后，得到对应组合文本输入到对应的 Transformer + linear中，组合进行分类。

训练数据上，论文使用BooksCorpus数据集来训练语言模型。它包含超过7000本独特的未出版的书籍，来自各种类型，包括冒险，幻想和浪漫。最重要的是，它包含了长距离的连续文本，这使得生成模型能够学习如何处理长距离信息。

模型结构上，用了 12 层 Transformer，隐层维度 768，12个 head，FFN 部分维度是 3072. Adam 优化，最大学习率 2.5e-4。用了 LN， 权值初始化是 N(0, 0.02), GELU 激活。。。值得一提的是，编码采用的是 BPE。应用下游任务时 fine-tuning 细节可以看论文。

实验表明么迁移学习时 GPT 的每一层都对目标任务有帮助。与 LSTM 结构相比， Transformer 能够捕捉更长的语言结构。因此 Transformer 的迁移学习效果更好，更稳定。除此之外，GPT 在微调任务中引入语言模型的损失这个辅助目标可以提升监督任务的性能，实际上辅助目标对大数据集有效，小数据集效果不佳。

## GPT2.0 - Language Models are Unsupervised Multitask Learners

大家想必已经发现了， GPT1.0 介绍的比较粗浅，好处缺点都没说。。。这是因为 GPT2.0 和 GPT1.0 一脉相承，该说的都在这说，并且更重要的是， GPT2.0 更火。。。。

GPT2.0 相比于 GPT1.0，主要改动如下：

* 量更大，领域更丰富的文本数据，WebText 包含 800 万 个链接的文本，大约 40 个 G。这么多网页，几乎你能想到的领域内容几乎都被包含了。    
* 模型更深，GPT1.0 只有12 层，而 GPT2.0 则搞了 48 层，最后还告诉你这还有点欠拟合。。。    
* Layer normalization被移到了sub-block之前    
* 缩放残差层的权重    
* 词表被扩大到50257、context size从512扩大到1024、batchsize使用512    
* 做下游任务时，GPT1.0 是用 fine-tuning 方法，而 GPT2.0 是用无监督的方法去做下游任务。

上面的改动的整体目标就是把网络做的更大更深，喂给它更多更丰富的数据。这么做的动机是论文认为一个足够强大的语言模型在训练过程中已经学习到了各种可能的下游任务需要的东西。比如说，一个网页里包含 “美国总统是特朗普” 这句话，当语言模型根据“美国总统是” 这几个字预测出 “特朗普”时，它就已经相当于做了 问答类的任务。再比如当网页中有 “苹果对应的英语单词是 apple”，那它就等价于 翻译类任务。在数据量足够大的时候，这些都是有可能遇见的。因此下游用无监督的方式去做 zero-shot 任务也是为了证明这个，我这个玩意很强，对各种任务我都悄悄的学到了。

至于更大的模型，这个是真的无敌， 48 层的网络，最后还告诉你这还有点欠拟合，也就是说网络还可以更深，更大。

这里额外记录一下 GPT 用的子词方法 Byte Pair Encoding(BPE)。它通过统计字符对出现的频率，把高频的 char n-gram 当成一个整体输入单位。优点是可以有效地平衡词汇表大小和步数(编码句子所需的token数量)。缺点是基于贪婪和确定的符号替换，不能提供带概率的多个分片结果。

可以参考知乎[深入理解NLP Subword算法：BPE、WordPiece、ULM](https://zhuanlan.zhihu.com/p/86965595)，我这里截取一段

算法流程为

* 准备足够大的训练语料    
* 确定期望的subword词表大小    
* 将单词拆分为字符序列并在末尾添加后缀“ </ w>”，统计单词频率。 本阶段的subword的粒度是字符。 例如，“ low”的频率为5，那么我们将其改写为“ l o w </ w>”：5    
* 统计每一个连续字节对的出现频率，选择最高频者合并成新的subword    
* 重复第4步直到达到第2步设定的subword词表大小或下一个最高频的字节对出现频率为1

停止符"</w>"的意义在于表示subword是词后缀。举例来说："st"字词不加"</w>"可以出现在词首如"st ar"，加了"</w>"表明改字词位于词尾，如"wide st</w>"，二者意义截然不同。

每次合并后词表可能出现3种变化：

* +1，表明加入合并后的新字词，同时原来在2个子词还保留（2个字词不是完全同时连续出现）    
* +0，表明加入合并后的新字词，同时原来2个子词中一个保留，一个被消解（一个字词完全随着另一个字词的出现而紧跟着出现）    
* -1，表明加入合并后的新字词，同时原来2个子词都被消解（2个字词同时连续出现）

实际上，随着合并的次数增加，词表大小通常先增加后减小。

例子：

```
输入：

{'l o w </w>': 5, 'l o w e r </w>': 2, 'n e w e s t </w>': 6, 'w i d e s t </w>': 3}

Iter 1, 最高频连续字节对"e"和"s"出现了6+3=9次，合并成"es"。输出：

{'l o w </w>': 5, 'l o w e r </w>': 2, 'n e w es t </w>': 6, 'w i d es t </w>': 3}

Iter 2, 最高频连续字节对"es"和"t"出现了6+3=9次, 合并成"est"。输出：

{'l o w </w>': 5, 'l o w e r </w>': 2, 'n e w est </w>': 6, 'w i d est </w>': 3}

Iter 3, 以此类推，最高频连续字节对为"est"和"</w>" 输出：

{'l o w </w>': 5, 'l o w e r </w>': 2, 'n e w est</w>': 6, 'w i d est</w>': 3}

……

Iter n, 继续迭代直到达到预设的subword词表大小或下一个最高频的字节对出现频率为1。
```

给定单词序列 

[“the</w>”, “highest</w>”, “mountain</w>”]

假设已有排好序的subword词表 

[“errrr</w>”, “tain</w>”, “moun”, “est</w>”, “high”, “the</w>”, “a</w>”]

则编码结果为

"the</w>" -> ["the</w>"]    
"highest</w>" -> ["high", "est</w>"]    
"mountain</w>" -> ["moun", "tain</w>"]

GPT2.0 在 8 个数据集上与其他模型进行比较，结果在其中 7 个数据集上 GPT2.0 取得了最佳效果。且数据集越小， GPT2.0 的优势就越大。目前 GPT2.0 在阅读理解和写作任务中(尤其是写作，用的人是真的多。。。)表现都特别好。

GPT2.0 为我们打开了一扇门， 只要数据量够大，模型够大，就能持续提升性能，不管你是单向的还是双向的。只不过这个门有点费钱。。。

## BERT - Bidirectional Encoder Representation from Transformers

BERT 是 Bidirectional Encoder Representations from Transformers 的简称。其实从名字也可以看出来，它的主要特点是采用了双向的 Transformer 做 Encoder。而 GPT 用的是单向的 Transformer， ELMO 用的是双向的 LSTM。这里的单向是指网络只可以利用当前词之前的输入，双向是网络可以利用前后文，在 token-level 时，如 SQuAD，双向明显效果会更好一点。

另一方面，BERT 还采用了两种新的自监督任务来训练模型–MLM(Masked Language Model) 和 NSP(Next Sentence Prediction)。MLM 任务类似于完形填空，就是给定前后文预测中间的词。NSP就是给两个句子，判断它们是不是上下文关系。两个任务可以分别捕捉词语级别和句子级别的表示。

模型结构如下图所示

![](/img/in-post/tensorflow/bert_struct.png)

其中最左侧的是 BERT 的网络结构图，最底层的是 embedding 的输入，包含 position embedding、segment embedding、Token embedding 三部分：

* Position embedding: 位置向量，就是位置的 id，而后查表得到，作者没用Transformer 中的那个公式来得到位置向量    
* Segment embedding: 句子向量，如果当前句是第一句，用那么 EA 就是 0，第二句 EB 就是1，再查表    
* Token embedding: 字向量

最终的输入是它们三个相加。如下图所示。其中需要注意的是，每个序列的开头都是 CLS，两句话之间用 SEP 分隔。

![](/img/in-post/tensorflow/bert_embed.png)

组合之后的向量作为网络的输入。隐藏单元由 Transformer block 构成，对于 BASE 版本，包含L 12 层、隐藏单元H大小为 768，multi-head self-attention 的 head 数量为A 为 12，总参数 110M。LARGE 版本层数不变，H隐藏单元大小变为1024，head 数变为 16，总参数为 340M。

BERT 的预训练任务有两个，一个是 MLM，启发于完形填空，它的核心思想是随机 mask 15%的词作为训练样本然后预测被 mask 掉的词。像下面这样：

my dog is hairy –> my dog is [MASK]

但作者又考虑到训练时句子中包含很多的 MASK，这与实际使用中不符，因此：

* 80% 的概率将目标词像之前一样用 [MASK] 替换    
* 10% 的概率用一个随机的词替换目标词： my dog is hairy --> my dog is apple     
* 10% 的概率保持不变：my dog is hairy -->my dog is hairy

Transformer encoder 不知道那个词是要预测的或者被随机替换的，因此模型将强迫模型去关注每一个词。不过 MLM 的收敛速度比left-to-right(预测每一个词的) 要慢一点，但效果却好很多。最终的损失只包含被 mask 掉的。

MLM 任务不同于传统的语言模型类任务，它只预测被 mask掉的词，为什么要这么设计呢？传统的自回归任务是根据上文去预测当前词，而 BERT 同时使用了上下文信息，要是再做预测当前词这种自回归任务的话，模型可以变相看到自己，这就没有效果了。因此采用了这种 mask 的方式。

另一个任务是下一句预测(NSP)任务，这个任务用来关于句与句之间的联系，如 QA或者 NLI 这种。因此 NSP 任务就是预测两句话是不是上下句关系。语料建议从 文档级别的里面抽取句子对，这样可以更好的获取连续长特征的能力。训练语料中，50%的概率是原文中的下下句，50%的概率是随机选取的句子。最终模型在这项任务上得到的准确率是 97%-98%，实验证明该项任务可以明显给QA 和 NLI 类任务带来提升。不过也有后续的论文像 XLNET 说这个任务起得作用不大，相反，除了 RACE 任务之外它还会损害性能。因此后续模型都没有它。

在下游任务的使用上，针对 NLP 中常见的任务分类，BERT 提供了对应的解决方案：

* 句对关系判断，加上一个起始和终止福海，句子之间加分隔符，输出时第一个起始符号[CLS]对应的Transformer编码器后，增加简单的Softmax层，即可用于分类；
* 分类任务：文本分类/情感计算...，增加起始和终结符号，输出部分和句子关系判断任务类似改造；
* 序列标注：分词/POS/NER/语义标注...,输入部分和单句分类是一样的，只需要输出部分Transformer最后一层每个单词对应位置都进行分类即可。
* 生成式任务：机器翻译/文本摘要：问答系统输入文本序列的question和Paragraph，中间加好分隔符，BERT会在输出部分给出答案的开始和结束 span。

在参数的设置上，作者建议大部分超参不用变，至修改 batch size 、learning rate、number of epochs 这仨就够了。：

* batch size: 16，32    
* Learning rate(Adam): 5e-5, 3e-5, 2e-5    
* number of epochs: 3, 4

作者还说，训练数据集越大时，对超参就越不敏感，而且 fine-tuning 一般来说收敛的很快，3 个 epoch 就很棒了。

横向比对一下 BERT 和 GPT 与 ELMO的网络结构。可以看到相比于 GPT，BERT 的个隐藏单元都结合了前后文的信息，而GPT 只利用了当前时间步之前的信息。ELMO 倒是使用了双向的LSTM来结合前后文信息，但只是在输出的时候结合了一下，并不是像 BERT 那样每一层都结合。因此文章反复强调，BERT 的这种深度双向表示模型很重要。这个很好理解，毕竟有很多任务都比较依赖于前后文，在预训练时只关注前文效果一定会差一点。这种 通过 mask 掉一些词而后进行预测的思想可以看做是一种 Denoising AutoEncoder(DAE) 的思路。那些被 mask 掉的单词就是在输入侧加入噪音。类似 BERT 这种预训练模式，被称为 DAE LM。但该方法也带来了一定的缺点，主要在输入侧引入 Mask 标记，导致预训练阶段和Fine-tuning阶段不一致的问题，因为Fine-tuning阶段是看不到[Mask]标记的。

## XLNET - XLNet: Generalized Autoregressive Pretraining for Language Understanding

无监督预训练有两种策略：

* 自回归语言模型(autoregressive language model: AR LM)：通过自回归模型来预测一个序列的生成概率。    
* 自编码语言模型(autoencoding language model: AE LM)：通过自编码模型来从损坏的输入中重建原始数据

自回归语言模型是单向模型，无法对双向上下文进行建模，而下游任务通常需要双向上下文的信息，这使得模型的能力不满足下游任务的需求。

自编码语言模型也有两个主要缺点：

* 在预训练期间引入噪声 [mask]，而下游任务中噪声并不存在，这使得预训练和 fine-tune 之间产生差别

* 自编码语言模型假设序列中个位置处的 [mask] 彼此独立，与实际情况不符。

为了结合 AE 和 AR 的优点，同时避免它们的局限性， XLNET 提出了一种广义自回归语言模型：

* 对于每个 token ，通过概率分解的排列顺序的所有可能组合形式，当前词可以变向的看到“上下文”信息    
* XLNET 通过上述方式避免了引入 MASK 的噪声，因此预训练阶段和 fine-tune 就不存在差异了    
* XLNET 基于乘积规则来分解整个序列的联合概率，消除了自编码语言模型中的预测目标独立。

可以看出，整个模型的重点是变向看到“上下文”这，在论文里叫 Permutation Language Model(PLM)，直译叫 组合语言模型，假设有一个长度为 3 的文本序列 $$ w = (w_{1}, w_{2}, w_{3}) $$，则一共有 3! = 6 种顺序来实现一个有效的自回归概率分解

$$ \begin{aligned}
p(w) &= p(w_{1})\times p(w_{2} | w_{1})\times p(w_{3} | w_{1}, w_{2}) \\
    &= p(w_{2})\times p(w_{3} | w_{2})\times p(w_{1} | w_{2}, w_{3}) \\
    &    \dots \\
\end{aligned}
$$

等价于 对序列 123, 132, 231, 213, 312, 321 的自回归语言模型概率分解。例如 231 序列表示先生成 $w_{2}$ ，在生成 $w_{3}$，最后生成 $w_{1}$。

XLNET 在此基础上考虑了所有可能排列组合的概率分解，其目标函数为

$$ L = \sum_{z\in Z_{t}}\sum_{t=1}^{T}\log p_{\Theta}(w_{z_{t}} | w_{z<t}) $$

其中 $Z_{T}$ 表示长度为 T 的序列的所有可能的排列组合，令 z 是 $Z_{T}$ 的某个排列， $z_{t}$ 为该排列的第 t 个位置编号， $z_{<t}$ 表示该排列中前 t-1 个位置编号， $w_{z<t}$ 表示编号位于 $z_{<t}$ 的token 组成的序列。

但问题来了，对于一个长序列，它的全排序可能性是爆炸的，这将无法计算，因此可以采用期望的形式，并利用采样来计算期望，以此降低复杂度，同时假设每一种排列是等概率的，则每次采样一个顺序 z ，而后根据该顺序进行概率分解。则最终目标函数为

$$ L = E_{z\in Z_{t}}\left[\sum_{t=1}^{T}\log p_{\Theta}(w_{z_{t}}| w_{z<t})  \right] $$

另外，论文还假设如果模型参数在所有分解顺序中共享，那么在期望中，模型将学习从两边所有位置收集信息，即模型能够访问上下文。

需要注意的是， XLNET 仅仅调整了联合概率 p(w) 的分解顺序，并没有调整 w 本身的 token 顺序。因此在 XLNET 中保留了序列 w 的原始顺序，并使用原始顺序的 position embedding。而联合概率 p(w) 的分解顺序调整依赖于  Transformer 的 mask 机制。下图给出当输入序列为 $$ x = (x_{1}, x_{2}, x_{3}, x_{4}) $$ 时，计算 $ x_{3} $ 的集中排列顺序：

![](/img/in-post/pretrain_model/xlnet_factor_order.JPG)

接下来就是怎么实施的问题了，传统的 Transformer 是不行的，比如 我们有两个打乱后的序列， 25143 和 25413 ，当 t = 3 时， 则跟别是依靠 2 和 5 预测 1 和 4。用公式表示为

$$ p_{\theta}(X_{z_{t}} = x | x_{z<t}) = \frac{exp(e(x)^{T}h_{\theta}(x_{z<t}))}{\sum_{x'}exp(e(x')^{T}h_{\theta}(x_{z<t}))} $$

其中 $$ h_{\theta}(x_{z<t}) $$ 表示$$ x_{z<t}$$ 的隐层表示。在上面的例子中因为 t< 3 前的序列是一致的，对 1 和 4 的预测分布是一样的，这不合理。 因此需要在 h 中加入对 $z_{t}$ 的依赖。这样我们有

$$ p_{\theta}(X_{z_{t}} = x | x_{z<t}) = \frac{exp(e(x)^{T}g_{\theta}(x_{z<t}, z_{t}))}{\sum_{x'} exp(e(x')^{T}g_{\theta}(x_{z<t}, z_{t}))} $$

}这样就合理了，那 函数 g 如何定义呢？论文提出  two-stream self attention 机制：

* Content stream: 提供了内容表达 Content representation $$ H_{\theta}(w \leq t) $$ ，记做 $$ H_{zt} $$， 它类似于标准 Transformer 中的隐状态向量，编码了上下文了 $w _{zt} $ 本身。    
* Query stream: 提供了查询表达 query representation $$ G_{\Theta}(w_{z<t}, zt) $$ ，记做 $G_{zt}$ ，它仅编码了 $w_{zt}$ 的上下文及其位置  $z_{t}$，并未编码 $w_{zt}$ 本身。

双路 self-Attention 的更新规则：

* 第一层 query 状态向量通过一个参数向量 w 来初始化 $$ \overrightarrow{g}_{i}^{0} = \overrightarrow{w} $$，参数向量 w 是待学习参数。    
* 第一层的 content 状态向量通过 word embedding 初始化：$$\overrightarrow{h}_{i}^{0} = \overrightarrow{w_{i}} $$    
* 对每一个 self-attention 层， m=1,2...，两路 self-attention 通过共享参数来更新

$$ \overrightarrow{g}_{zt}^{m} \leftarrow Attention(Q = \overrightarrow{g}_{zt}^{m-1}, KV=H_{z_{t-1}}^{m-1}; \Theta) $$

$$ \overrightarrow{h}_{zt}^{m} \leftarrow Attention(Q = \overrightarrow{h}_{zt}^{m-1}, KV=H_{z_{t-1}}^{m-1}; \Theta) $$

其中 Attention 的计算规则与 BERT 的一致。由上述公式可以看到， query-stream 的更新只可能用到 $$z_{<t}$$ 位置的内容信息(不包含它自身)，而 content-stream 状态更新则会用到 $$ z_{\leq t}$$ 位置的内容信息(包含自身)。注意是内容信息。

下图 C 是整体更新的示意图，采样顺序为 3 --> 2 --> 4 --> 1，Attention Masks 第 k 行表示 $w_{k}$ 的上下文。

![](/img/in-post/pretrain_model/xlnet_update.JPG)

先看上图中的 a， 此时在更新 query stream，根据 z 我们知道此时会用到 $g_{1}^{0}$ 和 $h_{2}^{0}$、$h_{3}^{0}$、$h_{4}^{0}$。

上图中 b 是在更新 content stream，根据 z 序列知道此时会用到 $h_{1}^{0}$、  $h_{2}^{0}$、$h_{3}^{0}$、$h_{4}^{0}$。

上图 c 是整体更新 示意图， 重点是 右侧的那个 Attention Masks 矩阵，它是 PLM 实现的基础。对于 content stream 来说，因为能看到自身， 因此根据 z(3 --> 2 --> 4 --> 1):
    
* 1 能看到 1,2,3,4    
* 2 能看到 2，3    
* 3 能看到 3    
* 4 能看到 2，3，4

对于 query stream， 不能看到自身，因此

* 1 能看到 2，3，4    
* 2 能看到 3    
* 3 什么都看到不到    
* 4 能看到 2，3

这样就得到了 mask 矩阵。剩下的根据更新公式进行更新优化就好了。在微调节点，仅仅使用 content stream self-attention，并采用 Transformer-XL 的推断机制(用来解决BERT 类输入文本长度只有  512 的限制)。但 PLM 会带来优化困难：由于位置排列的各种组合导致收敛速度很慢，为此模型仅预测位置排列 z 最后面的 c 个位置对应的 token，拆分点 c 的选择由超参数 K 来选择，定义

$$ \frac{1}{K} = \frac{T-c}{T} $$

也就是要预测原文的 K 分之一个词。这样对于拆分点之前的 token ，模型无需计算它们的 query representation(计算也不参与反向传播)，这会大大节省内存，降低计算代价。部分预测(Partial Prediction) 选择后半部分的原因是：

后半部分的 token 具有较长的上下文，使得上下文信息更丰富，从而为模型提供了更丰富的输入特征，更有利于模型对上下文的特征抽取。(比如 7563142，预测 4时，会用到 75631，这比预测 6 时只能用到 65 的上下文更多了)    ，在做下游任务时，由于上述原因，效果会更好。

现在对比一下 BERT 和 XLNET ，前面说过， BERT 的两个缺点是 MASK 表示在预训练和微调时导致差别，同时学习不到 MASK  token 之间的依赖性。 XLNET 没有 mask 这个很明显了，至于 MASK(XLNET 里是被预测词)得相互依赖性，比如 (3 --> 2 --> 4 --> 1) ，要预测 4 和 1 ，在预测 1 时， 明显会参考 4 的内容，这就捕捉到了依赖性。

XLNET 提供 LARGE 和 BASE 两个版本，结构超参数和 BERT 对应的一致，模型大小类似。数据集方面，加大增加了预训练阶段使用的数据规模；Bert使用的预训练数据是BooksCorpus和英文Wiki数据，大小13G。XLNet除了使用这些数据外，另外引入了Giga5，ClueWeb以及Common Crawl数据，并排掉了其中的一些低质量数据，大小分别是16G,19G和78G。可以看出，在预训练阶段极大扩充了数据规模，并对质量进行了筛选过滤。这个明显走的是GPT2.0的路线。

这里借用张俊林老师的结论

> XLNet综合而言，效果是优于Bert的，尤其是在长文档类型任务，效果提升明显。如果进一步拆解的话，因为对比实验不足，只能做个粗略的结论：预训练数据量的提升，大概带来30%左右的性能提升，其它两个模型因素带来剩余的大约70%的性能提升。当然，这个主要指的是XLNet性能提升比较明显的阅读理解类任务而言。对于其它类型任务，感觉Transformer XL的因素贡献估计不会太大，主要应该是其它两个因素在起作用。

## 清华 ERNIE - Enhanced Language Representation with Informative Entities

前面的预训练模型都是基于大量的文本的。没有考虑知识信息，本质只学到了大量的统计性分布，而不知道语言到底描述了什么。该论文认为知识图谱中的多信息实体（informative entity）可以作为外部知识改善语言表征。

但如何将外部知识融入到模型中又有两大挑战：

* 对于给定的文本，如何高效地抽取并编码对应的知识图谱事实是非常重要的，这些 KG 事实需要能用于语言表征模型。    
* 异质信息融合：语言表征的预训练过程和知识表征过程有很大的不同，它们会产生两个独立的向量空间。因此，如何设计一个特殊的预训练目标，以融合词汇、句法和知识信息就显得非常重要了。

ENRIE 分为抽取知识信息和训练语言模型两部分

* 对于抽取并编码的知识信息，研究者首先识别文本中的命名实体，然后将这些提到的实体与知识图谱中的实体进行匹配。而知识图谱中的图结构通过 知识嵌入算法(如TransE)进行编码，并将多信息实体嵌入作为 ERNIE 的输入。基于文本和知识图谱的对齐，ERNIE 将知识模块的实体表征整合到语义模块的隐藏层中。    
* 语言模型的训练除了包含和 BERT 一样的 MLM 和 NSP 任务外， 还额外加入了 预测被随机 Mask 掉一些对齐了输入文本的命名实体任务，要求模型从知识图谱中选择合适的实体以完成对齐。这样通过同时预测词和实体的新目标， ENRIE 聚合了上下文和知识事实信息，构建了新的知识化的语言表征模型。

模型结构如下图所示:

![](/img/in-post/pretrain_model/tsenrie_arc.JPG)

ERNIE 的整个模型架构由两个堆叠的模块构成(如左图所示)：

* 底层的文本编码器（T-Encoder），负责获取输入 token 的词法和句法信息    
* 上层的知识型编码器（K-Encoder），负责将额外的面向 token 的知识信息整合进来自底层的文本信息，以在一个统一的特征空间中表征实体和 token 的信息。

具体来说， T-Encoder 和 BERT 一样，它将输入的序列 $$ \{x_{1}, x_{2},\dots, x_{n}\} $$ 编码为 编码为 $$ \{w_{1}, w_{2},\dots, w_{n}\} $$

对于 K-Encoder，在获取到 w 和 e后，在第 i 层 aggregator 中，首先做 Multi-head attention:

$$ \{w_{1}^{i}, \dots, w_{n}^{i} \} = MH-ATT({w_{1}^{i-1}, \dots, w_{n}^{i-1}}) $$

$$ {e_{1}^{i}, \dots, e_{m}^{i}} = MH-ATT({e_{1}^{i-1}, \dots, e_{m}^{i-1}}) $$

接下来进入信息融合层来融合 token 和实体序列的信息。对于有对应实体向量的token

$$ h_{j} = \sigma(W_{t}^{i}w_{j}^{i} + W_{e}^{i}e_{k}^{i} + b^{i}) $$

$$ w_{j}^{i} = \sigma(W_{t}^{i}h_{j} + b_{t}^{i}) $$

$$ e_{k}^{i} = \sigma(W_{e}^{i}h_{j} + b_{e}^{i}) $$

其中 $h_{j}$ 是内部信息聚合的隐态，用线性连接的方式进行聚合并加了非线性激活函数。对于没有对应实体嵌入的 token ，它们的更新公式为

$$ h_{j} = \sigma(W_{t}^{i}w_{j}^{i} + b^{i}) $$

$$ w_{j}^{i} = \sigma(W_{t}^{i}h_{j} + b_{t}^{i}) $$

这里我不确定的是这种聚合之后，通过一个非线性操作凭什么就认为拆出来的东西还是实体和 token表示呢？难道是通过 mask 实体预测任务来迫使 e 保持实体表示？

对于 mask 实体预测任务，和  MLM 类似：

* 5% 替换为随机的其他实体：应对匹配错误情况    
* 15% 被 mask 掉：对应实体在图普里找不到情况    
* 80% 不变：学习实体信息

具体使用的话， 对于不同类型的常见 NLP 任务，ERNIE 可以采用类似于 BERT 的精调过程。研究者提出可将第一个 token 的最终输出嵌入（其对应于特有的 [CLS] token）用作特定任务的输入序列的表征。针对某些知识驱动型任务（比如关系分类和实体分型），可以设计出针对性的精调过程。比如实体类型分类任务，可以在实体前后分别添加 [ENT]  标记，修改过的输入序列可以引导 ERNIE 关注将上下文信息与实体提及信息两者结合起来。对于 关系分类任务，通过添加两个标记（mark）token 来凸显实体提及，从而修改输入 token 序列。这些额外的标记 token 的作用类似于传统的关系分类模型（Zeng et al., 2015）中的位置嵌入。然后，也取其 [CLS] token 嵌入以便分类。注意，研究者分别为头部实体和尾部实体设计了不同的 token [HD] 和 [TL]。

最后，研究者针对两种知识驱动型 NLP 任务进行了实验，即实体分型（entity typing）和关系分类。实验结果表明，ERNIE 在知识驱动型任务中效果显著超过当前最佳的 BERT，因此 ERNIE 能完整利用词汇、句法和知识信息的优势。研究者同时在其它一般 NLP 任务中测试 ERNIE，并发现它能获得与 BERT 相媲美的性能。

## 百度 ENRIE1.0 - ERNIE: Enhanced Representation through Knowledge Integration

和上面的模型目的相同， 你 BERT 只 mask 掉了字，但学习不到短语和实体的语义信息，没文化。既然这样，我 直接  mask 一个词、实体、短语呢？这样模型就可以学习到实体级的语义信息。再结合海量的文本，像

"黑龙江的省会是哈尔滨"

这种，哈尔滨是一个实体， 我给它 mask 掉，则有

"黑龙江的省会是[MASK][MASK][MASK]"

我这就学习到了三元组关系关系呀,简直完美。正经来说， ENRIE1.0 采用三种 mask 策略：

* Basic-Level Masking： 跟bert一样对单字进行mask，很难学习到高层次的语义信息；    
* Phrase-Level Masking： 输入仍然是单字级别的，mask连续短语；    
* Entity-Level Masking： 首先进行实体识别，然后将识别出的实体进行mask。

除此之外， ENRIE1.0 还收集了大量的中文语料来更好的建模真实中文语义关系。ERNIE预训练的语料引入了多源数据知识，包括了中文维基百科，百度百科，百度新闻和百度贴吧（可用于对话训练）。

最终实验表明， ENRIE1.0 在所有的 5 个中文数据集上表现都比 BERT 好。

## 百度 ERNIE 2.0: A Continual Pre-Training Framework for Language Understanding

论文认为当前的预训练模型值用几个简单的任务来获取词级或句子的共现信息，没有进一步考虑词法、语法、语义层次的信息。为此ENRIE2.0添加了词级别、结构级别和语义级别的 3 大类预训练任务：

* 词级别： knowledge masking(短语/实体 级别的 mask)、Capitalization Predition(大写预测)、Token-Document Relation Prediction(词是否会出现在文档的其他地方) 三个任务    
* 结构级别： Sentence Reordering(句子排序分类)、Sentence Distance(句子距离分类)    
* 语义级别：Distance Relation(句子语义关系)、IR Relation(句子检索相关性)

但这么多任务又带来一个问题那就是怎么训练？论文指出，要是用 Multi-task Learning 的方式做就会很慢，而用 持续学习的方法也就是一个任务学完再学另一个的方式会忘掉之前的知识。因此论文提出 Sequential Multi-task Learning。如下图左侧所示

![](/img/in-post/pretrain_model/bdenrie_lr.JPG)

这种学习方式是一种逐渐增加任务数量的方式：(task1)->(task1,task2)->(task1,task2,task3)->...->(task1，task2,...,taskN)，最终达到多个任务同步训练。按照这个方式，后续还可以增加其他训练任务，因而整个过程称作continual learning。这点和 MT-DNN 不同， MT-DNN是2+N的模式，也就是先用MLM+NSP这2个任务训练（MT-DNN是直接用了BERT模型的结果），此后接着用其他的任务来接着训练(这部分不再用之前的两个任务)，方式上是用多个任务的mini-batch同步训练。

那任务这么多，难道都用同一个 input 么？论文采用了 Task embedding 的方式，将任务类别加入到 BERT 的 Segment embedding、position embedding、Token embedding 中，以此达到区分不同任务的目的。如下图所示

![](/img/in-post/pretrain_model/bdenrie_input.JPG)

论文中说在很多任务中都有提升，不过没有对每个任务的效果做细致的分析实验，也没有消融实验，改进的具体来源和比例弄不太清，就不细展开了。

## ULMFiT - Universal Language Model Fine-tuning for Text Classification

这个论文发的比较早，不过影响力没 ELMo 大，因此放在了后面。借鉴于图像领域的预训练和迁移技术， 作者在 NLP 领域做了初步尝试，提出了基于微调的通用语言模型（ULMFiT），可以应用于NLP中的多种任务。 实验表明，ULMFiT方法在六个文本分类任务中取得显著效果。该方法包含三个步骤：（1）通用领域的LM预训练，（2）目标任务的LM微调，（3）目标任务的分类器微调。如下图所示：

![](/img/in-post/pretrain_model/ulmfit_arc.JPG)

模型内部用了标准的 AWD-LSTM 网络，在在Wikitext-103数据集上进行语言模型预训练，Wikitext-103含有28595个预处理的Wikipedia文章和1.03亿 个词。

训练好语言模型后，无论用于预训练的一般域数据有多多样化，目标任务的数据很可能来自不同的分布。因此，我们对目标任务数据的LM进行微调。

Fine-tune 分为两步，一个是目标任务的 LM 微调，另一个是目标任务的分类器微调。

在 LM 微调阶段，由于不同的层抓取的信息类型不同，因此微调是应当以不同的尺度进行微调。论文提出的区别性微调策略，就是允许不同层采取不同的学习率。具体来说采用逐层递减的方式设置学习率， $$ \theta^{l-1} = \frac{\theta^{l}}{2.6} $$。这么做的原因是高层跟各个任务的相关性比较大，因此高层需要采用较大的学习率，进行大幅度的调整，而低层通常认为在不同的任务之间可以迁移，因此采用较小的学习率进行调整。这样就一定层度上避免了“灾难性遗忘（catastrophic forgetting）”的问题。除了上述方式，还有倾斜三角学习率微调策略，该策略让学习率先随时间增加，然后在随时间线性下降。如下图所示

![](/img/in-post/pretrain_model/ulmfit_lr.JPG)

在目标任务的分类器微调阶段，可以让模型学习到任务相关的区分性信息。论文在语言模型后又加了两个线性层做分类任务，第一个线性层的输入是LM最后一层隐状态pooling之后的结果，第二个线性层接一个softmax做分类器的输出。论文采用的 pooling 策略是 max 和 mean 一起用，并和 LSTM 最后一个 step 连起来得到。在这个阶段也存在“灾难性遗忘”问题，为此沦为提出了Gradual unfreezing(渐次解冻)的训练策略，即一开始只训练最后一层，将其他的层固定，当训练好后，将倒数第二层再解冻，两层再一起训练，其他层依次类推。

实验证明论文的方法在六个数据集上都达到了STOA的水平。这些数据集有不同的分类任务、不同的数据量、不同的领域，所以论文称提出的方法是通用的方法。在训练数据量比较少的时候，采用ULMFiT预训练方法比不使用预训练的方法有极大的性能提升。当训练数据量比较大的时候，采用预训练和不采用预训练的差别就不大了。 

## SiATL - An Embarrassingly Simple Approach for Transfer Learning from Pretrained Language Models

论文提出了一种简单有效的迁移学习方法来解决灾难性遗忘问题。具体而言，我们将任务特定的优化函数与辅助语言模型目标相结合，在训练过程中对其进行调整。这保留了语言模型捕获的语言规则，同时为解决目标任务提供了足够的适应性。

模型结构如下图所示

![](/img/in-post/pretrain_model/siatl_arc.JPG)

在训练阶段，目的是得到 word-level 的语言模型，网络结构是是 2 个LSTM + 1个线性层。做下游任务时，添加了 LSTM + self-attention ，引入 LM 作为辅助目标，论文证明这对小数据集很有用，整体损失函数如下所示，其中 $$\gamma $$ 是指数衰减的。

$$ L = L_{task} + \gamma L_{LM} $$

除此之外，论文还采用了逐层解冻方法，具体的和上面的论文类似，如果预训练用的无监督数据和任务数据所在领域不同，逐层解冻带来的效果更明显。

## MASS: Masked Sequence to Sequence Pre-training for Language Generation

论文认为 BERT 的 MLM 任务对自然语言里理解任务较友好，为此论文针对生成类任务，提出了一种基于编解码器的语言生成的mask序列到序列预训练模式（MASS）。MASS采用编码器-解码器框架，在给定句子剩余部分的情况下重构句子序列：其编码器以一个随机 mask 序列（多个连续标记）的句子作为输入，其解码器尝试预测被 mask 的序列。通过这种方式，MASS可以联合训练编解码器，以提高表示抽取和语言建模的能力。通过对各种零/低资源语言生成任务（包括神经机器翻译、文本摘要和会话响应生成（3个任务和总共8个数据集））的进一步微调，MASS在不进行预训练或使用其他预训练方法的情况下比基线有了显著的改进。特别是，MASS 在无监督的英法翻译中达到了最先进的准确性（BLEU分数为37.5），甚至超过了早期基于注意的监督模型（Bahdanau等人，2015b）

MASS 的编码解码结构如下图所示

![](/img/in-post/pretrain_model/mass_arc.JPG)

编码器后端一段连续的词被 mask 掉，然后解码器端只预测这几个连续的词，而屏蔽掉其他的词。图中 "_"表示 mask 标记(被屏蔽掉的词)。

设 x 表示输入序列， u 表示 mask 开始的位置， v 表示 mask 结束的位置, k = v - u + 1 表示被屏蔽序列的长度。被屏蔽的词用 [M] 代替。$$ x^{u:v} $$ 表示被 mask 的 连续序列，$$ x^{\u:v} $$ 表示剩余的序列。整体任务就是根据输入的序列 $$x^{\u:v} $$ 预测 $$x^{u:v} $$，因此损失函数为

$$ L(\theta;\chi) = \frac{1}{|\chi|}\sum_{x\in\chi}\log P(x^{u:v} | x^{\u:v};\theta) $$

当 k = 1 时，根据 MASS 的设定，编码器一端屏蔽一个单词，解码器端预测一个单词。此时 MASS 和 BERT 等价。

当 k = 序列长度时，编码器屏蔽所有单词，和 GPT 等价。如下图所示

![](/img/in-post/pretrain_model/mass_sim.JPG)

至于 MASK 策略，试验表明 50% 的效果最好。最下有任务测试中，MASS达到了机器翻译的新SOTA，并在不同的序列到序列自然语言生成任务中，MASS均取得了非常不错的效果。

## UNILM - Unified Language Model Pre-training for Natural Language Understanding and Generation

上面的模型为了解决 BERT 在生成方面的不足做了一些改进，本篇论文也出自同样的目的，只不过是从 MASK 矩阵的角度来解决。具体来说， BERT 在 fine-tune 时支持输入两个句子 S1  [SEP] S2，论文将这个看做 encoder-decoder 结构。S2 根据 S1 的输入生成对应的句子。模型结构如下图所示

![](/img/in-post/pretrain_model/unilm_arc.JPG)

我们关注图右侧：

* 当我们想训练双向语言模型时， S1 和 S2 能都看到全部的输入。    
* 训练单向语言模型时(从左到右)，S1 和 S2 都智能看到左侧的输入。预测过程采用 MLM 的思路，只计算 MASK 的 词带来的损失，这是为了和 BERT 的保持一致。    
* 训练 seq-to-seq 语言模型时，S1 能看到自己全部，S2 能看到S1 全部和自己左侧的输入。输入两句，第一句采用 BiLM 的方式，第二句采用单向 LM 的方式。同时训练 encoder(BiLM) 和 decoder(Uni-LM)，处理输入时同样也是随机 mask 掉一些 token。

以上这些都通过 Mask 矩阵完成， Transformer 的结构没有其他变化。加Mask的频率和BERT一样，但是对于加Mask，80%的时间随机mask一个，20%时间会mask一个bigram或trigram，增加模型的预测能力。训练时，在一个batch里，优化目标的分配是1/3的时间采用BiLM和Next sentence，1/3的时间采用Seq2Seq LM，1/6的时间分别给从左到右和从右到左的LM。

## SpanBERT: Improving Pre-training by Representing and Predicting Spans

论文认为 BERT 只盖住一个 subword 的方式没有考虑子词之间更高层次上的关联性。比如一个正常的单词被拆成了多个子词，或者一个实体词组只被 mask 掉一个字。

基于以上原因,改论文采用盖住一个连续的序列这种方式，具体来说，根据几何分布，先随机选择一段（span）的长度，之后再根据均匀分布随机选择这一段的起始位置，最后按照长度遮盖。文中使用几何分布取 p=0.2，最大长度只能是 10，利用此方案获得平均采样长度分布。通过采样，平均被遮盖长度是 3.8 个词的长度。在消融实验中，作者发现除了指代消解任务外，Random Span 方法普遍更优。至于指代消解的问题，可以用Span Boundary Objective(SBO) 任务解决。

SBO 在训练时取 Span 前后边界的两个词(mask 外的)，而后用这两个词向量和 Span 中被遮盖掉的词的位置向量拼接起来，来预测原词。最终损失包含原 MLM 和 SBO 带来的损失。至于 NSP 任务，则没有用，论文认为：

* 相比起两句拼接，一句长句，模型可以获得更长上下文（类似 XLNet 的一部分效果）    
* 在 NSP 的负例情况下，基于另一个文档的句子来预测词，会给 MLM 任务带来很大噪音。

实验表明，SpanBERT 普遍强于 BERT，在抽取式问答上表现优异(受益于 SBO) 任务。去掉 NSP 任务的一段长句训练普遍要比原始 BERT 两段拼接的方式要好。

## RoBERTa: A Robustly Optimized BERT Pretraining Approach

虽然后续的许多任务都宣称可以超越 BERT ，但实际上 BERT 并没有被充分的训练，本论文通过精细的调参和增大数据集后，证明了 BERT 训练不足的事实。为此提出了并改进了训练 BERT 模型的方法，它可以达到或超过所有 BERT 后续方法的性能。

论文的主要贡献包括

* 对模型进行更长时间、更大 batch、更多数据的训练    
* 删除 NSP 任务    
* 使用更长的序列进行训练    
* 使用动态 MASK

动态  MASK VS 静态 MASK：原始 BERT 的 MASK 是在预处理时做的，所以在之后的每个 epoch 中我们将会面对相同的实例。而动态 MASK 则是在向模型提供序列时做 MASK，，这样每次模型都能看到不同的数据。当对更多步骤或更大的数据集进行预训练时，这一点变得至关重要。实验表明动态 MASK 至少能和静态 MASK 相当，甚至还好一点。

移除NSP 任务：论文分别尝试了 SEGMENT-PAIR+NSP、SENTENCE-PAIR+NSP、FILL-SENTENCE、DOC-SENTENCE 四种设置，发现使用单独的句子确实会影响下游任务的性能，但假设这是因为该模型无法学习长期依赖关系。同时相比于无 NSP 损失的训练和 doc-sentence 的比较，发现消除 NSP 达到或略微提高了下游任务性能。最后，论文发现将序列限制为来自单个文档(doc-sentence)的性能，略好于打包来自多个文档 (全句) 的序列。

大 batch 训练：论文比较了 BERTBASE 在增大 batch size 时的复杂性和最终任务性能，控制了通过训练数据的次数。发现large batches 训练提高了 masked language modeling 目标的困惑度，以及最终任务的准确性。最终 batch 定位 8k(真有钱)。

# 模型压缩
## ALBERT: A LITE BERT FOR SELF-SUPERVISED LEARNING OF LANGUAGE REPRESENTATIONS

虽然 BERT 模型本身是很有效的，但参数太多了。本论文引入一种瘦身版的 BERT 模型 ALBERT，提出2中降低内存消耗和提高训练速度的参数减少策略。

具体来说，第一个是 Factorized embedding parameterization，论文认为原始 BERT 中，词向量 E 和隐层维度 H 是相等的。但实际上词向量维度不用那么大，因此论文提出 (V * E + E* H)。其中 V 是词表大小，而原来是 V * H，这一步能减少一些参数，不过不多，模型表现还下来了(0.6)，因此得从隐层内的参数入手。

第二个是 Cross-layer Parameter Sharing，作者分别尝试了 全不共享、全共享、只共享 self-attention、只共享 FFN。结果如下图所示

![](/img/in-post/pretrain_model/albert_share.JPG)

结果表明， 全部共享时，参数量只有原来的1/7左右，不过代价是性能下降了。而 只共享 self-attention 时，效果在 E = 128 时反而更好了，参数也能少一些，但论文没有考虑这种情况，**至于为什么对只共享  self-attention 时，E 变小它更好了，有点想不懂**。当采用全部共享时，性能下降了，弥补办法就是加大模型规模。论文提出了 ALBERT-xxlarge(12层)，但 H 变成 4096，真正的扁而宽，整体参数少了 30%，但变慢了。ALBERT 用 70% 的参数量，在同样的 1M steps 上确实超越了 BERT。相应地，训练时间也拉长到 3.17 倍，至于屠榜的结果，则是 1.5 M steps 的结果，又多出一半。。。落下了贫穷的眼泪

第三个 Sentence Order Prediction(SOP) 任务，论文认为原版的 NSP 任务太简单了，因此 SOP 对其加强，将负样本换成了同一篇文章中的两个逆序的句子。除此之外，原始的 BERT 是先用 128 个 toekn 训练90% ，后用 512 的训练，但之前的研究表明，单一的长句子对预训练很重要，因此 ALBERT 在 90% 时用满 512 的，10 % 随机找短于 512 的 segment。


# 模型分析
## How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings

BERT 和 GPT 等预训练模型的一个主要贡献是证明带有上下文信息的词表示比传统 word2vec 得到的静态词向量对下游任务带来的提升更大。但这个上下文化的信息表示到底和上下文关系多大？是每一个不同的上下文都有一个对应的词表示么？若是多个的话，它们之间差距会很大么？改论文对此进行研究，并给出了一些结论：

* 在BERT、ELMo和GPT-2的所有层中，所有的词它们在嵌入空间中占据一个狭窄的锥，而不是分布在整个区域。    
* 在这三种模型中，**上层比下层产生更多特定于上下文的表示**，然而，这些模型对单词的上下文环境非常不同。**GPT-2是最具特定上下文化的, ELMo中相同句子中的单词之间的相似性最高，而GPT-2中几乎不存在，其最后一层中的表示几乎是与上下文相关程度最高的**。在BERT中，同一句话的上层单词之间的相似性更大，但平均而言，它们之间的相似性比两个随机单词之间的相似性更大。相比之下，对于GPT-2，同一句话中的单词表示彼此之间的相似性并不比随机抽样的单词更大。这表明，**BERT和GPT-2的上下文化比ELMo的更微妙，因为它似乎认识到，出现在相同上下文中的单词不一定有相同的意思**。    
* 如果一个单词的上下文化表示根本不是上下文化的，那么我们可以期望100%的差别可以通过静态嵌入来解释，其中静态嵌入是第一个主成分。相反，我们发现，平均而言，只有不到5%的差别可以用静态嵌入来解释。这表明，**BERT、ELMo和GPT-2并不是简单地为每个词意义分配一个嵌入**：否则，可解释的变化比例会高得多。在许多静态嵌入基准上，**BERT的低层上下文化表示的主成分表现优于GloVe和FastText**。这可以有一个应用，即如果我们通过简单地使用上下文化表示的第一个主成分为每个单词创建一种新的静态嵌入类型，通过对 BERT 底层表示的使用，可以在涉及语义相似、类比求解和概念分类的基准测试任务上胜过 Glove 和 FastText。(5-10% 个点)    

## Are Sixteen Heads Really Better than One?

MultiHead-Attention 是 Transformer 的基石，通过令不同的头关注不同的输入片段，使得模型关注输入的不同可能部分，从而使 attention  超出普通的加权平均的方式，表达更复杂的函数成为可能。但该论文却发现，并不是所有的 Head 都同样重要，在实践中，有很大一部分可以在测试时移除，而不会显著影响性能。甚至有些层可以只保留一个 head 。下图是测试再不同层只保留 1 个主要 head 时性能的表现

![](/img/in-post/pretrain_model/remove_heads.JPG)

可以看出， 对于大部分层，影响都不大，只有 enc-dec 那掉了 13.56 个点。因此论文认为， encoder-decoder 的 attention 比 self-attention 更依赖于 multi-head 机制。不过没给出进一步的解释，我个人猜测有没有可能是在 encoder-decoder 时，对齐矩阵不那么一致，也就是说这一次我还只关注自身周围的，下一次就可能来一个长程依赖。而 self-attention 时，关注的点更稳定些，我就只关注上下文或者长程的某些结构依赖。因此 enc-dec 更需要多头来满足不同的可能性。只是猜测，待证实。

作者还采用 MNLI 和 MTNT 任务来试验了一下，表明大部分 Head 都不重要这个结论的普适性。除此之外，作者又分析了一下这个情况是什么时候产生的，经实验发现，在前几个 epoch 时，每个头都很重要，但当 Epoch 大于10时，就出现了一些不那么重要的头 head 了。据此论文提出一个假设，即模型在整个训练过程中可以分为两个阶段：

* 经验风险最小化阶段：此时最大化与标签之间的中间表示的互信息    
* 压缩阶段：与输入的互信息最小化

嗯。。。没看懂，作者说给未来研究。。。

# Ref

* [清华 PLM 必读论文 ](https://github.com/thunlp/PLMpapers)    
* Unsupervised Pretraining for Sequence to Sequence Learning. Prajit Ramachandran, Peter J. Liu, Quoc V. Le. EMNLP 2017.     
* Deep contextualized word representations. Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee and Luke Zettlemoyer. NAACL 2018.     
* Universal Language Model Fine-tuning for Text Classification. Jeremy Howard and Sebastian Ruder. ACL 2018.    
* Improving Language Understanding by Generative Pre-Training. Alec Radford, Karthik Narasimhan, Tim Salimans and Ilya Sutskever.     
* BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova. NAACL 2019    
* Language Models are Unsupervised Multitask Learners. Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei and Ilya Sutskever. Preprint.    
* ERNIE: Enhanced Language Representation with Informative Entities. Zhengyan Zhang, Xu Han, Zhiyuan Liu, Xin Jiang, Maosong Sun and Qun Liu. ACL 2019.    
* ERNIE: Enhanced Representation through Knowledge Integration. Yu Sun, Shuohuan Wang, Yukun Li, Shikun Feng, Xuyi Chen, Han Zhang, Xin Tian, Danxiang Zhu, Hao Tian and Hua Wu.     
* Cross-lingual Language Model Pretraining. Guillaume Lample, Alexis Conneau. NeurIPS 2019.    
* Multi-Task Deep Neural Networks for Natural Language Understanding. Xiaodong Liu, Pengcheng He, Weizhu Chen, Jianfeng Gao. ACL 2019.     
* MASS: Masked Sequence to Sequence Pre-training for Language Generation. Kaitao Song, Xu Tan, Tao Qin, Jianfeng Lu, Tie-Yan Liu. ICML 2019    
* Unified Language Model Pre-training for Natural Language Understanding and Generation. Li Dong, Nan Yang, Wenhui Wang, Furu Wei, Xiaodong Liu, Yu Wang, Jianfeng Gao, Ming Zhou, Hsiao-Wuen Hon    
* XLNet: Generalized Autoregressive Pretraining for Language Understanding. Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, Quoc V. Le. NeurIPS 2019    
* RoBERTa: A Robustly Optimized BERT Pretraining Approach. Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov    
* SpanBERT: Improving Pre-training by Representing and Predicting Spans. Mandar Joshi, Danqi Chen, Yinhan Liu, Daniel S. Weld, Luke Zettlemoyer, Omer Levy    
* K-BERT: Enabling Language Representation with Knowledge Graph. Weijie Liu, Peng Zhou, Zhe Zhao, Zhiruo Wang, Qi Ju, Haotang Deng, Ping Wang    
* ERNIE 2.0: A Continual Pre-training Framework for Language Understanding. Yu Sun, Shuohuan Wang, Yukun Li, Shikun Feng, Hao Tian, Hua Wu, Haifeng Wang    
* Pre-Training with Whole Word Masking for Chinese BERT. Yiming Cui, Wanxiang Che, Ting Liu, Bing Qin, Ziqing Yang, Shijin Wang, Guoping Hu. Preprint(Chinese-BERT-wwm)    
* ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators. Kevin Clark, Minh-Thang Luong, Quoc V. Le, Christopher D. Manning.    
* Pre-training Tasks for Embedding-based Large-scale Retrieval. Wei-Cheng Chang, Felix X. Yu, Yin-Wen Chang, Yiming Yang, Sanjiv Kumar. ICLR 2020    
* ALBERT: A Lite BERT for Self-supervised Learning of Language Representations. Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut. ICLR 2020.     
* Q-BERT: Hessian Based Ultra Low Precision Quantization of BERT. Sheng Shen, Zhen Dong, Jiayu Ye, Linjian Ma, Zhewei Yao, Amir Gholami, Michael W. Mahoney, Kurt Keutzer
* TinyBERT: Distilling BERT for Natural Language Understanding. Xiaoqi Jiao, Yichun Yin, Lifeng Shang, Xin Jiang, Xiao Chen, Linlin Li, Fang Wang, Qun Liu. Preprint.     
* How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings. Kawin Ethayarajh. EMNLP 2019    
* Are Sixteen Heads Really Better than One?. Paul Michel, Omer Levy, Graham Neubig.    
* Patient Knowledge Distillation for BERT Model Compression
