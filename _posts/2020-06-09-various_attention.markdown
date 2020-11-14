---
layout:     post
title:      "Attention 及 Transformer 变体总结"
subtitle:   ""
date:       2020-07-09 00:15:18
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

Transformer 近年来在各种任务中大放异彩，什么 QA、文本匹配、翻译任务都不在话下。2018 年 GPT、 BERT 和后续的预训练模型的出现，又带动了新一轮的热潮。Transformer 由于内部 attention 的特性，相比于 RNN，可以处理更长期的依赖关系。但可惜的是，self-attention 的时间复杂度是序列长度的二次方，对于超长的文本就不行了，像 BERT 这种的最大长度被限制在了 512。因此主要的改进点就是降低 Transformer 的复杂度，毕竟 $$O(N^{2})$$ 太高了。同时也需要让 Transformer 可以处理更长的序列，像信息检索领域的 doc、文本摘要、案情笔录等文本都很长，512 肯定是不够的

接下来将按照这个点进行展开，介绍进来研究的进展。

# 标准 Transformer

这里标准 Transformer 以 [Attention Is All You Need](http://cn.arxiv.org/abs/1706.03762) 中的 Encoder 部分为准。

关于标准的 Transformer 介绍，请看我之前的一片博文[Transformer 介绍](http://pelhans.com/2019/04/26/deep_learning-note10/#transformer)

还有 Transformer 的代码讲解 [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)

# 改进 Transformer
## TransformerXL: Attentive Language Models Beyond a Fixed-Length Context

Transformer XL 通过引入递归机制和相对位置编码来解决上述问题。

* 1）能够学到的最长依赖性的长度：Transformer XL 比 RNN 长 80%，比Transformer 长 450%。    
* 2）推断速度：Transformer XL 比 Transformer 快 1800 多倍。

对于超长的文本，Transformer 是很乏力的，毕竟复杂度与序列长度的二次方成正比，那对于超长的文本传统的 Transformer 是怎么办的呢？一个可行的方案(如下图所示)是将整个文本切割成固定大小的、较短的片段（segment），然后用每个片段来训练模型，而忽略之前所有片段的信息。但这就会带来一个问题，即模型能够捕获的最大依赖长度不能超过 segment 大小，同时划分 segment 时没有考虑句子的边界信息，破坏了语义完整性，导致泛化能力不好，效果比较差，这个问题称作上下文碎片化（context fragmentation）。

![](/img/in-post/various_attention/tm_xl_exam.png)

为了解决上下文碎片化的问题，你不是没有前文么，一个直接的想法就是引入 cache，缓存前一个 segment 计算得到的隐状态序列，然后在下一个 segment 中使用它。这也是 Transformer XL 的思想，所谓 XL 是 extra long 的简称，也就是说，通过这种方法，可以处理更长的序列。示意图如下图所示，可以看到，相比于普通 Transformer，在处理 segment 2 时，会缓存 segment 1 的信息，同时固定住 segment 1 不用更新梯度。这样就在降低计算量的同时，解决了上下文碎片化的影响。

![](/img/in-post/various_attention/tm_xl_exam2.png)

上下文碎片化的问题解决了，但这又引入了另一个问题：如何保持位置信息的一致性？用公式来表述这个问题明显一点。令 segment 长度为 n

* 第 $$\tau$$ 个 segment 的输入 token 序列 $$x_{\tau} = (x_{\tau, 1}, x_{\tau, 2},\dots, x_{\tau, n})$$，对应的 embedding 矩阵$$ E_{x_{\tau}}$$    
* 第 $$\tau + 1$$ 个segment 的输入 token 序列 $$x_{\tau+1} = (x_{\tau+1, 1}, x_{\tau+1, 2},\dots, x_{\tau+1, n})$$，对应的 embedding 矩阵 $$E_{x_{\tau+1}}$$    
* Position embedding 的矩阵为 P_{1:n}$$

则

$$ h_{\tau} = f(H_{\tau-1}, E_{\tau} + P_{1:n}) $$

$$ h_{\tau+1} = f(H_{\tau}, E_{\tau+1} + P_{1:n}) $$

可以看到，虽然是两个 segment ，但都采用了相同的位置编码，也就是说，模型不知道一个 token 在两个 Segment 里有什么区别，为此 Transformer XL 引入了**相对位置编码**。

具体来说，假设位置 j 相对于位置 i 的距离为 b = j - i $$，则位置 j 相对于位置 i 的相对 position embedding 为：

$$ \overrightarrow{r}_{b} = (r_{b,1}, r_{b,2}, \dots, r_{b, d}) $$

$$ r_{b, 2j} = sin(\frac{b}{10000^{2j/d}}),~~~~r_{b, 2j+1} = sin(\frac{b}{10000^{(2j+1)/d}}) $$

其中 j 和 i 是两个  segment 的位置索引。Transformer 的绝对位置编码在输入 transformer 之前就和token emb求和，相对位置编码需要在计算attention score 时加入和计算。

最终在 wikiText-103 、text8、enwik8 等数据集上的表现都比标准 Transformer 要好。

![](/img/in-post/various_attention/tm_xl_res.png)

在 WikiText-103 数据集上验证 segment-level 递归、相对位置编码的作用，验证结果如下：

![](/img/in-post/various_attention/tm_xl_res2.png)

可以看出，相对位置编码非常重要对于 Transformer XL 非常重要，而绝对位置编码只有在 Half Loss 中工作良好。这是因为 Half Loss 只考虑当前 segment 后一半位置的损失，因此受到前一个 segment 的绝对位置编码的影响比较小。随着 attention length 的增加，模型的困惑度下降。

下图对 Transformer 和 Transformer XL 的 evaluation 速度进行评估，可以看到，由于状态重用方案，Transformer XL在评估期间实现了高达1874倍的加速。

![](/img/in-post/various_attention/tm_xl_res3.png)

## Universal Transformer

Universal Transformer 是 Transformer 的推广 ，它是一个时间并行的递归自注意力序列模型，解决了上述两个问题。

* 通过引入了深度上递归，因此在内存充足的条件下，它是图灵完备的    
* 通过引入自适应处理时间 Adaptive Computation Time(ACT) ，它使得不同 token 的迭代时间步不同。

用人话来讲，之前 Transformer 的层数都是固定的，比如 Encoder 是 6 层，现在引入深度上的递归，利用 ACT 对深度上的循环次数进行控制，使得不同的 token 可以有不同的循环深度，需要多层才能有好的表达的就多走几层，其他的只用几层就够了。

详细来说，这个 递归机制如下图所示，其中比如输入是 $$h_{1}^{t}$$，正常的 Transformer 就是经过 self-attention + ADD_AND_LN +  FFN + ADD_AND_LN 就进入下一层了，但这里不是，经过 self-attnetion 后还要进入一个叫做  Transition function 的模块，这个模块是共享权重的，具体形式上来说， Transition function 可以使之前一样的 FFN，也可以是 可分离卷积层。

![](/img/in-post/various_attention/univ_arc.png)

ACT 可以调整计算步数，加 入 ACT 机制的 Universal transformer被称为Adaptive universal transformer。要注意的细节是，每个position的ACT是独立的，如果一个position a在t时刻被停止了， $$h_{a}^{T}$$ 会被一直复制到最后一个position停止，当然也会设置一个最大 step，避免死循环。

下图给出UT 的计算，在动态停止的UT 的每一步中，我 们都会得到停止 的概率、余数、 到那时为止的更 新次数、以前的 状态（全部初始 化为零），以及0 到1之间的标量阈 值（超参数）。 然后我们计算每 个位置的新状 态，并基于每个 位置的状态计算 每个位置的新停 止概率。然 后，UT决定停止 一些超过阈值的 位置，并更新其 他位置的状态， 直到模型停止所 有位置或达到预 定义的最大步数

![](/img/in-post/various_attention/univ_ut.png)

通过以上改进， Universal Transformer 在问答、语言模型、翻译任务上都有更好的效果。下图给出了简单的对比，可以看到，Universal 的 比简单的 Transformer 要强，加了 ACT 后 Universal 悔更好一些。

![](/img/in-post/various_attention/univ_res.png)

## Star-Transformer 
论文认为 Transformer 计算复杂度是序列长度的平方，太高。同时 Transformer 也比较吃数据量，当数据比较少时，Transformer 表现得往往的没那么好。 那么 Transformer 为什么那么迟数据呢？文章认为，这是由于 Transformer 的设计缺乏先验知识导致的。当开始训练 Transformer 时就需要从零开始，从而增加了学习成本。因此在改动 Transformer 时加入一些任务需要的先验知识可以减轻这种情况。

基于以上原因，文章提出，通过将完全连接的拓扑结构移动到星形结构中来节省体系结构。改进的网络结构如下图右侧所示，对比于左侧的传统 attnetion ，内部的连接数量明显减少了。

![](/img/in-post/kg_paper/ner_star_trans.JPG)

具体来说，星形 Transformer 有两种连接方式。中间的节点叫根节点，周围的节点叫卫星节点，卫星节点实际就是一个一个 timestep 的输入，卫星节点到根节点的连接叫直接连接(Radical connections)，直接连接保留了全局的(non-local) 信息，消除冗余连接，全局信息可以通过根节点自由流通。卫星节点之间的相互连接叫环连接(Ring connections)，环连接提供了局部成分的先验性。这么设计的好处是1. 可以降低模型的复杂度从 $O(n^{2}d)$到 $O(6nd)$。2. 环连接可以减轻之前无偏学习负担，提高泛化能力。

Star-Transformer 的训练分为两个步骤：1：卫星节点更新；2：中继节点的更新；整个更新的流程图如下所示

![](/img/in-post/kg_paper/ner_star_update.JPG)

首先将输入文本序列嵌入得到 $$ E = [e_{1}, e_{2}, \dots,e_{n}]$$，用 E 去初始化卫星节点 $$ H = [h_{1}^{0}, \dots, h_{n}^{0}] $$。而后利用 E 得均值去初始化根节点 S。之后执行T轮更新，即卫星节点更新和根节点，更新公式图中所示。

模型在 CoNLL2003 与 CoNLL2012 数据集上相比于传统的 Transformer 有较大的提升。

![](/img/in-post/kg_paper/ner_star_result.JPG)

## Generating Long Sequences with Sparse Transformers
论文中将 full attention 进行分解，通过多个 sparse attention 来代替，在不牺牲性能的情况下降低复杂度至 $$O(N\sqrt(N))$$，如下图所示：

![](/img/in-post/various_attention/sparse_tm_exam.png)

上图最上面那行表示输入是一个 6x6 的图片，此时两个 head 在计算输出时的收到的输入，对于 Transformer 来说（图 a），做生成任务时，它接受当前时刻之前的全部序列作为输入，因此最终的 attention 矩阵就像图 a 显示的那样是个下三角形状。

图 b) 是论文提出的一个改进点，称为 Strided Attention，每一个位置通过 attend 其对应的行和列来获取信息。具体来说，从图 b 上面的左侧小图看，其中一个 head 在当前时刻关注它所在行的前面那几个，右侧小图显示另一个 head 关注当前时刻所在列的之前时刻输入。这两个 head 一组合就得到下面的 attention 关注范围。 这种方式主要应用于图像或者音频这种依赖信息比较密集且周期性比较明显的任务。

细看的话，其实图b 上面左侧的 head 目标就是获取 local 信息，右侧类似于膨胀卷积网络那种，通过稀疏化的方案达到低复杂度的 global 信息。

图 c) 是论文的另一个改进点，称为Fixed Attention，这种方式主要应用于像文本之类没有周期性的数据，首先将文本分成固定长度的块，然后第一个 head 处理该块中该位置之前的所有元素，第二个head 处理每个块的最后一部分的固定大小的部分，以此捕获跨块的信息，类似于 Transformer XL。

最终实验显示可以使用数百个层来模拟数万个时间步长的序列。论文使用相同的体系结构从原始字节对图像、音频和文本进行建模，为Enwik8、CIFAR-10和ImageNet-64的密度建模开创了新的技术水平。证明了全局一致性和巨大的多样性，并证明了在原则上可以对长度为一百万或更多的模型序列使用 self-attention。

这两个改进方案的思想很有用，后续有很多任务跟进它，比如 Longformer、Bigbird。

##  Explicit Sparse Transformer: Concentrated Attention Through Explicit Selection
self-attention 能够建模长期依赖，但它可能会受到上下文中无关信息的影响。为了解决这个问题，论文提出了一个新的模型，称为显式稀疏 Transformer。显式稀疏 Transformer 能够通过显式选择最相关的片段来提高对全局上下文的关注度。在一系列自然语言处理和计算机视觉任务（包括神经机器翻译、图像字幕和语言建模）上的大量实验结果都证明了显式稀疏 Transformer 在模型性能上的优势。论文还表明，论文提出的稀疏 attention 方法比先前的稀疏 attention 方法取得了可比或更好的结果，但显著减少了训练和测试时间。例如，在Transformer模型中，推理速度是sparsemax的两倍。

下图是普通 attention（蓝色）的权值分布，可以看到，虽然大部分注意力都集中在相关的那几个上，但由于是对全局做的，整个 attention 的注意被分散开了，导致真正相关的 token  权值与其他的 token 差距不大。作者认为这就相当于其他的 token 作为噪声干扰了对 heart 的关注。为此作者提出显示稀疏 attention (橙色)，它相当于一个 hard 的 attention，截取了文本的 k 个 attention 得分最大的元素，从而移除无关信息，流程为：

* 沿用 vanilla Transformer 的 attention 计算公式得到 attention score P    
* 假定分值越大的元素其相关性越，计算 Masking 矩阵，找出 P 中每行的 k 个最大元素，记录其位置，并得到一个 阈值向量    
* 将 Masking 矩阵应用到原始矩阵上，将 P 中得分大于 t 的保留不变，其余设置为负无穷，之后再计算 softmax 啥的

上面的 k 是超参，论文建议在 5 - 10 之间选，最终实验上 k = 8 的效果比较好。

![](/img/in-post/various_attention/sparse_att_exam.png)

对Sparse Transformer进行了一系列定性分析。分析表明，所提出的稀疏化方法可以应用于模型的任何部分，从而提高了模型的性能。与vanilla Transformer 相比， Sparse Transformer 具有更sharp 的 attention 和更好的效果。在翻译任务上的表现如下图所示，可以看出比 Transformer 要强一些：

![](/img/in-post/various_attention/sparse_att_res.png)

## Reformer The Efficient Transformer

还是解决 Transformer 复杂度 高，占用内存大的问题 论文提出两个核心改进点：局 部哈希敏感(LSH)和 可逆 Transformer：

* LSH：计算Q和K的点乘 就是为了找到Q和K相似 的部分，我们没有必要把 Q中的每个向量都跟K进 行相乘，我们可以只计算 相近的那部分，毕竟 sfotmax 后是这部分主 导。因此需要找到相近的 部分。解决办法是 LSH， 通过特定类型的哈希映 射，使得在向量空间里相 近的两个向量，经过hash 函数后，他们依然是相近 的。首先我们可以看到对 于每一句话先用LSH来对 每个块进行分桶，将相似 的部分放在同一个桶里 面。然后我们将每一个桶 并行化后分别计算其中的 点乘。这就将复杂度从 $$O(L^{2})$$降到 $$O(L logL)$$    
* 可逆Transformer：借鉴RevNet 思想，对残差连接进行修改，引入可逆层。可逆层将输入分成两部分，使得每一层的值可以由它下一层的输出推导出来。因此整个网络只需要存储最后一层的值即可。

先介绍 LSH，这么做的理由是我们只对 softmax（QK）感兴趣，毕竟我们只想知道它们间的相关性。并且实际上对于一个长度为 64K 的数据来说，相关的 key 可能只有32 或者 64 个这么点。所以如果能高效的找到这些相近的 keys，事情就好办了。

局部敏感哈希（LSH）可以解决在高维空间中快速找到最近邻的问题。将每个向量x分配给哈希函数h（x）的散列方案称为局部敏感的，如果相邻向量以高概率获得相同的哈希值，而远处的向量则没有。在我们的例子中，我们实际上只要求附近的向量以高概率获得相同的散列，而散列桶的大小与概率相似。形象点说，如下面这张图所示，有两个点 x, y，经过旋转(映射)后，距离远得点(第一                             行)有很大概率分到不同得桶中，而距离近得点(第二                           行)很大概率分到相同得桶中。

![](/img/in-post/various_attention/reformer_exam.png)

具体实现上，如果要得到 b 个哈希桶，那么我们可以使用一个随机产生的大小为 $$[d_{k}, b/2]$$ 的随机矩阵 R，定义哈希映射函数

$$ h(x) = argmax([xR;-XR]) $$

这样所有的 x，都可以把它们分配到 b 个哈希桶里。这个方法叫做 LSH schema。有了这个东西之后，我们将它用到 atention 中，如下图所示。

![](/img/in-post/various_attention/reformer_att.png)

上图中左侧是 LSH Attention 的流程图：

* 首先我们可以看到对于每一句话先用LSH来对每个块进行分桶    
* 将相似的部分放在同一个桶里面。    
* 将每一个桶并行化后分别计算其中的点乘。

这就将复杂度从$$O(L^{2})$$降到 $$O(L logL)$$。右侧a 和 b 是正常attention 的图，从 a 可以看到这种稀疏性。b的话， k和q根据它们的哈希桶（注意随机的那个矩阵R是共享的）排序好，然后再使用。

一次哈希可能出现分桶不均的情况，这可以用不同的哈希函数进行多轮哈希来进行减轻。

除了 LSH Attention 之外，论文还引入可逆 Transformer，它借鉴了 Gomez 等人提出的可逆残差网络 RevNets，研究表明，它们可以代替resnet进行图像分类。其主要思想是只使用模型参数，允许从下一层的激活中恢复任何给定层的激活值。当反向传播从网络输出到其输入时，可以逐个反推，而不必为反向传播过程中使用的中间值设置存储点，这就减少了消耗。

对于正常的 Transformer，我们有：

$$ Y_{1} = X_{1} + Attention(X_{2}) $$

$$ Y_{2} = X_{2} + FeedForward(Y_{1}) $$

其中 $$X_{1}$$ 和 $$X_{2}$$ 是两个输入，数据流如下图所示：

![](/img/in-post/various_attention/reformer_pos.png)

![](/img/in-post/various_attention/reformer_rev.png)

那么我们就可以由输出反推出输入：

$$ X_{1} = Y_{1} - Attention(X_{2}) $$

$$ X_{2} = Y_{2} - FeedForward(Y_{1}) $$

这样我们只存储最后一个就可以了，前面的中间节点都可以反推，而不需要存储在内存中。

最终 Reformer 的结果如下图所示，取得了比较好的效果

![](/img/in-post/various_attention/reformer_res.png)

## Longformer: The Long-Document Transformer

很好的一篇论文，基于 Transformer 的模型无法处理长序列，还是 self-attention 得锅，与序列长度成二次函数。为了解决这一局限性，我们引入了Longformer，该机制具有一个随序列长度线性伸缩的注意机制，使得处理数千个或更长 token 的文档变得更加容易。

Longformer的注意力机制是 self-attention 的替代品，它用局部滑动窗口来捕捉上下文信息（包含n-gram 那种上下文和膨胀卷积的那种来扩大接收范围），用任务相关的全局 attention 来捕捉全局信息（对于分类任务，用 CLS 标记，对于QA 类的，就在整个问句上计算 global attention）。论文还评估了Longformer在字符级语言建模上的性能，并在text8和enwik8上取得了最新的结果。与之前的大多数工作相比，论文还对Longformer进行了预训练，并对其进行了各种下游任务的微调。我们经过预先训练的Longformer在长文档任务上一直优于RoBERTa，并在WikiHop和TriviaQA上创造了最新的结果。下面详细的说一下。

对于长文档 Transformer，目前有两种主流的方法，一种是 LTR（left to right） 那种，它将文档分块从左向右移动。虽然这类模型在自回归语言建模中取得了成功，但它们不适用于具有双向语境的任务的迁移学习方法。这个论文属于另一种方法，它定义了某种形式的稀疏self-attention 模式，并且避免了计算完全二次注意力矩阵乘法。下图给出之前的各种对 Transformer 的改进工作总结：

![](/img/in-post/various_attention/longformer_sum.png)

值得关注的有两个，首先是 attention 矩阵的形式，像前面介绍的  Transformer XL 是属于 LTR 的，而前面的 Reformer 通过 LSH attention 实现 attention 的稀疏化，因此算是 sparse，BP-Transformer 中的 BP 指的是 Binary partitioning，即二分。在 BP-Transformer 中首先将一整个序列通过二分手段构建为一颗二叉树，从而实现在计算 attention 时的低复杂度。

论文提出了3种 attention 模式(pattern):sliding window 和 dilated sliding window、global + sliding window，如下图所示

![](/img/in-post/various_attention/longformer_arc.png)

先说 sliding window，它围绕每一个 token 采用固定大小的窗口计算 local attention，相当于赋予一个 local 的先验，使用这种窗口化注意力的多个堆叠层会产生一个大的感受野（和CNN 的感觉很像），其中顶层可以访问所有输入位置，并有能力构建包含整个输入信息的表示。如果窗口大小是 w，序列长度是 n ，那么这种 attention 的复杂度就是 $$ O(nxw)$$。为了使这种注意模式有效，w应该比n小（要不然复杂度又变成序列长度二次方了）。多层 Transformer 的模型将具有较大的接受场，如果模型有 l 层，那么低 l 层的感受野就是 lxw（假设各层 w 不变）。在实践中，不同层用不同的上下文感受范围是有益处的。

第二个是 Dilated sliding window(膨胀卷积滑动窗口)，膨胀卷积我们都知道，它可以帮助 CNN 在较少层数和参数的情况下获得更大的感受野。这里也是这个想法，假设 window 间的 gap 是 （不是膨胀率）d，窗口大小 w，层数为 l，则 第 l 层的感受野变成 lxdxw。我们看到，即使是很小的 d，也可以带来很大的感受野提升。具体实践中，我们发现每个head 具有不同膨胀参数的设置。通过允许一些没有膨胀的头部专注于局部环境，而其他具有膨胀功能的头部则专注于较长的上下文，从而提高了性能。

在论文的实验中，窗口化和膨胀注意力不够灵活，无法学习特定任务的表征。因此，作者在几个预先选定的输入位置添加“global attention”。使这个注意操作是对称的：也就是说，具有全局注意力的token 与整个序列中的所有token 发生关系，序列中的所有token 都和它发生关系它。上图表示出在自定义位置处的几个token 处具有global attention 的滑动窗口attention的示例。例如，对于分类，全局注意力用于[CLS]，而在QA中，对所有问题token 提供全局attention。虽然指定全局注意力是特定于任务的，但它比现有的特定于任务的方法简单得多，即将输入分块/缩短为更小的序列，并且通常使用复杂的体系结构来跨这些块组合信息。此外，它增加了模型的表现力，因为它允许在整个序列中构建上下文表示。

最后，作者用TVM构建了自己的自定义CUDA kernel。这使得longformer的速度与full self-attention的操作几乎一样快，而且显存占用要小得多，这个论文有开源代码，值得膜拜一下。

![](/img/in-post/various_attention/longformer_mem.png)

实验结果如下图所示，在 text8 和 enwik8 数据集上，大模型和小模型与其他模型的对比在小的模型上，long former比其他模型都要好在大的模型上，比18层的transformerxl要好，跟第二个和第三个齐平

![](/img/in-post/various_attention/longformer_res1.png)

递增窗口大小的表现最好，递减窗口大小表现最差，使用固定窗口的表现介于两者之间。不用膨胀比对2个头膨胀效果差一点

![](/img/in-post/various_attention/longformer_res1.png)

## Synthesizer Rethinking Self-Attention in Transformer Models

self-attention 我们都知道，在 Transformer 中， Q 中的每个 token 和所有 Key 的 token 进行运算， 计算 $$QK^{T}$$ ， 但这种 token 对 token 相乘得到相似度的操作对 attention 来讲是必须的么？该论文对此进行了质疑，并提出了几种“脑洞大开”的 attention 矩阵计算方案：Dense 和 random 以及它们的低秩分解形式 最终结果显示，这些“脑洞大开”的版本并没有比传统的 Transformer 差什么，在某些任务上甚至更好。同时发现：1）random 比对传统方法表现出惊人的竞争性；（2）从token-token (query-key)交互中学习注意权值并不重要。

先写一下标准 self-attention 的形式，先计算权值矩阵而后计算 softmax 得到 A：

$$ A = softmax(B),~~~ B = \frac{XW_{q}W_{k}^{T}X^{T}}{\sqrt{d_{k}}} $$

第一种被称为 Dense 的结构，既然不要 token 对 token 这种形式，那是否可以考虑移除掉 query 或者 key 中的一个呢？我们的输入 X 的维度是 $$ l x d$$，其中 l 是输入序列的长度，则 B 的维度应该是 $$ l x l$$。如果只用 query 或者 key 中的一个的话，那么我们可以让 X 乘以一个 $$ d x l $$ 的矩阵 W 就可以了，即 $$B = XW $$。当然这样有点太简单了，因此论文中实际采用的是：

$$ B = W(ReLU(W(X) + b)) + b $$ 

这个还是可以理解的形式，毕竟之前也有人用多层感知机形式的计算 attention。但下面的这个 random 的就。。。刚刚不是说 B 的 shape 是 $$l x l $$ 的嘛，那我们可不可以直接初始化一个 shape 为 $$ lxl $$ 的 random 矩阵呢？论文就这么做了，并称之为 Random 结构。

模型结构如下图所示：

![](/img/in-post/various_attention/synth_arc.png)

除了上面两个之外，论文还讨论了低秩分解的问题，想通过低秩分解来降低参数量，对于 Dense 和 Random 的对应低秩分解形式论文称之为 Factorized Dense 和 Factorized Random。

对于 Dense 结构，低秩分解的方式就是生成两个 $$ l x a $$ 和 $$ l x b $$ 的矩阵 $$B_{1} $$ 和 $$B_{2}$$，其中  $$a*b=l$$，而后将  $$B_{1}$$ 重复 b次， $$B_{2}$$ 重复 a 次，得到对应的  $$ lxl$$ 矩阵 $$\tilde{B}_{1}, ~~\tilde{B}_{2}$$，然后逐位相乘，得到 $$lxl$$矩阵 B。对于 Random 的低秩形式，是去两个 $$ lxk$$ 的矩阵 $$R_{1}$$ 和 $$R_{2}$$，令它们相乘即可得到 $$ b = R_{1}R_{2}^{T}$$

论文分别再翻译、摘要。对话任务上进行了评测。翻译任务上的结果如下图所示：

![](/img/in-post/various_attention/synth_res1.png)

在 EnDe 上，除了 Fixed random 比较差之外，剩下的竟然都差不多。现在个人还没理解为什么会这样，如果能看到内部 attention 的分布就好了。

在摘要任务上，标准 attention 效果比较好，QA 任务上标准 Transformer 是最差的。

![](/img/in-post/various_attention/synth_res2.png)

在预训练模型上论文也做了实验，不过在这里 Dense 和 Random就不大行了。

![](/img/in-post/various_attention/synth_res3.png)

坐着对比了训练 5 万步后的参数分布，对比的是 Encoder 层的 1,3,5,6 层，得出它们的参数分布大致相同的结论。

![](/img/in-post/various_attention/synth_res4.png)

这个论文值得细度，等有空再好好看看。

## Deformer：Decomposing Pre-trained Transformers for Faster Question Answering

对于 BERT 这种模型，我们知道不同的层模型学到了不同的东西，一般认为底层的模块学习语法信息，如 POS 这种相对 local 的信息，而高层部分则会学习上下文的语义信息，因此论文认为，至少在模型的某些部分，“文档编码能够不依赖于问题”的假设是成立的。 具体来说可以在 Transformer 开始的低层分别对问题和文档各自编码，然后再在高层部分拼接问题和文档的表征进行交互编码，详细来说，提出Transformer模型的一种变形计算方式（称作 DeFormer）：在前k层对文档编码离线计算得到第k 层表征，问题的第k层表征通过实时计算，然后拼接问题和文档的表征输入到后面 k+1 到 n 层。除此之外，添加两个蒸馏损失项，来最小化 DeFormer 的高层表征和分类层 logits 与原始 BERT 模型的差异。模型结果如下图右侧所示：

![](/img/in-post/various_attention/deformer_arc.png)

在3个 QA 任务上，BERT 和 XLNet 采用 Deformer 分解后，取得了 2.7-3.5 倍的加速，节省内存65.8-72.0%，效果损失只有0.6-1.8%。BERT-base（[公式]）在SQuAD上，设置[公式]能加快推理3.2倍，节省内存70%。

![](/img/in-post/various_attention/deformer_res.png)

消融实验证明,添加的两个蒸馏损失项能起到弥补精度损失的效果。

![](/img/in-post/various_attention/deformer_res2.png)

这个论文很有意思，值得一试。

## Transformers are RNNs Fast Autoregressive Transformers with Linear Attention 
利用快速幂的思想，引入核函数，认为导致 self-attention 复杂度为序列长度二次级的原因是  softmax，要是没有  softmax，就可以先计算 K^T V，再与 Q 乘，这样复杂度就可以降到线性，因此论文采用了这个方案，并提出配套激活函数（核）来做。

论文的核心思想还是比较简单的，我们先看一眼标准 self-attention 计算：

$$ Q = XW_{Q}, ~~ K = XW_{K},~~~V=XW_{V}$$

$$ A_{l}(x) = V^{'} = softmax(\frac{QK^{T}}{\sqrt{D}})V $$

写成通用的形式：

$$ V^{'}_{i} = \frac{\sum_{j=1}^{N}sim(Q_{i}, K_{j})V_{j}}{\sum_{j=1}^{N}sim(Q_{i}, K_{j})} $$

当 $$ sim(q,k) = exp(\frac{q^{T}k}{\sqrt{D}}) $$ 时，上面两个是等价的。

既然认为 softmax 是瓶颈，那我们看看有没有什么办法去绕过它，我们现在对 sim() 函数的唯一限制就是非负，那我们是否可以引入一个核函数呢？假设有一个核函数的特征表示 $$\phi(x)$$ ，我们就可以将 attention score 重写为：

$$ V^{'}_{i} = \frac{\sum_{j=1}^{N}\phi(Q_{i})^{T} \phi(K_{j})V_{j}}{\sum_{j=1}^{N}\phi(Q_{i})^{T} \phi(K_{j})} $$

根据矩阵运算的性质：

$$ (\phi(Q)\phi(K)^{T}) V = \phi(Q)(\phi(K)^{T}V) $$

现在我们知道 QKV 都是 nxd 的，之前先算 $$QK^{T}$$ 的话，得到 $$nxn$$ 的，此时需要计算 $$n^{2}d$$ 次，之后这个 nxn 的再和 V 相乘需要计算 $$n^{2}d$$ 次，整体来说复杂度就是 $$O(n^{2}d)$$。而现在先算 $$K^{T}V$$ 的话，需要计算 $$d^{2}n$$ 次，这个再与 Q 乘也需要 $$nd^{2}$$ 次，整体复杂度变成了  $$O(nd^{2})$$，也就是序列长度的线性复杂度。当 d 比 n小时，这就可以大大的降低运算的复杂度。

特征映射 $$\phi()$$ 要用在 Q 和 K 的 row 上。对于核函数和特征函数的选取，对应于指数核的特征函数是无穷维的，这使得精确的 softmax attention 线性化不可行。另一方面，例如，多项式核有一个精确的有限维特征映射，并且被证明与指数核或 RBF 核同样有效。因此论文采用下列函数：

$$ \phi(x) = elu(x) + 1 $$

其中 elu() 是指数线性单元，用 elu 而不是 relu 是为了避免x 小于 0 时梯度为0 的情况。

论文提出的线性 Transformer 模型结合了两者的优点。当涉及到训练时，计 算可以并行化，并充分利用gpu或其他加速器。当涉及到推理时，对于论文中的 模型，每次预测的每次成本和内存都是恒定的。这意味着我们可以简单地将 φ（Kj）V_j^T矩阵存储为内部状态，并像递归神经网络一样在每一个时间步 对其进行更新。这导致推断速度比其他 Transformer 模型快数千倍。

## Linformer Self-Attention with Linear Complexity
将原本的尺度化的点积注意力拆解成了多个更小的线性投射的注 意力。这刚好是对原注意力去做低秩因式分解 在 BookCorpus 和 英文的 Wiki 上用 MLM 的训练目标预训练了一 个模型，并在 GLUE 的各个任务上，以及 情感分析任务 IMDB reviews 上进行微调。结果显示，在显著的速度提升基础上，的模型与原版的 Transformer 相当，甚至还好于原版。

先看一下这个论文给的 Linformer 和其他 Transformer 变体的算法复杂度一览（前面都有介绍）：

![](/img/in-post/various_attention/linformer_com.png)

该论文提出的算法复杂度是序列长度的线性。这个论文的思想比较简单，认为self-attention 是低秩的。为了验证这个观点，论文对模型的不同层、不同 attention head 对应的矩阵 P，进行奇异值分解 ，并把超过 10k 的句子上的归一化的累积奇异值
做了平均。如下图所示,左图是 self-attention 矩阵的频谱分析。Y轴是上下文映射矩阵P的归一化累积奇异值，X轴是最大特征值的索引。右图描绘了Wiki103数据中不同层和头的第128个最大特征值处的归一化累积特征值 heatmap：

![](/img/in-post/various_attention/linformer_eng.png)

结果显示：

* 沿着不同层、不同注意力头和不同的任务，都呈现出一个清晰的长尾分布。    
* 高层的 Transformers 中，会比更低层的 Transformers 有更大的偏度。这意味着，在更高层，更多信息集中在少量最大
的奇异值中，且矩阵 P 的秩是更低的

那这样的话，我们能否用低秩矩阵 $$P_{low}$$ 来近似 P呢？标准的Transformer Attention 计算 P：

$$ P = softmax(\frac{QW_{i}^{Q}(KW_{i}^{K})^{T}}{\sqrt{d}}) $$

现在引入两个映射矩阵 $$ E_{i},~F_{i}\in R^{nxk} $$，你 $$ KW_{i}^{K}$$ 的维度不是 nxd 么？那我再乘一个 E ，就会得到一个 kxd 的矩阵，对 K 和 V 都做这种事，就得到了下式：

$$ softmax(\frac{QW_{i}^{Q}(E_{i}KW_{i}^{K})^{T}}{\sqrt{d}}) F_{i}VW_{i}^{V} $$

现在我们算一下复杂度，$$E_{i}KW_{i}^{K}$$ 中，把KW 当作整体 nxd 的，则需要计算 nkd 次，同理V 那个 $$F_{i}VW_{i}^{V}$$ 也是一样的。  $$QW_{i}^{Q}$$ 是 nxd 维的，$$E_{i}KW_{i}^{K}$$ 是 kxd 维的，那这里需要计算 nkd 次。softmax 后，nxk 的 P 和V 那个 kxd 的乘，复杂度也是 nkd ，这样一看总计算次数是 4*nkd，也就是说，当 k << n 时，变成序列长度的线性复杂度了，这在超长文本上应该还是有用的。

对于 k 的选取，论文采用动态的方法，让更高层的 Transformer 选用更小的 k，整体上的话，k 增加，表现更好。但 k 越大，复杂度就越接近原版的了，参数还会变多。。。。至于其他投影方法，还可以用均值、最大池化、卷积来搞。

模型用的 RoBERTa 的架构，进行了预训练和finetune 实验，如下图所示：

![](/img/in-post/various_attention/linformer_res1.png)

论文用模型的困惑度来作为模型表现指标。困惑度越低，则模型训练得越好。(a) 图展示出，在其它条件相同下，随着 k 值变大，模型的困惑度越低，表示其训练得越好。(b) 图试图说明，即便我们把 k 从 256 降低到它的一半 128，其困惑度也不会显著增加。(c) 图比较的是不同参数共享下的效果。三者的效果相当，我们可以极端一点采用 Layerwise 参数共享。(d) 图比较了在固定k的情况下，不同的序列长度对其困惑度的影响。结果显示随着长度增加，初始的困惑度长序列会比较大。但最终收敛之后，不同长度序列的困惑度都差不多。这验证了 Linformer 是线性复杂度。

在下游任务中。我们也可以看到，使用 Linformer 架构训练的 RoBERTa 与原版 Transformer 架构训练的 RoBERTa 效果相当。在 k = 128 时，略逊于 RoBERTa，平均的差异小于 0.01%。当增大 k 后，二者的差别几乎可以忽略不计。我们还可以在增大 k 后，再用参数共享的技巧来减少空间消耗。其性能的损耗几乎不变。而且实验结果也表明，Linformer 的表现主要取决于投影维度 k ，而不是 n/k。

![](/img/in-post/various_attention/linformer_res2.png)

## Big Bird: Transformers for Longer Sequences
提出了BigBird，一种稀疏的注意力机制，将这种二次依赖性 降低为线性。证明 BigBird是序 列函数的一个通用逼近，并且是图灵完备的，从而保持了二次full attention 模型的这些性 质。 在此过程中，的理论分析 揭示了O（1）全局token（如 CLS）的一些好处，它们作为 稀疏注意机制的一部分来处理 整个序列。 所提出的稀疏注意可以处理的 序列长度是以前使用类似硬件时的8倍。由于能够处理较长的上下文，BigBird极大地提高 了各种NLP任务的性能，例如 问答和摘要。

整体论文的思想还是比较简单的，这里略过论文里复杂的描述，我按照自己的理解来记录一下。首先论文认为 full attention 可以看作一个图，图中 query 得每个 token 和 key 的每个 token 都有连接。这就太浪费了，我们通过前面的文章了解到，self-attention 是低秩的，那么我们就可以通过稀疏化这种连接达到降低复杂度的目的。从这点出发，论文提出 random attention 、window attention 、 global attention 三个 attention 模式，个人认为这个论文是延续 Longformer 这个工作的，这个 window attention 和 global attention 和那面的思想都一样。不同的多了一个 random attention。

![](/img/in-post/various_attention/bigbird_exam.png)

论文认为稀疏随机图的注意力机制应该有两个要求，1）节点间的平均路径长度较小；2）局部性的概念；因此Random attention 的构建，即Erdos-Renyi 模型，其中每条边都是以固定概率随机选取的。如果这样一个随机图正好有 o(n) 条边，则任意两个节点间的最短路径长度是节点数的对数。这就导致，这种随机图在频谱上近似于完全图，其第二特征值（邻接矩阵的）与第一特征值相差甚远。这种特性导致图中随机游走的rapid mixing time，也就是说信息可以在任意一对节点之间快速流动。用简单的话来理解的话，就是通过在图中撒的这些随机点，相当于部署了一些岛屿，通过这些岛屿，信息可以很方便的在任意岛屿间流动，如下图 a 所示，当 r=2 时，随机选取两个 key的 attention 矩阵，其中有连接的边邻接矩阵对应元素的值为1。

local 的是为了获得局部信息。对于局部信息，论文给出的原因是：克拉克等人研究了NLP任务中的 self-attention 模型，得出了相邻内积极为重要的结论。语言结构中代词的邻近性，也构成了各种语言学理论的基础，如转换生成语法。在图论的术语中，聚类系数是连通性局部性的度量，当图包含许多团或近团（几乎完全互连的子图）时，聚类系数就很高。简单的Erdos-Renyi随机图没有很高的聚类系数。但是一类被称为 small word graph 的随机图具有很高的聚类系数.

Watts和Strogatz提出的一个特殊模型与我们高度相关，因为它在平均最短路径和局部性概念之间实现了很好的平衡。其模型的生成过程是：构造一个有规律的环格，让图中 的 n 个节点与其相邻的w 邻接节点相连，每一侧连 w/2 个。换句话说，我们有一个滑动窗口，开始滑。然后将所有连接的 k%（随机选的） 替换为随机连接。剩余的（100-k）%的 局部连接被保留。然而，删除这些随机边在现代硬件上可能效率低下，因此我们保留它，这不会影响它的性质。这就得到的上图中的 b。

对于 global attention，作者先只用 random 和 window 两个attention 试了一下，想看看和 BERT 比在保证线性复杂度的前提下，性能咋样发现这俩的效果和BEET 比差一些，不能够捕捉足够的上下文来做这件事。

![](/img/in-post/various_attention/bigbird_res1.png)

根据实验结果，发现得补充全局信息，因此利用 “全局token” 的重要性（像 CLS 那种和所有token 都发生关系的），论文给了两种定义方式：

* internal transformer construction(ITC)： 利用现有的 token，让某些元素和全部 token 发生作用    
* extended transformer construction(ETC)：引入外部的 token，像 CLS

最终用的是上面三个的混合，如图 d 所示。不过虽然模块的复杂度低了，整体却需要更多的层数，论文在中给出了一个下限的证明。

在QA、分类等任务上，该模型都获得了更好的表现。

![](/img/in-post/various_attention/bigbird_res2.png)
![](/img/in-post/various_attention/bigbird_res3.png)
