---
layout:     post
title:      "模型压缩总结"
subtitle:   ""
date:       2020-04-02 00:15:18
author:     "Pelhans"
header-img: "img/dittilling.jpg"
header-mask: 0.3 
catalog:    true
tags:
    - NLP
---


* TOC
{:toc}

# 模型压缩

模型压缩大体上可以分为 5 种：

* 模型剪枝：即移除对结果作用较小的组件，如减少 head 的数量和去除作用较少的层，共享参数等，ALBERT属于这种    
* 量化：比如将 float32 讲到 float8    
* 蒸馏：将 teacher 的能力蒸馏到 student上，一般 student 会比 teacher 小。我们可以把一个大而深得网络蒸馏到一个小的网络，也可以把集成的网络蒸馏到一个网络上。    
* 参数共享：通过共享参数，达到减少网络参数的目的，如 ALBERT 共享了 Transformer 层    
* 参数矩阵近似：通过矩阵的低秩分解或其他方法达到降低矩阵参数的目的

这里我主要关注蒸馏和模型剪枝两方面。

# 模型蒸馏
## Distilling the Knowledge in a Neural Network

模型蒸馏的早期代表性工作，认为大模型虽然好，但推断速度太慢了，因此提出将知识从笨重的模型转移到更适合部署的小模型上，称这种流程为“蒸馏”。 

将笨重模型的泛化能力转化到小模型上的一个明显方法是利用笨重模型产生的类别概率(softmax 输出)作为训练小模型的软目标(soft target)。软目标能提供 ground truth 多的多的信息，毕竟 ground truth 是 0/1 编码的，通过熵看携带的信息也没连续的多。形象点来说，比如，一个宝马的形象，可能只有很小的机会被误认为垃 圾车，但这种错误的可能性 仍然比把它误认为胡萝卜高出许多倍，这种类别间的概率分布对于学生模型的进一步学习时很有用的。

对于transfer 阶段， 我们可以使用相同的训练集或单独的 transfer set。当复杂模型是一个简单模型的大集 合时，我们可以使用它们各自的预测分布的算术或几何平均值作为软目标。当软目标具有较高的熵时，每个训练样本比硬目标提供更多的 信息，训练样本之间的梯度方差要小得多，因此小模型通常可以用比原来繁琐的模型少得多的数据进行训练，并使用更高的学习率。

但用 softmax 的输出进行学习有一个缺点，就是一个训练比较好的模型，容易出现 0.999999... 或者接近 0 这种极端的情况，此时学生模型是很难从教师模型那学到东西的。因此一个可行的方案是使用  softmax 之前的 logits 进行学习，比如计算教师模型和学生模型 logits 的 MSE 损失来指导学生模型。该论文基于这个思想，进一步的提出用带温度的 softmax 来计算概率：

$$ q_{i} = \frac{exp(z_{i}/T)}{\sum_{j}exp(z_{j}/T)} $$

其中 T 表示温度，它等于 1 的话就是 softmax，T 越大，产生的概率分布就越 soft（越不极端），但可能稀释教师模型的负例信息。T 越小，softmax 后的结果就越 hard，学到的就更像 ground truth。因此如何选择 T 是一门艺术，论文指出， student 弱时，中间温度最有效，此时可以有效学习到教师模型的特征，student 网络越小，T 越小才能学得越好。

除此之外，这个带温度的 softmax 还可以看作是 logits 的 MSE loss 的推广情况。我们计算对蒸馏模型 $$z_{i}$$ 的梯度(v表示教师模型的logits 输出)：

$$ \frac{\partial C}{\partial z_{i}} = \frac{1}{T}(q_{i} - p_{i}) = \frac{1}{T}(\frac{e^{z_{i}/T}}{\sum_{j}e^{z_{j}/T}} - \frac{e^{v_{i}/T}}{\sum_{j}e^{v_{j}/T}} ) $$

当 T 比 logits 的数量级大很多时，我们可以近似得到：

$$ \frac{\partial C}{\partial z_{i}} = \frac{1}{T}(\frac{1 + z_{i}/T}{N+\sum_{j}x_{j}/T} - \frac{1 + v_{i}/T}{N+\sum_{j}v_{j}/T}  ) $$

假设 logits 是 0 均值的，则 $$ \sum_{j}z_{j} = \sum_{j}v_{j} = 0 $$，则

$$ \frac{\partial C}{\partial z_{i}} = \frac{1}{NT^{2}}(z_{i} - z_{j}) $$

这回传梯度的形式就差前面的系数，因此在最终计算学生模型的损失函数时，要在 soft loss 前加一个系数来控制 soft target 贡献的大小。

最终学生模型的 loss 由 ground truth 的 loss 和 soft target loss 的加和得到。

## FITNETS: HINTS FOR THIN DEEP NETS
前面的论文是将模型蒸馏到宽而潜的网络中，但实际上网络越深非线性会越强，因此该论文 尝试蒸馏得到一个窄而深（相对于 teacher来说）的网络，并利用Logit 和 中间层特征进行学习，由于学生的中间隐藏层通常比教师的中间隐藏层小，因此引入附加参数将学生隐藏层映射到教师隐藏层的预测中。

为了训练比教师网络更深的学生网络，论文引入 hint 和 guided，hint 表示 T 的隐层输出， guided 表示 S 的隐层，通过让 guided 预测 hint 进行学习。需要注意的是，hints 是正则化的一种形式，因此，hint/guided 对的选取要保证 S 没有过度正则化。我们设置的 guided 层越深，给 网络的灵活性就越低，因此，FitNets更可能遭受过度正则化的影响。

因为两个网络的宽度不一样，因此论文引入参数矩阵 W 来对  guided 层进行映射，使其与 hint 层的维度相匹配。但这样一来，参数又太多了，因此论文提出用 卷积这种共享参数的形式完成映射的任务。

![](/img/in-post/pretrain_model/fitnets_arc.png)

网络最终通过最小化下面的函数来训练 FitNet：

$$ L_{HT}(W_{Guided}, W_{r}) = \frac{1}{2}||u_{h}(x;W_{Hint}) - r(v_{g}(x;W_{Guided});W_{r})||^{2} $$

其中 $$ u_{h}$$ 和 $$v_{g}$$ 是 教师和学生网络通过权重矩阵的映射函数，而后整体用 MSE loss。

最终在 MINIST 上的 错误率如下图所示，可以看到，FitNet 比 前面 KD 的表现要好一些，说明该方法的有效性。

![](/img/in-post/pretrain_model/fitnets_res1.png)

那学生模型的深度到底影响有多大呢？论文分别作了不同深度/参数量的学生模型，如下图所示，可以看到，较深的学生模型在较低的参数量时，依旧可以获得比教师网络更好的效果：

![](/img/in-post/pretrain_model/fitnets_res2.png)

这就带来一个疑问。。。那你说。。。我为什么不直接训练一个窄而深得网络呢？等懂了再回来解答。

## Rocket Launching: A Universal and Efficient Framework for Training Well-performing Light Net

模型蒸馏方法，相比于传统知识蒸馏，教师网络 T 和 学生网络 S 共享一部分底层，并且最后的 loss 由 3 部分组成：T 和 S 的交叉熵 加上 logit 的 MSE loss。值得一说的是这个论文的名字。。。它把这种 booster 的教师网络和 light 网络看作火箭发射过程，一级是比较笨重的网络负责推进，用完扔掉，后面比较轻的部分负责实用，名字起得好呀。。。

整个模型的结构如下图所示：

![](/img/in-post/pretrain_model/rocket_arc.png)

两个模型共享底层参数 $$W_{S}$$，左侧教师模型的网络深一些，右侧学生网络的浅一些。整体的损失函数如下所示：

$$ L = H_{B} + H_{L} + \lambda||l(x) - z(x)||^{2} $$

前两个是教师网络和学生网络的 ground truth 交叉熵损失函数。后面那个是 两个网络的 logits MSE 损失函数。

除了 logits 的 MSE loss，论文还分别尝试了：

* 最终 softmax 的 MSE loss：
$$||p(x) - q(x)||^{2}$$    
* softmax 前的 logits MSE loss：
$$||l(x) - z(x)||^{2}$$，称之为 mimic loss    
* Hinton 的那个带温度的 loss

最终实验表明，第二个 mimic loss 效果最好。整个网络的回传梯度流如下图所示：

![](/img/in-post/pretrain_model/rocket_grad.png)

最终实验效果如下图所示，可以看出，这个方法比单纯的  KD 还是要好的：

![](/img/in-post/pretrain_model/rocket_res.png)

## Distilling Task-Specific Knowledge from BERT into Simple Neural Networks
接下来开启 BERT 蒸馏的大幕了，该论文首次将 BERT 蒸馏到一个单层双向 LSTM 网络上，整体蒸馏方法没啥可说的，和前面变化不大，loss 用的是 最后一层 Logits 的 MSE loss 和 交叉熵 loss 的组合：

$$ L = \alpha L_{CE} + (1-\alpha)L_{distill} $$

除此之外，由于小数据集不能让学生模型充分学习，因此该论文提出一系列的数据增强方法，用一个大的、未标记的数据集，让教师网络跑一下得到 Soft 标签来扩充训练集，使小模型充分的训练。蒸馏后的模型效果如下图所示，效果和 ELMO 相近，还是不错的。

![](/img/in-post/pretrain_model/bilstm_res.png)

## Knowledge Distillation via Route Constrained Optimization

之前的方法都是先训练 T ，收敛后再用来教 S ，但若 T 太复杂的话，S 直接学收敛的 T 也很难学，会带来较大的一致性损失。 
本论文不是用一个收敛的教师模型来监督学生模型，而是在教师模型经过的参数空 间中选择一些锚定点(中间 epoch 结果)来监督它，称之为路径约束优化（RCO）。实验证明，这种简 单的操作大大降低了知识提取、hints和模仿学习的 一致 损失下限。在CIFAR[16]和 ImageNet[3]等分类任务中，RCO分别将知识提取提高了2.14%和1.5%。

论文提出了一种用教师优化路径来监督学生的新方法。下图显示了RCO 与传统蒸馏的对比图。与单一的收敛模型相比，教师的路径通过提供一个容易到难的 学习序列来包含额外的知识。通过逐渐模仿这样的顺序，学生可以学习到与 老师更一致的东西，从而缩小成绩差距。此外，论文还分析了不同的学习序 列对性能的影响，提出了一种基于贪婪策略的序列生成方法，让难度梯度更合理。

![](/img/in-post/pretrain_model/route_arc.png)

这个图详细来说，上面两个 a 是只用最后一层的 logits 蒸馏， b 是用中间层 hints 和最后 logits 蒸馏。文章认为，对于收敛的大模型来说，直接学太难了，就像小学生直接跟教授学一样，跟不上。因此就让学生网络学教师网络中间结果，小学跟你学一下，初中跟你学一下，高中学一下。。。就这么跟着学，难度梯度比较合理，效果应该会更好。

下图是 ROC 的框架图：

![](/img/in-post/pretrain_model/route_arc2.png)

每一个 step 表示一个学习节点。整个网络的 loss(某一个 step) 是交叉熵 loss 加上 T 与 S 输出概率的 KL loss：

$$ L_{KD} = H_{S} + \lambda KL(P_{t}, P_{s}) $$

那问题变成这些 step 怎么选呢？论文尝试了两种策略：

* 等 epoch 间隔策略：每隔几个 epoch 就搞一下。但等 epoch 方法虽然简单，但没有考虑到 S 的学习曲线，毕竟这东西有时候隔几步没学到什么东西，有时候中间来个突变，所以就有了下面的 贪婪方法    
* 贪婪搜索策略：如下图所示，30 100 180分别代表用30th/100th/180th epoch 的T指导训练得到的S，然后随机取1w张图，分别送入他们得到输出,再和T计算KL散度，得到上图，发现在teacher的前期，30指导的S能够比较好的学习，而到了后面30的曲线突然起飞，说明不够用了，尤其是每一次减低学习率的时候，此时需要切换到其他 epoch 锚点进行学习。因此论文提出一种锚点的切换策略：    
    * 计算S和当前T以及下一个anchor point的T之间的KL距离(随机选一些验证集图片计算output)。公式如下所示    
    * 当距离大于一定阈值后就换T

![](/img/in-post/pretrain_model/route_kl.png)

![](/img/in-post/pretrain_model/route_for.png)

其中 H 表示 KL 散度，$$\Chi^{'}$$ 表示验证机，$$ r_{ij}$$ 表示两个 epochs 间的 H 相对变化。若相对变化超过阈值(论文里对 MobileNetV2 用 0.8)，就切换 T。贪婪搜索算法的流程如下图所示：

![](/img/in-post/pretrain_model/route_gre.png)

最终模型的效果如下图所示，相比于传统 KD，改进还是比较大的。对于EEI 和  GS 方法的对比，当 EEI 为 4 时效果比 GS 没弱太多，one-stage EEI 效果也还好，但 EEI-2 和 EEI-3 不太行。

![](/img/in-post/pretrain_model/route_res.png)

## Patient Knowledge Distillation for BERT Model Compression

认为 S不能只学最终输出的 logits，要学中间的隐层输出，因此论文提出两种方案：

* PKD-Last：学习teacher的最后k层    
* PKD-Skip：对 teacher 的每隔 k 个隐层学一下

这两种 patient 蒸馏方案，使教师的隐藏层中丰富的信息得以开发，并鼓励学生模型通过一个多 层的蒸馏过程耐心地向教师学习和模仿。从经验上看，这转化为多个 NLP 任务的改进结果， 在不牺牲模型精度的前提下显著提高了训练效率。

最终模型的 loss 就是 S 的交叉熵 loss 加上 T 与 S 间的 KL 散度 再记上隐层间的 loss ：

$$ L_{KD} = \alpha L_{CE} + (1-\alpha)L_{DS} + \beta L_{PT}$$

原始的BERT 做分类任务用的是 CLS 标记的 logit。 有一些变种甚至直接用用不同层 CLS 的加权得到的表示做 。因此论文认为，如果压缩模型能从教师中间层对任何给定输入的 [CLS] 表示中学习，它就有可能获得类似于教师模型的泛化能力。 因此该论文的中间层学习的是 CLS 表示。

具体层的选取策略，skip 用的是 $${2,4,6,8,10} $$ ，last 用的是 $${7,8,9,10,11}$$。再加上最后一层，所以用的都是 6 层。

整个模型的效果如下所示，其中 PKD 用的是 skip 策略，可以看出它的效果是比之前的 KD 要好的

![](/img/in-post/pretrain_model/bert_pkd_res.png)

下图是两种策略的对比，可以看出 skip 要比 last 整体强一些。

![](/img/in-post/pretrain_model/bert_pkd_res2.png)

## DistilBERT, a distilled version of BERT smaller, faster, cheaper and lighter

在预训练阶段蒸馏一个小的通用BERT做法。预训练阶段的loss由三部分组成：

* MLM 任务的 loss    
* teacher-student之间logits对应的loss    
* teacher-student隐层之间的cosine distance loss

student网络层数减小一半，其他不变，参数隔层初始化。最终效果上，模型参数减少40%，inference速度提升60%，最终效果保留 97%。

对于 S 的初始化，从 T 中，每隔两层选出一层对S 进行初始化即可，剩下预训练那一套，遵循 RoBERT 的就行，比如大 batch、动态 mask，移除 NSP 任务。

实验结果如下图所示，DistilBERT 能达到 BERT 97% 的效果。表3显示，相比于原始 BERT，参数量小了将近一半，速度也快了 60%：

![](/img/in-post/pretrain_model/distilBERT_res.png)

## TinyBERT Distilling BERT for Natural Language Understanding
个人觉得比较全面的 BERT 蒸馏论文，值得细看。除了以往针对 BERT 整体的蒸馏，还提出要对 Transformer 内部进行蒸馏，对 embed + attention 矩阵 + 隐层输出 + 最终输出 进行蒸馏。文章的主要改进点为：

* 设计了几种损耗函数，以适应不同的 BERT层表示：    
    * 1）嵌入层的输出    
    * 2）Transformer 层输出的隐藏状态和内部的 attention 矩阵    
    * 3）预测层输出的logits。
    * 原因 是 attention 权重矩阵包含统计学语言 知识，因此只学最后结果是不够用的     
* 提出了一个新的两阶段学习框架，包 括通用的蒸馏 和 针对特定任务的蒸馏。    
    * 在通用蒸馏阶段，未经微调的原始 BERT作为教师模型。学生 TinyBERT 通过在大规模语料库上执行 Transformer 蒸馏来学习模仿教师的行为。这就得到了一个可以针对各种下游任务进行微调的通用 TinyBERT。     
    * 在 特定任务的提取阶段，我们进行数据扩充，为教师和学生学习提供更多的任务特定数据，然后对增加的数据重 新执行Transformer 蒸馏。    
    * 这两个阶段对于提高TinyBERT的性能和泛化能力都是必不可少的。

Transformer 蒸馏的结构如下图所示，整体来说，像右面那面，教师模型由 N 层，学生有 M 层，N 大于 M，学生网络还可能更窄一点。从内部细节来说如右图所示，分别计算 MHA 内部的 attention 权值矩阵的 loss $$Attn_{loss}$$ 和 隐层输出的 $$Hiddn_{loss}$$。

![](/img/in-post/pretrain_model/tiny_arc.png)

至于为什么要对 attention 的权值矩阵进行蒸馏，论文认为 attention 内部包含了语言和 语法结构信息，经过 softmax 后学就没那么直接了，收敛也没权值矩阵快，因此需要有一块直接对 attention 权重矩阵进行学习。这一块的 loss 形式为：

$$ L_{attn} = \frac{1}{h}\sum_{i=1}^{h}MSE(A_{i}^{S}, A_{i}^{T}) $$

其中 h 是 head 的数量，A 的维度是 lxl。但对于隐层输出来说，由于 T 和 S 的隐层大小不一样，因此就需要借助前面论文里那个通过权值矩阵 W 进行映射来计算 loss 了，因此 隐层的 loss 公式为：

$$ L_{hiddn}  = MSE(H^{S}W_{h}, H^{T}) $$

其中 $$H^{S}$$ 的维度是 $$lxd^{'}$$，$$H^{T}$$ 的维度是 $$lxd$$，所以 $$W_{h}$$ 的维度是 $$d^{'}xd$$。

对于 embedding 层的loss，也一样用一个权值矩阵来映射：

$$ L_{embed} = MSE(E^{S}W_{e}, E^{T}) $$

然后再加上通用的 logits loss，这里它对 S 的 logits 搞了一个温度 t，不过实验表明 t=1 最好。。。：

$$ L_{pred} = -softmax(z^{T})*log softmax(z^{S}/t)  $$

在实际试验中，论文先进行中间层蒸馏，而后进行 logits 的蒸馏。

接下来就是两阶段蒸馏，一般的蒸馏为任务特定的蒸馏提供了良好的初始化，而任务特定的蒸馏通过集中学 习任务特定的知识进一步提高了 TinyBERT。但有时候特定任务的数据没那么多，导致 S 不能很好的学习 T，因此需要用数据增强技术得到更多的数据，论文里用词级别的替换。先用 Glove 得到词向量。使用语言模型来预 测单件词的替换，并使用单词嵌入来检索最相似的单词作为多个片段单词的替换。

论文里是用 BERT 作为 T 的，TinyBERT 的层数为 4，隐层维度 312，FFN 维度 1200，12 个 head(head是不是有点太多了？分到每个 head 的维度有点太低了)，最终参数量为 14.5 M。 

下图是模型的对比结果，PKD 和 DistilBERT 就是前面介绍那俩，可以看出，同等参数量下（最后一个TinyBERT）,TinyBERT 要比它俩要强很多，即使最小规模的 TinyBERT（第一行），也比那俩强，说明这种蒸馏 attention 矩阵、embedding 矩阵 和两阶段的方法很有效。

![](/img/in-post/pretrain_model/tiny_res.png)

那上面那些组分哪个贡献更大呢？文章做了消融实验，如下表所示，可以看出， embedding 蒸馏的影响最小，Transformer 层蒸馏的影响最大，说明对于 BERT 这种大模型，隐层蒸馏是很有必要的。值得考虑的是，不蒸馏 hidden 和不蒸馏 atten 对比，不蒸馏 atten 效果更差，说明 attention 内部的语言语义结构信息在隐层后学习的并不充分，要蒸馏 Transformer 结构必须要深入到 attention 矩阵才好。

![](/img/in-post/pretrain_model/tiny_res2.png)

## Extreme Language Model Compression with Optimal Subwords and Shared rojections
论文认为以往模型蒸馏虽然中间层参数少了，但词表那么大，输入部分的参数还是太多了，因此要优化。文章引入 Dual Training（teacher 输入的切分词表有一定概率采用 S的，这样就达成了用双向学习） 和 Shared Projection（让S 对 T 的参数矩阵进行学习，对于词表维度不一致问题，用矩阵映射解决），其中比较神奇的就是那个对偶训练。

![](/img/in-post/pretrain_model/dual_arc.png)

左侧是 教师网络 T，右侧是学生网络 S。T 有自己的词表，比较大，3w+，S 也有自己的词表，约 5 k 个，都是通过 WordPieces 方法获得的。S 的词有 93.9 在 T 中，这就带来一个问题，教师的有些东西，你学生怎么也学不到，毕竟你的词我这有一些没有。。。因此就引入对偶学习的概念。对于输入到教师模型中的给定训练序列，我们使用学生词汇从序列中随机选择（概率$$p_{DT}$$，超参数）标记来混合教师和学生词汇，其他标记使用教师词汇分割。如上图所示，给定输入上下文['I'、'like'、'machine'、'learning']，单词'I'和'machine'使用教师词汇表（绿色）分割，而'like'和'learning'则使用学生词汇表（蓝色）分割，这鼓励根据教师和学生词汇表对同一单词的表示进行对齐。这样的话：

* 使用教师词汇分割的单词来预测学生词表中的单词    
* 使用学生词汇分割的单词来预测教师词表中的单词

最终模型的优化目标就比较正常了，就包含两个的交叉熵 loss，Shared Projection 那的 loss。

## BERT-of-Theseus: Compressing BERT by Progressive Module Replacing

提出了利用模块替换的方式来做模型压缩。出发点是，你前面压缩 BERT 的模型都需要在预训练 阶段搞一下，这资源消耗太大了，但直接在 finetune 时蒸馏效果又不好，因此论文指出，那就做2次 finetune，第一个 finetune 阶段将 S 中的模块随机替换掉 T 中对应的模块，参与训练，学个大概。然后第二个 finetune 让 S 正常学，就可以了。

模型结构如下图右侧所示，假设压缩比为2，即12 层压成 6 层，$$pred_{i}$$ 表示 T 的第 i个模块，这里是两个 Transformer block。$$scc_{i}$$ 表示 S 的第 i 个 Transformer block。初始化时，可以用 T 的一部分 block 来初始化 S，最简单的就是用 T 的前六层初始化 S。接下来就是两阶段 finetune。

![](/img/in-post/pretrain_model/theseus_arc.png)

* 第一阶段是模块替换训练：用 S 的 block 替换掉 T 中对应的 block ，具体来说，用一个概率为 p 的伯努利分布控制替换概率。训练的话就和正常 fine-tune 一样，用特定任务的 loss 训练，只不过 T 的 block 不更新，只更新 S 的。通过这种方式，让 S 的模块去强者的世界中充分的学习，看看更广阔的的天地，学习通用的知识    
* 第二阶段就是 S 的 finetune 和 推断使用：S 中的 block 都是分散的扔到 T 中训练，还没整体训练过 S ，因此这个阶段就要将 S 整体 finetune 一下，任务和第一阶段一样

对于替换概率 p 的选取，论文给出了一个可以动态变化的公式：

$$ p_{d} = min(1, \theta(t)) = min(1, kt+b) $$

b 是初始替换率，k 是大于0的系数，t 是 step。这样我们可以让模型在第一阶段初期有较小的替换率，T 的 block 当主力， S 的block 打下手。随着 step 的增大，S 的 block 慢慢变成顶梁柱。

该模型的实验效果如下图所示，对比的都是比较早期的模型（毕竟只在 finetune 阶段搞，和TinyBERT 这种比确实没什么意义），效果还是可以的：

![](/img/in-post/pretrain_model/theseus_res.png)

论文还非常人性化的给了一个不同 BERT压缩方法的比较,如下图所示。“CE”和“MSE”分别代表交叉熵和均方误差。“KD”表示损失是为了知识的蒸馏。“CETASK”和“celm”分别表示对下游任务和 MLM 预训练计算的交叉熵：

![](/img/in-post/pretrain_model/theseus_com.png)

# 参数共享
## ALBERT: A LITE BERT FOR SELF-SUPERVISED LEARNING OF LANGUAGE REPRESENTATIONS

虽然 BERT 模型本身是很有效的，但参数太多了。本论文引入一种瘦身版的 BERT 模型 ALBERT，提出2中降低内存消耗和提高训练速度的参数减少策略。

具体来说，第一个是 Factorized embedding parameterization，论文认为原始 BERT 中，词向量 E 和隐层维度 H 是相等的。但实际上词向量维度不用那么大，因此论文提出 (V * E + E* H)。其中 V 是词表大小，而原来是 V * H，这一步能减少一些参数，不过不多，模型表现还下来了(0.6)，因此得从隐层内的参数入手。

第二个是 Cross-layer Parameter Sharing，作者分别尝试了 全不共享、全共享、只共享 self-attention、只共享 FFN。结果如下图所示

![](/img/in-post/pretrain_model/albert_share.JPG)

结果表明， 全部共享时，参数量只有原来的1/7左右，不过代价是性能下降了。而 只共享 self-attention 时，效果在 E = 128 时反而更好了，参数也能少一些，但论文没有考虑这种情况，**至于为什么对只共享  self-attention 时，E 变小它更好了，有点想不懂**。当采用全部共享时，性能下降了，弥补办法就是加大模型规模。论文提出了 ALBERT-xxlarge(12层)，但 H 变成 4096，真正的扁而宽，整体参数少了 30%，但变慢了。ALBERT 用 70% 的参数量，在同样的 1M steps 上确实超越了 BERT。相应地，训练时间也拉长到 3.17 倍，至于屠榜的结果，则是 1.5 M steps 的结果，又多出一半。。。落下了贫穷的眼泪

第三个 Sentence Order Prediction(SOP) 任务，论文认为原版的 NSP 任务太简单了，因此 SOP 对其加强，将负样本换成了同一篇文章中的两个逆序的句子。除此之外，原始的 BERT 是先用 128 个 toekn 训练90% ，后用 512 的训练，但之前的研究表明，单一的长句子对预训练很重要，因此 ALBERT 在 90% 时用满 512 的，10 % 随机找短于 512 的 segment。

# 剪枝
## Reducing Transformer Depth on Demand with Structured Dropout
文章探索了 LayerDrop，一种结 构化的 dropout 形式，说白了就是直接 drop 掉某些层，它在训练过程中具有 正则化效果，并允许在推理时进行有效的修剪。特别的，论文证明了可以从一个大的网络中选择任 何深度的子网络，而不必 对它们进行微调，并且对性能的影响有限。

具体来说：

* 在训练阶段，设定给一个dropout的概率 p。假定原先 bert 为 12 层，前向计算的时候，对于某一层，先使用随机的均匀分布采样得到一个概率，若该概率大于阈值 p，则跳过这个层，否则就正常进入 block。。。    
* 在inference的阶段，可以传入需要保留层的索引列表，在加载 模型后，根据保留的 block 列表信息，移除剪掉的层。 关于inference阶段的裁剪，论文给出了几个策略：    
    * 间隔裁剪，比如裁剪一半的层数，那么需要保留的层数可以为 0,2,4,6...等偶数层。这种方法最简单，同时也能到一个相对平衡 的网络。    
    * 验证集上找：类似于调参步骤，根据裁剪的层数，组成不同的子模型，然后用验证集来调试，看哪种组合效果最好。这种方式虽然能够得到最好的效果，但是计算量很大，很费时，成本较高。    
    * 数据驱动方法：设最终裁剪的比率为 p。对每个层都设定给一个裁剪概率 pd,其中所有层的的均值需要等于 p。具体来说，将局部 pd 参数化为 pd 所属层的非线性函数，并应用了一个softmax。在推断时，我们只根据 softmax 的 top-k 来选取要用的层。p 的话论文里用 0.2 ，但推荐用 0.5 来得到小模型。

## Structured Pruning of a BERT-based Question Answering Model

该论文采用任务特定的结构化剪枝和任务 特定的蒸馏的廉价组合，不需要预先训练 蒸馏的昂贵操作，就可以在一系列的速 度/精度权衡操作点上产生高性能的模型。 论文针对 SQuAD2.0任务，消除 Transformer 的选定部分。结构化剪枝以减 少每个 Transformer layer 中的参数数目， 具体来说： 

* 1. 减少各个transformer的 attention head数量    
* 2. 少各个transformer前馈子层的中间宽度    
* 3. 减少 embedding 的维度 

除此之外，将结构化修剪与蒸馏相结合,最终解码速度提高了一倍，准确度 损失仅有 1.5 f point

## Compressing BERT: Studying the Effects of Weight Pruning on Transfer learning

在该论文中，针对BERT的权重修剪问题 进行了研究和探讨：预训中的压缩是如 何影响迁移学习的？作者发现，修剪对 迁移学习的影响分三种情况:

* 低水平修剪（30-40%）不会对预训 练损失及迁移到下流任务产生影响     
* 中水平修剪会增加预训练的损失， 并会阻止有用的预训练迁移至下流任务    
* 高水平修剪还会影响到模型拟合下游数据，导致进一步降级

最终，根据观察，发现针对特定任务微调 BERT 并不会提高其可修剪能力，并得出结论，不影响性能的前提下，对 BERT  在预训练阶段进行单次修剪即可，无需针对各个任务分别修剪。

从实现上来说，该论文是通过 “量级权重修剪” （magnitude weight Prunning）进行剪枝的，它通过 移除接近0的权重来压缩模。剪枝流程为：

* 1. 选择要修剪的权重的目标百分比，例如50%    
* 2. 计算一个阈值，使 50% 的权重量级低于该阈值    
* 3. 移除那些权重    
* 4. 继续训练网络以恢复损失的准确性    
* 5. （可选）返回到步骤 1 并增加修剪权重的百分比

从权值剪枝策略上来说，可以分为全局修剪和局部修剪两种：

* 全局修剪就是这个概率在全局上算    
* 局部修剪就是这个概率在每个矩阵内部单独计算    

上面两个相比，全局的方法权重修剪的没那么均衡，因此这里选用社区更受欢迎的局部修剪策略。

修剪目标的话，定在 embedding 矩阵、QKV 的映射矩阵以及 FFN 的权值矩阵上。

不同权值修剪率的性能表现如下图所示，左侧是 dev 数据集上的 acc 表现，右侧是训练集上的 loss，可以看出，30-40%的权重可以使用量级权重修剪进行修剪，而不会降低下游精度。：

![](/img/in-post/pretrain_model/weight_prun_res.png)

