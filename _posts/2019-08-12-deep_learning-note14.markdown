---
layout:     post
title:      "深度学习笔记（十四）"
subtitle:   "Normalization 总结"
date:       2019-08-12 00:15:18
author:     "Pelhans"
header-img: "img/dl_background.jpg"
header-mask: 0.3 
catalog:    true
tags:
    - Deep Learning
---

* TOC
{:toc}

# 什么是 Normalization

在将数据送入下一层神经网络之前,先对其做平移和伸缩变换,将x的分布规范化成再固定区间范围得标准分布.

通用公式为:

$$ h = f(g*\frac{x-\mu}{\sigma} + b) $$

1) 其中 $\mu$ 是均值,表示平移参数, $\sigma$ 为方差,表示缩放参数. 通过这两个参数进行缩放和平移后,数据变成均值为0, 方差为1 得标准分布.    
2) 而后经过 g 再缩放参数和 b 再平移参数, 得到均值为b, 方差为 $g^{2}$得 分布.

第一步可以对原分布进行调整, 转化为标准正态分布. 这么做一方面可以使得激活值落入梯度敏感的区间(非饱和区),梯度更新幅度变大,模型训练加快;另一方面每一次迭代得数据调整为相同的分布(相当于白化),消除极端值, 提升训练稳定性.     

那为什么会有第二步呢? 这是由于虽然分布落在了梯度敏感区域,但 输出却接近于线性(sigmoid, tanh等), 模型得表达能力大幅度下降.因此需要第二步防止模型得表达能力因为规范化而下降. 此时参数 g 和 b 作为模型参数,训练得到. $\mu$ 和 $\sigma$ 通过统计的方式得到.这也是不同 Normalization 方法的主要区别所在.

那弄了半天又变回非标准正态分布了, 各层之间分布又不一样了, 效果有保证麽? 效果还是有的, 新分布的参数模型训练和统计得到, 祛除了与前面的层层耦合, 使得 参数的学习更加稳定有效.简化了神经网络的训练. 当然严格来说现在的方法只是将分布映射到了一个确定的呃区间范围,并不是我们追求的独立同分布.

# 为什么要做 Normalization

按照谷歌论文里的说法, 将其归为 ICS(Internal Covariate Shift, 指的是由于深度网络由很多隐层构成，在训练过程中由于底层网络参数不断变化，导致上层隐层神经元激活值的分布逐渐发生很大的变化和偏移，而这非常不利于有效稳定地训练神经网络。),引用(魏秀参得回答)[https://www.zhihu.com/question/38102762/answer/85238569]

> 大家都知道在统计机器学习中的一个经典假设是“源空间（source domain）和目标空间（target domain）的数据分布（distribution）是一致的”。如果不一致，那么就出现了新的机器学习问题，如，transfer learning/domain adaptation等。而covariate shift就是分布不一致假设之下的一个分支问题，它是指源空间和目标空间的条件概率是一致的，但是其边缘概率不同，即对所有的 $x\in X$, 
$P_{s}(Y|X=x) = P_{t}(Y|X=x)$, 但是 $P_{s}(X) \neq P_{t}(X)$.大家细想便会发现，的确，对于神经网络的各层输出，由于它们经过了层内操作作用，其分布显然与各层对应的输入信号分布不同，而且差异会随着网络深度增大而增大，可是它们所能“指示”的样本标记（label）仍然是不变的，这便符合了covariate shift的定义。由于是对层间信号的分析，也即是“internal”的来由。

除此之外, Normalization 还可以防止反向传播中得梯度问题(消失或者爆炸), 这是通过 Re-scaling 不变性实现的. 所谓 Re-scaling 不变性 ,即权重向量/数据/权重矩阵 $\alpha$ 倍后, 输出保持不变的性质. 这意味着参数值过大过着过小对神经元得输出没什么影响, 可以从一定程度上减轻梯度爆炸或消失得问题. Re-scaling 分为三种情况:

* 权重向量 Re-scaling: 边对应的权重向量W, 如果它乘以一个 缩放因子 $\alpha$ 后, 经过 Normalization 保持激活值不变, 则具有这个性质.    
* 数据 Re-scaling: 数据乘以 $\alpha$     
* 权重矩阵 Re-scaling: 两层之间的所有权中参数乘以相同的缩放因子 $\alpha$.    

下表总结了不同 Normalization 方法的 Re-scaling 不变性.

![](/img/in-post/tensorflow/re-scaling.jpg)

最后, 由于 神经网络损失函数非凸得性质, 研究表明,BN 真正的用处在于通过 Normalization 操作, 使得网络参数参数重整(reparametrize), 可以让损失函数曲面变得平滑一些, 更利于 SGD 等算法进行优化.

# 常见的 Normalization 方法

## Batch Normalization

batch normalization 针对单个神经元进行, 首先利用网络训练时的一个 batch 数据来计算该神经元 $z_{i}$ 得均值 $\mu$ 和 标准差 $\sigma$:

$$ \mu = \frac{1}{m}\sum_{i=1}^{m}z^{i}, ~~~\sigma = \sqrt{\frac{1}{m}\sum_{i=1}^{m}(z^{i} - \mu)^{2}} $$

其中 $z^{i}$ 是该神经元对于第 i 个输入数据的响应, m 是 batch size, 而后将 $\mu$ 和 $\sigma$ 应用到 每一个响应上:

$$ \hat{z}^{i} = \frac{z^{i} - \mu}{\sqrt{\sigma^{2} + \epsilon}} $$

其中 $\epsilon$ 是一个很小的数,防止 出现标准差为0 得情况.在这之后还可以像一开始那样添加额外的两个再缩放因子. 

在模型预测的时候, 往往是单个输入的,此时可以将训练过程中得均值和标准差都保存下来,而后做加权平均给预测时用,其中最后几层得权重会大一点.

这里需要注意的一点是,我们在就计算均值和标准差时求和指标是 m ,因此 batch normalization 是针对同一神经元 的 m 个输入的响应z做的, 如下图所示:

![](/img/in-post/tensorflow/BN.jpg)

其中输入的一个 batch, 每个 实例输入到同一神经元中得到的激活值得集合作为 normalization 得对象. 对于  cnn 来说, 输入的维度是 (N, C, H, W)，则 BN 是对 N、H、W 这三个维度进行 element-wise 的平均和方差计算，最终得到 C 个 均值和方差对。 

Batch normalization 很强,但缺点也很明显就是很依赖batch 得大小和batch 内数据分布的差异性. 而在 RNN 中, 是单个时间步走的,因此 BN 不能直接使用.

## Layer Normalization

在只有一个训练实例时, 怎么做 normalization 呢? LN 得基本思想时, 虽然输入是单个得, 但神经网络的每一层都有很多神经元, 我们可以利用这些神经元得输出作为集合计算均值和方差.


![](/img/in-post/tensorflow/LN.jpg)

上图是 MLP中的 LN, 对于 CNN 来说, 输入的维度是 (N, C, H, W)，则 LN 是在 C、H、W 三个维度上计算均值方差，即在每个样本内部做 normalization。  相当于在输入的 N 个数据上做 normalization,  最终得到 N 个均值方差。

LN 好处是可以用在 RNN 中, 但在 CNN 里,效果不如 BN.

## Instance Normalization

IN 一般针对的是 CNN, 我们知道 CNN 的输出包含[样本数量 N, 通道数目 C, 高 H, 宽W], IN 是在HW上做,保留 N,C.即:

$$ x = \frac{1}{HW}\sum\sum x_{H,W} $$

这相当于在每一张输入的每一个卷积核中,针对宽,高做 normalization. 如下图所示:

![](/img/in-post/tensorflow/IN.jpg)

IN 再图片生成类的表现明显优于 BN, 但在很多其他任务如图像分类上不如 BN.

## Group Normalization

Layer Normalization 针对单个样本, 对同层得所有神经元进行统计, IN 则是对每一个样本的每一个输出通道作为统计范围. GN 是一种介于二者之间得方法, 它对输入或者输出通道进行分组, 再分组范围内进行统计.

![](/img/in-post/tensorflow/GN.jpg)

Group Normalization在要求Batch Size比较小的场景下或者物体检测／视频分类等应用场景下效果是优于BN的。

# 问题
## BN 应该放在激活函数前还是激活函数后？

BN 的原始论文里是放在 Sigmoid/tanh 函数前的，理由是 BN 可以重整输入，减缓 sigmoid 饱和等问题。但也有很多评测论文指出，BN 放在激活函数后好一点（特指像 RELU 这种），理由是 ReLU 这类函数对输入的分布破坏比较严重，放在前面不能阻止这种现象。

## BN 等到底解决了什么问题？

还处于研究中，有几种说法：

* 解决 ICS 问题    
* 减缓梯度弥散    
* 让损失函数曲面变得平滑，更容易优化    
* 对系统参数搜索空间进行约束（好的映射曲线应该是让数据曲线的分布尽量均匀尽量平滑）

上面四个都是大佬的观点，我个人思考下来，感觉BN 实际做的事是重置各层得分布，减少层级之间的偏移累积。保证了源空间与目标空间的一致性，减缓了模型为了拟合某些模式而在某些层出现极端分布的可能，从而让优化更丝滑。

## LN 相比于 BN 的优缺点？

优点是比较适合用在 NLP 任务中，尤其是 RNN 这种不好批处理的，针对单样本进行 Normalization 就很有必要了。缺点是在 Mini-batch 上表现的不如 BN，毕竟样本更多，得到的均值和方差更准确和稳定。
