---
layout:     post
title:      "知识图谱论文阅读笔记（一）"
subtitle:   "Neural Relation Extraction with Selective Attention over Instances"
date:       2019-08-14 00:15:18
author:     "Pelhans"
header-img: "img/tensorflow.jpg"
header-mask: 0.3 
catalog:    true
tags:
    - Knowledge Graph
---

> 多读书, 多看报,多吃零食, 多睡觉.

* TOC
{:toc}

# 论文概览

论文链接: [Neural Relation Extraction with Selective Attention over Instances](https://www.aclweb.org/anthology/P16-1200)

## 论文要解决的问题是什么

远程监督生成的关系抽取数据集噪声比较大, 比如 (奥巴马, 总统, 美国), 你在 KB 中是这样的关系, 但实际句子里, 奥巴马和美国间也有可能是 "出生地" 这样的关系. 但你标签打的却是 "总统", 这就带来了噪声. 对于 bag-level 得关系抽取来讲, 怎么减弱这种噪声得影响是一个很紧迫得问题.

## 这篇文章得主要贡献是什么

提出了一个 句子级别的基于 attention 的 卷积神经网络远程监督关系抽取模型.该模型采用 CNN 将输入的每句话转换为句向量, 而后 采用 sentence-level 得注意力机制, 为输入的多个句向量计算权重向量并进行加权求和得到一个综合的句向量. 这个句向量经过全连接层后进行分类得到类别输出.

模型得主要贡献有:

* 和之前的 max 选取最佳句子的方式不一样, 利用 attention 机制可以全面利用 bag 中所有句子的信息.    
* 利用 attention 机制可以有效去除bag 中的噪音, 是效果提升的主力, 前面可以配合不同的 网络结构,如CNN/PCNN/RNN 等. 并且根据thunlp 得OpenNRE 给出的测试结果, RNN/BRNN 得结果要比 CNN/PCNN 要好一些. 

# 论文详解

首先上一个模型结构图:

![](/img/in-post/kg_paper/CNN_attention.jpg)

模型整体分为几个部分: 输入, CNN层, attention. 下面逐个介绍.

## 输入

假设输入为 n 个句子 ${x_{1}, x_{2} \dots  x_{n}} $ . 两个实体 $e_{1},~e_{2}$. 输入句子由一个词一个词组成, $x_{1} = {w_{1}, w_{2}, \dots, w_{m}} $. 其中 m 表示输入句子的最大长度. $w_{i}$ 的维度是 [1, $d^{a}$], 论文里用的是 50.

输入除了词向量以外, 还有 位置向量的嵌入, 这是因为关系抽取喝句子分类不同, 它很依赖两个实体 和这两个实体的位置. 位置向量则可以告诉模型这两个实体所在的位置. 举个例子: 现在我们有一个句子 "Bill_Gates is the founder of Microsoft", 那么 founder 这个词距离实体1 "Bill_Gates" 得距离是3, 距离实体2 Microsoft 得距离是 2.  而后通过向量嵌入我们就获得了两个位置向量,假设它们的维度是 $d^{b}$, 则输入的总维度就是 $w_{i} \in R^{d} =  (d^{a}, d^{b1}, d^{b2}) $. 在论文里, 位置向量维度是5, 那么总得输入维度就是 [n, 1, m, 60] (对应[句子数量, 输入通道, 宽, 高])

## CNN

CNN 层的网络结构如下图所示:

![](/img/in-post/kg_paper/CNN_attention_cnn.jpg)

CNN 对输入的每句话分别进行. 卷积窗口大小为3, 深度为60, 两侧采用 0 Padding. 卷积核数目为目标句子向量的维度, 论文里采用230. 则输出的维度为: [n, 230,  m, 1] (代表[句子数量, 输出通道数 宽, 高]). 


卷积后跟着最大池化操作, 对 m 个向量进行 max 操作, 选取最大的最为最终的句子向量, CNN 层的最终输出维度为[n, 230]

## Attention

对于 CNN 得输出, 可以采取不同的解决方案:

* 对这 n 个句子取平均: 隐含假设是 这 n 个句子的贡献一样, 这种假设当然是不对的, 毕竟有噪声数据, 因此效果比较差.    
* 选取 n 个句子中最大的那个: 隐含假设是选取特征最强烈的做代表, 效果总体比平均要好一点    
* 用 attention 机制: 综合考虑各个句子, 进行分类, 给不表示目标关系得句子较小的权重, 反之给较大的权重

所以从理论上看, attention 天生适合这种去噪声任务 = =. 假设权重向量用 $\alpha$ 表示, 最终的句子向量为: $ s = \sum_{i}\alpha_{i}x_{i} $ . $x_{i}$ 是句向量输入. 下面介绍这个权重怎么来的.

首先在训练时, 根据数据集中得关系构建一个关系矩阵 r, r 的维度是 [句向量维度, 关系的数量]. 对于输入的每个句子, 我们根据句子标签去关系矩阵里找到这个关系对应的列, 维度为 [n, 230],

$$ e = xr $$

维度为 $ [n, 230] 点乘 [n, 230] = n $ . e 会被放入 softmax 得到归一化的概率, 也就是权重. 继而得到综合的句向量 s. 维度是 [1, 230],

得到 s 后就可以放入线性层,[1, 230]*[230, 关系数量 rel_tot]=[1, rel_tot]. 加softmax 愉快的进行分类了.

但是, 上面有一个问题就是训练时我有这个句子得标签, 可以这么搞, 但实际使用中可不是这样的, 怎么办? 最直接的想法那就全部用上呗. 对的, 不过这样的话:
    
* xr 的维度就变成了: [n, 230] * [230, rel_tot] = [n, rel_tot],     
* 转置一下做 softmax, 维度变成 [rel_tot, n].     
* 装作正常的权重和每个输入句向量做乘:[rel_tot, n] * [n, 230] = [rel_tot, 230].     
* 继续和线性层做计算, [rel_tot, 230]*[230, rel_tot] = [rel_tot, rel_tot]    
* 计算 softmax    
* 取对角线上得元素作为最终 logit 输出, [1, rel_tot], 这里是因为论文里有说 eAr, A是对角矩阵

现在不理解得是训练时为什么不这样做? 也许是想利用 sentence 的信息? 希望看更多的论文可以找到答案.

# 效果

作者分别用 CNN, CNN+ONE, CNN_AVE, CNN+ATT 对比, 得出结论. 

1) CNN + ONE 比 CNN 要好说明 去除噪声对性能很重要.    
2) CNN + AVE 比 CNN 好说明 综合考虑各句信息很重要.    
3) AVE 和 ONE 性能差不多, 说明平均考虑各句子信息会带来噪声,损害性能     
4) ATT 比其他的都好, 说明 ATT 在重分类用各句信息得基础上, 在一定程度上减弱了噪声带来的影响.

# 个人启发

* attention 很适合用来做这种噪声去除. 该模型只是加了一个 attention, 是否可以更充分得利用 CNN 得到的句向量来提升噪声去除能力?    
* CNN 对于关系抽取这种任务是否是最合适的? 答案是显然的, RNN 表现要比它好, 那是否可以对于关系抽取这种长距离依赖和位置敏感的任务换上更合适的网络呢?

# 参考

论文链接: [Neural Relation Extraction with Selective Attention over Instances](https://www.aclweb.org/anthology/P16-1200)

