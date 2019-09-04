---
layout:     post
title:      "隐马尔可夫模型 HMM"
subtitle:   ""
date:       2019-09-02 00:15:18
author:     "Pelhans"
header-img: "img/kg_bg.jpg"
header-mask: 0.3 
catalog:    true
tags:
    - PRML
---


* TOC
{:toc}

# 隐马尔可夫模型

## HMM 的基本概念和定义

HMM 是关于时序的概率模型, 描述一个隐藏的马尔科夫链随机生成不可观测的状态随机序列, 再由各个状态生成一个观测而产生观测随机序列的过程. HMM 的马尔科夫链随机生成的状态的序列称为状态序列. **每个状态生成一个观测**, 而由此产生的观测的随机序列, 称为观测序列. 序列的每一个位置又可以看作是一个时刻.

从潜在变量的角度来说, 我们可以使用潜在变量的马尔科夫链来表示顺序数据, 每个观测都以对应的潜在变量的状态为条件. 对于顺序数据来说, 下面的图描述了两个模型. 如果潜在变量是离散的, 那么我们就得到了HMM(HMM 的观测变量是离散的还是连续的都可以, 并且可以使用许多不同的条件概率分布建模). 如果潜在变量和观测变量都是高斯变量(结点的条件概率分布对于父节点的依赖是线性高斯的情形), 那么我们就得到了线性动态系统.

举个例子, 对于词性标注任务来说, 观测序列就是我们的词. 状态序列就是词对应的词性标签.

![](/img/in-post/ml_mianshi/hmm_chain.png)

HMM 由初始条件概率分布, 状态转移概率分布以及观测概率分布确定. 

设 $Q\in \{q_{1}, q_{2},\dots, q_{N} \}$ 是所有可能的状态的集合, $V\in \{v_{1}, v_{2},\dots, v_{M} \}$. 是所有可能的观测的集合. 

I 是长度为 T 的状态序列, O 是对应的观测序列. $I = (i_{1}, i_{2}, \dots, i_{T})$, $O = \{o_{1}, o_{2}, \dots, o_{T}\}$.

A 是状态转移概率矩阵: $$A = [A_{ij}]_{N\times N} $$. 其中
$$a_{ij} = P(i_{t+1}=q | i_{t}=q_{i}), ~~~ i=1, 2, \dots, N; j=1,2,\dots, N$$. 表示 在t时刻处于状态 $q_{i}$ 的条件下,在时刻 t+1 转移到状态 $q_{j}$ 的概率.

B 是发射概率矩阵. 
$$B = [b_{j}(k)]_{N\times N}$$, 其中
$$b_{j}(k) = P(o_{t} = v_{k}|i_{t} = q_{j}), k=1,2,\dots, M; j = 1, 2,\dots,N$$. 表示在时刻 t 处于状态 $q_{j}$ 的条件下生成观测 $v_{k}$ 的概率.

$\pi$ 是初始概率向量, $\pi = (\pi_{i})$, 其中 $\pi_{i} = P(i_{1}=q_{i}), i=1,2,\dots, N$. 表示 t=1 处于状态 $q_{i}$ 的概率.

HMM 模型由上面三个参数所决定. 因此HMM 的模型参数为 $\theta = (A, B, \pi)$

从定义可知, HMM 做了两个基本假设:

* 其次马尔可夫性假设: 假设隐藏的马尔科夫链在任意时刻 t 的状态只依赖于其前一时刻的状态, 与其他时刻的状态以及观测无关, 也与时刻 t 无关.    
* 观测独立性假设: 假设任意时刻的观测只依赖于该时刻的马尔科夫链的状态, 与其他观测及状态无关.

如果我们放松第二个假设,就得到最大熵马尔科夫模型(MEMM). MEMM 描述的是在当前观察值 $o_{t}$ 和前一个状态 $i_{t-1}$ 的条件下, 当前状态 $i_{t}$ 的概率. 从图结构来说, MEMM 描述了由观察变量生成隐藏状态序列的模型.

从生成式的观点看 HMM. 首先我们选择初始的潜在变量 $z_{1}$, 概率由参数 $\pi_{k}$ 来控制, 而后采样对应的观测 $x_{1}$. 采样过程由发射概率控制. 现在我们使用已经初始化的 $z_{1}$ 的值, 根据转移概率来选择 $z_{2}$ 的状态. 继续对 $z_{2}$ 采样, 生成 $x_{2}$. 以此类推, 得到数据.

# HMM 的基本问题

HMM 有三个基本问题:

* 似然性计算: 给定HMM的模型参数 $\lambda$ 和一个观察序列 O ，计算出观察序列 O 的概率分布矩阵 
$P(O|\lambda)$. 可以用 前向算法, 后向算法解决.    
* 学习问题: 已知观测序列 O, 估计模型参数 $\lambda$, 使得似然函数 
$ P(O|\lambda)$ 最大. 可以用极大似然估计的方法估计参数, 如 EM 算法.    
* 解码问题: 给定 HMM 的模型参数和 观测序列O. 找到最优的状态序列 Q. 可以用 维特比算法解决.

## 似然性计算

直接计算在考虑整个序列T全局的情况下, 时间复杂度为 $O(T\times Q^{T})$. 计算方式是每个时间步上都有 Q 个可能的状态, 整个序列上就是 $Q^{T}$. 复杂度爆炸, 因此需要前/后向这种局部算法.

### 前向算法

前向算法是基于状态序列的路径结构递推计算
$P(O|\lambda)$. 给定隐马尔可夫模型参数 $\lambda$, 定义前向算子 
$\alpha_{t}(i) = P(o_{1}, o_{2}, \dots. o_{t}. i_{t}=i|\lambda)$. 它表示在时刻 t 时的观测序列为 $o_{1}, o_{2}, \dots, o_{t}$, 且时刻 t 时状态为 $q_{i}$ 的概率.

根据定义, 我们可以写出递推公式:

$$ \alpha_{t+1}(i) = [\sum_{j=1}^{Q}\alpha_{t}(j)a_{j,i}]b_{i}(o_{t+1}) $$

其中 中括号内表示, 再时刻 t 的观测序列为 $o_{1}, o_{2}, \dots, o_{t}$, 并且在 t+1 时刻处于状态 $q_{i}$ 的概率. 后面再乘以 t+1 时刻状态 i 的发射概率, 就得了 t+1 时刻的前向算子. 从图上来看就是下图这样.

![](/img/in-post/ml_mianshi/forward_HMM_prob.png)

有了递推关系, 我们可以写出前向算法流程:

* 输入为: 模型参数 $\lambda$, 观测序列 O. 输出为观测序列概率
$P(O|\lambda)$    
* 计算初始值, $\alpha_{1}(i) = \pi_{i}b_{i}(o_{1}), ~~~i=1,2,\dots, Q$    
* 利用递推公式递推计算前向算子.    
* 在时刻 T 终止, 即 
$P(O|\lambda) = \sum_{i=1}^{Q}\alpha_{T}(i)$.

可以看到, 其算法高效的原因是可以局部计算前向概率, 而后利用路径结构将前向概率"递推"到全局, 算法复杂度为 $O(TQ^{2})$.

类似地, 我们可以定义后向算法.

### 后向算法

类比于前向算法, 我可以写出后向算子: 
$ \beta_{t}(i) = P(o_{t+1}, o_{t+2}, \dots, o_{T}| i_{t} = i| \lambda) $. 它表示在时刻 t 的状态为 $q_{i}$ 的条件下, 从时刻 t+1 到 T 的观测序列为 $ o_{t+1}, o_{t+2}, \dots, o_{T} $ 的概率.它的递推公式为:

$$ \beta_{t}(i) = \sum_{j=1}^{Q}a_{i,j}b_{j}(o_{t+1})\beta_{t+1}(j) $$

含义是, 在 t+1 时刻状态 为 $q_{j}$ 的条件下, 从时刻 t+1 到 T 的观测序列为 $o_{t+1}, o_{t+2}, \dots, o_{T} $ 的概率为 $b_{j}(o_{t+1})\times \beta_{t+1}(j) $. 再考虑到所有的从 t 到 t+1 的可能转移概率 $a_{i,j}$, 并对 j 求和, 就有上述公式.

从图上来看则为:

![](/img/in-post/ml_mianshi/backward_HMM_prob.png)

## 综合形式

利用前向概率和后向概率的定义, 可以将观测序列概率统一为:

$$ P(O|\lambda) = \sum_{i=1}^{Q}\sum_{j=1}^{Q}\alpha_{t}(i)a_{i,j}b_{j}(o_{t+1})\beta_{t+1}(j) $$

当 t=1 时, 就是前向算法, t=T-1 时, 这是后向算法. 更进一步

$$ P(O|\lambda) = \sum_{j=1}^{Q}\left[\sum_{i=1}^{Q}\alpha_{t}(i)a_{i,j}b_{j}(o_{t+1})\right]\beta_{t+1}(j) =\sum_{j=1}^{Q} \alpha_{t+1}(j)\beta_{t+1}(j) , ~~~ t=1,2,\dots, T-1 $$

定义两个新的算子:
$\gamma_{t}(i) = P(i_{t}=i | O, \lambda)$ 和 $\xi_{t}(i,j)$.

$$ \gamma_{t}(i) = P(i_{t}=i | O, \lambda) = \frac{P(i_{t}=i, O | \lambda)}{P(O|\lambda)} $$

将前后概率定义代入,有:

$$ \gamma_{t}(i) = \frac{P(i_{t}=i, O|\lambda)}{P(O|\lambda)} = \frac{\alpha_{t}(i)\beta_{t}(i)}{\sum_{j=1}^{Q}\alpha_{t}(i)\beta_{t}(j)} $$

给定模型参数和观测序列 O, 在时刻 t 处于状态 $q_{i}$ 且在 t+1 时刻处于状态 $q_{j}$ 的概率记作 
$$\xi_{t}(i,j) = P(i_{t}=i, i_{t+1}=j |O, \lambda) $$. 因此

$$ 
\begin{aligned}
\xi_{t}(i, j) &= P(i_{t}=i, i_{t+1}=j |O, \lambda) = \frac{P(i_{t}=i, i_{t+1}=j, O | \lambda)}{P(O|\lambda)} \\
&= \frac{\alpha_{t}(i)a_{i,j}b_{j}(o_{t+1})\beta_{t+1}(j)}{\sum_{j=1}^{Q}\alpha_{t}(i)\beta_{t}(j)}
\end{aligned}
$$

由上面的可知, 在给定观测 O 的条件下, 状态 i 出现的期望值为 $ \sum_{t=1}^{T} \gamma_{t}(i) $, 由状态 i 转移到状态 j 的期望值为 $\sum_{t=1}^{T-1}\xi_{t}(i,j) $.

## 学习算法

可分为有监督学习和无监督学习两种。 有监督学习是指隐含状态序列和观测序列都给你， 此时我们可以通过统计得到转移概率, 发射概率, 初始概率. 无监督的学习就麻烦很多, 常用的方法是 Baum-welch 算法， 它是 EM 算法在 HMM 中的具体实现。

首先, 根据 EM 算法, 我们写出Q函数:

$$ 
\begin{aligned}
Q(\lambda, \lambda^{old}) &= \sum_{Z}P(Z|O, \lambda^{old})\ln P(O,Z |\lambda) \\
&= \sum_{Z}\frac{1}{P(O|\lambda^{old})}P(O|\lambda^{old})P(Z|O, \lambda^{old})\ln P(O,Z |\lambda) \\
&= \sum_{Z}\frac{1}{P(O|\lambda^{old})}P(O, Z|\lambda^{old})\ln P(O, Z|\lambda)
\end{aligned}
$$

在给定参数 $\lambda^{old}$ 时, 
$ P(O|\lambda^{old})$ 是常数, 忽略掉. 同时, 我们知道 

$$ P(O,Z |\lambda) = \pi_{i1}b_{i1}(o_{1})a_{i1,i2}b_{i2}(o_{2})\dots b_{iT}(o_{T}) $$

带入 Q 函数, 有:

$$ 
\begin{aligned}
Q(\lambda, \lambda^{old}) & = \sum_{Z}\log\pi_{i1} P(O, Z|\lambda^{old}) + \sum_{Z}\left[\sum_{t=1}^{T-1}\log a_{i_{t}, i_{t+1}}\right]P(O, Z|\lambda^{old}) \\
& + \sum_{Z}\left[\sum_{t=1}^{T}\log b_{it}(o_{t})\right]P(O, Z|\lambda^{old})
\end{aligned}
$$

其中 T 是序列总长度. 在 M 步骤, 要求最大化 Q 的参数. 因为 A, B, $\pi$ 分别出现在3项中, 因此可以分别优化.

首先优化 $\pi$. 上式第一项可以写成:

$$ \sum_{Z}\log \pi_{i1}P(O, Z|\lambda^{old}) = \sum_{i=1}^{N}\log\pi_{i} P(O, i1=i |\lambda^{old})$$

其中约束条件 $\sum_{i=1}^{N}\pi_{i} = 1$, 利用拉格朗日乘子法, 写出拉格朗日函数:

$$ \sum_{i=1}^{N}\log\pi_{i} P(O, i1=i |\lambda^{old}) + \gamma[\sum_{i=1}^{N}\pi_{i}-1] $$

对其求偏导, 并令导数结果为 0

$$ \frac{\partial}{\partial\pi_{i}}\left[\sum_{i=1}^{N}\log\pi_{i} P(O, i1=i |\lambda^{old}) + \gamma[\sum_{i=1}^{N}\pi_{i}-1]\right] = 0 $$

$$ \frac{1}{\pi_{i}}P(O, i=i|\lambda^{old}) + \gamma = 0 $$

两侧乘以 $\pi_{i}$, 并对i求和, 则:

$$ P(O, i=i|\lambda^{old}) + \gamma\pi_{i} = 0 $$

$$ \gamma = -P(O|\lambda^{old}) $$

将 $\gamma$ 带入偏导式, 有:

$$ \frac{1}{\pi_{i}}P(O, i=i | \lambda^{old}) - P(O|\lambda^{old}) = 0 $$

$$ P(O, i=i | \lambda^{old}) = \pi_{i}P(O|\lambda^{old}) $$

$$ \pi_{i} = \frac{P(O, i=i | \lambda^{old})}{P(O|\lambda^{old})} $$

对于其他两个参数, 也采用类似的方法, 对于 $a_{ij}$:

$$ \sum_{Z}[\sum_{t=1}^{T-1}\log a_{it,i_{t+1} }]P(O, Z | \lambda^{old}) = \sum_{i=1}^{N}\sum_{j=1}^{N}\sum_{t=1}^{T-1}\log a_{ij}P(O, i_{t}=i, i_{t+1}=j |\lambda^{old}) $$

约束条件为 $$ \sum_{j=1}^{N}a_{ij} = 1 $$, 则有:

$$ \sum_{i=1}^{N}\sum_{j=1}^{N}\sum_{t=1}^{T-1}\log a_{ij}P(O, i_{t}=i, i_{t+1}=j |\lambda^{old}) + \gamma[\sum_{j=1}^{N}a_{ij}-1] = 0 $$

求导有:

$$ \sum_{t=1}^{T-1}\frac{1}{a_{ij}}P(O, i_{t}=i, i_{t+1}=j |\lambda^{old}) + \gamma = 0 $$

两侧乘以 $a_{ij}$ 并对 j 求和:

$$ \gamma = -\sum_{t=1}^{T-1}P(O, i_{t}=i|\lambda^{old}) $$

将 $\gamma$ 带入偏导式有:

$$ a_{ij} = \frac{\sum_{t=1}^{T}P(O, i_{t}=i, i_{t+1}=j |\lambda^{old})}{\sum_{t=1}^{T-1}P(O, i_{t}=i|\lambda^{old})} $$

对于 $b_{j(k)}$ 它的约束条件为 $\sum_{k=1}^{M}b_{j}(k) = 1 $, 需要注意的是, 只有在 $o_{t} = v_{k}$ 时 $b_{j}(o_{t})$ 对 $b_{j}(k)$ 的偏导才不为零, 用 $ I(o_{t}=v_{k})$ 表示.求得:

$$ b_{j}(k) = \frac{\sum_{t=1}^{T}P(O, i_{t}=j|\lambda^{old})I(o_{t}=v_{k})}{\sum_{t=1}^{T}P(O, i_{t}=j|\lambda^{old})} $$

上面几个结果,我们可以用前面在前/后向算法那里定义的 $\gamma_{t}(i)$ 和 $\xi_{t}(i,j)$ 来表示.

$$ a_{i,j} = \frac{\sum_{t=1}^{T-1}\xi_{t}(i,j)}{\sum_{t=1}^{T-1}\gamma_{t}(i)} $$

$$ b_{j}(k) = \frac{\sum_{t=1, o_{t}=v_{k}}^{T}\gamma_{t}(j)}{\sum_{t=1}^{T}\gamma_{t}(j)} $$

$$ \pi_{i} = \gamma_{1}(i) $$

### Baum-Welch 算法流程

输入观测数据, 输出学习到的模型参数. 首先初始化参数, 得到一个模型 $\lambda_{0}$. 根据观测序列和模型参数, 我们可以根据 $\gamma$ 和 $\xi$ 求出期望. 在 M 步骤, 我们根据它俩计算新的模型参数使得期望最大化. 如此递推直到模型收敛.

## 解码算法

解码就用维特比算法, 典型的动态规划算法, 对应的最优子结构是 
$$\delta_{t}(i) = \max_{i1,i2,\dots, i_{t-1}}P(i_{t}=i, i_{t-1}, \dots, i_{1}, o_{t}, \dots, o_{1}|\lambda), ~~~i=1,2,\dots, Q $$. 其中 $\delta$ 定义为再时刻 t 状态为 i 的所有单个路径种概率最大值. 

换句话说,我们只需要从时刻 t=1 开始, 递推地计算从时刻1 到 时刻 t 且 时刻 t 的状态为 i的各条部分路径最大的概率. 此时对应的整体路径就是最优路径.

# HMM 模型的缺点

HMM 模型的主要缺点来自于它的两个假设, 即观测独立性假设和齐次马尔可夫性假设. 观测独立性假设导致它不能考虑上下文的特征, 限制了模型的能力.

MEMM 算法抛弃了这个假设, 直接对条件概率建模, 因此可以选择任意的特征. 但 MEMM 是局部归一化, 导致每一个节点都要进行归一化, 所以只能找到局部最优值, 同时也导致了标注偏置的问题, 即凡是语料库中未出现的情况全部忽略掉。

CRF 则更进一步, 它是建立在无向图上的判别式模型. 它完全抛弃了 HMM 的两个不合理假设. 直接采用团和和势函数进行建模，对所有特征进行全局归一化，因此可以求得全局最优解。
