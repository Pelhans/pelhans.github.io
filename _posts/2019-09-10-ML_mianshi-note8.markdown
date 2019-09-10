---
layout:     post
title:      "主成分分析 PCA"
date:       2019-09-10 00:15:18
author:     "Pelhans"
header-img: "img/prml.jpg"
header-mask: 0.3 
catalog:    true
tags:
    - PRML
---


* TOC
{:toc}

# 主成分分析

主成分分析，或者称为PCA，是一种被广泛使用的技术，应用领域包括维度降低。有损数据压缩、特征抽取、数据可视化。它也被成为 Karhunen-Loeve 变换。有两种经常使用的PCA的定义，他们会给出同样的算法。**PCA可以被定义为数据在低维线性空间上的正交投影，这个线性空间被称为主子空间，使得投影数据的方差被最大化。等价地，它也可以被定义为使得平均投影代价最小的线性投影。平均投影代价是指数据点和它们投影之间的平均平方距离**。

## 最大方差形式

假设有一组观测数据集$x_{n}$，维度为D，目标是将数据投影到维度为M(M<D)的空间中，同时最大化投影数据的方差。

考虑在一维空间上的投影，我们可以使用D维向量$u_{1}$定义这个空间的方向。为了方便，我们假定选择一个单位向量，这样，每个数据点$x_{n}$被投影到一个标量值$u_{1}^{T}x_{n}$上。投影数据的均值是$u_{1}^{T}\bar{x}$，投影数据的方差为：

$$ 
\begin{aligned}
\frac{1}{N}\sum_{n=1}^{N}\{u_{1}^{T}x_{n} - u_{1}^{T}\bar{x}\}^{2} &=\frac{1}{N}\sum_{n=1}^{N}\{ u_{1}^{T}(x_{n} - \bar{x}) \}^{2} \\
&= \frac{1}{N}\sum_{n=1}^{N}\{u_{1}^{T}(x_{n} - \bar{x})(x_{n} - \bar{x})^{T}u_{1}\} \\
&= u_{1}^{T}Su_{1}
\end{aligned}
$$

其中S 是数据的协方差矩阵,定义为

$$ S = \frac{1}{N}\sum_{n=1}^{N}(x_{n} - \bar{x})(x_{n} - \bar{x})^{T} $$

我们现在关于$u_{1}$最大化投影方差$u_{1}^{T}Su_{1}$。采用拉格朗日乘数法，以$u_{1}$的归一化条件为限制，写出拉格朗日函数

$$ L = u_{1}^{T}S u_{1} + \lambda_{1}(1-u_{1}^{T}u_{1}) $$

对其进行最大化, 我们对 $u_{1}$ 求导, 并令导数为0. 已知求导公式

$$ \frac{d x^{T}A x}{d x} = (A + A^{T})x $$

$$ \frac{d x^{T}x}{d x} = 2x $$

我们可得

$$ (S + S^{T})u_{1} - 2\lambda_{1}u_{1} = 0 $$

S 是一个对阵矩阵, 且对角线是各个维度的方差. 那么 $S = S^{T} $, 因此移项后我们看到驻点满足：

$$ Su_{1} = \lambda_{1}u_{1} $$

这表明$u_{1}$一定是S的一个特征向量。如果我们左乘$u_{1}^{T}$，使用$u_{1}^{T}u_{1}=1$，我们看到方差为：

$$u_{1}^{T}Su_{1} = \lambda_{1} $$

因此当我们**将$u_{1}$设置为与最大的特征值$\lambda_{1}$的特征向量相等时，方差会达到最大值，这个特征向量被称为第一主成分**。对于其他主成分，我们可以考虑那些与现有方向正交的所有可能方向中，将新的方向选择为最大化投影方差的方向。以此类推得到协方差矩阵S的M个特征向量$u_{1},\dots,u_{M}$，对应于M个最大特征值$\lambda_{1},\dots,\lambda_{M}$。

使用的话,我们小计算数据集的均值喝协方差矩阵S, 然后寻找 S 的对于 M 个最大特征值的 M 个特征向量. 将样本投影到选取的特征向量上实现降维.

## 最小误差形式

给定数据集$$D = \{\overrightarrow{x}_{1}, \dots, \overrightarrow{x}_{N}\} $$, 其中 $$\overrightarrow{x}_{i} \in R^{n}$$. 假设样本经过了中心化, 即每个样本都减去了均值.中心化的原因是中心化后的常规线性变换变成了绕原点的旋转变换, 也就是坐标变换.

现在我们的目的是要把原数据降到 d 为空间中, $d < n$, 所以我们用一个变换矩阵 W, $$ W = (\overrightarrow{w}_{1}, \overrightarrow{w}_{2}, \dots, \overrightarrow{w}_{d}) $$. 它是 $n\times d$ 的, 因此经过 $$\overrightarrow{z} = W^{T}\overrightarrow{x}$$, 我们将样本 $$\overrightarrow{x}$$ 降低到 d 维度.

根据坐标变换的性质, 我们有

$$ ||\overrightarrow{w}_{i}|| = 1 ,~~~~i=1,2,\dots, d $$

$$ \overrightarrow{w}_{i}\overrightarrow{w}_{j} = \delta_{ij} $$

现在考虑对 $$ \overrightarrow{z}$$ 进行重构, 重构之后的样本为 $$ \hat{\overrightarrow{x}} = W\overrightarrow{z}$$. 也就是说, 我们利用 W 和 z 来得到 x 的新表示. 我们希望这个重构出来的 $$ \hat{\overrightarrow{x}} $$ 与原 $$\overrightarrow{x}$$ 之间差别尽可能的小, 因此对于整个数据集而言, 重建样本与原始样本的误差为:

$$ J = \sum_{i=1}^{N}||\hat{\overrightarrow{x}} - \overrightarrow{x}||^{2} = WW^{T}\overrightarrow{x}_{i} - \overrightarrow{x}_{i}||^{2} $$

对 W 进行展开

$$ WW^{T}\overrightarrow{x}_{i} = \sum_{j=1}^{d}\overrightarrow{w}_{j}(\overrightarrow{w}_{j}^{T}\overrightarrow{x}_{i}) $$

括号内部的是标量,  所以上式可以进一步表示为

$$ \hat{\overrightarrow{x}} =  WW^{T}\overrightarrow{x}_{i} = \sum_{j=1}^{d}(\overrightarrow{w}_{j}^{T}\overrightarrow{x}_{i})\overrightarrow{w}_{j} $$

继续, 我们现在想把平方内的展开, 利用向量内积性质 $$ \overrightarrow{x}_{i}^{T} \hat{\overrightarrow{x}}_{i} = \hat{\overrightarrow{x}}_{i}^{T}\overrightarrow{x}_{k} $$ 得到

$$
\begin{aligned}
||\overrightarrow{x}_{i} - \hat{\overrightarrow{x}}_{i} ||^{2} &= (\overrightarrow{x}_{i} - \hat{\overrightarrow{x}}_{i})^{T}(\overrightarrow{x}_{i} - \hat{\overrightarrow{x}}_{i}) \\
&= \overrightarrow{x}_{i}^{T}\overrightarrow{x}_{i} + \hat{\overrightarrow{x}}_{i}^{T}\hat{\overrightarrow{x}}_{i} - \overrightarrow{x}_{i}^{T}\hat{\overrightarrow{x}}_{i} - \hat{\overrightarrow{x}}_{i}^{T}\overrightarrow{x}_{i} \\
&= \overrightarrow{x}_{i}^{T}\overrightarrow{x}_{i} + \hat{\overrightarrow{x}}_{i}^{T}\hat{\overrightarrow{x}}_{i} - 2\overrightarrow{x}_{i}^{T}\hat{\overrightarrow{x}}_{i}
\end{aligned}
$$

第一项是个常数, 与 W 无关. 我们利用前面投影的向量继续展开第二第三项

$$
\begin{aligned}
\overrightarrow{x}_{i}^{T}\hat{\overrightarrow{x}}_{i} &=  \overrightarrow{x}_{i}^{T}\sum_{k=1}^{d}(\overrightarrow{w}_{k}^{T}\overrightarrow{x}_{i})\overrightarrow{w}_{k} \\
&= \sum_{k=1}^{d}(\overrightarrow{w}_{k}^{T}\overrightarrow{x}_{i})\overrightarrow{x}_{i}^{T}\overrightarrow{w}_{k} \\
&= \sum_{k=1}^{d} \overrightarrow{w}_{k}\overrightarrow{x}_{i}\overrightarrow{x}_{i}^{T}\overrightarrow{w}_{k} 
\end{aligned}
$$

$$
\begin{aligned}
\hat{\overrightarrow{x}}_{i}^{T}\hat{\overrightarrow{x}}_{i} &= \left(\sum_{k=1}^{d}(\overrightarrow{w}_{k}^{T}\overrightarrow{x}_{i})\overrightarrow{w}_{k} \right)^{T}\left(\sum_{h=1}^{d}(\overrightarrow{w}_{h}^{T}\overrightarrow{x}_{i})\overrightarrow{w}_{h} \right) \\
&= \sum_{k=1}^{d}\sum_{h=1}^{d}\left((\overrightarrow{w}_{k}^{T}\overrightarrow{x}_{i})\overrightarrow{w}_{k} \right)^{T}\left( (\overrightarrow{w}_{h}^{T}\overrightarrow{x}_{i})\overrightarrow{w}_{h}\right)
\end{aligned}
$$

由于正交性质, 上式当 $ k\neq h$时 对应的项为0, 因此只剩下 d 项

$$
\begin{aligned}
\hat{\overrightarrow{x}}_{i}^{T}\hat{\overrightarrow{x}}_{i} &= \sum_{k=1}^{d}\left((\overrightarrow{w}_{k}^{T}\overrightarrow{x}_{i})\overrightarrow{w}_{k} \right)^{T}\left((\overrightarrow{w}_{k}^{T}\overrightarrow{x}_{i})\overrightarrow{w}_{k}\right) \\
&= \sum_{k=1}^{d}(\overrightarrow{w}_{k}^{T}\overrightarrow{x}_{i})(\overrightarrow{w}_{k}^{T}\overrightarrow{x}_{i}) \\
&= \sum_{k=1}^{d}(\overrightarrow{w}_{k}^{T}\overrightarrow{x}_{i})(\overrightarrow{x}_{i}^{T}\overrightarrow{w}_{k}) \\
&= \sum_{k=1}^{d}\overrightarrow{w}_{k}^{T}\overrightarrow{x}_{i}\overrightarrow{x}_{i}^{T}\overrightarrow{w}_{k} 
\end{aligned}
$$

上式第二步是利用正交性质 $$ \overrightarrow{w}_{i}^{T}\overrightarrow{w}_{i} = 1 $$ 消除对应项. 第三部利用标量的转置等于它本身, 对第二个括号内进行了转置. 我们发现 $$\hat{\overrightarrow{x}}_{i}^{T}\hat{\overrightarrow{x}}_{i}$$ 和 $$\overrightarrow{x}_{i}^{T}\hat{\overrightarrow{x}}_{i} $$ 得到的结果是一样的.

注意到 $$\sum_{k=1}^{d}\overrightarrow{w}_{k}^{T}\overrightarrow{x}_{i}\overrightarrow{x}_{i}^{T}\overrightarrow{w}_{k}$$ 实际上就是矩阵 $$W^{T}\overrightarrow{x}_{i}\overrightarrow{x}_{i}^{T}W $$ 的迹(对角线元素之和), 于是我们可以将上面求得结果带入, 误差继续简化.

$$ 
\begin{aligned}
||\overrightarrow{x}_{i} - \hat{\overrightarrow{x}}_{i} ||^{2} &= -\sum_{k=1}^{d}\overrightarrow{w}_{k}^{T}\overrightarrow{x}_{i}\overrightarrow{x}_{i}^{T}\overrightarrow{w}_{k} \\
&= -tr(W^{T}\overrightarrow{x}_{i}\overrightarrow{x}_{i}^{T}W)
\end{aligned}
$$

对所有的数据 i 求和,得到整体损失

$$ arg\max_{W}\sum_{i=1}^{N}tr(W^{T}\overrightarrow{x}_{i}\overrightarrow{x}_{i}^{T}W) $$

$$ s.t. ~~~~W^{T}W = 1 $$

把 $$\overrightarrow{x}_{i}$$ 写成矩阵形式, 则问题最终的可表述为

$$ arg\max_{W}tr(W^{T}X X^{T}W) $$

$$ s.t. ~~~~W^{T}W = 1 $$

因此我们只需要对矩阵 $X^{T}X$ 进行特征值分解, 对求得的特征值排序, 取前 d 个特征值对应的单位特征向量即可.

## 低空间维数的选取

通常有两种方法

* 作为超参, 选取使得后面的算法表现最好的那个    
* 从算法的原理角度设置一个阈值, 然后选取使得下式成立的最小的 d

$$ \frac{\sum_{i=1}^{d}\lambda_{i}}{\sum_{i=1}^{n}\lambda_{i}} \geq t $$

## 证明两种方法的等价性

从物理意义上来看, 最小误差是给定协方差矩阵 $X^{T}X$, 通过坐标变换将其对角化为对角线为特征值的对角矩阵. 这相当于再新的坐标系中

* 任意一对特征之间的协方差为0.     
* 单个特征的方差为$\lambda_{i}$

我们选取前d大的 $\lambda$ 意味着要数据在每个维度上尽可能分散, 且任意两个维度互相不相关, 这个恰恰是最大方差方法要求的.

从公式角度来看, 对于样本点 $$\overrightarrow{x_{i}}$$, 它在降维后空间中的投影是 $$\overrightarrow{z_{i}} = W^{T}\overrightarrow{x}$$, 由于样本数据进行了中心化, 那么投影后样本点的方差就是

$$ \sum_{i=1}^{N}\overrightarrow{z_{i}}\overrightarrow{z}_{i}^{T} = \sum_{i=1}^{N}W^{T}\overrightarrow{x}_{i}\overrightarrow{x}_{i}^{T}W = tr(W^{T}X^{T}XW)$$

这个和最小化误差方法得到的结果是一样的.

## LDA 与 PCA 的关系

相同点是 都可以用于降维,

区别

* LDA 用到了类别信息, 是有监督算法, PCA 是无监督算法    
* LDA 考虑的是向类别区分最大的方向投影, PCA 考虑的是向方差最大的方向投影

# 概率PCA

PCA也可以被视为概率潜在变量模型的最大似然解，PCA的这种形式被称为概率PCA，它与因子分析密切相关。

概率PCA是线性高斯框架的一个简单的例子，其中所有的边缘概率分布和条件概率分布都是高斯分布。首先显示引入潜在变量z，对应于主成分子空间。接下来我们定义潜在变量上的一个高斯先验分布p(z)以及高斯条件概率分布
$p(x|z)$：

$$ p(z) = \mathcal{N}(z | 0, I) $$

$$ p(x | z) = \mathcal{N}(x | Wz + \mu, \sigma^{2}I) $$

其中x的均值是z的一个一般线性函数，由$D\times M$的矩阵W和D维向量$\mu$控制。W的列张成了数据空间的一个线性子空间，对应于主子空间。p(z)被定义为零均值单位协方差的高斯是因为更一般的高斯分布会产生一个等价的概率模型。

假如从生成式的观点看待概率PCA模型的话，观测值的一个采样值可以这样获得：首先为潜在变量选择一个值，然后以这个潜在变量的值为条件，对观测变量采样。具体来说，D维观测变量x由M维潜在变量z的一个线性变换附加一个高斯“噪声”定义，即：

$$ x = Wz + \mu + \epsilon $$

其中$\epsilon$是一个D维零均值高斯分布的噪声变量。可以看出，这个框架基于的是从潜在空间到数据空间的一个映射，从数据空间到潜在空间的逆映射可以通过使用贝叶斯定理的方式得到。

我们希望使用最大似然的方式确定$W, \mu, \sigma^{2} $的值。概率PCA模型可以表示为一个有向图，如下图所示，则对应的对数似然函数为：

$$ \ln p(X | mu, W, \sigma^{2}) = \sum_{n=1}^{N}\ln p(x_{n} | W, \mu, \sigma^{2}) = -\frac{ND}{2}\ln(2\pi) - \frac{N}{2}\ln|C| - \frac{1}{2}\sum_{n=1}^{N}(x_{n}-\mu)^{T}C^{-1}(x_{n}-\mu) $$

![](/img/in-post/prml_note8/p13.png)

$\mu$的解及W和$\sigma^{2}$的近似封闭解为：

$$ \mu = \bar{x} $$

$$ W_{ML} = U_{M}(L_{M} - \sigma^{2}I)^{\frac{1}{2}}R $$

$$ \sigma^{2}_{ML} = \frac{1}{D-M}\sum_{i=M+1}^{D}\lambda_{i} $$

其中$U_{M}$是一个$D\times M$的矩阵。当M个特征向量被选为前M个最大的特征值所对应的特征向量时，对数似然函数可以达到最大值，其他所有的解都是鞍点。假定特征向量按照对应的特征值的大小降序排列，从而M个主特征向量是$u_{1},\dots,u_{M}$，从而W的列定义了标准PCA的主子空间。而$\sigma_{ML}^{2}$是与丢弃的维度相关联的平均方差，它可以被看做是M为潜在空间的一个旋转矩阵。

## 因子分析

因子分析是一个线性高斯潜在变量模型，它与概率PCA密切相关。**它的定义与概率PCA唯一的差别是给定潜在变量z的条件下观测变量x的条件概率分布的协方差矩阵是一个对角矩阵而不是各项同性的协方差矩阵**，即：

$$ p(x | z) = \mathcal{N}(x | Wz + \mu, \Psi) $$

其中$\Psi$是一个$D\times D$的对角矩阵。本质上讲，**因子分析模型对数据的观测协方差的结构解释为：表示出矩阵$\Psi$中与每个坐标相关联的独立变量，然后描述矩阵W中的变量之间的协方差**。

从潜在变量密度模型角度来看因子分析，我们感兴趣的是潜在空间的形式，而不是描述它的具体的坐标系的选择。**如果我们想要移除与潜在空间旋转相关联的模型的退化，那么我们必须考虑非高斯的潜在变量分布，这就产生了独立成分分析(ICA)模型**。

# 核PCA

若我们将核替换的方法应用到主成分分析中，从而得到了一个非线性的推广，被称为核PCA(kernel PCA)。我们希望避免直接在特征空间中进行计算，因此我们完全根据核函数来建立算法的公式。在中心化之后，投影的数据点为：

$$ \tilde{\phi}(x_{n}) = \phi(x_{n}) - \frac{1}{N}\sum_{l=1}^{N}\phi(x_{l}) $$

从而Gram矩阵的对应元素为：

$$
\begin{aligned}
\tilde{K}_{nm} = & \tilde{\phi})(x_{n})^{T}\tilde{\phi}(x_{m}) \\
        = & \phi(x_{n})^{T}\phi(x_{m}) - \frac{1}{N}\sum_{l=1}^{N}\phi(x_{n})^{T}\phi(x_{l})-\frac{1}{N}\sum_{l=1}^{N}\phi(x_{l})^{T}\phi(x_{m}) - \frac{1}{N^{2}}\sum_{j=1}^{N}\sum_{l=1}^{N}\phi(x_{j})\phi(x_{l}) \\
        = & k(x_{n}, x_{m}) - \frac{1}{N}\sum_{l=1}^{N}k(x_{l}, x_{m}) - \frac{1}{N}\sum_{l=1}^{N}k(x_{n}, x_{l}) + \frac{1}{N^{2}}\sum_{j=1}^{N}\sum_{l=1}^{N}k(x_{j}. x_{l}) 
\end{aligned}
$$

因此，**我们可以只使用核函数来计算$\tilde{K}$，然后使用$\tilde{K}$确定特征值和特征向量。注意，如果我们使用线性核$k(x, x^{'}) = x^{T}x^{'}$，那么我们就恢复了标准的PCA算法**。
