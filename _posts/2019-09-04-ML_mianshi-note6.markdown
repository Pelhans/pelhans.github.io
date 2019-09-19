---
layout:     post
title:      "NLP 手册"
subtitle:   "朴素贝叶斯分类器"
date:       2019-09-04 00:15:18
author:     "Pelhans"
header-img: "img/kg_bg.jpg"
header-mask: 0.3 
catalog:    true
tags:
    - PRML
---


* TOC
{:toc}

# 朴素贝叶斯法

朴素贝叶斯法是基于贝叶斯定理与条件特征独立假设的分类方法. 它首先基于特征条件独立假设学习输入, 输出的联合概率分布. 而后基于此模型, 对给定输入 x, 利用贝叶斯定理求出后验概率最大的输出 y.因此可以看出它是一个生成式模型.

具体来说, 设输入 为 $\overrightarrow{x} = (x_{1}, x_{2}, \dots, x_{n})^{T}$ 为定义在 n 维空间上的随机向量, $ Y \in \{c_{1}, c_{2},\dots, c_{k}\}$. 假设训练数据集 $$D = \{(\overrightarrow{x}_{1}, \tilde{y}_{1}), (\overrightarrow{x}_{2}, \tilde{y}_{2}), \dots, (\overrightarrow{x}_{N}, \tilde{y}_{N}) \}$$ 由联合分布 $p(\overrightarrow{x}, y)$ 独立同产生. 

朴素贝叶斯法对联合概率分布建模. 具体来说是对:

* 先验概率分布 $p(y)$    
* 条件概率分布
$ p(\overrightarrow{x}|y) = p(x_{1}, x_{2}, \dots, x_{n}| y) $$

朴素贝叶斯假设特征条件独立, 即:

$$ p(\overrightarrow{x}|y) = p(x_{1}, x_{2}, \dots, x_{n}| y) = \prod_{j=1}^{n}p(x_{j}|y) $$

该假设使得朴素贝叶斯法变的简单, 但实际使用时常常不满足该假设, 因此会损失一定的准确率. 但该模型由于简单, 效果好, 依旧是常用算法.

## 优化目标

根据贝叶斯定理:

$$
\begin{aligned}
p(y | \overrightarrow{x}) &= \frac{p(\overrightarrow{x}|y)p(y)}{\sum_{y^{'}}p(\overrightarrow{x}|y^{'})p(y^{'})} \\
&= \frac{p(y)\prod_{i=1}^{n}p(x_{i}|y)}{\sum_{y^{'}}p(\overrightarrow{x}|y^{'})p(y^{'}) } 
\end{aligned}
$$

我们要使得该条件概率最大的类别作为分类结果, 则朴素贝叶斯分类器表示为:

$$ f(\overrightarrow{x}) = arg\max_{y\in Y}\frac{p(y)\prod_{i=1}^{n}p(x_{i}|y)}{\sum_{y^{'}}p(\overrightarrow{x}|y^{'})p(y^{'}) } $$

由于分母与 y 无关, 因此我们只优化分子:

$$ f(\overrightarrow{x}) = arg\max_{y\in Y}p(y)\prod_{i=1}^{n}p(x_{i}|y) $$

## 求解

前面说过, 在朴素贝叶斯法中, 学习意味着估计概率 $p(y=c_{k})$, 
$ p(x_{j}=a_{jl}|y=c_{k})$. 其中 $a_{jl}$ 表示第j个特征的第l个值. 可以用极大似然估计相应概率.

首先写出似然函数, 并取对数:

$$ 
\begin{aligned}
l &= \log\prod_{i=1}^{N}p(x_{i}, y_{i}) \\
&= \log\prod_{i=1}^{N}p(x_{i}|y_{i})p(y_{i}) \\
&= \log\prod_{i=1}^{N}\left(\prod_{j=1}^{n}p(x_{i}^{j}|y_{i})\right)p(y_{i}) \\
&= \sum_{i=1}^{N}\left(\log p(y_{i}) + \sum_{j=1}^{m}\log p(x_{i}^{j}|y_{i})\right) \\
&= \sum_{i=1}^{N}\left[\sum_{k=1}^{K}\log p(y=c_{k})^{I(y_{i}=c_{k})} + \sum_{j=1}^{n}\sum_{l=1}^{S_{j}}\log p(x^{j}=a_{jl}|y_{i}=c_{k})^{I(x_{i}^{j}=a_{jl}, y_{i}=c_{k})} \right] \\
&= \sum_{i=1}^{N}\left[\sum_{k=1}^{K}I(y_{i}=c_{k})\log p(y=c_{k}) + \sum_{j=1}^{n}\sum_{l=1}^{S_{j}}I(x_{i}^{j}=a_{jl}, y_{i}=c_{k})\log p(x_{i}^{j}=a_{jl}| y_{i}=c_{k}) \right]
\end{aligned}
$$

而后运用极大似然估计的方法求那两个概率. 我们高兴的发现, 这两个概率分别处于不同的两项中,因此可以分别进行优化.首先我们求 $p(y=c_{k})$.

根据约束条件 $\sum_{k=1}^{K}p(y_{i}=c_{k}) = 1 $, 运用拉格朗日乘子法, 可写出:

$$ \sum_{i=1}^{N}\sum_{k=1}^{K}I(y_{i}=c_{k})\log p(y_{i}=c_{k}) + \gamma\left[\sum_{k=1}^{K}p(y=c_{k}) - 1 \right] $$

两侧对 $p(y_{i}=c_{k})$ 求偏导, 并令导数等于0

$$ \frac{(y_{i}=c_{k})}{p(y_{i}=c_{k})} + \gamma = 0 $$

两侧同时乘以 p(y_{i}=c_{k}), 并对 k求和, 得到 

$$\gamma = -\sum_{k=1}^{K}I(y_{i}=c_{k}) $$

将 $\gamma$ 带入并求导可得

$$ \frac{I(y_{1}=c_{k})}{p(y_{i} = c_{k})} - \sum_{k=1}^{K}I(y_{i}=c_{k}) = 0 $$

$$ p(y_{i} = c_{k}) = \frac{I(y_{1}=c_{k})}{\sum_{k=1}^{K}I(y_{i}=c_{k})} $$

两侧对 i 求和;

$$ p(y=c_{k}) = \frac{\sum_{i=1}^{N}I(y_{i}=c_{k})}{\sum_{i=1}^{N}\sum_{k=1}^{K}I(y_{i}=c_{k})} = \frac{\sum_{i=1}^{N}I(y_{i}=c_{k})}{N} $$

类似地, 我们可以根据约束 
$$\sum_{l=1}^{S_{j}}p(x^{j}=a_{jl}|y=c_{k}) = 1 $$ 求得 
$p(x_{j}=a_{jl}|y=c_{k})$ 的极大似然估计为:

$$ p(x_{j}=a_{jl}|y=c_{k}) = \frac{\sum_{i=1}^{N}I(x_{ij}=a_{jl}, y_{i}=c_{k})}{\sum_{i=1}^{N}I(y_{i}=c_{k})} $$

## 算法流程

有了上面两个公式, 我们就可以进行分类了. 首先输入为训练集 $$D = \{(\overrightarrow{x}_{1}, y_{1}), (\overrightarrow{x}_{2}, y_{2}), \dots, (\overrightarrow{x}_{N}, y_{N})\} $$. 其中 $$ \overrightarrow{x}_{i} = (x_{i,1}, x_{i,2}, \dots, x_{i,n})$$, $ x_{ij}$为第 i 个样本的第 j 个特征. 其中 $$x_{ij} \in \{a_{j1}, a_{j2}, \dots, a_{j, S_{j}}\} $$, $a_{jl}$ 是第 j 个特征可能取到的第 l 个值. 输出就是样本 $$\overrightarrow{x}$$ 的分类.

首先再输入中计算先验概率以及条件概率:

$$ p(y=c_{k}) = \frac{1}{N}\sum_{i=1}^{N}I(y_{i}=c_{k}) $$

$$ p(x_{j}=a_{jl}|y=c_{k}) = \frac{\sum_{i=1}^{N}I(x_{ij}=a_{jl}, y_{i}=c_{k})}{\sum_{i=1}^{N}I(y_{i}=c_{k})} $$

其中 $j=1,2,\dots, n, ~~~ l=1,2,\dots, s_{j} , ~~~ k=1,2,\dots, K$.

对于给定的实例 $\overrightarrow{x}$, 计算

$$ p(y=c_{k})\prod_{j=1}^{n}p(x_{j}|y=c_{k}) $$

根据上式得到的各类别概率, 输出概率最高的类别作为分类结果.

# 贝叶斯估计

我们看到, 在估计概率 
$p(x_{i}|y)$ 的过程中, 分母 $\sum_{i=1}^{N}I(y_{i}=c_{k})$ 可能为 0. 这可能是由于样本太少导致的, 但在实际中可能并不为0. 对应的解决方案是采用贝叶斯估计.

假设第 j 个特征  $x_{j}$ 可能的取值为 $\{a_{j1}, a_{j2}, \dots, a_{j, s_{j}}\} $, 贝叶斯估计假设在每个取值上都有一个先验计数 $\lambda$, 这样即使 $c_{k}$ 的样本数为0, 也可以给出一个概率估计.

$$ p_{\lambda}(x_{j}=a_{jl} | y=c_{k}) = \frac{\sum_{i=1}^{N}I(x_{ij}=a_{jl}, y_{i}=c_{k})+\lambda}{\sum_{i=1}^{N}I(y_{i}=c_{k})+s_{j}\lambda} $$

上面分母中, $s_{j}$ 是若 $c_{k}$ 的样本数为0, 那么 特征 $x_{j}$ 取每个值的概率是等可能的, 为 $\frac{1}{s_{j}}$. 

$$ p_{\lambda}(y=c_{k}) = \frac{1}{N+K\lambda}[\sum_{i=1}^{N}I(y_{i}=c_{k})+\lambda] $$

当 $\lambda = 1 $ 时, 就是拉普拉斯平滑.

# 模型优缺点

优点:

* 生成式模型, 通过计算概率进行分类, 可以用来处理多分类问题.    
* 对小规模数据表现很好, 适合多分类任务, 适合增量式训练, 算法也比较简单.

缺点:

* 对输入数据的表达形式很敏感.    
* 由于朴素贝叶斯的特征独立假设, 所以会带来一些准确率上的损失.    
* 需要计算先验概率, 分类决策存在错误率.
