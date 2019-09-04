---
layout:     post
title:      "NLP 手册"
subtitle:   "EM 算法"
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

# 一般形式的 EM 算法

## 算法原理

期望最大化算法或者叫 EM 算法,是寻找具有潜在变量的概率模型的最大似然解的一种通用的方法.每次迭代由两步组成, 分别是E步骤计算期望, M步骤优化期望求极大.  我们假设直接优化 
$p(X|\theta)$ 比较困难, 但是最优化完整数据(完整数据是同时给定观测变量 X 和 隐含变量 Z, 只给观测数据X 叫不完全数据)似然函数
$p(X,Z|\theta)$ 就容易得多. 其中$\theta$ 是参数, 它控制X, Z的联合概率分布. 

现在考虑一个概率模型, 我们的目标是最大化似然函数:

$$ p(X|\theta) = \sum_{Z} p(X,Z|\theta) $$

因为最优化
$ p(X|\theta) $ 比较麻烦, 但是优化右面的就容易很多, 使用概率乘积规则:

$$ p(X,Z |\theta) = p(Z|X,\theta)p(X|\theta, Z) $$

$$ ln p(X, Z | \theta) = ln p(Z | X, \theta) + ln p(X|\theta, Z) $$
 
引入一个定义在潜在变量上的分布 q(Z), 将上式带入可得.

$$
\begin{aligned}
ln p(X | \theta) &= \sum_{Z}q(Z) ln(p(X,Z | \theta)) \\
& =\sum_{Z}q(Z)\{ln p(Z | X, \theta) + ln p(X|\theta) - ln q(Z)\} + \sum_{Z}q(Z)\{-ln p(Z|X, \theta) + ln q(Z)\} \\
& =\sum_{Z}q(Z)ln\{\frac{p(X,Z|\theta)}{q(Z)}\} + (-\sum_{Z}q(Z)ln\{\frac{p(Z|X,\theta)}{q(Z)}\} ) \\
& =l(q, \theta) + KL(q||p)
\end{aligned}
$$

其中上式第二项可以看作是q(Z)与后验概率分布 
$p(Z|X, \theta)$ 之间的KL 散度.  我们知道 KL 散度是大于等于0的, 当且仅当 
$q(Z) = p(Z|X, \theta)$ 时等号成立.因此
$l(q,\theta) \leq ln p(X|\theta)$, 也就是说, $l(q,\theta)$ 是
$ln p(X|\theta)$ 的下界. $l(q, \theta)$ 的具体形式是 概率分布 q(Z) 的泛函, 并且是 参数 $\theta$ 的一个函数.

我们使用上述公式来定义 EM 算法, 并证明它确实最大化了对数似然函数. 假设参数向量的当前值是 $\theta^{old}$, 在 E 步骤, 下界 $l(q, \theta^{old})$ 被最大化, 而 $\theta^{old}$ 保持固定. 我们发现, 
$ln p(X|\theta^{old})$ 不依赖于 q(Z), 因此 $l(q, \theta^{old})$ 的最大值出现在 KL 散度等于零的时候, 换句话说, 最大值出现在 q(Z) 与 后验概率分布 
$p(Z | X, \theta^{old})$ 相等的时候. 此时下界等于对数似然函数. 下界的形式为:

$$ 
\begin{aligned}
l(q, \theta) &= \sum_{Z}p(Z | X, \theta^{old})ln p(X,Z|\theta) - \sum_{Z}p(Z|X, \theta^{old})ln p(Z | X, \theta^{Old}) \\
&= Q(\theta, \theta^{old}) + const. 
\end{aligned}
$$

在 M 步骤, 分布 q(Z) 保持固定, 下界 $l(q, \theta)$ 关于 $\theta$ 进行最大化, 得到了某个新的 $\theta^{new}$. 在这个过程中,  由于概率分布 q 是由旧的参数确定, 并且在M 步保持固定,  因此它将不会等于新的后验概率分布 
$p(Z | X,\theta)$, 从而 KL 散度不等于零, 这就给下一轮的下界留出了优化空间, 即对数似然函数的增加量大于下界的增加量. 如此迭代直到在 M 收敛即期望不再增加达到最大. 上面的 $Q(\theta, \theta^{old})$ 可以叫 Q 函数.

上面的讨论是基于离散变量的, 若变为连续变量, 则将求和换为积分即可.
