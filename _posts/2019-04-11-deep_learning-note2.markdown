---
layout:     post
title:      "深度学习笔记（二）"
subtitle:   "拉格朗日乘数法与KKT条件"
date:       2019-04-11 00:15:18
author:     "Pelhans"
header-img: "img/background.jpg"
header-mask: 0.3 
catalog:    true
tags:
    - Deep Learning
---

* TOC
{:toc}

# 概览

通常情况下，最优化问题会答题可分为三种情况：无约束条件、等式约束条件、不等式约束条件，对应的算法为费马定理、拉格朗日乘数法、KKT条件。

# 无约束条件

最简单的情况，根据费马定理，解决方法通常是函数对变量求导，零导数函数等于0的点可能是极值点。将结果待会原函数进行验证即可。

# 等式约束条件

设目标函数为$f(x)$，约束条件为$h_{k}(x)$，公式表示为：

$$ \min f(x) $$

$$ h_{k}(x) = 0 ~~ k=1,2,...l $$

对应的解决方法就是消元法或拉格朗日法。求解过程为：

首先定义拉格朗日函数 F(x)：

$$ F(x, \lambda) = f(x) + \sum_{k=1}^{l}\lambda_{k}h_{k}(x) $$

然后解变量的偏导方程：

$$ \frac{\partial F}{\partial x_{i}} = 0 $$

$$\frac{\partial F}{\partial \lambda_{k}} = 0 $$

若有 l 个约束条件，n个变量，那么就会有 l+i 个方程。求出方程组的解为驻点，包含原函数的极值点。将结果带回原方程验证就可得到解。

那么为什么这么做就可以得到最优解呢？根据参考中知乎的回答可以进行理解。

![](/img/in-post/tensorflow/lagelangri.jpg)

我们可以画出F的等高线图，如上图。此时，约束h=c由于只有一个自由度，因此也是图中的一条曲线（红色曲线所示）。显然地，当约束曲线h=c与某一条等高线f=d1相切时，函数f取得极值。**两曲线相切等价于两曲线在切点处拥有共线的法向量。因此可得函数f(x,y)与h(x,y)在切点处的梯度（gradient）成正比**。于是我们便可以列出方程组求解切点的坐标(x,y)，进而得到函数F的极值。

## 一道练习题

已知函数 $ f(x, y, z) = 8xyz$，约束条件为 $\frac{x^{2}}{a^{2}} + \frac{y^{2}}{b^{2}} + \frac{z^{2}}{c^{2}} = 1 $，求函数 $f(x)$的最大值。

解：首先对约束条件改写为右侧为0的形式：

$$ \frac{x^{2}}{a^{2}} + \frac{y^{2}}{b^{2}} + \frac{z^{2}}{c^{2}} - 1 = 0 $$

在前面乘以拉格朗日乘子 $\lambda $与目标函数联合得：

$$ F(x,y,z) = 8xyz + \lambda(\frac{x^{2}}{a^{2}} + \frac{y^{2}}{b^{2}} + \frac{z^{2}}{c^{2}} - 1) $$

对各变量及 $\lambda$求偏导有：

$$ frac{\partial F(x,y,z,\lambda)}{\partial x} = 8yz + \frac{2\lambda x}{a^{2}} = 0 $$

$$ frac{\partial F(x,y,z,\lambda)}{\partial y} = 8xz + \frac{2\lambda y}{b^{2}} = 0 $$

$$ frac{\partial F(x,y,z,\lambda)}{\partial z} = 8xy + \frac{2\lambda z}{c^{2}} = 0 $$

$$ frac{\partial F(x,y,z,\lambda)}{\partial \lambda} = \frac{x^{2}}{a^{2}} + \frac{y^{2}}{b^{2}} + \frac{z^{2}}{c^{2}} - 1 = 0 $$

联立头三个方程有: $bx = ay$$, $$az = cx $$，带入第四个方程得到解：

$$ x = \frac{\sqrt{3}}{3}a~~~y = \frac{\sqrt{3}}{3}b~~~z = \frac{\sqrt{3}}{3}c~~~ $$

带入第四个方程有 $f(x,y,z)$中得到最大体积为：

$$ V_{\max} = f(x, y, z) = \frac{8\sqrt{3}}{9}abc $$

# 不等式约束

KKT条件可以看做拉格朗日乘数法的一种泛化。这里[Karush-Kuhn-Tucker (KKT)条件](https://zhuanlan.zhihu.com/p/38163970)说的很好。我尝试理解，并整理成自己记忆习惯的。

首先我们有带不等式约束的优化问题：

$$ \min f(x)~~~~约束条件为 g(x) \leq 0 $$

根据约束条件定义可行域
$K = x\in R^{n}|g(x) \leq 0 $。设最优解为$x^{*}$。当$g(x^{*}) < 0$时，最优解在K的内部，此时约束条件无效，也就是g(x)不起作用，问题就变成了无约束优化问题，即$\nabla f = 0$且 $\lambda = 0$。

当$g(x^{*})=0$，此时最优解落在K的边界，约束条件生效，$g(x)=0$，和拉格朗日乘数法相同。因此无论是内部解还是边界解，$\lambda g(x)=0 $都成立，成为互补松弛性。整合上述两种情况，最佳解的必要条件包括拉格朗日函数的定常方程式、原始可行性、对偶可行性以及互补松弛性：

$$ \nabla_{x}L = \nabla f + \lambda\nabla g = 0 $$

$$ g(x) \leq 0 $$

$$ \lambda \geq 0 $$

$$ \lambda g(x) = 0 $$

这些条件合称 KKT条件。当最小化f(x)变成最大化时，$\lambda \leq 0$。

## 例子

![](/img/in-post/tensorflow/kkt_exam.png)

# 参考

[深入理解拉格朗日乘子法（Lagrange Multiplier) 和KKT条件](https://www.cnblogs.com/sddai/p/5728195.html)    
[如何理解拉格朗日乘子法？](https://www.zhihu.com/question/38586401/answer/105273125)    
[Karush-Kuhn-Tucker (KKT)条件](https://zhuanlan.zhihu.com/p/38163970)
