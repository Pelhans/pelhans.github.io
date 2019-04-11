---
layout:     post
title:      "Tensorflow 笔记（三）"
subtitle:   "牛顿法与BFGS"
date:       2019-04-11 00:15:18
author:     "Pelhans"
header-img: "img/tensorflow.jpg"
header-mask: 0.3 
catalog:    true
tags:
    - Deep Learning
---

> 眼睛看过只是别人的，整理出来并反复学习才是自己的。

* TOC
{:toc}

# 概览

与一阶的相比，二阶梯度方法使用二阶导数改进了优化。最广泛使用的二阶方法是牛顿法。

# 牛顿法

牛顿法是给予二阶泰勒级数展开在某点 $\theta_{0}$附近来近似 $J(\theta)$的方法，它忽略了高于二阶的导数：

$$ f(x) = f(x_{0}) + \nabla f(x_{0})^{T}(x-x_{0}) + \frac{1}{2}(x-x_{0})^{T}\nabla^{2}f(x_{0})(x-x_{0}) + O((x-x_{0})^{2}) $$

对上式两侧同时求梯度，得到函数的导数为：

$$ \nabla f(x) = \nabla f(x_{0}) + \nabla^{2}f(x_{0})(x-x_{0}) $$

令梯度等于0：

$$ \nabla f(x_{0}) + \nabla^{2}f(x_{0})(x-x_{0}) = 0 $$

可得：

$$ x = x_{0} - (\nabla^{2}f(x_{0}))^{-1}\nabla f(x_{0}) $$

将 $\nabla^{2}f(x_{0}) $ 记为 H, $\nabla f(x_{0})$ 记为 g，则得到更新公式：

$$ x_{k+1} = x_{k} - H^{-1}_{k}g_{k} $$

其中 $-H^{-1}g $ 称为牛顿方向。迭代的终止条件是梯度的模接近于0或者函数值下降小于指定阈值。

对于局部的二次函数(具有正定的H)，牛顿法会直接调到极小值。对于非二次的表面，只要 Hessian 矩阵保持正定，牛顿法能够迭代地应用。

但在深度学习中，目标函数的表面通常是非凸的，因此使用牛顿法是有问题的。如在靠近鞍点处牛顿法会朝着错误的方向移动，这种情况可以通过正则化 Hessian 来避免。另一个缺点是，牛顿法需要计算 Hessian 矩阵的逆，这将带来非常大的计算量，若 Hessian 不可逆，则牛顿法将失效。

# 拟牛顿法

对于上面提到的牛顿法的缺点，提出了一些改进方法，典型的代表是拟牛顿算法(Quasi-Newton)。拟牛顿算法的思想是不计算目标函数的 Hessian 矩阵然后求逆，而是通过其他手段得到 Hessian 矩阵或者 H的逆矩阵的近似矩阵。具体做法是构造一个近似 Hessian矩阵或逆矩阵的正定对称矩阵，用它来进行牛顿法的迭代。

这个近似矩阵不是随意构造的，需要满足拟牛顿条件：

$$ s_{k} = x_{k+1} - x_{k} $$

$$ y_{k} = g_{k+1} - g_{k} $$

$$ s_{k} \simeq H^{-1}_{k+1} y_{k} $$

该条件的来源就是对泰勒展开式的两侧同时求导，而后移项即可。

# BFGS

BFGS 是 Broyden-Fletcher-Goldfarb-Shanno 的缩写。算法的核心思想是构造 Hessian 矩阵的近似矩阵 B，而后用 B 来代替 H进行更新：

$$ B_{k+1} = B_{k} + \Delta B_{k} $$

$$ \Delta B_{k} = \alpha uu^{T} + \beta vv^{T} $$

$$ u = y_{k}~~~ v = B_{k}s_{k} $$

$$ \alpha = \frac{1}{y_{k}^{T}s_{k}} ~~~ \beta = -\frac{1}{s_{k}^{T}B_{k}s_{k}} $$

等价于：

$$ \Delta B_{k} = \frac{y_{k}y_{k}^{T}}{y_{k}^{T}s_{k}} - \frac{ B_{K}s_{k}s_{k}^{T}B_{k}  }{s^{T}_{k}B_{k}s_{k}} $$

算法更新流程是，一开始 B是单位矩阵 I，k=0，而后以此计算个变量得到$\Delta B$，更新它，然后重复迭代。

BFGS 算法必须存储 H 矩阵 的逆矩阵，比较耗费内存。为此提出改进方案 L-BFGS，其思想是不存储完成的逆矩阵，而是只存储向量 S 和 Y。

# 参考
[深度学习](https://link.springer.com/article/10.1007%2Fs10710-017-9314-z)
[理解牛顿法](https://zhuanlan.zhihu.com/p/37588590)    
