---
layout:     post
title:      "深度学习笔记（一）"
subtitle:   "常见优化器"
date:       2019-04-09 00:15:18
author:     "Pelhans"
header-img: "img/tensorflow.jpg"
header-mask: 0.3 
catalog:    true
tags:
    - Deep Learning
---

> 系统整理 tensorflow 相关技术，并随着学习随时更新。本篇包含深度学习中常用的优化器。

* TOC
{:toc}

# 概览

下面列出深度学习中常用的一些优化方法：

* SGD: tf.train.GradientDescentOptimizer()    
* Adam: tf.train.AdamOptimizer()    
* Adagrad: tf.train.AdagradOptimizer()    
* Adadetla: tf.train.AdadeltaOptimizer()    
* RMSProp: tf.train.RMSPropOptimizer()    
* FTRL: tf.train.FtrlOptimizer()    
* Momentum: tf.train.MomentumOptimizer(lr, momentum)    
* Nesterov Acceletate Gradient: tf.train.MomentumOptimizer(lr, momentum, use_nesterov)

# 梯度下降

梯度下降算法的公式非常简单：

$$ \theta^{new} = \theta - \eta\nabla_{\theta} J(\theta) $$

其中 $\theta$表示神经网络的参数， $\eta$ 表示学习率。

## 梯度下降的变种

包含三个主要变种，它们间的主要区别在于计算梯度时用的数据量不一样。在实际使用中，需要在准确性和更新速度上进行权衡。

### 批梯度下降(Batch gradient descent)

批梯度下降在训练过程中使用全部的数据来计算梯度，因此它的更新会非常的慢，并且很吃内存。**批梯度下降保证收敛到凸误差表面的全局最小值和非凸表面的局部最小值。**

### 单样本随机梯度下降

单样本随机梯度下降通过对每个样本$x^{i}$和标签$y^{i}$执行更新：

$$ \theta = \theta - \eta\nabla_{\theta}J(\theta; x^{i}, y^{i}) $$

单样本SGD更新的比较快，不过由于更新的很频繁，因此目标函数优化曲线波动会比较大。

### 随机梯度下降

按照数据生成分布，抽取m个小批量(独立同分布的)样本，通过**计算他们梯度均值，我们可以得到梯度的无偏估计**。

$$ \theta = \theta - \eta\nabla_{\theta}J(\theta; x^{(i:i+n)}; y^{(i:i+n)}) $$

常见的的batch大小为 50 到 256 之间，但可以根据不同的应用而变化。另一方面需要注意的是学习率，在实践中，有必要随着时间的推移逐渐降低学习率，我们设第k步的学习率为$\epsilon_{k}$。保证 SGD 收敛的一个充分条件是

$$ \sum_{k=1}^{\infty}\epsilon_{k} = \infty $$

$$ \sum_{k=1}^{\infty}\epsilon^{2}_{k} < \infty $$

实践中，一般会线性衰减学习率(也可以指数衰减)直到第 $\tau$ 次迭代。

初始学习率的选取也非常重要。**初始学习率过大，学习曲线会剧烈震荡，代价函数值通常会明显增加。温和的震荡是良好的，容易在训练随机代价函数时出现。如果学习率太小，那么学习过程会很缓慢，过低的话甚至可能会卡在一个相当高的代价值**。

# 梯度下降算法的优化

## 动量
动量方法旨在加速学习，特别是处理高曲率、小但一致的梯度，或是带有噪声的梯度。动量算法累积了之前梯度指数级衰减(指数衰减是因为$\alpha$得累积)的移动平均，并且继续沿着该方向移动。

从形式上看，动量算法引入了变量 v充当速度的角色-它代表参数在参数空间移动的方向和速率。速度被设定为负梯度的指数衰减平均($\alpha\nu$ 和 负梯度得加和平均)。超参数 $\alpha$ 决定了之前梯度的贡献衰减的多快，在实践中，一般取值为0.5、0.9、0.99。

$$ \nu \leftarrow \alpha\nu - \epsilon\nabla_{\theta}(\frac{1}{m}\sum_{i=1}^{m}L(f(x^{i};\theta), y^{i})) $$

$$ \theta \leftarrow \theta + \nu $$

**动量的主要目的是解决两个问题：Hessian矩阵的病态条件和随机梯度的方差**。这里的动量与经典物理学中的动量是一致的，就像从山上投出一个球，在下落过程中收集动量，小球的速度不断增加。当其梯度指向实际移动方向时，动量项$\alpha$ 增大；当梯度与实际移动方向相反时，$\alpha$ 减小。这种方式意味着动量项只对相关样本进行参数更新，减少了不必要的参数更新，从而得到更快且稳定的收敛，也减少了振荡过程。

下面是我从深度学习那本书上找到的，为什么是动量。

从物理角度来看，负梯度$-\nabla_{\theta}J(\theta)$ 代表力，它推动粒子沿着代价函数表面下坡的方向移动。而$\alpha\nu$ 可以看做粘性阻力(但也可以加速，这里的粘性阻力是否不恰当)，它会导致粒子随着时间推移逐渐失去能量，最终收敛到局部极小点。使用 $\alpha\nu$ 的部分原因是数学上的便利--速度的整数幂很容易处理。

## Nesterov 动量
Nesterov 梯度加速法(NAG)是一种赋予动量项预知能力的方法，它通过在计算梯度时加入过去的动量来影响当前的梯度，更新策略为：

$$ \nu \leftarrow \alpha\nu - \epsilon\nabla_{\theta}[\frac{1}{m}\sum_{i=1}^{m}L(f(x^{i};\theta+\alpha\nu), y^{i})] $$

$$ \theta \leftarrow \theta + \nu $$

其中参数 $\alpha$ 和 \epsilon$ 发挥了和标准动量中相似的作用。 Nesterov 动量中，梯度计算在施加当前速度之后，因此它可以解释为在标准动量方法中添加了一个矫正因子。这种预更新方法能防止大幅振荡，不会错过最小值，并对参数更新更加敏感。

# 自适应学习率算法

传统的梯度下降算法比较依赖于学习率的设置，而改进的动量算法又多了一个动量参数。因此就有一些自适应学习率的方法。

## Adagrad 

一种基于梯度的优化算法，它可以使学习率适应参数，对稀疏参数进行大幅更新和对频繁参数进行小幅更新。因此，Adagrad方法非常适合处理稀疏数据。同时也极大地提高了SGD的稳健性。更新公式为：

$$ g \leftarrow \frac{1}{m}\nabla_{\theta}\sum_{i}L(f(x^{i};\theta),y^{i}) $$

$$ r \leftarrow r + g\odot g $$

$$ \Delta\theta \leftarrow -\frac{\epsilon}{\delta + \sqrt{r}}\odot g $$

$$ \theta \leftarrow \theta + \Delta\theta $$

Adagrad 的好处之一是不需要手动调整学习率，一般用默认的 0.01 就行。主要缺点是每个附加项都是正数，因此累计总和在训练期间不断增长。这反过来导致学习率缩小并且最终变得无限小，此时算法不能够再获得额外的知识。

## Adadetla
Adagrad 对之前的所有梯度都累计了导致低效，因此就在历史梯度信息的累计上乘上一个衰减系数$\gamma$，然后用$(1-\gamma)$作为当前梯度的平方加权系数相加。

$$ E[g^{2}]_{t} = \gamma*E[g^{2}]_{t-1} + (1-\gamma)*g^{2}_{t} $$

$$ \Delta\theta \leftarrow -\frac{\epsilon}{\sqrt{E[g^{2}]_{t} + \delta}}*g_{t} $$

$$ RMS(g_{t}) = \sqrt{E[g^{2}]_{t} + \delta} $$

现在算法为每个参数计算出不同的学习率，也计算了动量项，同时也防止学习率衰减或梯度消失等问题的出现。

特点是在训练初中期，加速效果不错，但是在训练后期，会反复在局部最小值附近抖动。

## RMSProp

RMSProp 算法 修改 Adagrad 以在非凸设定下效果最好，改变梯度累计为指数加权的移动平均。它使用指数衰减平均以丢弃遥远过去的历史，使其能够在找到凸碗结构后快速收敛。更新策略为：

$$ E[g^{2}]_{t} = \gamma E[g^{2}]_{t-1} + (1-\gamma)g^{2}_{t} $$

$$ \Delta\theta = -\frac{\epsilon}{\sqrt{E[g^{2}]_{t} + \delta}}*g_{t} $$

一般来说，$\gamma$ 建议设置为　0.9，　学习率　$\epsilon$　设置为0.001. Kears 文档说它对 RNN效果很好。(和 Adadetla 算法的区别是啥？)

## Adam
Adam 算法派生于短语"adaptive moments"，它本质上是带有动量项的 RMSProp，它利用梯度的一阶矩估计和二阶矩估计动态调整每个参数的学习率。更新公式为：

$$ g \leftarrow \frac{1}{m}\nabla\sum_{i}L(f(x^{i};\theta),y^{i}) $$

$$ t \leftarrow t+1 $$

更新一阶有偏估计(为什么是有偏的? 梯度前有附加项,导致一阶矩的期望和实际得不一致,差一个稀疏,再后面修正中可以看到这点)(为什么要加这个附加项? 没有明确得理论)

$$ s \leftarrow \rho_{1}s + (1-\rho_{1})g $$

更新有偏二阶估计(为什么要用二阶矩?加速训练,可以在非凸情况下表现更好, 其实二阶矩可以看作加速度) 

$$ r \leftarrow \rho_{1}r + (1-\rho_{2})g\odot g $$

修正一阶矩与二阶矩的偏差(为什么要修正偏差? 在初期偏置比 RMSProb 要低)

$$ \hat{s} \leftarrow \frac{s}{1-\rho_{1}^{2}} $$

$$ \hat{r} \leftarrow \frac{r}{1-\rho_{2}^{t}} $$

$$ \Delta\theta = -\epsilon\frac{\hat{s}}{\sqrt{\hat{r}+\delta}} $$

$$ \theta \leftarrow \theta + \Delta\theta $$

对与该公式的解释，深度学习这本书上说，在Adam中，动量直接并入了梯度一阶矩(指数加权)的估计。将动量加入RMSProp最直观的方法是将动量应用于缩放后的梯度。结合缩放的动量使用没有明确的理论动机。其次Adam包含偏置修正，修正从圆点初始化的一阶矩(动量项)和(非中心的)二阶矩的估计。

个人理解上说，Adam算法结合了 RMSProp 对二阶梯度指数衰减平均，同时加上了一阶的动量部分，动量部分也用相同的衰减方法。除此之外又加了一阶和二阶的修正，得到无偏估计。

Adam 算法的特点是结合了 Adagrad 善于处理稀疏梯度和 RMSProp 善于处理非平稳目标的优点，对内存需求较小。可以为不同的参数计算不同的自适应学习率。也适用于大多非凸优化，适用于大数据集和高维空间。

# 总结

根据深度学习这本书上的说法，上述这些方法不相伯仲，选择哪个用主要取决于你对哪个更熟悉，好调参。各自特点总结的话就是：

* 对于稀疏数据或者复杂网络，尽量使用自适应学习率的算法，获得更快的速度。    
* SGD 通常可以达到最小值，但可能比其他优化器更慢，更依赖与强大的初始化和退火计划，并且可能卡在鞍点而不是局部最小值    
* 自适应学习率算法中，Adam 稍微更强一点，更鲁棒一点

# 机器学习优化方法

除了梯度下降外，机器学习中还有很多很多的优化算法，牛顿法、共轭梯度、BFGS、拉格朗日乘数法、KKT条件、SMO、坐标下降法等等。SIGAI 对于这些算法有一个比较好的总结[机器学习中的最优化算法总结](https://zhuanlan.zhihu.com/p/42689565)。

![](/img/in-post/tensorflow/ml_opti.jpg)
