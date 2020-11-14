---
layout:     post
title:      "[源码解读] Tensorflow 中的 linear-chain CRF"
subtitle:   ""
date:       2020-11-14 00:15:18
author:     "Pelhans"
header-img: "img/attention.jpg"
header-mask: 0.3 
catalog:    true
tags:
    - NLP
---


* TOC
{:toc}

# 概览

这里主要对 Tensorflow 中的 CRF 源码和相关原理进行整理。这里的 CRF 模块指的是 线性链 CRF（当然一般情况NLP 里用的都是这个），同时一般都是接在像 LSTM 或者线性层后，用于处理序列类任务的。

源代码路径 https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/crf/python/ops/crf.py  。 其中比较重要的函数有 **crf_log_likelihood、crf_decode、viterbi_decode** 三个：

* crf_log_likelihood：CRF 的损失函数，它由 真实路径的分数 和 所有路径的总分数 两部分构成,真实路径的分数应该是所有路径分数里最高的。真实路径的分数由 状态分数和 转移分数相加得到 $$S_{i}$$, 则分数 $$P_{S_{i}}$$ 为 $$e^{S_{i}}$$，所有路径的总分就是所有路径取指数的分数和。 这样, 损失函数的公式为：
$$ L = -\log \frac{P_{RealPath}}{P_{1} + P_{2} + \dots + P_{N}} $$    
* crf_decode：CRF 的解码模块，输入为 tensor，根据LSTM 等模块的输出加上转移矩阵来得到最佳解码序列和对应的得分。    
* viterbi_decode：维特比解码模块，输入为数组一类的东西而不是 tensor，在测试或者要对解码后处理时用得到。

对于这种应用情况，我们可以做一些符号假设，首先假设 LSTM 的输出 shape，我们的 input x 是

```python
[batch_size, max_seq_len, num_tags]
```

这个input 可以看作发射分数(Emission score, 或状态分数)，比如单词 w1 在 第 j 个标签的分数为0.3。

序列长度 sequence_lengths 的 shape 为

```python
[batch_size]
```

序列长度可以生成 mask 来屏蔽 padding 部分。

转移矩阵 transition_params 的 shape 为

```python
[num_tags, num_tags]
```

转移矩阵表示由一个状态转移到另一个状态的得分, 在 lstm 后接的 crf 里,它一开始是一个随即初始化的矩阵, 大小就是标签的数量.随着网络进行训练。

有了这些后就可以直接看代码了。对于具体的 CRF 、维特比原理这里不会细讲，可以看[博客](https://createmomo.github.io/2017/11/24/CRF-Layer-on-the-Top-of-BiLSTM-6/)

# crf_log_likelihood

接口格式为：

```python
log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
    unary_scores, gold_tags, sequence_lengths, transition_params=None)
```

里面的 unary_scores 就是前面的 input，比如 LSTM 的输出或者其他什么的。gold_tags 是 ground truth，shape 为 [batch_size, max_seq_len]。transition_params 可以自己预先指定传入，也可以空着让它自己随机初始化。返回的话就是 loss 加上转移矩阵，这个转移矩阵可以用来解码。

内部实现上，代码如下：

```python
def crf_log_likelihood(inputs,
                       tag_indices,
                       sequence_lengths,
                       transition_params=None):
  """Computes the log-likelihood of tag sequences in a CRF.

  Args:
    inputs: A [batch_size, max_seq_len, num_tags] tensor of unary potentials
        to use as input to the CRF layer.
    tag_indices: A [batch_size, max_seq_len] matrix of tag indices for which we
        compute the log-likelihood.
    sequence_lengths: A [batch_size] vector of true sequence lengths.
    transition_params: A [num_tags, num_tags] transition matrix, if available.
  Returns:
    log_likelihood: A [batch_size] `Tensor` containing the log-likelihood of
      each example, given the sequence of tag indices.
    transition_params: A [num_tags, num_tags] transition matrix. This is either
        provided by the caller or created in this function.
  """
  // 得到 tag 的数量
  num_tags = inputs.get_shape()[2].value

  // 如果没给转移矩阵就自己初始化
  if transition_params is None:
    transition_params = vs.get_variable("transitions", [num_tags, num_tags])

  // 得到最佳路径得分
  sequence_scores = crf_sequence_score(inputs, tag_indices, sequence_lengths,
                                       transition_params)
  // 所有路径得分做指数运算求和后取对数的值（logsumexp）
  log_norm = crf_log_norm(inputs, sequence_lengths, transition_params)

  // Normalize the scores to get the log-likelihood per example.
  // 最佳路径得分减去 logsumexp(所有路径) 的分数
  // 简要推导看下面
  // 详情请见 https://createmomo.github.io/2017/11/11/CRF-Layer-on-the-Top-of-BiLSTM-5/
  log_likelihood = sequence_scores - log_norm
  return log_likelihood, transition_params
```

前面说过损失函数的形式：

$$ L = -\log \frac{P_{RealPath}}{P_{1} + P_{2} + \dots + P_{N}} $$

公式形式比较复杂，我们进一步推导一下：

$$ 
\begin{aligned}
L & = -\log \frac{P_{RealPath}}{P_{1} + P_{2} + \dots + P_{N}} \\
& = -\log \frac{e^{S_{RealPath}}}{e^{S_{1}} + e^{S_{2}} + \dots + e^{S_{N}}} \\
& = -(\log (e^{S_{RealPath}}) - \log (e^{S_{1}} + e^{S_{2}} + \dots + e^{S_{N}})) \\
& = -(S_{RealPath} -\log (e^{S_{1}} + e^{S_{2}} + \dots + e^{S_{N}}))
\end{aligned}
$$

这就得到 最佳路径得分和 logsumexp 差的形式了。

再看最佳路径得分这，用的函数是 crf_sequence_score ，看代码发现核心是两个函数 crf_unary_score、crf_binary_score。这两个函数分别计算 tag 序列的一元得分（状态序列得分）和二元得分（tag 间转移矩阵的得分）。

对于 logsumexp 部分的函数是 crf_log_norm，这里比较有意思，源代码 + 注释如下：

```python
def crf_log_norm(inputs, sequence_lengths, transition_params):
  """Computes the normalization for a CRF.

  Args:
    inputs: A [batch_size, max_seq_len, num_tags] tensor of unary potentials
        to use as input to the CRF layer.
    sequence_lengths: A [batch_size] vector of true sequence lengths.
    transition_params: A [num_tags, num_tags] transition matrix.
  Returns:
    log_norm: A [batch_size] vector of normalizers for a CRF.
  """
  // Split up the first and rest of the inputs in preparation for the forward
  // algorithm.
  first_input = array_ops.slice(inputs, [0, 0, 0], [-1, 1, -1])
  first_input = array_ops.squeeze(first_input, [1])

  // If max_seq_len is 1, we skip the algorithm and simply reduce_logsumexp over
  // the "initial state" (the unary potentials).
  def _single_seq_fn():
    log_norm = math_ops.reduce_logsumexp(first_input, [1])
    // Mask `log_norm` of the sequences with length <= zero.
    log_norm = array_ops.where(math_ops.less_equal(sequence_lengths, 0),
                               array_ops.zeros_like(log_norm),
                               log_norm)
    return log_norm

  def _multi_seq_fn():
    """Forward computation of alpha values."""
    rest_of_input = array_ops.slice(inputs, [0, 1, 0], [-1, -1, -1])

    // Compute the alpha values in the forward algorithm in order to get the
    // partition function.
    // 说是用 RNN 来做 CRF 里的前向算法
    forward_cell = CrfForwardRnnCell(transition_params)

    // Sequence length is not allowed to be less than zero.
    // 限制 max length 的长度，若小于 1 ，则用 0 代替
    sequence_lengths_less_one = math_ops.maximum( constant_op.constant(0, dtype=sequence_lengths.dtype),sequence_lengths - 1)
    // alpha 是 cell state
    _, alphas = rnn.dynamic_rnn(
        cell=forward_cell,
        inputs=rest_of_input,
        sequence_length=sequence_lengths_less_one,
        initial_state=first_input,
        dtype=dtypes.float32)
    log_norm = math_ops.reduce_logsumexp(alphas, [1])
    // Mask `log_norm` of the sequences with length <= zero.
    log_norm = array_ops.where(math_ops.less_equal(sequence_lengths, 0),
                               array_ops.zeros_like(log_norm),
                               log_norm)
    return log_norm

  max_seq_len = array_ops.shape(inputs)[1]
  return control_flow_ops.cond(pred=math_ops.equal(max_seq_len, 1),
                               true_fn=_single_seq_fn,
                               false_fn=_multi_seq_fn)
```

这里直接看 _multi_seq_fn 里面的东西，核心思想是借助 RNN 函数来实现时间步上的操作，简直太妙了，CrfForwardRnnCell 核心代码如下：

```python
  def __call__(self, inputs, state, scope=None):
    """Build the CrfForwardRnnCell.

    Args:
      inputs: A [batch_size, num_tags] matrix of unary potentials.
      state: A [batch_size, num_tags] matrix containing the previous alpha
          values.
      scope: Unused variable scope of this cell.

    Returns:
      new_alphas, new_alphas: A pair of [batch_size, num_tags] matrices
          values containing the new alpha values.
    """
    state = array_ops.expand_dims(state, 2)

    // This addition op broadcasts self._transitions_params along the zeroth
    // dimension and state along the second dimension. This performs the
    // multiplication of previous alpha values and the current binary potentials
    // in log space.
    transition_scores = state + self._transition_params
    new_alphas = inputs + math_ops.reduce_logsumexp(transition_scores, [1])

    // Both the state and the output of this RNN cell contain the alphas values.
    // The output value is currently unused and simply satisfies the RNN API.
    // This could be useful in the future if we need to compute marginal
    // probabilities, which would require the accumulated alpha values at every
    // time step.
    return new_alphas, new_alphas
```

大体上来说是利用前向算法对每个时间步进行累加，得到全局的路径总分。我们这里把前面博客的内容 copy 到这里一部分，二者对应的话，state 等于下面的 previous，inputs 等于 obs：

![](/img/in-post/kg_paper/crf_1.png)    
![](/img/in-post/kg_paper/crf_2.png)

细心看可以发现这个公式和代码还是不一致的，inputs 和 另外两个在代码里是分开算的，但我实际测试过，一起算的话结果是一样的：

```python
    def __call__(self, inputs, state, scope=None):
        state = array_ops.expand_dims(state, 2)
        // self._transition_params 是 [1, num_tags, num_tags]
        // state 是 [batch_size, num_tags, 1]
        // 加完变成 [batch_size, num_tags, num_tags]
        // 相当于 previous + obs 那步
        transition_scores = inputs + state + self._transition_params

        // 相当于计算 scores 和 更新 previous 结合，
        // new_alphas = inputs + math_ops.reduce_logsumexp(transition_scores, [1])
        new_alphas = math_ops.reduce_logsumexp(transition_scores, [1])
```

这样通过前向算法对每个时间步进行累计，而后通过 RNN 函数来遍历每个时间步，实现前向算法，简直太神奇了。

至此损失函数计算这就算完事了，同时理解了这步，后面的两个就简单很多。

# viterbi_decode

更标准的维特比算法，要熟练到能手撸出来的程度，下面给出注释版本：

```python
import numpy as np  
  
def viterbi_decode(score, transition_params):  
  """Decode the highest scoring sequence of tags with Viterbi. 
  Args: 
    score: A [seq_len, num_tags] matrix of unary potentials. 
    transition_params: A [num_tags, num_tags] matrix of binary potentials. 
  Returns: 
    viterbi: A [seq_len] list of integers containing the highest scoring tag 
        indices. 
    viterbi_score: A float containing the score for the Viterbi sequence. 
  """  
  // 和输入同样 shape 的 0 矩阵  
  // 这个东西用来记录解码中累积的分数  
  trellis = np.zeros_like(score)  
  // 和上面一样的，只不过里面元素强制要求是 int  
  backpointers = np.zeros_like(score, dtype=np.int32)  
  // 第一个 token 的初始化  
  trellis[0] = score[0]  
  
  for t in range(1, score.shape[0]):  
    // np.expand_dims(trellis[t - 1], 1) 的 shape 是[num_tags, 1]  
    // 通过 numpy 矩阵的自动扩维，使得每个转移矩阵和当前分数进行结合  
    v = np.expand_dims(trellis[t - 1], 1) + transition_params  
    trellis[t] = score[t] + np.max(v, 0)  
    // 上一步的 tag  
    backpointers[t] = np.argmax(v, 0)  
  
  // 从最后时间步向前解码  
  print("trellis: ", trellis)  
  print("backpointers: ", backpointers)  
  viterbi = [np.argmax(trellis[-1])]  
  for bp in reversed(backpointers[1:]):  
    viterbi.append(bp[viterbi[-1]])  
  viterbi.reverse()  
  
  viterbi_score = np.max(trellis[-1])  
  return viterbi, viterbi_score  

```

# crf_decode

接口格式为：

```python
viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(
    unary_scores, transition_params, sequence_lengths)
```

输入还是那几个东西，输出是解码序列和对应的分数。

这里的源码没啥好说的，整体还是维特比解码的路子，只是通过调用 RNN 来实现，先调用前向 Cell 进行累计，而后调用后向 Cell 得到 tag 序列。维特比理解好的话，代码上没啥难点。

