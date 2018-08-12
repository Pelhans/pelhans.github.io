---
layout:     post
title:      "Python Cookbook总结"
subtitle:   "1-6 章"
date:       2018-08-11 00:15:18
author:     "Pelhans"
header-img: "img/speech_process.jpg"
header-mask: 0.3 
catalog:    true
tags:
    - speech process
---


> 看这本书的过程中各种茅塞顿开，但开完后记住的并不多。。。写总结把自己不熟悉的记下来。

* TOC
{:toc}

# 第一章 数据结构和算法

## 1.1 保存最后 N 个元素 collections.deque

deque(max_len=N) 创建了一个固定长度的序列，当有新序列加入而队列已满时会自动移除最老的那个记录。若不指定队列的大小，也就得到了一个无限界的队列，可以在两端执行添加和弹出操作。

```python
from collections import deque

q = deque(maxlen=3)
q.append(2)
q.append(3)
q.appendleft(1)
q
Out[18]: deque([1, 2, 3])
q.append(4)
Out[20]: deque([2, 3, 4])
q.pop()
Out[22]: 4
```

## 1.2 找出集合中最大或最小的N个元素 heapq

```python
import heapq

nums = [1, 8, 2, 23, 7, -4, 18, 23, 42, 37, 2]
print(heapq.nlargest(3, nums))
out: [42, 37, 23]
print(heapq.nsmallest(3,nums))
out: [-4, 1, 2]

# 还可以接受一个参数key
In [1]: portfolio = [
   ...: {'name': 'IBM', 'shares': 100, 'price': 91.1},
   ...: {'name': 'AAPL', 'shares': 50, 'price': 543.22},
   ...: {'name': 'FB', 'shares': 200, 'price': 21.09},
   ...: {'name': 'HPQ', 'shares': 35, 'price': 31.75},
   ...: {'name': 'YHOO', 'shares': 45, 'price': 16.35},
   ...: {'name': 'ACME', 'shares': 75, 'price': 115.65}
   ...: ]

cheap = heapq.nsmallest(3, portfolio, key=lambda s: s['price'])
cheap
out: 
[{'name': 'YHOO', 'price': 16.35, 'shares': 45},
 {'name': 'FB', 'price': 21.09, 'shares': 200},
 {'name': 'HPQ', 'price': 31.75, 'shares': 35}]
```

## 1.3 在字典中将键映射到多个值上　collections.defaultdict

在字典中将键映射到多个值上也就是一个ｋｅｙ对应一个列表或集合中。可以使用defaultdict方便的创建，同时该模块创建的字典会自动初始化第一哦值，这样只需关注添加元素即可。

```python
from collections import defaultdict

d.defaultdict(list)
d['a'].append(1)
d['a'].append(1)
d
out: defaultdict(list, {'a': [1, 1]})

s = defaultdict(set)
s['a'].add(1)
s['a'].add(1)
s['a'].add(2)
s
out: defaultdict(set, {'a': {1, 2}})

```

## 1.4 让字典保持有序　collections.OrderedDict

使用ＯrderedDict 创建的dict 会严格按照初始添加的顺序进行。其内部维护了一个双向链表，它会根据元素加入的顺序来排列键的位置。因此ＯrderedDict的大小是普通字典的２倍多。

## 1.5 对切片命名 slice

作为一条基本准则，代码中如果有很多硬编码的索引值，将导致可读性和可维护性都不佳。因此可以使用slice()创建一个切片对象，用在任何允许进行切片操作的地方。

```python
items = [0, 1, 2, 3, 4, 5, 6]
a = slice(2, 4)
Out[24]: [2, 3]
items[a]
Out[25]: [2, 3]
```

## 1.6 通过公共键对字典列表排序 operator.itemgetter

```python
from operator import itemgetter

In [26]: rows = [
    ...: {'fname': 'Brian', 'lname': 'Jones', 'uid':1003},
    ...: {'fname': 'David', 'lname': 'Beazley', 'uid':1002},
    ...: {'fname': 'John', 'lname': 'Cleese', 'uid':1001},
    ...: {'fname': 'Big', 'lname': 'Jones', 'uid':1004}
    ...: ]

itemgetter('fname')
Out[31]: <operator.itemgetter at 0x7f01606657d0>

rows_by_frame = sorted(rows, key=itemgetter('fname'))
Out[30]: 
[{'fname': 'Big', 'lname': 'Jones', 'uid': 1004},
 {'fname': 'Brian', 'lname': 'Jones', 'uid': 1003},
 {'fname': 'David', 'lname': 'Beazley', 'uid': 1002},
 {'fname': 'John', 'lname': 'Cleese', 'uid': 1001}]

# 也可以使用lambda 代替 itemgetter, 但itemgetter更快
rows_by_fname = sorted(rows, key=lambda r: r['fname'])
In [34]: rows_by_fname
Out[34]: 
[{'fname': 'Big', 'lname': 'Jones', 'uid': 1004},
 {'fname': 'Brian', 'lname': 'Jones', 'uid': 1003},
 {'fname': 'David', 'lname': 'Beazley', 'uid': 1002},
 {'fname': 'John', 'lname': 'Cleese', 'uid': 1001}]
```
## 1.7 将多个映射合并为单个映射 Chainmap

我们有多个字典或者映射，想在逻辑上将他们合并为一个单独的映射结构。python3 中使用

```python
from collections import ChainMap

a = {'x': 1, 'z': 3}
b = {'y' : 2, 'z' : 4}
c = ChainMap(a,b)
In [5]: c
Out[5]: ChainMap({'z': 3, 'x': 1}, {'z': 4, 'y': 2})
In [6]: print(c['x'])
1
In [7]: print(c['y'])
2
In [8]: print(c['z'])
```
# 第二章 字符串和文本

## 文本过滤和清理　str.translate

```python
s = 'python\fis\tawesome\r\n'
In [10]: s
Out[10]: 'python\x0cis\tawesome\r\n'

In [11]: remap = {
   ....: ord('\t') : ' ',
   ....: ord('\f') : ' ',
   ....: ord('\r') : None # Deleted
}
In [21]: a = s.translate(remap)

In [22]: a
Out[22]: 'python is awesome\n'
```

# 第三章　数字丶日期和时间

## 3.1 分数的计算 fractions.Fraction

```python
from fractions import Fraction
a = Fraction(5, 4)
b = Fraction(7, 16)
c = a + b

In [30]: c.numerator
Out[30]: 27
In [31]: c.denominator
Out[31]: 16
```

## 3.2 时间换算 datetime.timedelta

```python
from datetime import timedelta

In [33]: a = timedelta(days=2, hours=6)
In [34]: b = timedelta(hours=4.5)
In [35]: c = a + b
In [36]: c.days
Out[36]: 2
In [37]: c.seconds
Out[37]: 37800
In [38]: c.seconds/3600
Out[38]: 10.5
In [39]: c.total_seconds() / 3600
Out[39]: 58.5
```

# 第四章 迭代器和生成器

##　4.1 手动访问迭代器中的元素 next()函数

```python
with open('/etc/passwd') as f:
    line = next(f)
    print(line)
```

## 4.2 委托迭代 __iter__()方法

对自定义的容器对象，其内部持有一个列表丶元组或其他的可迭代对象，我们想让自己的新容器能够完成迭代操作。一般来说，我们所要做的就是定义一个__iter__()方法，将迭代请求委托到对象内部持有的容器上。

```python
class Node:
    def __init__(self, value):
        Self._value = vaule
        self._children = []
    def __repr__(self):
        return 'Node({!r})'.format(self._value)
    def __iter__(self):
        return iter(self._children)
```
在这个例子中，__iter__()方法将迭代请求转发给对象内部持有的_children属性上。


