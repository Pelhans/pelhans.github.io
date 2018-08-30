---
layout:     post
title:      "Python Cookbook总结"
subtitle:   "1-6 章"
date:       2018-08-11 00:15:18
author:     "Pelhans"
header-img: "img/python_cookbook.jpg"
header-mask: 0.3 
catalog:    true
tags:
    - python
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

## 4.3 yield from

大意是说 yield from 表达式允许一个生成器代理另一个生成器, 这样就允许生成器被替换为另一个生成器, 子生成器允许返回值。yield from 在涉及写成和给予生成器的并发型高级程序中有着更加重要的作用。
```python
def g1(x):     
     yield  range(x)
def g2(x):
     yield  from range(x)

it1 = g1(5)
it2 = g2(5)

print( [ x for x in it1] )
#out [range(0, 5)]
print( [ x for x in it2] )
#out [0, 1, 2, 3, 4]
```
可以看到 , yield返回一个生成器 , 这个生成器就是range自身 , yield from 也返回一个生成器, 这个生成器是由range代理的, yield from 在递归程序中较为常用。

## 4.4 反向迭代 reversed()

假如想要反向迭代序列中的元素，可以使用内建的 reversed()函数。也可以在自己的类中实现__reversed__()方法。具体实现类似于__iter__()方法。

```python
a = [1, 2, ,3 ,4]
for x in reversed(a):
    print(x)
```

## 4.5 定义带有额外状态的生成器函数

想定义一个生成器函数，但它还涉及一些额外的自定义值，可以通过将self的一些属性放到__iter__()函数里。这样在迭代的过程中就可以持续的访问内部属性。

## 4.6 对迭代器做切片操作 itertools.islice

对生成切做切片操作，普通的切片不能用，可以使用itertools.islice()函数

```python
In [3]: def count(n):
   ...:     while True:
   ...:         yield n
   ...:         n += 1
   ...:   
In [5]: c = count(0)
In [6]: c
Out[6]: <generator object count at 0x7f92899b3c80>
----> 1 c[0]
TypeError: 'generator' object has no attribute '__getitem__'

import itertools
In [10]: for x in itertools.islice(c, 10, 20):
    ...:     print(x)
10
11
12
13
14
15
16
17
18
19
```

## 4.7 迭代所有可能的组合或排列 itertools.permutations, itertools.combinations, itertools.combinations_with_replacement

itertools.permutations 接受一个元素集合，将其中所有的元素重排列为所有可能的情况，并以元组序列的形式返回。combinations 不考虑元素间的实际顺序，同时已经原则过的元素将从从可能的候选元素中移除。若想解除这一限制，可用combinations_with_replacement。

```python
In [11]: from itertools import permutations
In [12]: items = ['a', 'b', 'c']
In [13]: for p in permutations(items):
    ...:     print(p)
    ...:     
('a', 'b', 'c')
('a', 'c', 'b')
('b', 'a', 'c')
('b', 'c', 'a')
('c', 'a', 'b')
('c', 'b', 'a')

In [14]: from itertools import combinations
In [16]: for c in combinations(items, 3):
    ...:     print(c)
    ...:     
('a', 'b', 'c')

```

## 4.8 在不同的容器中进行迭代 itertools.chain()

我们需要对许多对象执行相同的操作，但是这些对象包含在不同的容器内，而我们希望可以避免写出嵌套的循环处理，保持代码的可读性。使用itertools.chain()方法可以用来简化这个任务。

```python
from itertools import chain

In [18]: a = [1, 2, 3, 4]
In [19]: b = ['x', 'y', 'z']
In [20]: for x in chain(a, b):
    ...:     print (x)
    ...:     
1
2
3
4
x
y
z
```

## 4.9 创建处理数据的管道

定义一些裂小型的生成器函数，每个函数执行特定的独立任务。这样就每次把生成器产生的一批数据处理完。由于处理过程的迭代特性，这里只会用道非常少的内存

## 4.10 用迭代器取代 while循环

关于内建函数iter()，一个少有人知的特性是他可以选择性接受一个无参的可调用对象以及一个哨兵（结束）值作为输入。例如:

```python
CHUNKSIZE = 8192
def reader(s):
    for chunk in iter(lambda: s.recv(CHUNKSIZE), 'b'):
        process_data(data)
```

# 第五章 文件和 I/O

## 5.1 在字符串上执行I/O操作 io.StringIO() 和 io.BytesIO()

可以模仿文件输入，下列是StringIO()， BytesIO()和这个的操作是一样的。

```python
In [1]: import io
In [2]: s = io.StringIO()
In [3]: s.write(u'hello world\n')
Out[3]: 12
In [4]: print('This is a test', file=s)
In [6]: s.getvalue()
Out[6]: 'hello world\nThis is a test\n'
```

## 5.2 将二进制数据读到可变缓冲区中

我们想将二进制数据直接读取到一个可变缓冲区中，中间不经过任何拷贝环节。例如我们想原地修改数据再将它写回到文件中去。

```python
import os.path
def read_into_buffer(filename):
    buf = bytearray(os.path.getsize(filename))
    with open(filename, 'rb') as f:
        f.readinto(buf)
    return buf

with open('sample.bin', 'wb') as f:
    f.write(b'hello world')

buf = read_into_buffer('sample.bin')
In [16]: buf
Out[16]: bytearray(b'hello world')
```

## 5.3 将已有的文件描述符包装为文件对象 

以python 文件对象来包装这个文件描述符，它与一般打开的文件相比是有区别的。区别在于，文件描述符只是一个由操作系统分配的整数句柄，用来指代某种系统I/O通道。

```python
import os 
fd = os.open('somefile.txt', os.O_WRONLY | os.O_CREAT)

f = open(fd, 'wt')
f.write('hello world \n')
f.close()
```

# 第六章 数据编码与处理

## 6.1 读写CSV数据

对大部分类型的CSV数据，都可以用csv库来处理。如csv.reader()、、csv.writer()、csv.DictReader()、csv.DictWriter()

## 6.2 读写JSON数据

这个比较常见，主要使用的是JSON模块。两个主要的函数为json.dunps 和 json.loads()。如果是对文件进行处理而不是字符串的话，可以选择使用json.dump 和json.load 来编码和解码JSON数据。

## 6.3 解析简单的XML文档 xml.etree.ElementTree

xml.etree.ElementTree可以从简单的XML文档中提取数据。

```python
from urllib.request import urlopen
from xml.etree.ElementTree import parse

u = urlopen('http://planet.python.org/rss20.xml')
doc = parse(u)
In [24]: for item in doc.iterfind('channel/item'):
   ....:     title = item.findtext('title')
   ....:     date = item.findtext('pubDate')
   ....:     link = item.findtext('link')
   ....:     print (title)
   ....:     print(date)
   ....:     print(link)
   ....:     print()
   ....: 
```
