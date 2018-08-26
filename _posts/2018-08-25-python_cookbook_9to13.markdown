---
layout:     post
title:      "Python Cookbook总结"
subtitle:   "9-13 章"
date:       2018-08-25 00:15:18
author:     "Pelhans"
header-img: "img/speech_process.jpg"
header-mask: 0.3 
catalog:    true
tags:
    - speech process
---


> 包含第9、10、12、13 章的内容。

* TOC
{:toc}

# 第九章 元编程

元编程的主要目标就是创建函数和类，并用它们来操纵代码(比如说修改、生成或包装已有的代码)。python 中基于这个目的的主要特性包括装饰器、类装饰器以及元类。

## 9.1 给函数添加一个包装 装饰器

装饰器就是一个函数，它可以接受一个函数作为输入并返回一个新的函数作为输出。当我们有如下代码

```python
@timethis
def countdown(n):
...
运行起来和下面代码的效果是一样的
```python
def countdown(n):
    ....
countdown = timethis(countdown)
```

常见的一些内建装饰器如@staticmethod、@classmethod以及@preperty 的工作方式也是一样的。需要重点强调的是，装饰器一般来说不会修改调用签名，也不会修改被包装函数返回的结果。

## 9.2 编写装饰器时如何保存函数的元数据 functools @wraps

编写装饰器的一个重要部分就是拷贝装饰器的元数据，如果忘记使用@wraps，就会发现被包装的函数丢失了所有的有用信息。

```python
import time
from functools import wraps

def timethis(func):
    @wraps(func)
    def wrapper(*args, **kargs):
        start = time.time()
        result = func(*args, **kargs)
        end = time.time()
        print(func.__name__, end-start)
        return result
    return wrapper

@timethis
def countdown(n):
    while n > 0:
        n -= 1
countdown(1000000)
out: ('countdown', 0.02944493293762207)
countdown.__name__
out: 'countdown'
```
@wraps 装饰器的一个重要特性就是它可以通过__wrapped__属性来访问被包装的那个函数。如
```python
countdown.__wrapped__(10000)
```

## 9.3 定义一个可接受参数的装饰器

这个实现的思想很简单，在现有装饰器的基础上外层再定义一个函数来接受所需的参数，并让它们对装饰器的内层函数可见就可以了。下面给个例子：
```python
from functools import wraps
import logging

def logged(level, name=None, message=None):
    def decorate(func):
        logname = name if name else func.__moudle__
        log = logging.getLogger(logname)
        logmsg = message if message else func.__name__

        @wraps(func)
        def wrapper(*args, **kargs):
            log.log(level, logmsg)
            return func(*args, **kargs)
        return wrapper
    return decorate
```

## 9.4 
