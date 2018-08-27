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

## 9.4 定义一个属性可由用户修改的装饰器

我们想编写一个装饰器来包装函数，但是可以让用户调整装饰器的属性，这样在运行时能够控制装饰器的行为。为了达到这个目的我们需要使用访问器函数。访问器函数以属性的形式附加到了包装函数上，每个访问器函数允许对nonlocal 变量赋值来调整内部参数。如果所有的装饰器都使用了@functools.wraps的话，访问器函数可以跨越多个装饰器内层进行传播。

```python
from functools import wraps, partial
import logging

def attach_wrapper(obj, func=None):
    if func is None:
        return partial(attach_wrapper, obj)
    setattr(obj, func.__name__, func)
    return func

def logged(level, name=None, message=None):
    def decorate(func):
        logname = name if name else func.__moudle__
        log = logging.getLogger(logname)
        logmsg = message if message else func.__name__

        @wraps
        def wrapper(*args, **kwargs):
            log.log(level, logmsg)
            return func(*args, **kwargs)

        @attach_wrapper(wrapper)
        def set_level(newlevel):
            nonlocal level
            level = newlevel

        @attach_wrapper(wrapper):
        def set_message(newmsg):
            nonlocal logmsg
            logmsg = newmsg

        return wrapper
    return decorate

@logged(logging.DEBUG)
def add(x, y):
    return x + y

@logged(logging.CRITICAL, 'example')
def spam():
    print('Spam!')
```

## 9.5 定义一个能接受可选参数的装饰器

通过使用 functools.partial 来实现。

```python
from functools import wraps, partial
import logging

def logged(func=None, *, level=logging.DEBUG, name=None, message=None):
    if func is None:
        return partial(logged, level=level, name=name, message=message)
    
    logname = name if name else func.__moudle__
    log = logging.getLogger(logname)
    logmsg = message if message else func.__name__
    @wraps
    def wrapper(*args, **kwargs):
        log.log(level, logmsg)
        return func(*args, **kwargs)
    return wrapper
```

## 9.6 利用装饰器对函数参数强制执行类型检查
可以使用 inspect.signature()函数来实现，这个函数允许我们从一个可调用的对象中提取出参数签名信息。

```python
from inspect import signature

def spam(x, y, z=42):
    pass

sig = signature(spam)
print(sig)
out: (x, y, z=42)
```

bind_partial() 方法可以对提供的类型到参数名做部分绑定。

```python
bound_types = sig.bind_partial(int, z=int)
bound_types
out: <inspect.BoundArguments object at 0x10069bb50>
bound_types.arguments
out: OrderDict([('x', <class 'int'), ('z', <class 'int')])
```

可以看到，缺失的参数被简单的忽略掉了。还有另外一种方法sig.bind()，只是它不允许出现缺失的参数。
