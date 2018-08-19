---
layout:     post
title:      "Python Cookbook总结"
subtitle:   "7-8 章"
date:       2018-08-19 00:15:18
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

# 第七章 函数

## 7.1 将元数据信息附加到函数参数上

函数的参数注解可以提示程序员该函数应该如何使用，这是很有帮助的。比如，考虑下面这个带参数注解的函数：

```python
def add(x:int, y:int) -> int:
    return x + y
```

python解释器并不会附加任何语法意义到这些参数注解上。但参数注解会给阅读代码的人提供提示，并且一些第三方工具和框架可能会为注解加上语法含义。这些注解也会出现在文档中：
```python
help(add)

Help on function add in module __main__:

    add(x:int, y:int) -> int
```

另外，函数注解还可以用来实现函数重载。

## 7.2 在匿名函数中绑定变量的值

我们利用lambda表达式定义了一个匿名函数，但是也希望可以在函数定义的时候完成对特定变量的绑定。也许我们想的是这样的：

```python
In [1]: x = 10
In [2]: a = lambda y : x + y
In [3]: x = 20
In [4]: b = lambda y : x + y
In [5]: a(10)
Out[5]: 30
In [6]: b(10)
Out[6]: 30
```

可以看到，和我们预想的结果有差距，这是因为lambda函数是在运行时才进行变量的绑定，而不是在定义时进行。因此，为了达成目标，我们需要在定义匿名函数的时候就进行变量绑定：

```python
In [7]: x = 10
In [8]: a = lambda y , x=x: x + y
In [9]: b(10)
Out[9]: 20
In [10]: x = 20
In [11]: b = lambda y, x=x: x + y
In [12]: a(10)
Out[12]: 20
In [13]: b(10)
Out[13]: 30
```

## 7.3 让带有N个参数的可调用对象以较少的参数形式调用  functools.partial()

函数partial()允许我们给一个或多个参数指定固定的值，一次来减少参数的数量。

```python
In [14]: def spam(a, b, c, d):
    ...:     print(a, b, c, d)
In [15]: from functools import partial
In [17]: s1 = partial(spam, 1)
In [18]: s1(2, 3, 4)
(1, 2, 3, 4)
In [19]: s1(4, 5, 6)
(1, 4, 5, 6)
In [20]: s2 = partial(spam, d=42)
In [21]: s2(4, 5, 5)
(4, 5, 5, 42)
In [24]: s3 = partial(spam, 1, 2, d=42)
In [25]: s3(5)
(1, 2, 5, 42)
```

这个东西的主要用途是和那些只接受单一参数的函数来一起工作。如sort()函数：

```python
In [26]: points = [ (1, 2), (3, 4), (5, 6), (7, 8) ]
In [27]: import math
In [28]: def distance(p1, p2):
    ...:     x1, y1 = p1
    ...:     x2, y2 = p2
    ...:     return math.hypot(x2 - x1, y2 - y1)
In [29]: pt = (4, 3)
In [30]: points.sort(key=partial(distance, pt))
In [31]: points
Out[31]: [(3, 4), (1, 2), (5, 6), (7, 8)]
```
更一般来讲，partial() 常常可以用来调整其他库中用到的回调函数的参数签名。

## 7.4 在回调函数中携带额外的状态

一种在回调函数中携带额外信息的方法是使用绑定方法而不是普通的函数，比如下面这个类保存了一个内部的序列号码，每当接收到一个结果时就递增这个号码：

```python
In [32]: class ResultHandler:
    ...:     def __init__(self):
    ...:         self.sequence = 0
    ...:     def handler(self, result):
    ...:         self.sequence += 1
    ...:         print('[{}] Got : {}'.format(self.sequence, result))
In [33]:apply_async(add, (2, 3), callback=r.handler)
Got: 5
In [33]:apply_async(add, ('hello', 'world'), callback=r.handler)
Got: helloworld
```

作为替代方案，也可以使用闭包来捕获状态：

```python
def make_handler():
    sequence = 0
    def handler(result):
        nonlocal sequence
        sequence += 1
        print('[{}] Got : {}'.format(self.sequence, result))
    return handler
```

除此之外还可以使用协程(coroutine)来完成同样的任务：

```python
def make_handler():
    sequence = 0
    while True:
        result = yield
        sequence += 1
        print('[{}] Got : {}'.format(self.sequence, result))
```

对于协程来说，可以使用它的send()函数作为回调函数：

```python
handler = make_handler()
next(handler)
apply_async(add, (2, 3), callback=handler.send)
[1] Got: 5
```

这里对协程做一个笔记，使用协程的程序调用时不是栈的关系，在子程序内部可以中断，然后转而执行别的子程序，在适当的时候再返回来接着执行。类似于CPU终端，而非函数调用。在python中，yield可在一定程度上实现协程。使用send 到另一个程序运行。

最后，同样重要的是也可以通过额外的参数在回调函数中携带状态，然后用partial()来处理参数个数的额问题。在现实问题张，闭包可能显得更轻量级一些，而且由于闭包是函数构建的，这样会显得更自然。将协程作为回调函数的有趣之处在于这种方式和采用闭包的方案关系紧密，从某种意义上说，协程甚至更为清晰，不过较难理解。

## 7.5 内联回调函数

我们正在编写使用回调函数的额代码，但是担心小型函数在代码中大肆泛滥，程序的控制流会因此而失控。这时我们可以通过生成器和协程讲回调函数内联到一个函数中。从而使得回调函数得到隐藏。

```python
def apply_async(func, args, *, callback):
    # Compute the Result
    result = func(*args)

    # Invoke the callback with result
    callback(result)

from queue import Queue
from functools import wraps

class Async:
    def __init(self, func, args):
        self.func = func
        self.args = args

def inlined_async(func):
    @wraps(func)
    def wrapper(*args):
        f = func(*args)
        result_queue = Queue()
        result_queue.put(None)
        while True:
            result = result_queue.get()
            try:
                a = f.send(result)
                apply_async(a.func, a.args, callback=result_queue.put)
            except StopIteration:
                break
        return wrapper

# When used
def add(x, y):
    return x + y

@inlined_async
def test():
    r = yield Async(add, (2, 3))
    print(r)
    r = yield Async(add, ('hello', 'world'))
    print(r)
    for n in range(10):
        r = yield Async(add, (n, n))
        print(r)
    print('Goodbye')

# Result
5
helloworld
0
2
4
6
8
10
12
14
16
18
Goodbye
```

可以看到，除了那个特殊的装饰器和对yield的使用之外，我们会发现代码中根本没有出现回调函数(它被隐藏到幕后了)。将精巧的控制流隐藏在生成器函数之后，这种做法可以在标准库及第三方包中找到。

## 7.6 访问闭包中的变量

一般来说，在闭包内层定义的变脸对于外界来说是完全隔离的。但是可以通过编写存取函数(getter/setter 方法)并将他们作为函数属性附加到闭包上来提供对内层变量的访问支持。如：

```python
def sample():
    n = 0

    # Closure function
    def func():
        print('n=', n)

    # Accessor methods for n
    def get_n():
        return n
    
    def set_n():
        nonlocal n
        n = value
    # Attach as function attr
    func.get_n = get_n
    func.set_n = set_n

    return func
```

采用上述方法还可以让闭包模拟成类实例，我们要做的就是将内层函数拷贝到一个实例字典中然后将它返回。通常来说，采用闭包的版本有可能更快一些，因为不用涉及额外的self变量。

# 第八章 类与对象

## 8.1 修改实例的字符串表示  __repr__() 和 __str__()

要修改实例的字符串表示，可以通过定义 __str__() 和 __repr__() 方法来实现。特殊方法 __repr__()返回的是实例的代码表示。通常可以用它发挥的字符串文版本重新创建这个实例，即满足 obj == eval(repr(obj))。如：

```python
class Pair:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __repr__(self):
        return 'Pair ({0.x!r}, {0.y!r})'.format(self)

    def __str__(self):
        return '({0.x!s}, {0.y!s})'.format(self)

p = Pair(3, 4)
p
Pair(3, 4)
print(p)
(3, 4)
```

通常认为定义 __repr__() 和 __str__()是好的编程实践，因为这么做可以简化调试过程和实例的输出。

## 8.2 自定义字符串的输出格式 __format__()

目的是想通过format()函数和字符串方法来支持自定义的输出格式。可以通过在类内定义 __format__()来实现，一个 __format__()的例子：

```python
_formats = {
    'ymd' : '{d.year} - {d.month} - {d.day}',
    'mdy' : ...,
    'dmy' : ...
}
class Date:
    def __init__(self, year, month, day):
        ...
    def __format__(self, code):
        if code == '':
            code = 'ymd'
        fmt = _formats[code]
        return fmt.format(d=self)
```

## 8.3 让对象支持上下文管理协议  __enter__() 和 __exit__()

要让对象能够兼容with 语句，需要实现 __enter__() 和 __exit__() 方法。

```python
from socket import socket, AF_INET, SOCK_STREAM

class LazyConnection:
    def __init__(self, address, family=AF_INET, type=SOCK_STREAM):
        self.address =address
        self.family = AF_INET
        self.type = SOCK_STREAM
        self.sock = None

    def __enter__(self):
        if self.sock is not None:
            raise RuntimeError('Already connected')
        self.sock = socket(self.family, self.type)
        self.sock.connect(self.address)
        return self.sock  # 当用 with conn as s 这种时，这个s就是返回值 self.sock

    def __exit__(self, exc_ty, exc_cal, tb):
        self.sock.close()
        self.sock = None
```

## 8.4 将名称封装到类中

python 中不像c++ 有private 那种个东西，但是通常认为:    
* 任何以单下划线开头的名字应该总是被认为只属于内部实现。    
* 以双下划线开头的名称会导致出现命名重整的行为，如在类B中实现的__private_method 则会被重命名为_B__private_method。这样重整的目的在于以双下划线开头的属性不能通过继承而覆盖。

## 8.5 创建可管理的属性 @property

要自定义对属性的访问，一种简单的方式是将其定义为 property,即把类中定义的函数当做一种属性来使用。下面的例子定义了一个 property，增加了对属性的类型检查：

```python
class Person:
    def __init__(self, first_name):
        self._first_name = first_name

    # Getter function
    @property
    def first_name(self):
        return self._first_name

    # Setter function
    @first_name.setter
    def first_name(self, value):
        if not isinstance(value, str):
            raise TypeError('Expected a string')
        self._first_name = value

    # Deleter function
    @first_name.deleter
    def first_name(self):
        raise AttributeError("Can't delete attribute")
```

在上述代码中，一共有三个互相关联的方法，它们必须有着相同的名称。第一个方法是getter 函数，并将first_name 定义为 property属性，其他两个可选方法附加到了first_name属性上。 property的重要特性就是它看起来就像一个普通的属性，但是根据访问它的不同方式，会自动出发getter、setter、deleter 方法。

property也可以用来定义需要计算的属性。这类属性并不会实际保存起来，而是根需要计算完成。如：

```python
import math
class Circle:
    def __init__(self, radius):
        self.radius = radius

    @property
    def area(self):
        return math.pi * self.radius ** 2

    @property
    def perimeter(self):
        return 2 * math.pi * self.radius

c = Cirle(4.0)
c.radius
4
c.area
50.2654824
c.perimeter
25.132741228
```

需要注意的是，不要编写那种定义了大量重复性 property 的代码，这会导致代码膨胀，容易出错。

## 8.6 在子类中扩展属性

在子类中扩展在父类中已经存在的属性，首先需要弄清楚是需要重新定义所有的方法还是只针对其中一个方法做扩展。要重新定义所有的方法很容易，只要吧 getter 、setter、Deleter都实现一遍就好。但当只针对其中的一个方法做扩展时，只使用@property是不够的，如下面代码是无法工作的：

```python
class SubPerson(Person):
    @property
    def name(self):
        print('Getting name')
        return super().name
```

当使用上述代码时，会发现setter函数消失不见了相反，我们应该这么做：

```python
class SubPerson(Person):
    @Person.getter
    def name(self):
        print('Getting name')
        return super().name
```

通过这种方式，之前定义的所有属性方法都会被拷贝过来，而getter函数则会被替换掉。

## 8.7 描述符

所谓的描述符就是以特殊方法 __get__()、__set__()、__delete__() 的形式实现了三个核心的属性访问操作的类。这些方法通过接受类实例作为输入来工作。之后，底层的实例字典 __dict__ 会根据需要适当的进行调整。要使用一个描述符，我们把描述符的实例放置在类的定义中作为类变量来使用。

对于大多数Python 类的特性，描述符都提供了底层的魔法，包括@classmethod、@staticmethod、@property甚至 __slots__。通过定义一个描述符，我们可以在很底层的情况下捕获关键的实例操作(get、set、delete)。关于描述符，长容易困惑的地方就是它们只能在类的层次上定义，不能根据实例来产生。

## 8.8 定义一个接口或抽象基类

抽象基类的核心特征就是不能被直接实例化，它是用来给其他的类当做基类使用的，主要用途是强制规定所需要的编程接口。要定义一个抽象基类，可以使用abc 模块：

```python
from abc import ADCMeta, abstractmethod

class IStream(metaclass=ABCmeta):
    @abstractmethod
    def read(self, maxbytes=-1):
        pass
    @abstractmethod
    def write(self, data):
        pass
```

同时，抽象基类也允许其他的类向其注册，然后实现所需的接口：

```python
import io

# REgister the built-in I/O classes as supporting our interface

IStream.register(io.IOBase)

# Open a normal file and type check
f = open('foo.txt')
isinstance(f, IStream)
```

此处内容较多，建议看书。

## 8.8 委托属性的访问

我们想在访问实例的属性时能够将其委托到一个内部持有的对象上，这可以作为继承的替代方案或者是为了实现一种代理机制。

简单的说，委托是一种编程模式，我们将某个特定的操作转交给(委托)另一个不同的对象实现。最简单的委托看起来是这样的：

```python
class A:
    def spam(self, x):
        pass
    def foo(self):
        pass

class B:
    def __init__(self):
        self._a = A()

    def spam(self, x):
        # Delegate to the internal self._a instance
        return self._a.spam(x)

    def foo(self):
        return self._a.foo()

    def bar(self):
        pass
```

当仅有几个方法需要委托时，上面的代码是非常简单的。但当有许多方法被委托时，另一种实现法师是定义__getattr__()方法。

```python
class A:
    def spam(self, x):
        pass
    def foo(self):
        pass

class B:
    def __init__(self):
        self._a = A()

    def bar(self):
        pass

    # Expose all of the methods defined on class A
    def __getter__(self, name):
        return getter(self._a, name)
```

有时候当直接使用继承可能没多大意义，或者我们想要更多地控制对象之间的关系，或者说进一步封装，如只暴露出特定的方法、实现接口等，此时使用委托会很有用。

当使用委托来实现代理是，需要注意的是，__getattr__()实际上是一个回滚(fallback)方法，它只会在某个属性没有找到的时候才会调用。
