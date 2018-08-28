---
layout:     post
title:      "Python Cookbook总结"
subtitle:   "9-12 章"
date:       2018-08-25 00:15:18
author:     "Pelhans"
header-img: "img/speech_process.jpg"
header-mask: 0.3 
catalog:    true
tags:
    - speech process
---


> 包含第9、10、12 章的内容。

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

## 9.7 在类中定义装饰器

在类中定义装饰器是很容易的，问题是我们想要以什么方式使用装饰器，即以实例方法还是类方法的形式。当以类实例来用时，那装饰器的第一个参数应该是self，类方法来用的话参数就是cls。

## 9.8 把装饰器定义成类

我么想用装饰器来包装函数，但是希望得到的结果是一个可调用的实例，我们需要装饰器既能在类中工作，也可以在类外部使用。要把装饰器定义成类实例，需要确保在类中实现__cal__()和__get__()方法。 每当函数实现的方法需要在类中进行查询时，作为描述协议的一部分，他们的 __get__() 方法都会被调用。在这种情况下， __get__()的目的是用来创建一个绑定方法对象(最终会给方法提供self 参数)。

```python
import types
from functools import wraps

class Profiled:
    def __init__(self, func):
        wraps(func)(self)
        self.ncalls = 0
    
    def __call__(Self, *args, **kwargs):
        slef.ncalls += 1
        return self.__wrapped__(*args, **kwargs)

    def __get__(self, instance, cls):
        if instance is None:
            return self
        else:
            return typespes.MethodType(self, instance)

# when used
@Profiled
def add(x, y):
    return x + y

class Spam:
    @Profiled
    def bar(self, x):
        print(self, x)
```

## 9.9 把装饰器作用到类和静态方法上

将装饰器作用到类和静态方法上很简单，但要保证装饰器在应用的时候需要放在@classmethode 和 @staticmethod 之前。问题在于 这两个装饰器并不会实际创建可直接调用的对象。相反它们创建的是特殊的描述符对象，因此如果尝试在另一个装饰器中像函数那样使用它们，装饰器就会崩溃。确保这些装饰器出现在 @classmethod 和 @staticmethod 之前就能解决这个问题。

## 9.10 编写装饰器为被包装的函数添加参数

我们想编写一个装饰器为被包装的函数添加额外的参数，但是添加的参数不能影响到该函数已有的调用约定。此时我们可以使用keyword-only 参数将额外的参数注入到函数的调用签名中。

```python
from functools import wraps

def optional_debug(func):
    @wraps(func)
    def wrapper(*args, debug=False, **kwargs):
        if debug:
            print('Calling', finc.__name__)
        return func(*args, **kwargs)
    return wrapper

# when used
@optional_debug
def spam(a, b, c):
    print(a, b, c)

spam(1, 2, 3)
out: 1 2 3
spam(1, 2, 3, debug=True)
out:Calling spam 
    1, 2, 3
```

## 9.11 利用装饰器给类定义打补丁

现在我们打算不通过集成或者元类的方式来做，而是通过装饰器来检查或改写一部分类的定义，以此来修改类的行为。类装饰器常常可以直接作为涉及混合类(mixin)或者元类等高级技术的替代方案

```python
def log_getattribute(cls):
    # Get the original implementation
    orig_getattribute = cls.__getattribute__

    # Make a new definition
    def new_getattribute(self, name):
        print('getting:', name)
        return orig_getattribute(self, name)

    # Attach to the class and return
    cls.__getattribute__ = new_getattribute
    return cls

# Example use
@log_getattribute
class A:
    def __init__(self, x):
        self.x = x
    def spam(self):
        pass

a = A(42)
a.x
out: getting: x
    42
a.spam()
out: getting: spam
```

## 9.12 利用元类来控制实例的创建

我们想改变实例创建的方式，以此来实现单例模式、缓存或者其他类似的特性。为了定制化这个步骤，则可以通过定义一个元类并以某种方式重新实现它的__call__()方法。如果不用元类，那就得将类隐藏在某种额外的工厂函数之后。

```python
class NoInstances(type):
    def __call__(self, *args, **kwargs):
        raise TypeError("Can't instantiate directly")

class Spam(metaclass=NoInstances):
    @staticmethod
    def grok(x):
        print('Spam.grok')

# when used
spam.grok(42)
out: Spam.grok
s = Spam()
out: TypeError: Can't instantiate directly
```

## 9.13 获取属性的定义顺序

即自动记录下属性和方法在类中定义的顺序，这样就可以利用这个顺序进行各种操作。如

```python
from collections import OrdereDict

class Typed:
    _excepted_type = type(None)
    def __init__(self, name=None):
        self._name = name

    def __set__(self, instance, value):
        if not isinstance(value, slef._expected_type):
            raise TypeError('Expected ' + str(self._expected_type))
        instance.__dict[self._name] = value

class Integer(Typed):
    _expected_type = int

class Float(Typed):
     _expected_type = float

class String(Typed):
    _expected_type = str

class OrderedMeta(type):
    def __new__(cls, clsname, bases, clsdict):
        d = dict(clsdict)
        order = []
        for name, value in clsdict.items():
            if isinstance(value, Typed):
                value._name = name
                order.append(name)
                d['_order'] = order
                return type._new_(cls, clsname, bases, d)
    @classmethod
    def __prepare__(cls, clsname, bases):
        return OrderDict()

class Structure(metaclass = OrderMeta):
    def as_csv(self):
        return ','.join(str(getattr(self, name)) for name in self._order )

class Stock(Structure):
    name = String()
    shares = Integer()
    price = Float()

    def __init__(self, name, shares, price):
        self.name = name
        self.shares = shares
        self.price = price
```

在上述元类中，描述符的定义顺序是通过 OrderDict在执行类的定义体时获取到的，得到的结果会从字典提取出来然后保存到类的属性_order中。之后类方法就能够以各种方式使用属性 _order。实现这个的核心就在 __prepare__()方法上，该特殊方法定义在元类 OrderMeta中。该方法会在类定义一开始的时候立刻得到调用，调用时以类名和基类名作为参数，它必须返回一个映射型对象工处理类定义时使用。由于返回的是 OrderDict实例而不是普通的字典，因此类中各个属性间的顺去就可以方便地得到维护。

## 9.14 定义一个能接受可选参数的元类

在自定义元类时，我们想提供额外的关键字参数，如

```python
class Spam(metaclass=MyMeta, debug=True, synchronize=True):
    ...
```

要在元类中支持这样的关键字参数，需要保证在定义 __prepare__()、 __new__() 以及 __init__()方法时使用 keyword-only 参数来指定他们。如

```python
class Mydate(type):
    @classmethod
    def __prepare__(cls, name, bases, ns, *, debug=False, synchronize=False):
        ...
        return super().__prepare__(name, bases)

    def __new__(cls, name, bases, ns, *, debug=False, synchronize=False):
        ...
        return super().__new__(cls, name, bases, ns)

    def __init__(self, name, bases, ns, debug=False, synchronize=False):
        ...
        super().__init__(name, bases, ns)
```

额外的参数会传递给每一个与该过程相关的方法。__prepare__()方法是第一个被调用的，用来创建类的名称空间，这是在处理类的定义体之前需要完成的。一般来说这个方法只是简单地返回一个字典或者其他的映射型对象。 __new__()方法用来实例化最终得到的类型对象，他会在类的定义提被完全执行完毕后才调用。最后调用的是 __init__()方法，用来执行任何其他额外的初始化步骤。编写元类时，比较常见的做法是只定义一个 __new__() 或者 __init__()方法，而不同时定义两者。但是如果打算接受额外的关键字参数，那么两个方法就都必须提供，并且要提供可兼容的函数签名。

## 9.15 在类中强制规定编码约定

元类的一个核心功能就是允许在定义类的时候对类本身的内容进行检查。在重新定义的 __init__()方法中，我们可以自由地检查类字典、基类以及其他更多信息。此外一旦为某个类指定了元类，该类的所有子类都会自动继承这个特性。
