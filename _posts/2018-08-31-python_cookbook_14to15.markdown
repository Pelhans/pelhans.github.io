---
layout:     post
title:      "Python Cookbook总结"
subtitle:   "14-15 章"
date:       2018-08-30 00:15:18
author:     "Pelhans"
header-img: "img/python_cookbook.jpg"
header-mask: 0.3 
catalog:    true
tags:
    - python
---


> 包含第14章的测试、调试以及异常和第15章C语言扩展的内容。

* TOC
{:toc}

# 第十四章 并发

## 14.1 在单元测试中为对象打补丁

unuttest.mock.path() 函数的用法较多，可以当做装饰器、上下文管理器或者单独使用。如：

```python
from unuttest.mock import patch
import example

# 做装饰器用
@patch('example_func')
def test1(x, mock_func):
    example.func(x)
    mock_func.assert_called_with(x)

# 做上下文管理器用
with patch('example.func') as mock_func:
    example.func(x)
    mock_func.assert_called_with(x)

# 用来手动打补丁
p = patch('example.func')
mock_func = p.start()
example.func(x)
mock_func.assert_called_with(x)
p.stop()
```

patch()接受一个已有对象的完全限定名称并将其替换为一个新值，在装饰器函数或者上下文管理器结束执行后会将对象恢复为原始值。默认情况下，对象会被替换为 MagicMock实例。

## 14.2 在单元测试中检测异常情况

使用 assertRaise() 方法。

```python
import unittest

def parse_int(s):
    return int(s)

class TestConversion(unittest.TestCase):
    def test_bad_int(self):
        self.assertRaise(ValueError, parse_int, 'N/A')
```

## 14.3 跳过测试或者预计测试结果为失败

unittest模块中由一些装饰器渴作用域所选的测试方法上，以此控制它们的处理行为。

```python
import unittest
import os
import platform

class Tests(unittest.TestCase):
    def test_0(self):
        self.assertTrue(True)

    @unittest.skip('skipped test')
    def test_1(self):
        self.fail('should have failed!')

    @unittest.skipIf(os.name=='posix', 'Not supported on Unix')
    def test_2(self):
        import winreg

    @unittest.skipUnless(paltform.system() == 'Drawin', 'Max specific test')
    def test_3(self):
        self.assertTrue(True)

    @unittest.expectedFailure
    def test_4(self):
        self.assertEqual(2+2, 5)

if __name__ == '__main__':
    unittest.main()
```
## 14.4 创建自定义的异常

创建自定义的异常是非常简单的,只要将它们定义成继承自Exception 的类即可(也可以继承自其他已有的异常类型,如果这么做更有道理的话)。自定义的类应该总是继承自内建的Exception类，或者继承自一些本地定义的基类，而这个基类本身又是继承自Exception 的。虽然所有的异常也都继承自 BaseException，但不应该将它作为基类来产生新的异常。BaseException 是预留给系统退出异常的，比如 KeyboardInterrupt。因此捕获这些异常并不适用于它们本来的用途。

```
class NetworkError(Exception):
    pass

class HostnameError(NetworkError):
    pass

# when used
try:
    msg = s.s.recv()
except HostnameError as e:
    ...
```

如果打算定义一个新的异常并且改写 Exception 的 __init__()方法，请确保总是用所有传递过来的参数调用 Exception.__init__()。

```python
class CustomError(Exception):
    def __init__(self, message, status):
        super().__init__(message, status)
        self.message = message
        self.status = status
```

## 14.5 通过引发异常来响应另一个异常

要将异常串联起来，可以用 raise from 语句来代替普通的 raise。还可以通过查看异常对象的 __cause__属性来跟踪所希望的异常链。

```python
def example():
    try:
        int('N/A')
    except ValueError as e:
        raise RuntimeError('A PARSING ERROR OCCURED') from e...

```

## 14.6 让你的程序运行的更快

下面列出一些常见简单的优化策略：

* 有选择的消除属性访问：每次使用句点操作符(.)来访问属性时都会带来开销。在底层，这会触发调用特殊方法，比如 __getattribute__() 和 __getattr__()，而调用这些方法常常会导致字典查询操作。    
* 理解变量所处的位置：通常来说，访问局部变量要比全局变量要快。对于需要频繁访问的名称，想要提高运行速度，可以通过让这些名称尽可能成为局部变量来达成。    
* 避免不必要的抽象：任何时候当使用额外的处理层比如装饰器、属性或者描述符来包装代码时，代码的速度就会变慢。    
* 使用内建的容器：内建的数据类型处理速度一般要比自己写的快的多。    
* 避免产生不必要的数据结构或者拷贝操作

# 第15章 C语言扩展
