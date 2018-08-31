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


> 包含第14章的测试、调试以及异常和第15章C语言扩展的内容。对于C语言扩展由多种方法，个人比较倾向于韦易笑老师的选取标准“不要求性能ctypes或者cffi，需要性能cython或者手写module，其它都是邪路。最好的方法是全部写成ctypes，或者cffi，送上线跑，有空再把最慢的一两个接口换成cython”

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

## 15.1 利用Ctypes来访问C代码

对于C语言编写的小程序，使用Python标准库中的ctypes模块来访问通常是非常容易的。要使用ctypes，必须首先确保想要访问的C代码已经被编译为与Python解释器兼容(即采用同样的体系结构，字长，编译器等)的共享库了。

```python
import ctypes
import os

_file = 'libsample.so'
_path = os.path.join(*(os.path.split(__file__)[:-1] + (_file,)))
_mod = ctypes.cdll.LoadLibrary(_path)

# int gcd(int, int)
gcd = _mod.gcd
gcd.argtypes = (ctypes.c_int, ctypes.c_int)
gcd.restype = ctypes.c_int

# int divide(int, int, int *)
_divide = _mod.divide
_divide.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int))
_divide.restype = ctypes.c_int

def divide(x, y):
    rem = ctypes.c_int()
    quot = _divide(x, y, rem)
    return quot, rem.value

# in c code
int gcd(int x, int y){
    int g = y;
    while (x > 0){
        g = x;
        x = y % x;
        y = g;
    }
    return g;
}

# Divide two numbers
int divide(int a, int b, int *remainder){
    int quot = a / b;
    *remainder = a % b;
    return quot
}
```

## 15.2 编写简单的C语言扩展模块

不依赖任何其他工具直接使用 Python 的扩展 API编写一个简单的C 语言扩展模块。对于简单的 C代码，手工创建一个简单的扩展模块是很简单直接的。

第一步，确保自己的C代码有一个合适的头文件。通常这个头文件会对应于一个单独编译好的库。

```c++
/* sample.h */
#include <math.h>
extern int gcd(int, int)
extern int divide(int a, int b, int *remainder)

typeder struct Point(
    double x, y;
    )Point;
```

下面是一个 C语言扩展模块的样例，用来说明编写扩展函数的基础：

```python
#include "Python.h"
#include "sample.h"

/* int gcd(int, int) */
// PyObject 是一个C数据类型，表示任意的python 对象
static Pyobject *py_gcd(PyObject *self, PyObject *aegs){
    int x, y, retult;
    //PyArg_ParseTuple()用来将值从python转换为C语言中的表示
    if (!PyArg_ParseTuple(args, "ii", &x, &y)){
        return NULL;
    }
    retult = gcd(x, y);
    // 函数 Py_BuildValue()用来从C数据类型创建出对应的python对象
    return Py_BuildValue("i", result);
}
...

为了构建扩展模块，需要创建一个 setup.py 文件：

```python
# setup.py
setup(name='sample',
      ext_modules=[
          Extension('sample'
                    '[pysample.c]',
                    include_dirs = ['/some/dir'],
                    define_macros = [('FOO', '1')],
                    undef_macros = ['BAR'],
                    ;ibrary_dirs = ['/usr/local/lib'],
                    libraries = ['sample']
              )
        ]
     )
...
```

现在要构建出目标库，只需要用 python3 buildlib.py build_ext --inplace即可。这样就创建了一个名为 sample.so 的共享库。编译结束后，应该就可以开始将其当做一个Python模块来导入了。

```python
import sample
sample.gcd(35, 35)
out: 7
```

## 15.3 用Swig来包装C代码

Swig 可以解析C头文件并自动创建出扩展代码来。要使用这个工具，首先要有一个C头文件。而后下一步就是编写一个 Swig "接口"文件。根据约定，这些接口文件都以i 作为后缀。

```python
// sample.i -Swig interface
%moudle sample
%{
#include "sample.h"
}

/* Customizations */
%extend Point{
    /* Constructor for Point objects */
    Point(double x, double y){
        Point *p = (Point *) malloc(sizeof(Point));
        p->x = x;
        p->y = y;
        return p;

    };
};

/* Map int *remainder as an output argument */
%include typemaps.i
%apply int *OUTPUT {int * remainder};

/* Map the argument pattern (double *a, int n) to arrats */
%typemap(in) (double *a, int n)(Py_buffer view){
    view.obj = NULL;
    if (PyObject_GetBuffer($input, &view, PyBUF_ANY_CONTIGUOUS | PyBUF_FORMAT) == -1){
        SWIG_fail;
    }
    if (strcmp(view.format, "d") != 0){
        PyErr_SetString(PyExc_TypeError, "Expected an array of doubles");
        SWIG_fail;
    }
    $1 = (double *) view.buf;
    $2 = view.len / sizeof(double);
}

%typemap(freearg)(double *a, int n){
    if (view$argnum.obj){
        PyBuffer_Release(&view$argum);
    }
}

/* C declarations to be include in the extension module, you can copy the .h to here directly */

extern int gcd(int, int);
...
```

一旦写好了这个接口文件， Swig就可以作为命令行工具在终端中调用了：

```python
swig -python -py3 sample.i
```

swig会产生两个文件：sample_wrap.c 和 sample.py。而后编写 setup.py.py 并进行编译即可正常导入。

## 15.4 用 Cpython 来包装 C代码

从某种程度上来看，用Cpython创建一个扩展模块和手动编写扩展模块有些类似，它们都需要创建一组包装函数。现在假设C代码已经被编译为C库。

首先我们创建一个名为 csample.pxd 文件。

```python
# csample.pxd
#
# Declarations of "external" C functions and structures

# declara the head file we need
cdef extern from "sample.h"
   # next code copy from heah.h
   int gcd(int, int)
   bint in_mandel(double, double, int)
   int divide(int, int, int *)
   double avg(double *, int) nogil
   ...
```

这个文件在Cpython中的目的和作用就相当于一个C头文件。接下来创建一个名为sample.pyx的文件。这个文件定义包装函数，作为Python解释器到csample.pxd文件中定义的底层C代码之间的桥梁。

```Python
# sample.pyx

# Import the low-level C declarations
cimport csample

from cpython.pycapsule cimport *
from libc.stdlib cimport malloc, free

def gcd(unsigned int x, unsigned int y):
    return csample.gcd(x, y)

# other function 
...
```

要构建出扩展模块，还需要创建一个setup.py文件。如：

```python
from disutils.core import setup
from disutils.extension import Extension
from Cpython.Distuils import build_ext

ex_modules = [
    Extension('sample',
              ['sample.pys'],
              libraries=['sample'],
              librarie_dirs=['x']
        )
]

setup(
    name = 'Sample extension module',
    cmdclass = {'build_ext' : build_ext},
    ext_modules = ext_modules
    )
```

最终使用 ```  python3 setup.py build_ext --inplace ```构建出名为 sample.so 的扩展模块。
