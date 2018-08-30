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

