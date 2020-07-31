---
layout:     post
title:      "Python Cookbook总结"
subtitle:   "12-13 章"
date:       2018-08-30 00:15:18
author:     "Pelhans"
header-img: "img/python_cookbook.jpg"
header-mask: 0.3 
catalog:    true
tags:
    - Python
---


> 包含第12、13章的内容。

* TOC
{:toc}

# 第十二章 并发

## 12.1 启动和停止线程 threading

threading 库可用来在单独的线程中执行任意的python 可调用对象。要实现这一要求，可以创建一个 Thread 实例并为它提供期望执行的可调用对象。

```python
# Code to execute in an independent thread
import time
def countdown(n):
    while n > 0:
        print('T-minus', n)
        n -= 1
        time.sleep(5)

# Creat and lanch a thread
from threading import Thread
t = Thread(target=countdown, args=(10,))
t.start()
```

当创建一个线程实例时，在调用它的 start() 方法之前(需要提供目标函数以及相应的参数)，线程并不会立即执行。 线程实例会在他们自己所属的系统级线程(即，POSIX线程或Windows 线程)中执行，这些线程完全由操作系统来管理。一旦启动后，线程就开始独立的运行，直到目标函数返回为止。可以用 t.is_alive() 来判断它是否还在运行。 还可以使用 t.join()请求连接到某个线程上，这么做会等待该线程结束。

解释器会一直保持运行，直到所有的线程都终结为止。对于需要长时间运行的线程或者一直不断运行的后台任务，应该考虑将这些线程设置为 daemon(即，守护线程)。

```python
t = Thread(target=countdown, args=(10,), daemon=True)
t.start()
```

daemon线程是无法被连接的，但是当主线程完成后它会自动销毁掉。如果想要终止线程，这个线程必须要能够在某个指定的点上轮询退出状态。如果线程会执行阻塞性的操作比如 I/O，那么轮询线程的退出状态时如何实现同步将变得很棘手。对于该问题，需要小心地为线程加上超时循环。

由于全局解释锁(GIL) 的存在，python线程的执行模型被限制为任意时刻只允许在解释器中运行一个线程。基于这个原因，不应该使用python线程来处理计算密集型的任务，因为在这种任务中，我们希望在多个CPU 核心上实现并行处理。python 线程更适合于 I/O 处理以及阻塞操作的并发执行任务，如等待 I/O、等待从数据库中取出结果等。

## 12.2 判断线程是否已经启动

线程的核心特征就是它们能够以非确定性的方式(即，何时开始执行、何时被打断、何时回复执行完全由操作系统来调度管理，这是用户和程序员都无法确定的)独立执行。如果程序中有其他线程需要判断某个线程是否已经到达执行过程中的某个点，根据这个判断后续的操作，那么就产生了非常棘手的线程同步问题。对应此问题，我们可以使用threading 库中的 Event 对象。Event 对象和条件标记类似，允许线程等待某个事件发生。

```python
from threading import Thread, Event
import time

# Code to execute in an independent thread
def countdown(n, started_evt):
    print('countdown starting')
    started_evt.set()  # 设置点，然后 wait 在这里执行
    while n > 0:
        print('T-minus', n)
        n -= 1
        time.sleep(5)

# Creat the event object that will be used to signal startup
started_evt = Event()

# Lanch the thread and pass the startup Event
print('Lanuching countdown')
t = Thread(target=countdown, args=(10, started_evt))
t.start()

# wait for thread to start
started_evt.wait()
print('countdown is running')
```

Event 对象最好只用于一次性的时间，如果线程打算一遍又一边的重复通知某个事件，那么最好使用 Condition 对象来处理。Event对象的关键特性就是它会唤醒所有等待的线程。

## 12.3 线程间的通信

也许将数据从一个线程发往另一个线程最安全的额做法就是使用queue模块中的 Queue了。要做到这些，首先创建一个Queue 实例，它会被所有线程共享。之后线程可以使用 put() 或 get() 操作来给队列添加或移除元素。 Queue 实例已经拥有了所有所需的锁，因此它们可以安全地在任意多的线程之间共享。

```python
from queue import Queue
from threading import Thread

# A thread that produces data
def producer(out_q):
    while True:
        # Produce some data
        ...
        out_q.put(data)
# A thread that consumes data
def consumer(in_q):
    while True:
        # Get some data
        data = in_q.get()
        # Process the data
        ...
# Creat the shared queue and launch both threads
q = Queue()
t1 = Thread(target=consumer, args=(q,))
t2 = Thread(target=consumer, args=(q,))
t1.start()
t2.start()
```

## 12.4 对临界区加锁 threading.Lock

对临界区加锁以避免出现静态条件。想让可变对象安全地用在多线程环境中，可以利用 threading 库中的 Lock对象。

```python
import threading

class SharedCounter:
    def __init__(self, initial_value=0):
        self._value = initial_value
        self._value_lock = threading.Lock()

    def incr(self, delta=1):
        with self._value_lock:
            self._value += delta

    def decr(self, delta=1):
        with self._value_lock:
            self._value =+ delta
```

当使用 with 语句时，Lock 对象可确保产生互斥的行为，也就是说，同一时间只允许一个线程执行 with 语句块中的代码。with 语句会在执行锁紧的代码块时获取到锁，当控制流离开锁进的语句块时释放这个锁。

threading 库中还有其他的同步原语，比如 RLock 和Semaphore 对象。 RLock 被称为可重入锁，它可以被同一个线程多次获取，主要用来编写基于锁的代码，或者基于“监视器”的同步处理。 Semaphore 对象是一种基于共享计数器的同步原语。如果计数器非零，那么with 语句会递减计数器并允许线程继续执行。当with 语句块结束时计数器会得到递增。如果计数器为零，那么执行过程会被阻塞，直到由另一个线程来递增计数为止。

## 12.5 避免死锁

在多线程程序中，出现死锁的常见原因就是线程一次尝试获取了多个锁。避免出现死锁的一种解决方案就是给程序中每个锁分配一个唯一的数字编号，并且在获取多个锁时只按照编号的升序来获取。利用上下文管理器实现这个机制非常简单。当想同一个或多个锁打交道时就要使用acquire()函数。

```python
import threading

x_lock = threading.Lock()
y_lock = threading.Lock()

def thread_1():
    while True:
        with acquire(x_lock, y_lock):
            print('Thread-1')

def thread_2():
    while True:
        with acquire(y_lock, x_lock):
            print('Thread_2')

t1 = threading.Thread(target=thread_1)
t1.daemon = True
t1.start()

t2 = threading.Thread(target=thread_2)
t2.daemon = True
t2.start()
```

## 12.6 创建线程池

concurrent.features 库中包含一个 ThreadPoolExecutor 类可以用来实现创建线程池。如果想手动创建自己的线程池，使用Queue来实现通常是足够简单的。

```
from socket import AF_INET, SOCK_STREAM, socket
from concurrent.features import ThreadPoolExecutor

def echo_client(sock, client_addr):
    print('Got connection from', client_addr)
    while True:
        msg = sock.recv(65536)
        if not msg:
            break
        sock.sendall(msg)
    print('Client closed conection')
    sock.close()

def echo_server(addr):
    pool = ThreadPoolExecutor(128)
    sock = socket(AF_INET, SOCK_STREAM)
    sock.bind(addr)
    sock.listen(5)
    while True:
        client_sock, client_addr = sock.accept()
        pool.submit(echo_client, client_sock, client_addr)
echo_server(('', 15000))
```

## 12.7 实现简单的额并行编程

concurrent.features 库中提供了一个 ProcessPoolExecutor类，可以用来在单独的python解释器实例中执行计算密集型函数。例如

```python
def find_all_robots(logdir):
    files = glob.glob(logdir + '/*.log.gz')
    all_robots = set()
    with features.ProcessPOolExecutor() as pool:
        for robots in pool.map(find_robots, files): # find_robots is a func
            all_robots.update(robots)
    return all_robots
```

## 12.8 如何避免GIL带来的限制

要规避GIL的限制主要有两种常用的策略。第一，如果完全用python 来编程，可以使用 multiprocessing 模块来创建线程池，把它当做协处理器来使用。每当有线程要执行CPU密集型的任务时，它就把任务提交到池中，然后进程池任务转交给运行在另一个进程中的python 解释器。当线程等待结果的时候就会释放 GIL。此外，由于计算是在另一个单独的解释器中进行的，这就不再收到 GIL 的限制了。

第二种方式的重点放在C语言扩展编程上，主要思想就是把计算密集型任务转移到C语言中，使其独立于python，在C代码中释放 GIL。

## 12.9 定义一个Actor任务

actor模式是最古老也是最简单的用来解决并发和分布式计算问题的方法之一。总的来说，actor就是执行一个并发的任务，它只是简单的对发送给它的消息进行处理。作为这些消息的响应，actor会决定是否要对其他的actor发送进一步的消息。actor任务之间的通信是单向且异步的。因此，消息的发送者并不知道消息何时才会实际传递，当消息已经处理完毕时也不会接收到相应确认。把线程和队列结合起来很容易定义actor。

## 12.10 实现发布者/订阅者消息模式

要实现发布者/订阅者消息模式，一般来说需要引入一个单独的“交换”或“网关”这样的对象，作为所有消息的中介。也就是说，不是直接消息从一个任务发往另一个任务，而是将消息发往交换中介，由中介将消息转发给一个或多个相关联的任务。

```python
from collections import defaultdict

class Exchange:
    def __init__(self):
        self._subscribers = set()

    def attach(self, task):
        self._subscribers.add(task)

    def detach(self, task):
        self._subscribers.remove(task)

    def end(self, msg):
        for subtitlebscriber in self._subscribers:
            subtitle.send(msg)

_exchanges = defaultdict(Exchange)

def get_exchange(name):
    return _exchanges[name]
```

交换中介其实就是一个对象，它保存了活跃的订阅者集合，并提供了关联、取消关联以及发送消息的方法。每个交换中介都有一个名称来标示，get_exchange()函数简单地返回同给定的名称相关联的那个Exchange对象。

## 12.11 轮询多个线程队列

对于轮询问题，我们常用的解决方案中涉及一个鲜为人知的技巧，即利用隐藏的环回网络连接。基本上来说思路是这样的：针对每个想要轮询的队列，创建一对互联网的socket。然后对其中一个socket执行写操作，以此表示数据保存在。另一个socket就传递给select()或者类似的函数来轮询数据。

```python
import queue
import socket
import os

class PollableQueue(queue.Queue):
    def __init__(self):
        super().__init__()
        if os.name == 'posix':
            self._putsocket, self._getsocket = socket.socketpair()
        else:
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.bind(('127.0.0.1', 0))
            server.listen(1)
            self._putsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._putsocket.connect(server.getsockname())
            self._getsocket, _ = server.accept()
            server.close()

    def fileno(self):
        return self._getsocket.fileno()

    def put(self, item):
        super().put(item)
        self._putsocket.send(b'x)

    def get(self):
        self._getsocket.recv(1)
        return super().get()
```

# 第十三章 实用脚本和系统管理

## 13.1 通过重定向、管道或输入文件来作为脚本的输入

python 内置的 fileinput模块可以对一个或多个文件中的内容进行迭代、遍历等操作。也可以将从命令中产生输出给脚本、把文件重定向给脚本等。

```python
import fileinput
              
with fileinput.input() as f_input:
    for line in f_input:      
        print(line, end='')

# when used
chmod +x filein.py
ls | ./filein.py
```

## 13.2 执行外部命令并获取输出

可以使用 subprocess.check_output()来完成，如

```python
import subprocess
out_bytes = subprocess.check_output(['netstat', '-a'])
```

默认情况下，check_out()只会返回写入到标准输出中的结果。如果希望标准输出和标准错误输出都能获取到，可以使用参数stderr。另外如果需要命令的执行通过shell来解释，那么需要提供sehll=True。

## 13.3 读取配置文件

可以使用 configparser 模块来读取如.ini 格式所编写的配置文件。

```python
#cinfig.ini
; config.init
; sample configuration file

[installation]
library=%(prefix)s/lib
include=%(prefix)s/include

[debug]
log_errors=true
show_warnings=False

# use configparser to read this ini
from configparser import configParser
cfg = configParser()
cfg.read('config.ini')
out: ['config.ini']
cfg.sections()
out: ['installation', 'debug']
cfg.get('installation', 'library')
out: '/usr/local/lib'
cfg.set('server', 'port', '9000') # write new config
```

## 13.4 给脚本添加日志记录

给程序简单的添加日志功能，最简单的方法就是使用 logging 模块了。 logging 的调用 (critical()、error()、warning()、info()、debug())分别代表着不同的严重级别，以降序排列。basicConfig()的 level参数是一个过滤器，所有等级低于此设定的消息都会被忽略掉。

```python
import logging

def main():
    logging.basicConfis(
        filename='app.log'
        levelel=logging.ERROR
        )
    hostname = 'www.python.org'
    item = 'spam'
    filename = 'data.csv'
    mode = 'r'
    
    logging.critical('Host %s unknown', hostname)
    logging.error("Couldn't find %r", item)
    logging.warning('Feature is deprecated')
    logging.info('Opening file %r, mode=%r', filename, mode)
    logging.debug('Got here')

if __name__ == '__main__':
    main()
```
