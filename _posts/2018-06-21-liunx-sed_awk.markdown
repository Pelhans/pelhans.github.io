---
layout:     post
title:      "常用文本处理 sed awk" 
subtitle:   "sed awk 使用总结"
date:       2018-06-21 00:15:18
author:     "Pelhans"
header-img: "img/linux.jpg"
header-mask: 0.3 
catalog:    true
tags:
    - Linux
---


> 本文收集了一些优秀的sed 以及awk 使用教程，并通过自身实践对常用命令做了总结。原文链接在文末。

* TOC
{:toc}

# P1  sed 命令的使用

## 概述

sed是stream editor的简称，也就是流编辑器。它一次处理一行内容，处理时，把当前处理的行存储在临时缓冲区中，称为“模式空间”（pattern space），接着用sed命令处理缓冲区中的内容，处理完成后，把缓冲区的内容送往屏幕。接着处理下一行，这样不断重复，直到文件末尾。文件内容并没有 改变，除非你使用重定向存储输出。

## 使用语法

sed [option] 'command' input_file

其中option是可选的，常用的option有如下几种：

   * -n 使用安静(silent)模式（想不通为什么不是-s）。在一般sed的用法中，所有来自stdin的内容一般都会被列出到屏幕上。但如果加上-n参数后，则只有经过sed特殊处理的那一行(或者动作)才会被列出来；
   *  -e 直接在指令列模式上进行 sed 的动作编辑；
   *  -f 直接将 sed 的动作写在一个文件内， -f filename 则可以执行filename内的sed命令；
   * -r 让sed命令支持扩展的正则表达式(默认是基础正则表达式)；
   *  -i 直接修改读取的文件内容，而不是由屏幕输出。

常用的命令有以下几种：

 * p： print即打印，该命令会打印当前选择的行到屏幕上；
 * a \： append即追加字符串， a \的后面跟上字符串s(多行字符串可以用\n分隔)，则会在当前选择的行的后面都加上字符串s；
 * d： delete即删除，该命令会将当前选中的行删除；
 * i \： insert即插入字符串，i \后面跟上字符串s(多行字符串可以用\n分隔)，则会在当前选中的行的前面都插入字符串s；
 * c \： 取代/替换字符串，**整行替换**，c \后面跟上字符串s(多行字符串可以用\n分隔)，则会将当前选中的行替换成字符串s；
 * s： 替换，通常s命令的用法是这样的：1，2s/old/new/g，将old字符串替换成new字符串，**它是单词级别替换**。
 * y： 替换，**字母级别的替换**，用法如 sed '1,3y/abc/ABC/' data ，是分别把字母a 替换为A，b替换为B，c替换为C。

## 命令示例

假设有一个本地文件test.txt，文件内容如下：
```
$ cat test.txt
this is first line
this is second line
this is third line
this is fourth line
this fifth line
happy everyday
end
```
本节将使用该文件详细演示每一个命令的用法。
### a命令
```
	[qifuguang@winwill~]$ sed '1a \add one' test.txt
	this is first line
	add one
	this is second line
	this is third line
	this is fourth line
	this is fifth line
	happy everyday
	end
```

本例命令部分中的1表示第一行，同样的第二行写成2，第一行到第三行写成1,3，用$表示最后一行，比如2,$表示第二行到最后一行中间所有的行(包含第二行和最后一行)。
本例的作用是在第一行之后增加字符串”add one”，从输出可以看到具体效果。
```
	[qifuguang@winwill~]$ sed '1,$a \add one' test.txt
	this is first line
	add one
	this is second line
	add one
	this is third line
	add one
	this is fourth line
	add one
	this is fifth line
	add one
	happy everyday
	add one
	end
	add one
```

本例表示在第一行和最后一行所有的行后面都加上”add one”字符串，从输出可以看到效果。
```
	[qifuguang@winwill~]$ sed '/first/a \add one' test.txt
	this is first line
	add one
	this is second line
	this is third line
	this is fourth line
	this is fifth line
	happy everyday
	end
```

本例表示在包含”first”字符串的行的后面加上字符串”add one”，从输出可以看到第一行包含first，所以第一行之后增加了”add one”
```
	[qifuguang@winwill~]$ sed '/^ha.*day$/a \add one' test.txt
	this is first line
	this is second line
	this is third line
	this is fourth line
	this is fifth line
	happy everyday
	add one
	end
```

本例使用正则表达式匹配行，^ha.*day$表示以ha开头，以day结尾的行，则可以匹配到文件的”happy everyday”这样，所以在该行后面增加了”add one”字符串。
### i命令

i命令使用方法和a命令一样的，只不过是在匹配的行的前面插入字符串，所以直接将上面a命令的示例的a替换成i即可，在此就不啰嗦了。
### c命令
```	
	[qifuguang@winwill~]$ sed '$c \add one' test.txt
	this is first line
	this is second line
	this is third line
	this is fourth line
	this is     fifth line
	happy everyday
	add one
```

本例表示将最后一行替换成字符串”add one”，从输出可以看到效果。
```
	[qifuguang@winwill~]$ sed '4,$c \add one' test.txt
	this is first line
	this is second line
	this is third line
	add one
```

本例将第四行到最后一行的内容替换成字符串”add one”。
```
	[qifuguang@winwill~]$ sed '/^ha.*day$/c \replace line' test.txt
	this is first line
	this is second line
	this is third line
	this is fourth line
	this is fifth line
	replace line
	end
```

本例将以ha开头，以day结尾的行替换成”replace line”。
### d命令
```
	[qifuguang@winwill~]$ sed '/^ha.*day$/d' test.txt
	this is first line
	this is second line
	this is third line
	this is fourth line
	this is fifth line
	end
```

本例删除以ha开头，以day结尾的行。
```	
	[qifuguang@winwill~]$ sed '4,$d' test.txt
	this is first line
	this is second line
	this is third line
```

本例删除第四行到最后一行中的内容。
### p命令
```
	[qifuguang@winwill~]$ sed -n '4,$p' test.txt
	this is fourth line
	this is fifth line
	happy everyday
	end
```

本例在屏幕上打印第四行到最后一行的内容，p命令一般和-n选项一起使用.
```
	[qifuguang@winwill~]$ sed -n '/^ha.*day$/p' test.txt
	happy everyday
```

本例打印以ha开始，以day结尾的行。
### s命令

实际运用中s命令式最常使用到的。
```
	[qifuguang@winwill~]$ sed 's/line/text/g' test.txt
	this is first text
	this is second text
	this is third text
	this is fourth text
	this is fifth text
	happy everyday
	end
```

本例将文件中的所有line替换成text，最后的g是global的意思，也就是全局替换，如果不加g，则只会替换本行的第一个line。
```
	[qifuguang@winwill~]$ sed '/^ha.*day$/s/happy/very happy/g' test.txt
	this is first line
	this is second line
	this is third line
	this is fourth line
	this is fifth line
	very happy everyday
	end
```

本例首先匹配以ha开始，以day结尾的行，本例中匹配到的行是”happy everyday”这样，然后再将该行中的happy替换成very happy。
```
	[qifuguang@winwill~]$ sed 's/\(.*\)line$/\1/g' test.txt
	this is first
	this is second
	this is third
	this is fourth
	this is fifth
	happy everyday
	end
```

这个例子有点复杂，先分解一下。首先s命令的模式是s/old/new/g这样的，所以本例的old部分即\(.*\)line$,sed命令中使用\(\)包裹的内容表示正则表达式的第n部分，序号从1开始计算，本例中只有一个\(\)所以\(.*\)表示正则表达式的第一部分，这部分匹配任意字符串，所以\(.*\)line$匹配的就是以line结尾的任何行。然后将匹配到的行替换成正则表达式的第一部分（本例中相当于删除line部分），使用\1表示匹配到的第一部分，同样\2表示第二部分，\3表示第三部分，可以依次这样引用。比如下面的例子：
```
	[qifuguang@winwill~]$ sed 's/\(.*\)is\(.*\)line/\1\2/g' test.txt
	this  first
	this  second
	this  third
	this  fourth
	this  fifth
	happy everyday
	end
```

正则表达式中is两边的部分可以用\1和\2表示，该例子的作用其实就是删除中间部分的is。

# awk 
AWK是一种处理文本文件的语言，是一个强大的文本分析工具。
## 语法
```
awk [选项参数] 'script' var=value file(s)
or
awk [选项参数] -f scriptfile var=value file(s)
```
常用选项参数说明：

* F fs or --field-separator fs 指定输入文件折分隔符，fs是一个字符串或者是一个正则表达式，如-F:。
* v var=value or --asign var=value 赋值一个用户定义变量。
* f scripfile or --file scriptfile 从脚本文件中读取awk命令。

## 常用命令解析

重点解释一下常用的几个内建变量、作用范围以及他们之间的差别。

### NR

**NR 是已经读出的记录数，就是行号，从1开始，需要注意的是它是跨文件的，即在处理第二个文件时会继续上一个文本的行号**，如：

```
$ awk '{print FILENAME, "FNR= ", FNR,"  NR= ", NR}' student-marks bookdetails
student-marks FNR=  1   NR=  1
student-marks FNR=  2   NR=  2
student-marks FNR=  3   NR=  3
student-marks FNR=  4   NR=  4
student-marks FNR=  5   NR=  5
bookdetails   FNR=  1   NR=  6
bookdetails   FNR=  2   NR=  7
bookdetails   FNR=  3   NR=  8
bookdetails   FNR=  4   NR=  9
bookdetails   FNR=  5   NR=  10
```

### FNR

FNR 是各文件分别计数的行号，在处理不同文件时重新开始计数，例子见上。

### NF

NF 是一条记录的字段的数目，说白了就是每行有多少个字。如：

```
$ awk '{print NR,"->",NF}' student-marks
1 -> 5
2 -> 5
3 -> 4
4 -> 5
5 -> 4
```

### FS

FS(Field Separator) 读取并解析输入文件中的每一行时，默认按照空格分隔为字段变量,$$\$1$$,$ 2...等。首先这个字段Field对应文本中的单词或者中文的字。

因此我的理解为它是在你读取输入时判断分割的标准。如默认是采用空格作为分割，那'we are,family ?' 它就认为是有三个输入，"we1 are,family2 ?3"，但当你把它设为其他字符如“,”时，那这句就会被认为是两个字符"we are1 family ?2":

```
$cat test.txt
we are,family ?
$awk '{print NF}' test.txt
3
$awk 'BEGIN{FS=","}{print NF}' test.txt
2
```

### OFS

OFS，顾名思义就是输出时的FS，直白理解就是输出时采用的分隔符，如：

```
$awk 'BEGIN{OFS="###"}{print $1,$2,$3}' test.txt
we###are,family###?
```

### RS

RS(Record Separator)定义了一行记录。读取文件时，默认将一行作为一条记录。也就是说它的处理对象是行。

```
$ cat student.txt
Jones
2143
78
84
77

Gondrol
2321
56
58
45

$ cat student.awk
BEGIN {
    RS="\n\n";
    FS="\n";
}
{
    print $1,$2;
}
$ awk -f student.awk  student.txt
Jones 2143
Gondrol 2321
RinRao 2122
Edwin 2537
Dayan 2415
```

在 student.awk 中，把每个学生的详细信息作为一条记录，　这是因为RS(记录分隔符)是被设置为两个换行符。并且因为 FS (字段分隔符)是一个换行符，所以一行就是一个字段。

### ORS

ORS就是输出时的RS。

```
$awk 'BEGIN{ORS="=";} {print;}' student-marks
Jones 2143 78 84 77=Gondrol 2321 56 58 45=RinRao 2122 38 37 65=Edwin 2537 78 67 45=Dayan 2415 30 47 20=
```

## 基本用法

先创建一个实验文本，log.txt文本内容如下：
```
	2 this is a test
	3 Are you like awk
	This's a test
	10 There are orange,apple,mongo
```

### 用法一：
```
	awk '{[pattern] action}' {filenames}   # 行匹配语句 awk
```
#### 实例：

每行按空格或TAB分割，输出文本中的1、4项

```
$ awk '{print $1,$4}' log.txt
```
	 ---------------------------------------------
	 2 a
	 3 like
	 This's
	 10 orange,apple,mongo
	 # 格式化输出
	 $ awk '{printf "%-8s %-10s\n",$1,$4}' log.txt
	 ---------------------------------------------
	 2        a
	 3        like
	 This's
	 10       orange,apple,mongo
 

### 用法二：
```
awk -F   #-F相当于内置变量FS, 指定分割字符
```
#### 实例：
```
# 使用","分割
 $  awk -F, '{print $1,$2}'   log.txt
 ---------------------------------------------
 2 this is a test
 3 Are you like awk
 This's a test
 10 There are orange apple
 # 或者使用内建变量
 $ awk 'BEGIN{FS=","} {print $1,$2}'     log.txt
 ---------------------------------------------
 2 this is a test
 3 Are you like awk
 This's a test
 10 There are orange apple
 # 使用多个分隔符.先使用空格分割，然后对分割结果再使用","分割
 $ awk -F '[ ,]'  '{print $1,$2,$5}'   log.txt
 ---------------------------------------------
 2 this test
 3 Are awk
 This's a
 10 There apple
```
### 用法三：

	awk -v  # 设置变量

#### 实例：
```
 $ awk -va=1 '{print $1,$1+a}' log.txt
 ---------------------------------------------
 2 3
 3 4
 This's 1
 10 11
 $ awk -va=1 -vb=s '{print $1,$1+a,$1b}' log.txt
 ---------------------------------------------
 2 3 2s
 3 4 3s
 This's 1 This'ss
 10 11 10s
```
### 用法四：

	awk -f {awk脚本} {文件名}

#### 实例：
```
 $ awk -f cal.awk log.txt
```
### 运算符

```
运算符 	描述
= += -= *= /= %= ^= **= 	赋值
?: 	C条件表达式
|| 	逻辑或
&& 	逻辑与
~ ~! 	匹配正则表达式和不匹配正则表达式
< <= > >= != == 	关系运算符
空格 	连接
+ - 	加，减
* / % 	乘，除与求余
+ - ! 	一元加，减和逻辑非
^ *** 	求幂
++ -- 	增加或减少，作为前缀或后缀
$ 	字段引用
in 	数组成员
```
```
过滤第一列大于2的行

$ awk '$1>2' log.txt    #命令
#输出
3 Are you like awk
This's a test
10 There are orange,apple,mongo
```
```
过滤第一列等于2的行

$ awk '$1==2 {print $1,$3}' log.txt    #命令
#输出
2 is
```
```
过滤第一列大于2并且第二列等于'Are'的行

$ awk '$1>2 && $2=="Are" {print $1,$2,$3}' log.txt
#输出
3 Are you
```
###内建变量
```
变量 	描述
$n 	当前记录的第n个字段，字段间由FS分隔
$0 	完整的输入记录
ARGC 	命令行参数的数目
ARGIND 	命令行中当前文件的位置(从0开始算)
ARGV 	包含命令行参数的数组
CONVFMT 	数字转换格式(默认值为%.6g)ENVIRON环境变量关联数组
ERRNO 	最后一个系统错误的描述
FIELDWIDTHS 	字段宽度列表(用空格键分隔)
FILENAME 	当前文件名
FNR 	各文件分别计数的行号
FS 	字段分隔符(默认是任何空格)
IGNORECASE 	如果为真，则进行忽略大小写的匹配
NF 	一条记录的字段的数目
NR 	已经读出的记录数，就是行号，从1开始
OFMT 	数字的输出格式(默认值是%.6g)
OFS 	输出记录分隔符（输出换行符），输出时用指定的符号代替换行符
ORS 	输出记录分隔符(默认值是一个换行符)
RLENGTH 	由match函数所匹配的字符串的长度
RS 	记录分隔符(默认是一个换行符)
RSTART 	由match函数所匹配的字符串的第一个位置
SUBSEP 	数组下标分隔符(默认值是/034)
```
```
$ awk 'BEGIN{printf "%4s %4s %4s %4s %4s %4s %4s %4s %4s\n","FILENAME","ARGC","FNR","FS","NF","NR","OFS","ORS","RS";printf "---------------------------------------------\n"} {printf "%4s %4s %4s %4s %4s %4s %4s %4s %4s\n",FILENAME,ARGC,FNR,FS,NF,NR,OFS,ORS,RS}'  log.txt
FILENAME ARGC  FNR   FS   NF   NR  OFS  ORS   RS
---------------------------------------------
log.txt    2    1         5    1
log.txt    2    2         5    2
log.txt    2    3         3    3
log.txt    2    4         4    4
$ awk -F\' 'BEGIN{printf "%4s %4s %4s %4s %4s %4s %4s %4s %4s\n","FILENAME","ARGC","FNR","FS","NF","NR","OFS","ORS","RS";printf "---------------------------------------------\n"} {printf "%4s %4s %4s %4s %4s %4s %4s %4s %4s\n",FILENAME,ARGC,FNR,FS,NF,NR,OFS,ORS,RS}'  log.txt
FILENAME ARGC  FNR   FS   NF   NR  OFS  ORS   RS
---------------------------------------------
log.txt    2    1    '    1    1
log.txt    2    2    '    1    2
log.txt    2    3    '    2    3
log.txt    2    4    '    1    4
# 输出顺序号 NR, 匹配文本行号
$ awk '{print NR,FNR,$1,$2,$3}' log.txt
---------------------------------------------
1 1 2 this is
2 2 3 Are you
3 3 This's a test
4 4 10 There are
# 指定输出分割符
$  awk '{print $1,$2,$5}' OFS=" $ "  log.txt
---------------------------------------------
2 $ this $ test
3 $ Are $ awk
This's $ a $
10 $ There $
```

### 使用正则，字符串匹配
```
# 输出第二列包含 "th"，并打印第二列与第四列
$ awk '$2 ~ /th/ {print $2,$4}' log.txt
---------------------------------------------
this a
```
```
~ 表示模式开始。// 中是模式。

# 输出包含"re" 的行
$ awk '/re/ ' log.txt
---------------------------------------------
3 Are you like awk
10 There are orange,apple,mongo
```
###忽略大小写
```
$ awk 'BEGIN{IGNORECASE=1} /this/' log.txt
---------------------------------------------
2 this is a test
This's a test
```
###模式取反
```
$ awk '$2 !~ /th/ {print $2,$4}' log.txt
---------------------------------------------
Are like
a
There orange,apple,mongo
$ awk '!/th/ {print $2,$4}' log.txt
---------------------------------------------
Are like
a
There orange,apple,mongo
```
### awk脚本

关于awk脚本，我们需要注意两个关键词BEGIN和END。

    BEGIN{ 这里面放的是执行前的语句 }
    END {这里面放的是处理完所有的行后要执行的语句 }
    {这里面放的是处理每一行时要执行的语句}

假设有这么一个文件（学生成绩表）：
```
$ cat score.txt
Marry   2143 78 84 77
Jack    2321 66 78 45
Tom     2122 48 77 71
Mike    2537 87 97 95
Bob     2415 40 57 62
```
我们的awk脚本如下：
```
$ cat cal.awk
#!/bin/awk -f
#运行前
BEGIN {
    math = 0
    english = 0
    computer = 0
 
    printf "NAME    NO.   MATH  ENGLISH  COMPUTER   TOTAL\n"
    printf "---------------------------------------------\n"
}
#运行中
{
    math+=$3
    english+=$4
    computer+=$5
    printf "%-6s %-6s %4d %8d %8d %8d\n", $1, $2, $3,$4,$5, $3+$4+$5
}
#运行后
END {
    printf "---------------------------------------------\n"
    printf "  TOTAL:%10d %8d %8d \n", math, english, computer
    printf "AVERAGE:%10.2f %8.2f %8.2f\n", math/NR, english/NR, computer/NR
}
```
我们来看一下执行结果：
```
$ awk -f cal.awk score.txt
NAME    NO.   MATH  ENGLISH  COMPUTER   TOTAL
---------------------------------------------
Marry  2143     78       84       77      239
Jack   2321     66       78       45      189
Tom    2122     48       77       71      196
Mike   2537     87       97       95      279
Bob    2415     40       57       62      159
---------------------------------------------
  TOTAL:       319      393      350
AVERAGE:     63.80    78.60    70.00
```
###另外一些实例

AWK的hello world程序为：

	BEGIN { print "Hello, world!" }

计算文件大小
```
$ ls -l *.txt | awk '{sum+=$6} END {print sum}'
--------------------------------------------------
666581
```
从文件中找出长度大于80的行

	awk 'length>80' log.txt

打印九九乘法表
```
seq 9 | sed 'H;g' | awk -v RS='' '{for(i=1;i<=NF;i++)printf("%dx%d=%d%s", i, NR, i*NR, i==NR?"\n":"\t")}'
```

#Ref
[sed winwill2012的回答 - 知乎](https://www.zhihu.com/question/30074714/answer/64706509)
[awk 菜鸟教程](http://www.runoob.com/linux/linux-comm-awk.html)
