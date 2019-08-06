---
layout:     post
title:      "手写代码 习题集"
subtitle:   "剑指 Offer 篇"
date:       2019-07-04 00:15:18
author:     "Pelhans"
header-img: "img/kg_bg.jpg"
header-mask: 0.3 
catalog:    true
tags:
    - Python
---

> 春天出去找工作不知道外面要手写代码和手推ML公式，简直 SB 到了极致，慢慢补吧。下面是自己近期刷几遍剑指后的个人分类总结。其中动态规划、数组、字符串、分治、贪心这几类剑指包含的不多，需要在 leetcode 上专门训练。

* TOC
{:toc}

# 数组

## 剑指 3. 数组中重复的数字

题目: 在一个长度为n的数组里的所有数字都在0到n-1的范围内。 数组中某些数字是重复的，但不知道有几个数字是重复的。也不知道每个数字重复几次。请找出数组中任意一个重复的数字。 例如，如果输入长度为7的数组{2,3,1,0,2,5,3}，那么对应的输出是第一个重复的数字2。

思路：由于给定数组的特殊性，当数字与索引号相等时跳过，否则与索引号的数对换，直到遇到重复数字。

代码

```
# -*- coding:utf-8 -*-
class Solution:
    # 这里要特别注意~找到任意重复的一个值并赋值到duplication[0]
    # 函数返回True/False
    def duplicate(self, numbers, duplication):
        # write code here
        for i in range(len(numbers)):
            if numbers[i] == i:
                continue
            elif numbers[i] == numbers[numbers[i]]:
                    duplication[0] = numbers[i]
                    return True
            else:
                numbers[numbers[i]], numbers[i] = numbers[i], numbers[numbers[i]]
        return False
```

## 剑指 4.  二维数组中的查找

题目： 在一个二维数组中（每个一维数组的长度相同），每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。

思路：比较简单的查找

代码

```
# -*- coding:utf-8 -*-
class Solution:
    # array 二维列表
    def Find(self, target, array):
        # write code here
        if len(array) < 1 or len(array[0]) < 1:
            return False
        
        row = len(array)
        col = len(array[0])
        r = 0
        while r < row and col > 0:
            if array[r][col-1] == target:
                return True
            elif array[r][col-1] < target:
                r += 1
            else:
                col -= 1
        return False
```

## 剑指 11. 旋转数组的最小数字 重要

题目：把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。 输入一个非减排序的数组的一个旋转，输出旋转数组的最小元素。 例如数组{3,4,5,1,2}为{1,2,3,4,5}的一个旋转，该数组的最小值为1。 NOTE：给出的所有元素都大于0，若数组大小为0，请返回0。

思路：二分查找，用递归和非递归两种方法

代码：

```
# -*- coding:utf-8 -*-
class Solution:
    def minNumberInRotateArray(self, rotateArray):
        # write code here
        if not rotateArray:
            return 0
        start = 0
        end = len(rotateArray) - 1
        mid = 0
        while start < end:
            if end - start == 1:
                mid = end
                break
            mid = (end + start) / 2
            if rotateArray[start] > rotateArray[mid]:
                end = mid
            elif rotateArray[mid] > rotateArray[end]:
                start = mid
        return rotateArray[mid]
```

## 顺时针打印矩阵； 

题目： 输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字，例如，如果输入如下4 X 4矩阵： 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 则依次打印出数字1,2,3,4,8,12,16,15,14,13,9,5,6,7,11,10.

思路： 知道四角的坐标，依次打印即可。难点是边界条件的判定。

代码：

```
# -*- coding:utf-8 -*-
class Solution:
    # matrix类型为二维列表，需要返回列表
    def printMatrix(self, matrix):
        # write code here
        if len(matrix) == 0 or len(matrix[0]) == 0:
            return None
        
        start = 0
        rows = len(matrix)
        cols = len(matrix[0])
        res = []
        while cols > 2*start and rows > 2*start:
            self.printNum(matrix, rows, cols, start, res)
            start += 1
        return res
    
    def  printNum(self, matrix, rows, cols, start, res):
        endCol = cols - start - 1
        endRow = rows - start - 1 
        
        for i in range(start, endCol+1):
            res.append(matrix[start][i]) 
        if start < endRow:
            for i in range(start+1, endRow+1):
                res.append(matrix[i][endCol])
        if start < endRow and start < endCol:
            for i in range(endCol-1, start-1, -1):
                res.append(matrix[endRow][i])
        if start < endRow-1 and start < endCol:
            for i in range(endRow-1, start, -1):
                res.append(matrix[i][start])
```


## 剑指 21. 调整数组顺序使奇数位于偶数前面

题目：输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有的奇数位于数组的前半部分，所有的偶数位于数组的后半部分，并保证奇数和奇数，偶数和偶数之间的相对位置不变。

思路：要求稳定，用插入排序思想

代码

```
# -*- coding:utf-8 -*-
class Solution:
    def reOrderArray(self, array):
        # write code here
        if not array:
            return []
        if len(array) == 1:
            return array
        for i in range(len(array)):
            tmp = array[i]
            idx = i
            for j in range(i-1, 0, -1):
                if array[j] % 2 != 0:
                    array[j+1] = array[j]
                    idx = j
                else:
                    break
            array[idx] = tmp
        return array
```

## 剑指 39. 数组中出现次数超过一半的数字

题目：数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。例如输入一个长度为9的数组{1,2,3,2,2,2,5,4,2}。由于数字2在数组中出现了5次，超过数组长度的一半，因此输出2。如果不存在则输出0。

思路：用times标记，最后出现次数大于一半的是最后将times置为1的数字。

代码

```
# -*- coding:utf-8 -*-
class Solution:
    def MoreThanHalfNum_Solution(self, numbers):
        # write code here
        if len(numbers) <= 0:
            return 0
        times = 0
        res = numbers[0]
        for i in xrange(len(numbers)):
            if times == 0:
                res = numbers[i]
                times = 1
            if numbers[i] == res:
                times += 1
            else:
                times -= 1
        sum = numbers.count(res)
        return res if sum > len(numbers)/2 else 0
```


## 剑指 40. 最小的k个数 重要

题目：输入n个整数，找出其中最小的K个数。例如输入4,5,1,6,2,7,3,8这8个数字，则最小的4个数字是1,2,3,4,。

思路：一种是维护一个最大堆，来存储k个数，另一种 是根据快排来做。

代码

```
# 最大堆版
#!/usr/bin/env python
# coding=utf-8

# -*- coding:utf-8 -*-
class Solution:
    def GetLeastNumbers_Solution(self, tinput, k):
        # write code here
        if tinput == [] or k > len(tinput) or k==0:
            return []
        def maxHeap(tinput):
            for i in range(len(tinput), -1, -1):
                if 2*i +1 > len(tinput) and 2*i +2 > len(tinput):
                    continue
                left = 2*i +1
                right = 2*i + 2
                idx = i
                if left < len(tinput) and tinput[left] > tinput[i]:
                    idx = left
                if right < len(tinput) and tinput[right] > tinput[i]:
                    idx = right
                tinput[i], tinput[idx] = tinput[idx], tinput[i]
            for i in range(len(tinput)):
                if 2*i +1 > len(tinput) and 2*i +2 > len(tinput):
                    continue
                left = 2*i + 1
                right = 2*i + 2
                idx = i
                if left < len(tinput) and tinput[left] > tinput[i]:
                    idx = left
                if right < len(tinput) and tinput[right] > tinput[i]:
                    idx = right
                tinput[i], tinput[idx] = tinput[idx], tinput[i]
            return tinput
        k_head = [float("inf")]*k
        for num in tinput:
            if num < k_head[0]:
                k_head[0] = num
                k_head = maxHeap(k_head)
        k_head.sort()
        return k_head
```

```
# 快排版
# -*- coding:utf-8 -*-
class Solution:
    def GetLeastNumbers_Solution(self, tinput, k):
        # write code here
        def quick_sort(lst, lp, rp):
            if not lst or lp >= rp:
                return lst
            povit = lst[lp]
            left = lp
            right = rp

            while left < right:
                while lst[right] >= povit and left < right:
                    right -= 1
                while lst[left] <= povit and left < right:
                    left += 1
                lst[left], lst[right] = lst[right], lst[left]
            lst[lp], lst[left] = lst[left], lst[lp]
            if left== k:
                return lst
            elif left < k:
                quick_sort(lst, right+1, rp)
            else:
                quick_sort(lst, lp, left-1)
            return lst

        if tinput == [] or k > len(tinput):
            return []
        tinput = quick_sort(tinput, 0, len(tinput)-1)
        return tinput[: k]
```


## 剑指 41. 数据流中的中位数  重要

题目：如何得到一个数据流中的中位数？如果从数据流中读出奇数个数值，那么中位数就是所有数值排序之后位于中间的数值。如果从数据流中读出偶数个数值，那么中位数就是所有数值排序之后中间两个数的平均值。我们使用Insert()方法读取数据流，使用GetMedian()方法获取当前读取数据的中位数。

思路：用最大堆和最小堆做。以下思路来自剑指 Offer

主要的思想是：因为要求的是中位数，那么这两个堆，大顶堆用来存较小的数，从大到小排列； 

* 小顶堆存较大的数，从小到大的顺序排序*，显然中位数就是大顶堆的根节点与小顶堆的根节点和的平均数。 
* 保证：小顶堆中的元素都大于等于大顶堆中的元素，所以每次塞值，并不是直接塞进去，而是从另一个堆中poll出一个最大（最小）的塞值 
* 当数目为偶数的时候，将这个值插入大顶堆中，再将大顶堆中根节点（即最大值）插入到小顶堆中； 
* 当数目为奇数的时候，将这个值插入小顶堆中，再讲小顶堆中根节点（即最小值）插入到大顶堆中； 
* 取中位数的时候，如果当前个数为偶数，显然是取小顶堆和大顶堆根结点的平均值；如果当前个数为奇数，显然是取小顶堆的根节点

代码

```
# 可以考虑复用大顶堆作为小顶堆，我这里是为了练手
# -*- coding:utf-8 -*-
class Solution:
    def __init__(self):
        self.max = []
        self.min = []
        self.count = 0
    
    def Insert(self, num):
        # write code here
        self.count += 1
        if self.count%2 == 0:
            self.min.append(num)
            self.min = self.min_heap(self.min)
            self.max.append(self.min[0])
            self.max = self.max_heap(self.max)
            self.min.pop(0)
            self.min = self.min_heap(self.min)
        else:
            self.max.append(num)
            self.max = self.max_heap(self.max)
            self.min.append(self.max[0])
            self.min = self.min_heap(self.min)
            self.max.pop(0)
            self.max = self.max_heap(self.max)
        
    def GetMedian(self, a):
        # write code here
        if self.count%2 == 0:
            return (self.max[0]+self.min[0])/2.0
        else:
            return self.min[0]
    
    def max_heap(self, stack, ref=float("-inf")):
        n = len(stack)
        for i in range(n-1, -1, -1):
            cur = stack[i]
            left = ref
            right = ref
            if 2*i + 1 < n:
                left = stack[2*i+1]
            if 2*i + 2 < n:
                right = stack[2*i+2]
            if left > cur and left > right:
                stack[i], stack[2*i+1] = stack[2*i+1], stack[i]
            elif right > cur and right > left:
                stack[i], stack[2*i+2] = stack[2*i+2], stack[i]
        for i in range(n-1):
            cur = stack[i]
            left = ref
            right = ref
            if 2*i + 1 < n:
                left = stack[2*i+1]
            if 2*i + 2 < n:
                right = stack[2*i+2]
            if left > cur and left > right:
                stack[i], stack[2*i+1] = stack[2*i+1], stack[i]
            elif right > cur and right > left:
                stack[i], stack[2*i+2] = stack[2*i+2], stack[i]
        return stack
    
    def min_heap(self, stack, ref=float("inf")):
        n = len(stack)
        for i in range(n-1, -1, -1):
            cur = stack[i]
            left = ref
            right = ref
            if 2*i + 1 < n:
                left = stack[2*i+1]
            if 2*i + 2 < n:
                right = stack[2*i+2]
            if left < cur and left < right:
                stack[i], stack[2*i+1] = stack[2*i+1], stack[i]
            elif right < cur and right < left:
                stack[i], stack[2*i+2] = stack[2*i+2], stack[i]
        for i in range(n-1):
            cur = stack[i]
            left = ref
            right = ref
            if 2*i + 1 < n:
                left = stack[2*i+1]
            if 2*i + 2 < n:
                right = stack[2*i+2]
            if left < cur and left < right:
                stack[i], stack[2*i+1] = stack[2*i+1], stack[i]
            elif right < cur and right < left:
                stack[i], stack[2*i+2] = stack[2*i+2], stack[i]
        return stack
```
## 剑指42. 连续子数组的最大和 重要 

题目：HZ偶尔会拿些专业问题来忽悠那些非计算机专业的同学。今天测试组开完会后,他又发话了:在古老的一维模式识别中,常常需要计算连续子向量的最大和,当向量全为正数的时候,问题很好解决。但是,如果向量中包含负数,是否应该包含某个负数,并期望旁边的正数会弥补它呢？例如:{6,-3,-2,7,-15,1,2,2},连续子向量的最大和为8(从第0个开始,到第3个为止)。给一个数组，返回它的最大连续子序列的和，你会不会被他忽悠住？(子向量的长度至少是1)

思路：用两个数，一个存全局最大值，另一个存局部最大值，当局部最大值为负时重置它。

代码

```
# -*- coding:utf-8 -*-
class Solution:
    def FindGreatestSumOfSubArray(self, array):
        # write code here
        if not array:
            return None
        max_sum = float('-inf')
        cur_sum = 0
        
        for i in range(len(array)):
            if cur_sum <= 0:
                cur_sum = array[i]
            else:
                cur_sum += array[i]
            max_sum = max_sum if max_sum > cur_sum else cur_sum
        return max_sum
```


##  剑指 45. 把数组排成最小的数 重要

题目：输入一个正整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。例如输入数组{3，32，321}，则打印出这三个数字能排成的最小数字为321323。

思路：如果一个数放到另一个数前面组合出来的数更小，那么就把它放在前面，因此可以当做排序看待，用插入排序解决。

代码

```
# -*- coding:utf-8 -*-
class Solution:
    def PrintMinNumber(self, numbers):
        # write code here
        if not numbers:
            return ""
        str_num = [str(i) for i in numbers]
        # true 表示 y 比 x 小
        check = lambda x, y: eval(x+y) > eval(y+x)
        for idx in range(1, len(numbers)):
            cur_num = str_num[idx]
            pre_idx = idx - 1
            while not check(cur_num, str_num[pre_idx]) and pre_idx >= 0:
                str_num[pre_idx+1] = str_num[pre_idx]
                pre_idx -= 1
            str_num[pre_idx+1] = cur_num
        return "".join(str_num)
```

## 剑指 49. 丑数

题目：把只包含质因子2、3和5的数称作丑数（Ugly Number）。例如6、8都是丑数，但14不是，因为它包含质因子7。 习惯上我们把1当做是第一个丑数。求按从小到大的顺序的第N个丑数。

思路：找规律题，用三个指针定位用来乘2 、 3、 5 的最小数，而后迭代。

代码

```
# -*- coding:utf-8 -*-
class Solution:
    def GetUglyNumber_Solution(self, index):
        # write code here
        if index <= 0:
            return 0
        res = [1]
        p = 1
        t2 = 0
        t3 = 0
        t5 = 0
        
        while p < index:
            min_value = min(res[t2]*2, res[t3]*3, res[t5]*5)
            res.append(min_value)
            while res[t2]*2 <= min_value:
                t2 += 1
            while res[t3]*3 <= min_value:
                t3 += 1
            while res[t5]*5 <= min_value:
                t5 += 1
            p += 1
        return res[index-1]
```


## 剑指 56. 数字在排序数组中出现的次数 重要

题目：统计一个数字在排序数组中出现的次数

思路：用二分查找

代码

```
# -*- coding:utf-8 -*-
class Solution:
    def GetNumberOfK(self, data, k):
        # write code here
        if not data:
            return 0
        self.count = 0
        self.division(data, 0, len(data)-1, k )
        return self.count
    
    def division(self, data, start, end, k):
        if end - start == 0 and data[start] != k:
            return 0
        if end - start == 1 and k not in [data[start], data[end]]:
            return 0
        mid = (start + end)/2
        if data[mid] == k:
            self.count += 1
            left = mid-1
            right = mid+1
            while left >= 0 and data[left] == k:
                self.count += 1
                left -= 1
            while right < len(data) and data[right] == k :
                self.count += 1
                right += 1
        elif data[mid] > k:
            return self.division(data, start, mid, k)
        else:
            return self.division(data, mid, end, k)
```

## 剑指 57 扩展. 和为S的连续正数序列  重要

题目：
小明很喜欢数学,有一天他在做数学作业时,要求计算出9~16的和,他马上就写出了正确答案是100。但是他并不满足于此,他在想究竟有多少种连续的正数序列的和为100(至少包括两个数)。没多久,他就得到另一组连续正数和为100的序列:18,19,20,21,22。现在把问题交给你,你能不能也很快的找出所有和为S的连续正数序列? Good Luck!

思路：双指针法，一个指向上限，一个指向下限。若当前和大则下限加一，否则上限加一。

代码

```
# -*- coding:utf-8 -*-
class Solution:
    def FindContinuousSequence(self, tsum):
        # write code here
        if tsum <= 1:
            return []
        res = []
        small = 1
        big = 2
        cur_sum = 3
        while small <= tsum/2 and big <= tsum:
            if cur_sum == tsum:
                res.append(range(small, big+1))
                big += 1
                cur_sum += big
            elif cur_sum < tsum:
                big += 1
                cur_sum += big
            else:
                cur_sum -= small
                small += 1
        return res
```

## 剑指 57. 和为S的两个数字

题目：
输入一个递增排序的数组和一个数字S，在数组中查找两个数，使得他们的和正好是S，如果有多对数字的和等于S，输出两个数的乘积最小的。

思路：思路和扩展相同

代码

```
# -*- coding:utf-8 -*-
class Solution:
    def FindNumbersWithSum(self, array, tsum):
        # write code here
        if not array or tsum <= 0:
            return []
        small = 0
        big = len(array) - 1
        res = []
        while small < big:
            if array[small] + array[big] == tsum:
                res.append((array[small], array[big]))
                big -= 1
            elif array[small] + array[big] < tsum:
                small += 1
            else:
                big -= 1
        res.sort(key=lambda x: x[0]*x[1])
        return res[0] if len(res) >= 1 else []
```

## 剑指 61. 扑克牌顺子

题目：
LL今天心情特别好,因为他去买了一副扑克牌,发现里面居然有2个大王,2个小王(一副牌原本是54张^_^)...他随机从中抽出了5张牌,想测测自己的手气,看看能不能抽到顺子,如果抽到的话,他决定去买体育彩票,嘿嘿！！“红心A,黑桃3,小王,大王,方片5”,“Oh My God!”不是顺子.....LL不高兴了,他想了想,决定大\小 王可以看成任何数字,并且A看作1,J为11,Q为12,K为13。上面的5张牌就可以变成“1,2,3,4,5”(大小王分别看作2和4),“So Lucky!”。LL决定去买体育彩票啦。 现在,要求你使用这幅牌模拟上面的过程,然后告诉我们LL的运气如何， 如果牌能组成顺子就输出true，否则就输出false。为了方便起见,你可以认为大小王是0。


思路：主要是注意输入都是什么意思, 有空缺就用 0 补，不够就 False

代码

```
# -*- coding:utf-8 -*-
class Solution:
    def IsContinuous(self, numbers):
        # write code here
        if not numbers:
            return False
        numbers.sort()
        num_wang = numbers.count(0)
        for i in range(num_wang+1, len(numbers)):
            cur_num = numbers[i]
            if cur_num - numbers[i-1] > 1:
                num_wang -= cur_num - numbers[i-1] - 1
                if num_wang < 0:
                    return False
            elif cur_num - numbers[i-1] == 0:
                return False
        return True
```


## 剑指 62. 圆圈中最后剩下的数字

题目：每年六一儿童节,牛客都会准备一些小礼物去看望孤儿院的小朋友,今年亦是如此。HF作为牛客的资深元老,自然也准备了一些小游戏。其中,有个游戏是这样的:首先,让小朋友们围成一个大圈。然后,他随机指定一个数m,让编号为0的小朋友开始报数。每次喊到m-1的那个小朋友要出列唱首歌,然后可以在礼品箱中任意的挑选礼物,并且不再回到圈中,从他的下一个小朋友开始,继续0...m-1报数....这样下去....直到剩下最后一个小朋友,可以不用表演,并且拿到牛客名贵的“名侦探柯南”典藏版(名额有限哦!!^_^)。请你试着想下,哪个小朋友会得到这份礼品呢？(注：小朋友的编号是从0到n-1)

思路：用列表模拟一下。

代码

```
# -*- coding:utf-8 -*-
class Solution:
    def LastRemaining_Solution(self, n, m):
        # write code here
        if n <= 0 or m <=0 :
            return -1
        child = range(n)
        
        start = 0
        
        while len(child) > 1:
            end = (start + m - 1) % (len(child))
            child.pop(end)
            start  = end
        return child[0]
```

## 剑指 66. 构建乘积数组

题目：
给定一个数组A[0,1,...,n-1],请构建一个数组B[0,1,...,n-1],其中B中的元素B[i]=A[0]*A[1]*...*A[i-1]*A[i+1]*...*A[n-1]。不能使用除法。


思路：用两个辅助数组保存前一步结果.

代码

```
# -*- coding:utf-8 -*-
class Solution:
    def multiply(self, A):
        # write code here
        if not A:
            return None
        n = len(A)
        C = [1]*len(A)
        D = [1]*len(A)
        for i in range(1, len(C)):
            C[i] = C[i-1]*A[i-1]
        for i in range(len(D)-2, -1, -1):
            D[i] = D[i+1]*A[i+1]
        B = [C[i]*D[i] for i in range(len(A))]
        return B
```

# 字符串


## 剑指 5. 替换空格

题目：请实现一个函数，将一个字符串中的每个空格替换成“%20”。例如，当字符串为We Are Happy.则经过替换之后的字符串为We%20Are%20Happy

思路：避免通过+号反复开辟内存，其他语言可以预先分配内存，python不知道怎么办，我用 join 操作

代码

```
# -*- coding:utf-8 -*-
class Solution:
    # s 源字符串
    def replaceSpace(self, s):
        # write code here
        s = list(s)
        count=len(s)
        for i in range(0,count):
            if s[i]==' ':
                s[i]='%20'
        return ''.join(s)
```

## 剑指19. 正则表达式匹配  重要 

题目：请实现一个函数用来匹配包括'.'和'*'的正则表达式。模式中的字符'.'表示任意一个字符，而'*'表示它前面的字符可以出现任意次（包含0次）。 在本题中，匹配是指字符串的所有字符匹配整个模式。例如，字符串"aaa"与模式"a.a"和"ab*ac*a"匹配，但是与"aa.a"和"ab*a"均不匹配

思路：分后面有×和无星号两种，如果有星号又分为字符动，模式后移2位(未匹配)、模式后移两位，字符串后移1位(匹配一个)、模式不动，字符串后移1位(匹配多个)三种情况。无星号就直接看相等还是有点。

代码

```
# -*- coding:utf-8 -*-
class Solution:
    # s, pattern都是字符串
    def match(self, s, pattern):
        # write code here
        if len(s) == 0 and len(pattern) == 0:
            return True
        if len(s) > 0 and len(pattern) == 0:
            return False
        
        if len(pattern) > 1 and pattern[1] == "*":
            if len(s) > 0 and (s[0] == pattern[0] or pattern[0] == "."):
                return self.match(s, pattern[2:]) or \
                        self.match(s[1:], pattern[2:]) or \
                        self.match(s[1:], pattern)
            else:
                return self.match(s, pattern[2:])
        if len(s) > 0 and (s[0] == pattern[0] or pattern[0] == "."):
            return self.match(s[1:], pattern[1:])
        return False
```


## 剑指 20. 表示数值的字符串

题目：请实现一个函数用来判断字符串是否表示数值（包括整数和小数）。例如，字符串"+100","5e2","-123","3.1416"和"-1E-16"都表示数值。 但是"12e","1a3.14","1.2.3","+-5"和"12e+4.3"都不是。

思路：虑各种可能的数字情况，并进行对应处理

代码

```
# -*- coding:utf-8 -*-
class Solution:
    # s字符串
    def isNumeric(self, s):
        # write code here
        if not s and len(s) < 1:
            return False
        s = s.lower()
        if s.find("e") != -1:
            nums = s.strip().split("e")
        else:
            nums = s.strip().split("E")
        if len(nums) > 2:
            return False
        isnum = True
        i = 0
        for n in nums:
            if  i == 1 and n.count(".") > 0:
                isnum = False
            if n.count(".") > 1:
                isnum = False
            n = n.replace(".", "")
            isnum &= self.check_num(n)
            
            i += 1
        return isnum
    
    def check_num(self, num):
        if len(num) == 0:
            return False
        if len(num) == 1 and num[0] in ["-", "+"]:
            return False
        for i in range(1, len(num)):
            if num[i].isalpha():
                return False
            if not num[i].isalnum():
                return False
        return True
```

## 剑指 38. 字符串的排列  重要

题目：输入一个字符串,按字典序打印出该字符串中字符的所有排列。例如输入字符串abc,则打印出由字符a,b,c所能排列出来的所有字符串abc,acb,bac,bca,cab和cba。

思路：用递归，把字符串看为两部分，即第一个字符和后面的字符

代码

```
# -*- coding:utf-8 -*-
class Solution:
    def Permutation(self, ss):
        # write code here
        if len(ss) <= 0:
            return []
        if len(ss) == 1:
            return list(ss)
        lss = list(ss)
        lss.sort()
        res = []
        for i, s in enumerate(lss):
            if i > 0 and lss[i] == lss[i-1]:
                continue
            tmp = self.Permutation("".join(lss[:i]) + "".join(lss[i+1:]))
            for j in tmp:
                res.append(lss[i] + j)
        return res
```

##  剑指 43. 1~n整数中1出现的次数

题目：求出1~13的整数中1出现的次数,并算出100~1300的整数中1出现的次数？为此他特别数了一下1~13中包含1的数字有1、10、11、12、13因此共出现6次,但是对于后面问题他就没辙了。ACMer希望你们帮帮他,并把问题更加普遍化,可以很快的求出任意非负整数区间中1出现的次数（从1 到 n 中1出现的次数）。

思路：直接用归纳法的思想总结出公式。

代码

```
def countDigitOne(self, n):
        """
        :type n: int
        :rtype: int
		 例：对于824883294，先求0－800000000之间（不包括800000000）的，再求0－24883294之间的。
		 如果等于1，如1244444，先求0－1000000之间，再求1000000－1244444，那么只需要加上244444＋1，再求0－244444之间的1
        如果大于1，例：0－800000000之间1的个数为8个100000000的1的个数加上100000000，因为从1000000000－200000000共有1000000000个数且最高位都为1。
        对于最后一位数，如果大于1，直接加上1即可。
        """
        result = 0
        if n < 0:
            return 0
        length = len(str(n))
        listN = list(str(n))
        for i, v in enumerate(listN):
            a = length - i - 1  # a为10的幂
            if i==length-1 and int(v)>=1:
                result+=1
                break
 
            if int(v) > 1:
                result += int(10 ** a * a / 10) * int(v) + 10**a
            if int(v) == 1:
                result += (int(10 ** a * a / 10) + int("".join(listN[i+1:])) + 1)
        return result
```

## 剑指 50. 第一个只出现一次的字符  重要

题目：在一个字符串(0<=字符串长度<=10000，全部由字母组成)中找到第一个只出现一次的字符,并返回它的位置, 如果没有则返回 -1（需要区分大小写）.

思路：用哈希表做。

代码

```
# -*- coding:utf-8 -*-
class Solution:
    def FirstNotRepeatingChar(self, s):
        # write code here
        if len(s) <= 0:
            return -1
        from collections import OrderedDict
        hash_char = OrderedDict()
        for char in s:
            if char in hash_char:
                hash_char[char] += 1
            else:
                hash_char[char] = 1
        for i in hash_char:
            if hash_char[i] == 1:
                return s.index(i)
        return -1
```

## 剑指 58. 左旋转字符串  重要 

题目：汇编语言中有一种移位指令叫做循环左移（ROL），现在有个简单的任务，就是用字符串模拟这个指令的运算结果。对于一个给定的字符序列S，请你把其循环左移K位后的序列输出。例如，字符序列S=”abcXYZdef”,要求输出循环左移3位后的结果，即“XYZdefabc”。是不是很简单？OK，搞定它！

思路：两次翻转法

代码

```
# -*- coding:utf-8 -*-
class Solution:
    def LeftRotateString(self, s, n):
        # write code here
        if not s or len(s) <= 0:
            return ""
        if len(s) <= n:
            return s
        return self.reverse(self.reverse(s[:n])+self.reverse(s[n:]))
    
    def reverse(self, s):
        if not isinstance(s, list):
            s = list(s)
        start = 0
        last = len(s) - 1
        while start < last:
            s[start], s[last] = s[last], s[start]
            start += 1
            last -= 1
        return "".join(s)
```

## 剑指 67. 把字符串转换成整数  重要

题目：将一个字符串转换成一个整数(实现Integer.valueOf(string)的功能，但是string不符合数字要求时返回0)，要求不能使用字符串转换整数的库函数。 数值为0或者字符串不是一个合法的数值则返回0。

思路：考虑边界情况，只有正负号、空字符串、数据上下溢出、有无正负号、错误字符等。

代码

```
# -*- coding:utf-8 -*-
class Solution:
    def StrToInt(self, s):
        # write code here
        if len(s) == 0:
            return 0
        
        nums = s.strip().split("e")
        if len(nums) > 2:
            return False
        isnum = True
        for n in nums:
            isnum &= self.check_num(n)
        if isnum:
            return self.trans(s)
        else:
            return 0
    
    def trans(self, s):
        
        numbers = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        nums = s.strip().split("e")
        res = 0
        base = 0
        for num in nums:
            tmp_res = 0
            sign = 1
            if s[0] == "-":
                sign = -1
            for i in num:
                if i in ["+", "-"]:
                    continue
                tmp_res = tmp_res*10 + numbers.index(i)
            if base == 0:
                base = tmp_res*sign
                res = base
            else:
                res = pow(base, tmp_res*sign)
        return res
    def check_num(self, num):
        sign = False
        if len(num) == 1 and num[0] in ["-", "+"]:
            return False
        for i in range(1, len(num)):
            if num[i].isalpha():
                return False
            if not num[i].isalnum():
                return False
        return True
```

## 剑指 50 扩展。 字符流中第一个不重复的字符 

题目：请实现一个函数用来找出字符流中第一个只出现一次的字符。例如，当从字符流中只读出前两个字符"go"时，第一个只出现一次的字符是"g"。当从该字符流中读出前六个字符“google"时，第一个只出现一次的字符是"l"。

思路：哈希

代码

```
# -*- coding:utf-8 -*-
class Solution:
    # 返回对应char
    def __init__(self):
        self.string = ""
    def FirstAppearingOnce(self):
        # write code here
        from collections import OrderedDict
        o = OrderedDict()
        for s in self.string:
            if s in o:
                o[s] += 1
            else:
                o[s] = 1
        for k, v in o.items():
            if v == 1:
                return k
        return "#"
        
    def Insert(self, char):
        # write code here
        self.string += char
```

# 链表

## 剑指 6. 从尾到头打印链表

题目：
输入一个链表，按链表值从尾到头的顺序返回一个ArrayList。

思路：简单遍历，最终翻转结果列表

代码

```
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    # 返回从尾部到头部的列表值序列，例如[1,2,3]
    def printListFromTailToHead(self, listNode):
        # write code here
        if not listNode:
            return []
        res = []
        while listNode:
            res.append(listNode.val)
            listNode = listNode.next
        return res[::-1]
```

##  剑指 18 扩展. 删除链表中重复的结 重要点

题目：在一个排序的链表中，存在重复的结点，请删除该链表中重复的结点，重复的结点不保留，返回链表头指针。 例如，链表1->2->3->3->4->4->5 处理后为 1->2->5

思路：遇见重复的就将指针指向重复数之后的数字

代码

```
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def deleteDuplication(self, pHead):
        # write code here
        if pHead is None or pHead.next is None:
            return pHead
        res = ListNode("x")
        res.next = pHead
        pre = res
        cur = pHead
        while cur:
            if cur.next and cur.next.val == cur.val:
                while cur.next and cur.next.val == cur.val:
                    cur = cur.next
                cur = cur.next
                pre.next = cur
            else:
                pre.next = cur
                pre = cur
                cur = cur.next
        return res.next
```

## 剑指 22. 链表中倒数第k个节点  重要

题目：输入一个链表，输出该链表中倒数第k个结点。

思路：用两个指针，一个先走K-1步，另一个再开始走，当前面的额到达末尾时，后面那个在k位置。

代码

```
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def FindKthToTail(self, head, k):
        # write code here
        if not head or k < 1:
            return
        pHead = head
        pBehead = head
        for i in range(k-1):
            if pHead.next:
                pHead = pHead.next
            else:
                return
        while pHead.next:
            pHead = pHead.next
            pBehead = pBehead.next
        return pBehead
```

## 剑指 23. 链表中环的入口节点  重要

题目：
给一个链表，若其中包含环，请找出该链表的环的入口结点，否则，输出null。


思路：两个指针，一个一下走两步，一个走一步，找到在环中的相遇点，而后快指针从头开始走，慢指针从相遇点开始，都走一步，最终再入口处相遇。

代码

```
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def EntryNodeOfLoop(self, pHead):
        # write code here
        fast_p = pHead
        slow_p = pHead
        if pHead is None or pHead.next is None:
            return None
        while fast_p and fast_p.next:
            fast_p = fast_p.next.next
            slow_p = slow_p.next
            if slow_p == fast_p:
                break
        if fast_p is None or fast_p.next is None:
            return None
        fast_p = pHead
        while fast_p != slow_p:
            fast_p = fast_p.next
            slow_p = slow_p.next
        return fast_p
```

## 剑指 24. 反转链表  重要

题目：输入一个链表，反转链表后，输出新链表的表头。

思路：用三个临时指针应对指针断裂的问题。

代码

```
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    # 返回ListNode
    def ReverseList(self, pHead):
        # write code here
        if not pHead or not pHead.next:
            return pHead
        pcur = pHead
        ppre = None
        res = None
        while pcur:
            pnext = pcur.next
            if not pnext:
                res = pcur
            pcur.next = ppre
            ppre = pcur
            pcur = pnext
        return res
```


## 剑指 25. 合并两个排序的链表  重要

题目：输入两个单调递增的链表，输出两个链表合成后的链表，当然我们需要合成后的链表满足单调不减规则。

思路：可以用递归和非递归处理，用改变指针指向的方法，避免创建过多额外空间。

代码

```
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    # 返回合并后列表
    def Merge(self, pHead1, pHead2):
        # write code here
        if not pHead1:
            return pHead2
        if not pHead2:
            return pHead1
        cur = ListNode(float("inf"))
        start = cur
        while pHead1 and pHead2:
            if pHead1.val <= pHead2.val:
                cur.next = pHead1
                cur = cur.next
                pHead1 = pHead1.next
            else:
                cur.next = pHead2
                cur = cur.next
                pHead2 = pHead2.next
        if not pHead1:
            cur.next = pHead2
        elif not pHead2:
            cur.next = pHead1
        return start.next
```

## 剑指 35. 复杂链表的复制  重要

题目：输入一个复杂链表（每个节点中有节点值，以及两个指针，一个指向下一个节点，另一个特殊指针指向任意一个节点），返回结果为复制后复杂链表的head。（注意，输出结果中请不要返回参数中的节点引用，否则判题程序会直接返回空）

思路：先在原链路中复制一份，而后遍历建立random连接，最后拆分。

代码

```
# -*- coding:utf-8 -*-
# class RandomListNode:
#     def __init__(self, x):
#         self.label = x
#         self.next = None
#         self.random = None
class Solution:
    # 返回 RandomListNode
    def Clone(self, pHead):
        # write code here
        if pHead is None:
            return pHead
        nodeList = []
        nodeLabel = []
        nodeRandom = []
        tmp = pHead
        while tmp:
            nodeList.append(tmp)
            nodeLabel.append(tmp.label)
            nodeRandom.append(tmp.random)
            tmp = tmp.next
        randomToNode = map(lambda c: nodeList.index(c) if c in nodeList else -1, nodeRandom)
        nodes = map(lambda c: RandomListNode(c), nodeLabel)
        
        newHead = RandomListNode('x')
        tmp = newHead
        for i, j in enumerate(nodes):
            if randomToNode[i] != -1:
                nodes[i].random = nodes[randomToNode[i]]
        for i, j in enumerate(nodes):
            tmp.next = nodes[i]
            tmp = tmp.next
        return newHead.next
```


## 剑指 52. 两个链表的第一个公共节点  重要

题目：
输入两个链表，找出它们的第一个公共结点。


思路：统计两个链表的长度，而后令长的先走，最后一起走。

代码

```
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def FindFirstCommonNode(self, pHead1, pHead2):
        # write code here
        if not pHead1 or not pHead2:
            return None
        len_node1 = 0
        p = pHead1
        while p:
            len_node1 += 1
            p = p.next
        len_node2 = 0
        p = pHead2
        while p:
            len_node2 += 1
            p = p.next
        diff = abs(len_node1 - len_node2)
        if len_node1 < len_node2:
            pHead1, pHead2 = pHead2, pHead1
        start = pHead1
        for _ in range(diff):
            start = start.next
        while start and pHead2:
            if start == pHead2:
                return start
            start = start.next
            pHead2 = pHead2.next
        return None
```

# 栈和队列

## 剑指 9. 用两个栈实现队列

题目：
用两个栈来实现一个队列，完成队列的Push和Pop操作。 队列中的元素为int类型。


思路：用两个栈，一个负责插入，一个负责弹出

代码

```
# -*- coding:utf-8 -*-
class Solution:
    def __init__(self):
        self.stackA = []
        self.stackB = []
        
    def push(self, node):
        # write code here
        self.stackA.append(node)
        
    def pop(self):
        # return xx
        if self.stackB:
            return self.stackB.pop()
        elif not self.stackA:
            return None
        elif self.stackB == []:
            while self.stackA:
                    self.stackB.append(self.stackA.pop())
            return self.stackB.pop()
```


## 剑指 30. 包含min函数的栈  重要

题目：定义栈的数据结构，请在该类型中实现一个能够得到栈中所含最小元素的min函数（时间复杂度应为O（1））。

思路：用两个栈，一个用来当做栈，另一个存储最小值

代码

```
# -*- coding:utf-8 -*-
class Solution:
    def __init__(self):
        self.stack = []
        self.min_stack = []
    def push(self, node):
        # write code here
        self.stack.append(node)
        if self.min_stack == [] or node < self.min():
            self.min_stack.append(node)
        else:
            self.min_stack.append(self.min())
    def pop(self):
        # write code here
        self.stack.pop()
        self.min_stack.pop()
    def top(self):
        # write code here
        return self.stack[-1]
    def min(self):
        # write code here
        return self.min_stack[-1]
```


## 剑指 31. 栈的压入、弹出元素

题目：输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否可能为该栈的弹出顺序。假设压入栈的所有数字均不相等。例如序列1,2,3,4,5是某栈的压入顺序，序列4,5,3,2,1是该压栈序列对应的一个弹出序列，但4,3,5,1,2就不可能是该压栈序列的弹出序列。（注意：这两个序列的长度是相等的）

思路：模拟栈的压入和弹出操作，最终栈内有剩余则False

代码

```
# -*- coding:utf-8 -*-
class Solution:
    def IsPopOrder(self, pushV, popV):
        # write code here
        if len(pushV) != len(popV) or not pushV or not popV:
            return False
        stack = []
        idx_pop = 0
        for i in pushV:
            stack.append(i)
            while stack[-1] == popV[idx_pop]:
                stack.pop()
                if idx_pop < len(popV) - 1:
                    idx_pop += 1
                else:
                    break
        if len(stack) > 0:
            return False
        else:
            return True
```

# 树

剑指的树这很全，建议都好好做做,默认全部重点

## 剑指 7. 重建二叉树

题目：输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。例如输入前序遍历序列{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}，则重建二叉树并返回。

思路：前序的首位是根节点，依次去中序结果中获取左右子树，递归重建

代码

```
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回构造的TreeNode根节点
    def reConstructBinaryTree(self, pre, tin):
        # write code here
        if len(pre) != len(tin) or not pre or not tin:
            return None
        root = TreeNode(pre[0])
        idx = tin.index(pre[0])
        root.left = self.reConstructBinaryTree(pre[1:idx+1], tin[:idx])
        root.right = self.reConstructBinaryTree(pre[idx+1:], tin[idx+1:])
        return root
```


## 剑指 26：树的子结构;

题目：
输入两棵二叉树A，B，判断B是不是A的子结构。（ps：我们约定空树不是任意一个树的子结构）


思路：用递归的方式依次检查左右子树

代码

```
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def HasSubtree(self, pRoot1, pRoot2):
        # write code here
        if not pRoot1 or not pRoot2:
            return False
        res = False
        if pRoot1.val == pRoot2.val:
            res = self.checkSub(pRoot1, pRoot2)
        if not res:
            res = self.HasSubtree(pRoot1.left, pRoot2)
        if not res:
            res = self.HasSubtree(pRoot1.right, pRoot2)
        return res
    
    def checkSub(self, pRoot1, pRoot2):
        if pRoot2 == None:
            return True
        if pRoot1 == None:
            return False
        if pRoot1.val != pRoot2.val:
            return False
        return self.checkSub(pRoot1.left, pRoot2.left) and self.checkSub(pRoot1.right, pRoot2.right)
```


## 剑指 27：二叉树的镜像

题目：操作给定的二叉树，将其变换为源二叉树的镜像。

思路：用递归左右子树互换

代码

```
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回镜像树的根节点
    def Mirror(self, root):
        # write code here
        if not root:
            return
        if root.left or root.right:
            root.left, root.right = self.Mirror(root.right), self.Mirror(root.left)
        return root
```


## 剑指 28. 对称的二叉树题

题目：请实现一个函数，用来判断一颗二叉树是不是对称的。注意，如果一个二叉树同此二叉树的镜像是同样的，定义其为对称的。

思路：用递归分别比较左右子树

代码

```
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def isSymmetrical(self, pRoot):
        # write code here
        if not pRoot:
            return True
        return self.check(pRoot, pRoot)
        
    def check(self, root1, root2):
        if not root1 and not root2:
            return True
        if not root1 or not root2:
            return False
        if root1.val != root2.val:
            return False
        return self.check(root1.left, root2.right) and self.check(root1.right, root2.left)
```


## 剑指 32. 从上到下打印二叉树

题目：
从上往下打印出二叉树的每个节点，同层节点从左至右打印。


思路：用一个辅助栈，层次遍历

代码

```
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回从上到下每个节点值列表，例：[1,2,3]
    def PrintFromTopToBottom(self, root):
        # write code here
        if not root:
            return []
        stack = [root]
        res = []
        while stack:
            cur_stack = []
            for s in stack:
                res.append(s.val)
                if s.left:
                    cur_stack.append(s.left)
                if s.right:
                    cur_stack.append(s.right)
            stack = cur_stack
        return res
```


##  剑指 33. 二叉搜索树的后序遍历序列

题目：输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历的结果。如果是则输出Yes,否则输出No。假设输入的数组的任意两个数字都互不相同。

思路：根据后序遍历特点，先确定根，而后确定左右子树，递归求解。边界情况比较麻烦

代码

```
# -*- coding:utf-8 -*-
class Solution:
    def VerifySquenceOfBST(self, sequence):
        # write code here
        if not sequence  or len(sequence)<=0:
            return False
        idx = 0
        for i in range(len(sequence)):
            idx = i
            if sequence[i] > sequence[-1]:
                break
        for i in sequence[idx:-1]:
            if i < sequence[-1]:
                return False
        left = True
        if idx > 0:
            left = self.VerifySquenceOfBST(sequence[:idx])
        right = True
        if idx < len(sequence) - 1:
            right = self.VerifySquenceOfBST(sequence[idx:-1])
        return left and right
        
```


## 剑指 34. 二叉树中和为某一值的路径

题目：输入一颗二叉树的跟节点和一个整数，打印出二叉树中结点值的和为输入整数的所有路径。路径定义为从树的根结点开始往下一直到叶结点所经过的结点形成一条路径。(注意: 在返回值的list中，数组长度大的数组靠前)

思路：深度优先遍历，当到某一节点处合大于期望数时，从路径中删除节点。

代码

```
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回二维列表，内部每个列表表示找到的路径
    def FindPath(self, root, expectNumber):
        # write code here
        if not root or expectNumber < root.val:
            return []
        res = []
        
        def dfs(root, path, curNum):
            path.append(root)
            curNum += root.val
            leaf = (not root.left and not root.right)
            
            if leaf and curNum == expectNumber:
                res.append([p.val for p in path])
            if curNum < expectNumber:
                if root.left:
                    dfs(root.left, path, curNum)
                if root.right:
                    dfs(root.right, path, curNum)
            path.pop()
        dfs(root, [], 0)
        return res
```


## 剑指 36. 二叉搜索树与双向链表

题目：输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的双向链表。要求不能创建任何新的结点，只能调整树中结点指针的指向。

思路：中序遍历二叉树可以得到有序排列，而后根据left 和 right 作为前后指针建立双向链表

代码

```
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def Convert(self, pRootOfTree):
        # write code here
        if not pRootOfTree:
            return 
        self.res = []
        def inOrder(pRootOfTree):
            if not pRootOfTree:
                return None
            inOrder(pRootOfTree.left)
            self.res.append(pRootOfTree)
            inOrder(pRootOfTree.right)
        inOrder(pRootOfTree)
        
        for i, node in enumerate(self.res[:-1]):
            node.right = self.res[i+1]
            self.res[i+1].left = node
        
        return self.res[0]
```


## 剑指 55. 二叉树的深度

题目：输入一棵二叉树，求该树的深度。从根结点到叶结点依次经过的结点（含根、叶结点）形成树的一条路径，最长路径的长度为树的深度。

思路：用递归，取左右子树最大的深度加上1

代码

```
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def TreeDepth(self, pRoot):
        # write code here
        if not pRoot:
            return 0
        left = self.TreeDepth(pRoot.left)
        right = self.TreeDepth(pRoot.right)
        return max(left, right) + 1
```


## 剑指 55 扩展 检验平衡二叉树；

题目：
输入一棵二叉树，判断该二叉树是否是平衡二叉树。

思路：计算左右子树的深度，若深度差大于1则返回 False

代码

```
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def IsBalanced_Solution(self, pRoot):
        # write code here
        if not pRoot:
            return True
        left = self.depth(pRoot.left)
        right = self.depth(pRoot.right)
        res = True if abs(left - right) <= 1 else False
        return res
        
    def depth(self, root):
        if not root:
            return 0
        left = self.depth(root.left)
        right = self.depth(root.right)
        return max(left, right) + 1
```


## 剑指 8. 二叉树的下一个节点

题目：给定一个二叉树和其中的一个结点，请找出中序遍历顺序的下一个结点并且返回。注意，树中的结点不仅包含左右子结点，同时包含指向父结点的指针。

思路：据遍历规则，该节点的可能情况为：有右叶子节点则位右子节点的最左叶子，无右子节点但是父节点的右子节点，则找他的父节点为左子节点时的父节点，否则直接返回父节点。

代码

```
# -*- coding:utf-8 -*-
# class TreeLinkNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
#         self.next = None
class Solution:
    def GetNext(self, pNode):
        # write code here
        if not pNode:
            return None
        elif pNode.right != None:
            pNode = pNode.right
            while pNode.left != None:
                pNode = pNode.left
            return pNode
        elif pNode.next!=None and pNode.next.right==pNode:
            while pNode.next!=None and pNode.next.left!=pNode:
                pNode=pNode.next
            return pNode.next
        else:
            return pNode.next
```


## 剑指 32 扩展2. 之字形打印二叉树

题目：请实现一个函数按照之字形打印二叉树，即第一行按照从左到右的顺序打印，第二层按照从右至左的顺序打印，第三行按照从左到右的顺序打印，其他行以此类推。

思路：

代码

```
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def Print(self, pRoot):
        # write code here
        if not pRoot:
            return []
        res = [[pRoot.val]]
        _stack = [pRoot]
        k = 1
        while _stack:
            k += 1
            tmp_stack = []
            for r in _stack:
                if r.left:
                    tmp_stack.append(r.left)
                if r.right:
                    tmp_stack.append(r.right)
            _stack = tmp_stack
            tmp_val = [i.val for i in _stack]
            if k%2 == 0:
                tmp_val.reverse()
            if tmp_val:
                res.append(tmp_val)
        return res
```


## 剑指32. 把二叉树打印成多行

题目：
从上到下按层打印二叉树，同一层结点从左至右输出。每一层输出一行。


思路：

代码

```
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回二维列表[[1,2],[4,5]]
    def Print(self, pRoot):
        # write code here
        if not pRoot:
            return []
        nodes = [pRoot]
        res = []
        while nodes:
            cur_stack = []
            next_stack = []
            for n in nodes:
                cur_stack.append(n.val)
                if n.left:
                    next_stack.append(n.left)
                if n.right:
                    next_stack.append(n.right)
            nodes = next_stack
            res.append(cur_stack)
        return res
```


## 剑指 37. 序列化二叉树

题目：
请实现两个函数，分别用来序列化和反序列化二叉树


思路：序列化表示用前序遍历的结果存下来，反序列化就是把前序遍历结果重建二叉树。

代码

```
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    flag=-1
    def Serialize(self, root):
        # write code here
        if not root:
            return '#'
        return str(root.val)+','+self.Serialize(root.left)+','+self.Serialize(root.right)
        
    def Deserialize(self, s):
        # write code here
        self.flag+=1
        lis = s.split(",")
        
        if self.flag > len(lis):
            return
        root = None
        if lis[self.flag] != "#":
            root = TreeNode(int(lis[self.flag]))
            root.left = self.Deserialize(s)
            root.right = self.Deserialize(s)
        return root
```


## 剑指 54. 二叉搜索树的第k个结点 

题目：
给定一棵二叉搜索树，请找出其中的第k小的结点。例如， （5，3，7，2，4，6，8）    中，按结点数值大小顺序第三小结点的值为4。


思路：中序遍历得到第K大的

代码

```
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回对应节点TreeNode
    def KthNode(self, pRoot, k):
        # write code here
        if not pRoot or k <= 0:
            return
        
        self.res = []
        self.mid(pRoot)
        if k > len(self.res):
            return
        return self.res[k-1]
        
    def mid(self, root):
        if not root:
            return None
        self.mid(root.left)
        self.res.append(root)
        self.mid(root.right)
        
```

# 位运算

## 剑指 15. 二进制中1的个数 

题目：
输入一个整数，输出该数二进制表示中1的个数。其中负数用补码表示。


思路：思路：把一个整数减去1，再和原整数做“与运算”，会把该整数最右边的1变成0。

代码

```
# -*- coding:utf-8 -*-
class Solution:
    def NumberOf1(self, n):
        # write code here
        count = 0
        if n < 0:
            n = n & 0xffffffff
        while n:
            n = (n-1) & n
            count += 1
        return count
```


## 剑指 16. 数值的整数次方

题目：
给定一个double类型的浮点数base和int类型的整数exponent。求base的exponent次方。


思路：把指数运算转化为一半乘以另一半，这样算法复杂度就变成 O(logN)

代码

```
# -*- coding:utf-8 -*-
class Solution:
    def Power(self, base, exponent):
        # write code here
        if base == 0 and exponent < 0:
            return 0
            raise ValueError("Wrong Input")
        if exponent >= 0:
            return self.compute_power(base, exponent)
        return 1.0/self.compute_power(base, -exponent)
    
    def compute_power(self, base, exponent):
        if exponent == 0:
            return 1
        if exponent == 1:
            return base
        res = self.compute_power(base, exponent>>1)
        res *= res
        if exponent & 0x1 == 1:
            res *= base
        return res
```


## 剑指 56 数组中只出现一次的数字

题目：一个整型数组里除了两个数字之外，其他的数字都出现了两次。请写程序找出这两个只出现一次的数字。

思路：用异或,a xor b xor a = b  次优解用哈希

代码

```
# -*- coding:utf-8 -*-
class Solution:
    # 返回[a,b] 其中ab是出现一次的两个数字
    def FindNumsAppearOnce(self, array):
        # write code here
        if not array:
            return []
        _hash = {}
        for i in range(len(array)):
            if array[i] not in _hash:
                _hash[array[i]] = 1
            else:
                _hash[array[i]] += 1
        res = []
        for k in _hash:
            if _hash[k] == 1:
                res.append(k)
        return res
```


## 剑指 64. 求1+2+3+...+n 

题目：求1+2+3+...+n，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）。

思路：

代码

```
# -*- coding:utf-8 -*-
class Solution:
    def Sum_Solution(self, n):
        # write code here
        if n < 1:
            return 0
        return self.Sum_Solution(n-1) + n
```

# 动态规划

动态规划一般可分为线性动规，区域动规，树形动规，背包动规四类。一般按照以下步骤：

* 划分阶段：按照问题的时间或空间特征，把问题分为若干个阶段。在划分阶段时，注意划分后的阶段一定要是有序的或者是可排序的，否则问题就无法求解。

* 确定状态和状态变量：将问题发展到各个阶段时所处于的各种客观情况用不同的状态表示出来。当然，状态的选择要满足无后效性。

* 确定决策并写出状态转移方程：因为决策和状态转移有着天然的联系，状态转移就是根据上一阶段的状态和决策来导出本阶段的状态。所以如果确定了决策，状态转移方程也就可写出。但事实上常常是反过来做，根据相邻两个阶段的状态之间的关系来确定决策方法和状态转移方程。

* 寻找边界条件：给出的状态转移方程是一个递推式，需要一个递推的终止条件或边界条件。

一般，只要解决问题的阶段、状态和状态转移决策确定了，就可以写出状态转移方程（包括边界条件）。最优子结构 +边界条件　＋状态转移方程。但难就难在这个转移方程一下看不出来。


## 剑指 10： 斐波那切数列

题目：
大家都知道斐波那契数列，现在要求输入一个整数n，请你输出斐波那契数列的第n项（从0开始，第0项为0）。n<=39


思路：避免用递归，只用前两个数计算节省空间。

代码

```
# -*- coding:utf-8 -*-
class Solution:
    def Fibonacci(self, n):
        # write code here
        if n in [0 or 1]:
            return n
        pre_1 = 1
        pre_2 = 0
        count = 2
        res = 0
        while count <= n:
            count += 1
            res = pre_1 + pre_2
            pre_2 = pre_1
            pre_1 = res
        return res
```


## 剑指 10 扩展 1：跳台阶

题目：一只青蛙一次可以跳上1级台阶，也可以跳上2级。求该青蛙跳上一个n级的台阶总共有多少种跳法（先后次序不同算不同的结果）。

思路：和斐波那契数列一样的方法

代码

```
# -*- coding:utf-8 -*-
class Solution:
    def jumpFloor(self, number):
        # write code here
        if number in [0, 1, 2]:
            return number
        pre_1 = 2
        pre_2 = 1
        res = 0
        for i in range(3, number+1):
            res = pre_1 + pre_2
            pre_2 = pre_1
            pre_1 = res
        return res
```


## 剑指 10 扩展2： 跳台阶变态版

题目：一只青蛙一次可以跳上1级台阶，也可以跳上2级……它也可以跳上n级。求该青蛙跳上一个n级的台阶总共有多少种跳法。

思路：f(n) = f(n-1) + f(n-2) + *** f(1) + 1, 而 f(n-1) = f(n-2) + **  + f(1) + 1，因此 f(n) – f(n-1) = f(n-1)，即 f(n) = 2*f(n-1)  = 2**(n-1)

代码

```
# -*- coding:utf-8 -*-
class Solution:
    def jumpFloorII(self, number):
        # write code here
        return 1 << (number-1)
```


## 剑指 10 扩展3：矩形覆盖问题

题目：我们可以用2*1的小矩形横着或者竖着去覆盖更大的矩形。请问用n个2*1的小矩形无重叠地覆盖一个2*n的大矩形，总共有多少种方法？

思路：本质上仍然是斐波那契数列。

代码

```
# -*- coding:utf-8 -*-
class Solution:
    def rectCover(self, number):
        # write code here
        if number <= 0:
            return 0
        if number == 1:
            return 1
        if number == 2:
            return 2
        front = 1
        res = 2
        for i in range(3, number+1):
            tmp = res
            res += front
            front = tmp
        return res
```


## 剑指 10 扩展 4：零钱兑换

题目：现有硬币种类 1, 3, 4，求给定面额后有多少种找零方式)

思路：斐波那契数列问题

代码

```
略
```

# 回溯法


## 剑指 12 . 矩阵中的路径

题目：请设计一个函数，用来判断在一个矩阵中是否存在一条包含某字符串所有字符的路径。路径可以从矩阵中的任意一个格子开始，每一步可以在矩阵中向左，向右，向上，向下移动一个格子。如果一条路径经过了矩阵中的某一个格子，则之后不能再次进入这个格子。 例如 a b c e s f c s a d e e 这样的3 X 4 矩阵中包含一条字符串"bcced"的路径，但是矩阵中不包含"abcb"路径，因为字符串的第一个字符b占据了矩阵中的第一行第二个格子之后，路径不能再次进入该格子。

思路：用递归

代码

```
# -*- coding:utf-8 -*-
class Solution:
    def hasPath(self, matrix, rows, cols, path):
        # write code here
        if rows < 0 or cols < 0 or path == None or not matrix:
            return False
        path_index = 0
        bool_matrix = [0]*(cols*rows)
        
        for col in range(cols):
            for row in range(rows):
                if self.juge_path(matrix, rows, cols, row, col, path, path_index, bool_matrix):
                    return True
        return False
    
    def juge_path(self, matrix, rows, cols, row, col, path, path_index, bool_matrix):
        if path_index == len(path):
            return True
        has_path = 0
        
        if row >= 0 and row < rows and col >= 0 and col < cols and matrix[row*cols+col] == path[path_index] and not bool_matrix[row*cols+col]:
            path_index += 1
            bool_matrix[row*cols+col] = True
            has_path = self.juge_path(matrix, rows, cols, row+1, col, path, path_index, bool_matrix) or \
                        self.juge_path(matrix, rows, cols, row-1, col, path, path_index, bool_matrix) or \
                        self.juge_path(matrix, rows, cols, row, col+1, path, path_index, bool_matrix) or \
                        self.juge_path(matrix, rows, cols, row, col-1, path, path_index, bool_matrix)
            if not has_path:
                path_index -= 1
                bool_matrix[row*cols+col] = False
        return has_path
```


## 剑指 13. 机器人的运动范围

题目：地上有一个m行和n列的方格。一个机器人从坐标0,0的格子开始移动，每一次只能向左，右，上，下四个方向移动一格，但是不能进入行坐标和列坐标的数位之和大于k的格子。 例如，当k为18时，机器人能够进入方格（35,37），因为3+5+3+7 = 18。但是，它不能进入方格（35,38），因为3+5+3+8 = 19。请问该机器人能够达到多少个格子？

思路：简单递归，加一个检查数字的函数

代码

```
# -*- coding:utf-8 -*-
class Solution:
    def movingCount(self, threshold, rows, cols):
        # write code here
        if rows < 1 or cols < 1 or threshold < 0:
            return 0
        
        bool_matrix = [False]*(rows*cols)
        count = self.juge_enter(threshold, rows, cols, 0, 0, bool_matrix)
        return count
    
    def juge_enter(self, threshold, rows, cols, row, col, bool_matrix):
        enter = 0
        if self.check_sum(row) + self.check_sum(col) <= threshold and col >=0 and col < cols and row >= 0 and row < rows and not bool_matrix[row*cols+col]:
            bool_matrix[row*cols+col] = True
            enter = 1 + self.juge_enter(threshold, rows, cols, row+1, col, bool_matrix) + \
                    self.juge_enter(threshold, rows, cols, row-1, col, bool_matrix) + \
                    self.juge_enter(threshold, rows, cols, row, col+1, bool_matrix)+ \
                    self.juge_enter(threshold, rows, cols, row, col-1, bool_matrix)
        return enter
    
    def check_sum(self, num):
        sum = 0
        while num > 0:
            sum += num % 10
            num = num / 10
        return sum
```

# 排序

基础排序算法： 冒泡排序、选择排序、**插入排序**、**归并排序**、希尔排序、**快速排序**、**堆排序**

![](/img/in-post/alg/sort_summary.jpg)

## 冒泡排序

代码

```
#!/usr/bin/env python
# coding=utf-8

"""
    1. 比较相邻的元素。如果第一个比第二个大，就交换他们两个。
    2. 对第0个到第n-1个数据做同样的工作。这时，最大的数就“浮”到了数组最后的位置上。
    3. 针对所有的元素重复以上的步骤，除了最后一个。
    4. 持续每次对越来越少的元素重复上面的步骤，直到没有任何一对数字需要比较.
"""

# 稳定排序，平均 O(n**2)，最好 O(N), 最差 O(N**2),辅助空间 O(1)
def bubble_sort(array):
    if not array:
        return []
    n = len(array)
    for i in range(1, n):
        for j in range(i, 0, -1):
            if array[j] < array[j-1]:
                tmp = array[j-1]
                array[j-1] = array[j]
                array[j] = tmp
            else:
                continue
    return array

print bubble_sort([1, 3, 2, 5, 4])
```

## 选择排序

代码

```
#!/usr/bin/env python
# coding=utf-8

"""
    1. 在未排序序列中找到最小（大）元素，存放到排序序列的起始位置。
    2. 再从剩余未排序元素中继续寻找最小（大）元素，然后放到已排序序列的末尾。
    3. 以此类推，直到所有元素均排序完毕。
"""

# 不稳定排序，平均 O(n**2)，最好 O(N**2), 最差 O(N**2),辅助空间 O(1)
def select_sort(array):
    for i in range(len(array)):
        idx = i
        for j in range(i+1, len(array)):
            if array[j] < array[idx]:
                idx = j
        array[idx], array[i] = array[i], array[idx]
    return array

print select_sort([1, 3, 2, 5, 4])
```


## 插入排序 重要

代码

```
#!/usr/bin/env python
# coding=utf-8

"""
   1. 从第一个元素开始，该元素可以认为已经被排序
   2. 取出下一个元素，在已经排序的元素序列中从后向前扫描
   3. 如果被扫描的元素（已排序）大于新元素，将该元素后移一位
   4. 重复步骤3，直到找到已排序的元素小于或者等于新元素的位置
   5. 将新元素插入到该位置后
   6. 重复步骤2~5
"""

# 稳定排序，平均 O(n**2)，最好 O(N), 最差 O(N**2),辅助空间 O(1)

def insertSort(array):
    for i in xrange(len(array)):
        tmp = array[i]
        idx = i
        for j in xrange(i, 0, -1):
            if tmp < array[j-1]:
                array[j] = array[j-1]
                idx = j - 1
            else:
                break
        array[idx] = tmp
    return array

print insertSort([1, 3, 2, 5, 4])
```


## 归并排序  重要 

代码

```
#!/usr/bin/env python
# coding=utf-8

def guibing(data):
    if len(data) <= 0:
        return
    if len(data) == 1:
        return data
    mid = len(data) // 2
    left = data[:mid]
    right = data[mid:]
    print "left: ", left

    l1 = guibing(left)
    r1 = guibing(right)

    return merge(l1, r1)

def merge(left, right):
#    print "left: ", left
    if not left or not right:
        return left + right
    result = []
    while len(left) > 0 and len(right) > 0:
        if left[0] <= right[0]:
            result.append(left.pop(0))
        else:
            result.append(right.pop(0))
    result += left
    result += right
    return result

print guibing([5,4 ,3 ,2 ,1])
```

## 希尔排序

代码

```
#!/usr/bin/env python
# coding=utf-8

"""
数组列在一个表中并对列分别进行插入排序，重复这过程，不过每次用更长的列（步长更长了，列数更少了）来进行。
"""

# 不稳定排序，平均 O(nlogn)-O(n^2)，最好 O(nlogn), 最差 O(N**2),辅助空间 O(1)
def shellSort(array):
    if not array:
        return []
    n = len(array)
    gap = n // 2
    while gap > 0:
        for i in range(gap):
            array = gapInsertSort(array, i, gap)
        gap = gap//2
    return array

def gapInsertSort(array, start, gap):
    for i in range(start+gap, len(array), gap):
        tmp = array[i]
        idx = i
        for j in range(i-gap, 0, -gap):
            if tmp < array[j]:
                array[j+gap] = array[j]
                idx = j
            else:
                break
        array[idx] = tmp
    return array

print shellSort([1,3, 2,5, 4, 5])
```


## 快速排序  重要 

代码

```
#!/usr/bin/env python
# coding=utf-8


"""
    1. 从数列中挑出一个元素作为基准数。
    2. 分区过程，将比基准数大的放到右边，小于或等于它的数都放到左边。
    3. 再对左右区间递归执行第二步，直至各区间只有一个数。
"""

def quickSort(array):
    return qsort(array, 0, len(array)-1)

def qsort(array, left, right):
    if left >= right:
        return array
    # 用最左边的为基准数
    key = array[left]
    lp = left
    rp = right
    while lp < rp:
        while array[rp] >= key and lp < rp:
            rp -= 1
        while array[lp] <= key and lp < rp:
            lp += 1
        array[lp], array[rp] = array[rp], array[lp]
    array[left], array[lp] = array[lp], array[left]
    qsort(array, left, lp-1)
    qsort(array, rp+1, right)
    return array

print quickSort([3, 8, 2, 5, 4])
```


## 堆排序  重要 

代码

```
#!/usr/bin/env python
# coding=utf-8

def maxHeap(arr, n, i):
    largest = i
    l = 2 * i + 1
    r = 2 * i + 2
    if l < n and arr[i] < arr[l]:
        largest = l
    if r < n and arr[i] < arr[r]:
        largest = r
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        maxHeap(arr, n, largest)

def heapSort(arr):
    n = len(arr)
    for i in range(n, -1, -1):
        maxHeap(arr, n, i)
    for i in range(n-1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        maxHeap(arr, i, 0)
    return arr

arr = [12, 11, 13, 5, 6, 7]
print heapSort(arr)
```
