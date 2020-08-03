## 38.外观数列

![38.外观数列](./images/38.png)

```python
# 38 外观数列
import collections
def countAndSay(n):
    """
    :type n: int
    :rtype: str
    """
    ""
    if n == 1:
        return '1'

    cur = '1'
    count = 1
    
    while count < n:
        nextl = ''
        rec = collections.defaultdict(int)
        for i in range(len(cur)):
            rec[cur[i]] += 1
            try:
                # 防止索引溢出
                if cur[i + 1] != cur[i]:
                    nextl += str(list(rec.values())[0])
                    nextl += str(list(rec.keys())[0])
                    rec = collections.defaultdict(int)
            except:
                nextl += str(list(rec.values())[0])
                nextl += str(list(rec.keys())[0])
                
        cur = nextl
        count += 1
        
    return cur

```

### Tips:
* 注意迭代的边界条件



## 160.相交链表

![160.相交链表](./images/160-1.png)

![160.相交链表](./images/160-2.png)

```python
# 相交链表
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        pA = headA
        pB = headB

        while pA != pB:
            pA = pA.next if pA else headB
            pB = pB.next if pB else headA

        return pA
```

### Tips
* 注意超时，尽量简化判断



## 371.两整数之和

![371.两整数之和](./images/371.png)

```python
# 371两整数之和

class Solution(object):
    def getSum(self, a, b):
        """
        :type a: int
        :type b: int
        :rtype: int
        """
        # 2^32
        MASK = 0x100000000
        # 整型最大值
        MAX_INT = 0x7FFFFFFF
        MIN_INT = MAX_INT + 1
        while b != 0:
            # 计算进位
            carry = (a & b) << 1 
            # 取余范围限制在 [0, 2^32-1] 范围内
            a = (a ^ b) % MASK
            b = carry % MASK
        return a if a <= MAX_INT else ~((a % MIN_INT) ^ MAX_INT)
```

### Tips
* `a ^ b`低位运算
* `(a & b) << 1 `进位运算
* 循环直到无需进位



## 155.最小栈

![155.最小栈](./images/155.png)

```python
# 155.最小栈
class MinStack(object):

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack = []


    def push(self, x):
        """
        :type x: int
        :rtype: None
        """
        self.stack.append(x)

    def pop(self):
        """
        :rtype: None
        """
        self.stack.pop(-1)

    def top(self):
        """
        :rtype: int
        """
        return self.stack[-1]

    def getMin(self):
        """
        :rtype: int
        """
        return min(self.stack)


# Your Minself.stack object will be instantiated and called as such:
# obj = Minself.stack()
# obj.push(x)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()
```

### Tips
* 注意栈的定义



## 121.买卖股票的最佳时机

![121.买卖股票的最佳时机](./images/121.png)

```python
# 121.买卖股票的最佳时机

class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        if not prices:
            return 0

        minValue = prices[0]
        maxProfit = 0

        for price in prices:
            minValue = min(price, minValue)
            maxProfit = max(maxProfit,price - minValue)

        return maxProfit
```

### Tips
* 维护两个变量，分别记录历史最低股价和有可能的最高利润
* 注意初始条件



## 217.存在重复元素

![217.存在重复元素](./images/217.png)

```python
# 217.存在重复元素
class Solution(object):
    def containsDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        if not nums:
            return False

        return len(list(set(nums))) != len(nums)
```

### Tips
* 可以使用哈希表，排序，暴力法



## 101.对称二叉树

![101.对称二叉树](./images/101.png)

```python
# 101.对称二叉树

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

# 递归
class Solution(object):
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        if not root:
            return True

        def isMirrored(rootA, rootB):
            # 双指针都达到None
            if not rootA and not rootB:
                return True
            try:
                # 若指针都非NONE
                if rootA.val == rootB.val:
                    flag = isMirrored(rootA.left, rootB.right) and isMirrored(rootA.right,rootB.left)
                    if flag:
                        return True
            except:
                # 其中一个指针为None
                return False
            
            return False
        
        return isMirrored(root.left, root.right)
```

### Tips
* 双指针，相反方向，往下递归，判断子树是否对称

```python
# 迭代

class Solution(object):
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """

        if not root:
            return True

        queue = [root.left,root.right]
        while queue:
            left = queue.pop(0)
            right = queue.pop(0)
            # 两个都为空
            if not left and not right:
                continue
            # 其中一个为空
            if not left or not right:
                return False
            # 两个都不为空
            if left.val != right.val:
                return False
            
            # 继续将他们的子树加入队列
            
            queue.append(left.left)
            queue.append(right.right)

            queue.append(left.right)
            queue.append(right.left)
        
        return True
```

### Tips
* 队列只能头部出，尾部进
* 经常可以使用栈和队列将递归转化为迭代



## 53.最大子序和

![53.最大子序和](./images/53.png)

```python
# 53.最大子序和
# 动态规划

class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) == 1:
            return nums[0]

        ans = nums[0]
        preSum = 0
        
        for num in nums:
            # 意思是要么把当前数字之前的和继续使用，要么重起炉灶
            preSum = max(preSum + num, num)
            # 是否更新最大值
            ans = max(ans, preSum)

        return ans
```

### Tips

* 注意动态规划函数的构建`dp[i] = max(dp[i - 1] + nums[i], nums[i])`
* `dp[i]`为以`nums[i]`结尾的连续子序列的最大值



## 26.删除排序数组中的重复项

![26.删除排序数组中的重复项](./images/26.png)

```python
# 26.删除排序数组中的重复项

class Solution(object):
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """

        p1 = 0
        p2 = 1

        while p2 < len(nums):
            if nums[p1] == nums[p2]:
                p2 += 1
            else:
                p1 += 1
                nums[p1] = nums[p2]
                
        
        return p1 + 1
```

### TIPS
* 使用双指针
* 遇到不同的值，慢指针先移动，然后再赋值



## 70.爬楼梯

![70.爬楼梯](./images/70.png)

```python
# 70.爬楼梯

class Solution(object):
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n== 1:
            return 1
        if n == 2:
            return 2

        tmp = [1,2]

        for i in range(2,n):
            tmp.append(tmp[i - 1] + tmp[i - 2])

        return tmp[-1]

```

### Tips
* 数学解法，`f(n) = f(n - 1) + f(n - 2)`
* 还可以用通项公式，或者矩阵快速幂



## 1.两数之和

![1.两数之和](./images/1.png)

```python
# 1.两数之和
import collections
def twoSum(nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: List[int]
    """
    record = collections.defaultdict()

    for i in range(len(nums)):
        num = nums[i]
        residual = target - num
        try:
            return [record[str(num)], i]
        except:
            record[str(residual)] = i
        
```

### Tips
* 单字典，边循环边查询



## 141.环形链表

![141.环形链表](./images/141.png)

```python
# 141.环形链表

class Solution(object):
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        p1 = head
        p2 = head
        
        while p2:
            try:
                p1 = p1.next
                p2 = p2.next.next
            except:
                return False
            
            if p1 == p2:
                return True
```

### Tips
* 快慢双指针，如果能相遇，说明有环
* 也可以使用哈希表储存访问过的节点



## 88.合并两个有序数组

![88.合并两个有序数组](./images/88.png)

```python
# 88.合并两个有序数组

class Solution(object):
    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: None Do not return anything, modify nums1 in-place instead.
        """

        p1 = m - 1
        p2 = n - 1
		
        # 从合并后的位置，从后往前
        for i in range(m + n)[::-1]:
            if p2 == -1:
                break
            
            if p1 == -1:
                nums1[i] = nums2[p2]
                p2 -= 1
                continue

            if nums1[p1] >= nums2[p2]:
                nums1[i] = nums1[p1]
                p1 -= 1

            else:
                nums1[i] = nums2[p2]
                p2 -= 1
```

### Tips

* 双指针分别指向两个数组，从后往前循环
* 注意循环截至条件



## 326.3的幂

![3的幂](./images/326.png)

```python
# 326. 3的幂

class Solution(object):
    def isPowerOfThree(self, n):
        """
        :type n: int
        :rtype: bool
        """

        while n > 1:
            n, ret = divmod(n, 3)
            if ret != 0:
                return False
        
        if n == 1:
            return True
```

### Tips
* 可以使用多种数学方法
* 可递归，可迭代



## 198. 打家劫舍

![198. 打家劫舍](./images/198.png)

```python
# 199.打家劫舍
class Solution(object):
    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return 0
        
        if not nums:
            return 0

        maxProfitPre1 = nums[0]
        maxProfitPre2 = 0

        for i in range(1, len(nums)):
            maxProfit = max(maxProfitPre2 + nums[i], maxProfitPre1)
            maxProfitPre2 = maxProfitPre1
            maxProfitPre1 = maxProfit

        return maxProfit
```

### Tips
* 动态规划，维护两个变量，递推公式为`fn = max(f(n - 2) + s(i), f(n - 1))`



## 125.验证回文串

![125.验证回文串](./images/125.png)

```python
# 125.验证回文串
import string
class Solution(object):
    def isPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        if not s:
            return True

        punc = string.punctuation + ' '

        stripped = [l.lower() for l in s if l not in punc]

        for i in range(len(stripped) // 2):
            if stripped[i] != stripped[-(i + 1)]:
                return False
        
        return True
```

### Tips:
* 使用API去除标点和空格
* 也可以在原字符串上操作



## 387.字符串中的第一个唯一字符

![387.字符串中的第一个唯一字符](./images/387.png)

```python
# 387.字符串中的第一个唯一字符
class Solution(object):
    def firstUniqChar(self, s):
        """
        :type s: str
        :rtype: int
        """
        
        if not s:
            return -1

        for i, letter in enumerate(s):
            if letter not in record:
                record[letter] = [i]
            else:
                record[letter].append(i)

        indexs = []

        for i in record.values():
            if len(i) == 1:
                indexs.append(i[0])

        if indexs:
            return min(indexs)
        else:
            return -1
                
```

### Tips:
* 也可以在字典中保存字母计数，第二次遍历字符串，找出计数为1的字母索引



## 66.加1

![66.加1](./images/66.png)

```python
# 61. 加1
class Solution(object):
    def plusOne(self, digits):
        """
        :type digits: List[int]
        :rtype: List[int]
        """

        if len(digits) == 1:
            if digits[0] == 9:
                digits = [1,0]
            else:
                digits[0] += 1
        
        else:
            if digits[-1] == 9:
                digits = self.plusOne(digits[:-1])
                digits.append(0)
            else:
                digits[-1] += 1
        
        return digits
```

### Tips
* 注意append的用法



## 234.回文链表

![234.回文链表](./images/234.png)

```python
# 234.回文链表

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def isPalindrome(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """

        p = head
        vals = []

        while p:
            vals.append(p.val)
            p = p.next

        for i in range(len(vals) // 2):
            if vals[i] != vals[- (i + 1)]:
                return False
            
        return True
```

### Tips
* 该解法将回文链转化为回文列表
* 可以使用快慢指针将链从中砍断，颠倒后半部分，然后比较，再把后半部分颠倒回来



## 189.旋转数组


![189.旋转数组](./images/189.png)

```python
# 189.旋转数组
# 计数逐次移动法
class Solution(object):
    def rotate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: None Do not return anything, modify nums in-place instead.
        """

        move = k % len(nums)

        for i in range(len(nums) - move):
            e = nums.pop(0)
            nums.append(e)
            

# 一次性移动到位法
class Solution(object):
    def rotate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: None Do not return anything, modify nums in-place instead.
        """

        move = k % len(nums)

        start = 0
        count = 0
        
        # 每个元素最多只需要移动一次，所以一共n次就可以移动结束，最外层循环
        while count < len(nums):
            tmp = nums[start]
            current = (start + move) % len(nums)
            # 内层循环，当当前处理，来到小循环出发点的时候，跳出循环
            while current != start:
                nums[current], tmp = tmp, nums[current]
                current = (current + move) % len(nums)
                count += 1
            # 跳出循环后，一定要做一次置换，把出发位置和缓存互换
            nums[current], tmp = tmp, nums[current]
            count += 1
            start += 1
```

### Tips
* 可以计算翻转次数和长度的最大公约数n，然后分n轮将某元素一次移到位，中间需要tmp记录被替换的元素
* 仔细考虑清楚法2的控制流，是难点



## 20.有效的括号

![20.有效的括号](./images/20.png)

```python
# 20.有效的括号

class Solution(object):
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        
        stack = []
        left = {'(':1,'[':2,'{':3}
        right = {')':1,']':2,'}':3}

        for i in s:
            if i in left:
                stack.append(left[i])
            elif i in right:
                if not stack:
                    return False
                if stack[-1] != right[i]:
                    return False
                else:
                    stack.pop(-1)

        if stack:
            return False
        else:
            return True
```

### Tips

* 使用栈结构，非常简单



## 172.阶乘后的零的数量

![172.阶乘后的零的数量](./images/172.png)

```python
# 172.阶乘后零的数量

# 迭代
def trailingZeroes(n):
    """
    :type n: int
    :rtype: int
    """

    # 等价于计算隔了多少个5 + 隔了多少25 + 隔了多少125 + ... 

    res = 0

    p = 1

    while (n // 5**p) > 0:
        res += n // 5**p
        p += 1
    return res

# 优化后的迭代

def trailingZeroes(n):
    res = 0
    
    while n > 0:
        n = n // 5
        res += n
        
    return res

# 递归

def trailingZeroes(n):
    
    if not n:
        return 0
    
    return trailingZeroes(n // 5) + (n // 5)
```

### Tips
* 这道题为数论题
* 有多少0，等价于质因数中有多少10，等价于质因数中有多少2 * 5，等价于质因数中有多少5
* 数列n中每隔5个元素产生1个5，每隔25个元素产生2个5,....每隔`5**p`产生p个5
* 等价于`f(n) = n//5 + n//25 + .. + n // 5**p`
* `f(n) = f(n // 5) + n // 5`



## 28.实现 strStr()

![28.实现 strStr()](./images/28.png)

```python
# 28.实现strStr()

class Solution(object):
    def strStr(self, haystack, needle):
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        if not needle and not haystack:
            return 0
        if not needle:
            return 0
        if len(haystack) < len(needle):
            return -1     
        
        p1 = 0
        p2 = 0

        while p1 <= len(haystack) - len(needle):
            if haystack[p1] == needle[0]:
                stop = p1
                while haystack[p1] == needle[p2]:
                    p1 += 1
                    p2 += 1

                    if p2 == len(needle):
                        return stop
                    
                p2 = 0
                p1 = stop
            p1 += 1

        return -1
```

### Tips
* 注意控制流的结束条件以及特殊条件的处理



## 69.X的平方根

![69.X的平方根](./images/69.png)

```python
# 69. X的平方根
def mySqrt(x):
    """
    :type x: int
    :rtype: int
    """

    if x == 1:
        return x

    left = 1
    right = x

    while left < right:
        mid = left + (right - left + 1) // 2
        square = mid ** 2
        
        if square > x:
            right = mid - 1
        elif square < x:
            left = mid
        else:
            return mid 

    return left
```

### Tips:
* 二分法主要注意中位数的取法，有很多种不同取法，只需要记住这种取右中位数。
* [二分法的一些模板](https://leetcode-cn.com/problems/search-insert-position/solution/te-bie-hao-yong-de-er-fen-cha-fa-fa-mo-ban-python-/)



## 14. 最长公共前缀

![14. 最长公共前缀](./images/14.png)

```python
# 14.最长公共前缀

class Solution(object):
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """

        if not strs:
            return ""

        if not strs[0]:
            return ""
        pub = ''
        p = 0
        while True:
            try:
                # 使用try试探，防止索引溢出
                cur = strs[0][p]
            except:
                return pub
            for s in strs:
                try:
                    if s[p] !=  cur:
                        return pub
                except:
                    return pub
            pub += cur
            p += 1

        return pub
```

### Tips
* 纵向匹配
* 还可以使用横向搜索法逐个匹配，或者使用二分递归



## 7.整数反转

![7.整数反转](./images/7.png)

```python
# 7.整数反转

class Solution(object):
    def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """
        
        res = 0
        if x < 0:
            value = -x
            flag = -1
        else:
            value = x
            flag = 1

        while value > 0:           
            value, ret = divmod(value, 10)
            res = res * 10 + ret
        res = res * flag
        if res > 2 ** 31 -1 or res < - 2 **31:
            return 0
        
        return res
```

### Tips
* 使用数学法判断



## 204.计数质数

![204.计数质数](./images/204.png)

```python
# 204.统计质数


def countPrimes(n):
    """
    :type n: int
    :rtype: int
    """

    def isPrime(n):
        x = 2
        while x ** 2 < n:
            if n % x == 0:
                return False
            x += 1
        return True

    res = [True] * n
    i = 2
    while i ** 2 < n:
        if isPrime(i):
            for j in range(i * i, n, i):
                res[j] = False
        i += 1

    count = 0
    for i in range(2, n):
        if res[i] == True:
            count += 1
    return count
```

### Tips：
* 判断一个数是否为质数，可以判断从是否能被从2到sqrt(n)整除
* n为质数，则n的倍数为非质数
* 该算法名为Sieve of Eratosthenes

