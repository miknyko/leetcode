## 98.验证二叉搜索树

![98.验证二叉搜索树](.\images\98.PNG)

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        
        stack = []
        p = root
        val = float('-inf')

        while stack or p:
            while p:
                stack.append(p)
                p = p.left
            p = stack.pop(-1)
            if p.val > val:
                val = p.val
            else:
                return False
            p = p.right
        
        return True
```

### Tips

* 中序遍历，判断递增性



## 5.最长回文子串

![5.最长回文子串](.\images\5.PNG)

```python
class Solution(object):
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        if not s:
            return s

        n = len(s)
        max_length = 0

        for center in range(2 * n - 1):
            left = center // 2
            right = left + center % 2

            while left >= 0 and right < n and s[left] == s[right]:
                if right - left + 1 > max_length:
                    max_length = right - left + 1
                    palindrome = s[left : right + 1]
                left -= 1
                right += 1
                
        return palindrome
```

### Tips

* 使用中心扩散法，遍历可能的回文串中心
* 还可以动态规划`dp[i][j] = (s[i] == s[j] && dp[i + 1][j - 1])`



## 15. 三数之和

![15. 三数之和](.\images\15.PNG)

```python
class Solution(object):
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        n = len(nums)
        nums.sort()
        res = []

        for i in range(n):
            # 若选择的最小的数都大于0，则没有有效组合，停止搜索
            if nums[i] > 0:
                break

            # 避免选择重复最小数
            if nums[i] == nums[i - 1] and i > 0:
                continue

            l = i + 1
            r = n - 1

            while l < r:
                sums = nums[i] + nums[l] + nums[r] 
                # 满足条件
                if sums == 0:
                    res.append([nums[i], nums[l], nums[r]])
                    # 跳越重复值
                    while l < r and nums[l] == nums[l + 1]:
                        l += 1
                    while l < r and nums[r] == nums[r - 1]:
                        r -= 1
                    l += 1
                    r -= 1
                # 如果大了，那么最大数减小
                elif sums > 0:
                    r -= 1
                # 如果小了，中间数增大
                elif sums < 0:
                    l += 1
                
            
        return res
```

### Tips

* 先对数组排序，可以利用单调性

* 固定最小值，搜索中间值和最大值



### 166.分数到小数

![166.分数到小数](.\images\166.PNG)

```python
class Solution(object):
    def fractionToDecimal(self, numerator, denominator):
        """
        :type numerator: int
        :type denominator: int
        :rtype: str
        """
        res = ''
        if numerator == 0:
            return '0'
        # 判断符号
        if (numerator > 0) ^ (denominator > 0):
            res += '-'
        # 都变为正值处理
        numerator, denominator = abs(numerator), abs(denominator)
        # 判断是否整除
        res += str(numerator // denominator)

        remainder = numerator % denominator
        if remainder == 0:
            return res
        # 若不能整除，有小数
        res += '.'
        record = {remainder : len(res)}

        while remainder:
            # 模拟长除法
            frac, remainder = divmod(remainder * 10, denominator)
            res += str(frac)
            # 如果余数出现过，说明开始了循环
            if remainder in record:
                # 添加括号
                res = list(res)
                res.insert(record[remainder], '(')
                res += ')'
                return "".join(res)
            else:
                record[remainder] = len(res)
        
        return res
                
```

### Tips

* 主要注意符号，是否整除的处理
* 处理循环小数的时候，每次长除使用哈希表记录相同余数第一次出现的位置



## 91.解码方法

![91.解码方法](.\images\91.PNG)

```python
class Solution(object):
    def numDecodings(self, s):
        """
        :type s: str
        :rtype: int
        """

        if s[0] == '0':
            return 0

        n = len(s)
        dp = [0 for _ in range(n)]
        dp[0] = 1

        for i in range(1, n):
            if s[i] != '0':
                dp[i] = dp[i - 1]

            if 10 <= int(s[i - 1] + s[i]) <= 26:
                if i == 1:
                    dp[i] += 1
                else:
                    dp[i] += dp[i - 2]

        return dp[-1]
```

### Tips

* 动态规划，和爬楼梯类似
* 主要注意0元素得处理



## 8.字符串转换整数(atoi)

![8.字符串转换整数(atoi)](.\images\8.PNG)

```python
class Solution(object):
    def myAtoi(self, str):
        """
        :type str: str
        :rtype: int
        """
        if not str:
            return 0

        INT_MAX = 2 ** 31 - 1
        INT_MIN = - 2 ** 31
        res = 0
        sign = 1
        i = 0

        while i < len(str) and str[i] == ' ':
            i += 1
        if i == len(str):
            return 0
        if str[i] == '-':
            sign = -1
        if str[i] in '+-':
            i += 1
 
        while i < len(str) and str[i].isdigit():
            res = res * 10 + int(str[i])
            tmp = res if sign > 0 else -res
            if tmp >= INT_MAX:
                return INT_MAX
            if tmp <= INT_MIN:
                return INT_MIN
            i += 1
 
        if sign > 0:
            return res
        else:
            return -res
```

### Tips

* 注意边界条件的处理



## 29.两数相除

![29.两数相除](.\images\29.PNG)

```python
class Solution(object):
    def divide(self, dividend, divisor):
        """
        :type dividend: int
        :type divisor: int
        :rtype: int
        """
        # 模拟长除法

        sign = (dividend > 0) ^ (divisor > 0)
        dividend = abs(dividend)
        divisor = abs(divisor)
        res = 0
        count = 0

        if dividend < divisor:
            return 0

        # 得到最高位count的位置
        while divisor <= dividend:
            divisor <<= 1
            count += 1
        # 开始长除法
        while count > 0:
            count -= 1
            divisor >>= 1
            if divisor <= dividend:
                res += 1 << count # 二进制商1，并累计到10进制结果中,否则此位商0
                dividend -= divisor
        
        res = -res if sign else res

        return res if -(1 << 31) <= res <= (1 << 31) - 1 else (1 << 31) - 1
```

### Tips

* 使用位移操作模拟二进制长除法



## 42.接雨水

![42.接雨水](.\images\42.PNG)

```python
# 双指针
class Solution(object):
    def trap(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        if not height:
            return 0

        n = len(height)
        left = 1
        right = n - 2
        # height[left]左边最高的墙的高度
        left_max = height[0]
        right_max = height[n - 1]
        sum = 0

        # 等号的时候
        while left <= right:
            # 如果左边小，就从左边求
            if left_max <= right_max:
                # 此列能存储的雨水
                water = max(left_max - height[left], 0)
                # 是否更新left_max
                left_max = max(left_max, height[left])
                # 更新结果
                sum += water
                left += 1

            # 否则从右边求
            else:
                water = max(right_max - height[right], 0)
                right_max = max(right_max, height[right])
                sum += water
                right -= 1

        return sum
```

### Tips

* 按列求，每一列能存储的雨水是它（左边最高的墙，或者右边最高的墙）矮的那堵，与自己的高度差，决定的。
* 从左往右遍历的时候，只有左边的`leftmax`可信，从右往左遍历的时候，只有右边的`rightmax`可信，因此交叉前进



```python
# 单调栈
class Solution(object):
    def trap(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        # 维护一个单调递减栈
        stack = [0]
        i = 1
        sums = 0

        # 遍历
        while i < len(height):
            # 如果比栈顶小，继续入栈
            if height[i] <= height[stack[-1]]:
                stack.append(i)

            # 否则，持续出栈，开始结算
            if height[i] > height[stack[-1]]:
                while stack and height[i] > height[stack[-1]]:
                    bottom = stack.pop(-1)
                    if not stack:
                        break
                    # 注意高度
                    w = i - stack[-1] - 1
                    h = min(height[stack[-1]], height[i]) - height[bottom]
                    sums += w * h
                # 结算完毕，将该元素入栈
                stack.append(i)
            
            i += 1
     
        return sums
```

### Tips

* 按横条结算，维护一个单调递减栈，遇小入栈，遇大出栈开始结算



## 23.合并K个升序链表

![23.合并K个升序链表](.\images\23.PNG)

```python
    # Definition for singly-linked list.
    # class ListNode(object):
    #     def __init__(self, val=0, next=None):
    #         self.val = val
    #         self.next = next
    class Solution(object):
        def mergeKLists(self, lists):
            """
            :type lists: List[ListNode]
            :rtype: ListNode
            """
            if not lists:
                return None
                
            def merge(nodeA, nodeB):
                dummyhead = ListNode(0)
                p = dummyhead

                while nodeA and nodeB:
                    if nodeA.val < nodeB.val:
                        p.next = nodeA
                        p = p.next
                        nodeA = nodeA.next
                    else:
                        p.next = nodeB
                        p = p.next
                        nodeB = nodeB.next
                
                p.next = nodeA if nodeA else nodeB
                return dummyhead.next
            
            def recursion(left, right):
                if left == right:
                    return lists[left]
                mid = left + (right - left) // 2
                l = recursion(left, mid)
                r = recursion(mid + 1, right)

                return merge(l, r)

            node = recursion(0, len(lists) - 1)
            return node
```

### Tips

* 分治递归，两两合并



## 128.最长连续序列

![128.最长连续序列](.\images\128.PNG)

```python
class Solution(object):
    def longestConsecutive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return 0
        
        # 存储的是以某个元素为端点的连续序列长度
        record = {}
        max_length = 0

        for num in nums:
            if num not in record:             
                left = record.get(num - 1, 0)
                right = record.get(num + 1, 0)

                # 得到一个新的，有可能是更长长度的长度
                new_length = left + right + 1
                max_length = max(max_length, new_length)

                # 把这个长度的两个端点进行更新
                record[num - left] = new_length
                record[num + right] = new_length

                # 把以当前数为端点的序列长度
                record[num] = max(left, right) + 1

        return max_length
```

### Tips

* 当一个新的数`n`进入哈希表时，以`n-1`和`n+1`为端点的连续序列，一定不可能包含`n`，因为`n`是新数。
* 这个是一个很关键的逻辑



## 297.二叉树的序列化与反序列化

![297.二叉树的序列化与反序列化](.\images\297.PNG)

```python
class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        if not root:
            return ""
        queue = [root]
        res = []

        while queue:
            node = queue.pop(0)
            if node:
                res.append(str(node.val))
                queue.append(node.left)
                queue.append(node.right)
            else:
                res.append('null')

        return ','.join(res)


        

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        if not data:
            return None

        data = data.split(",")
        root = TreeNode(int(data[0]))
        queue = [root]
        i = 1

        while queue:
            node = queue.pop(0)
            if data[i] != 'null':
                node.left = TreeNode(int(data[i]))
                queue.append(node.left)
            i += 1
            if data[i] != 'null':
                node.right = TreeNode(int(data[i]))
                queue.append(node.right)
            i += 1

        return root
```

### Tips

* 这种序列化和纯粹的遍历区别就是，需要将`NONE`输出



## 239.滑动窗口最大值

![239.滑动窗口最大值](.\images\239.PNG)

```python
class Solution(object):
    def maxSlidingWindow(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """

        # 单调队列,头部是滑动窗口最大元素的索引
        class queue():
            def __init__(self):
                self.queue = []

            def push(self, i):
                while self.queue and nums[i] > nums[self.queue[-1]]:
                    self.queue.pop(-1)
                self.queue.append(i)
            # 如果队列头部元素是被滑出的元素，则头部元素弹出
            def pop_front(self, i):
                if self.queue and self.queue[0] == i:
                    self.queue.pop(0)
            

        n = len(nums)
        deque = queue()
        res = []

        # 初始化双端队列
        for i in range(k):
            deque.push(i)     
        res.append(nums[deque.queue[0]])

        for i in range(k, n):
            deque.push(i)
            deque.pop_front(i - k)
            res.append(nums[deque.queue[0]])

        return res
        
```

### Tips

* 使用单调队列，记录当前窗口中最大元素的索引



## 295.数据流的中位数

![295.滑动窗口最大值](.\images\295.PNG)

```python
class MedianFinder(object):

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.min_heap = []
        self.max_heap = []
        

    def addNum(self, num):
        """
        :type num: int
        :rtype: None
        """
        
        max_top = heapq.heappushpop(self.max_heap, -num)
        heapq.heappush(self.min_heap, -max_top)
        # 如果此时小顶堆长度大于大顶堆，则将堆顶元素平衡给大顶堆
        if len(self.min_heap) > len(self.max_heap):
            min_top = heapq.heappop(self.min_heap)
            heapq.heappush(self.max_heap, -min_top)
 

    def findMedian(self):
        """
        :rtype: float
        """
        if len(self.max_heap) > len(self.min_heap):
            # 说明奇数个数据
            return -self.max_heap[0]

        else:
            return (float(self.min_heap[0]) - float(self.max_heap[0])) / 2
        
```

### Tips

* 构建一个大顶堆和一个小顶堆
* 添加一个数的时候，注意两个堆之间的平衡