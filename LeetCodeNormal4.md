## 79.单词搜索

![79.单词搜索](.\images\79.PNG)

```python
class Solution(object):
    def exist(self, board, word):
        """
        :type board: List[List[str]]
        :type word: str
        :rtype: bool
        """

        h = len(board)
        w = len(board[0])

        def dfs(i, j, count, word, board):
            if board[i][j] == word[count]:
                mask[i][j] = 1
                count += 1
            else:
                # 需要及时掐断递归
                return
            
            # 递归终止条件
            if count == len(word):
                return True

            for k, l in [(i - 1, j),(i + 1, j),(i, j - 1),(i, j + 1)]:
                # 未超出边界
                if 0 <= k < h and 0 <= l < w:
                    # 某节点没有访问过
                    if mask[k][l] == 0:
                        if dfs(k, l, count, word, board):
                            return True
            # 回溯
            mask[i][j] = 0
            return
                    
        mask = [[0 for _ in range(w)] for _ in range(h)]
        count = 0  
        for i in range(h):
            for j in range(w):
                # 对每一个元素进行这种搜索
                # 只要有一个元素返回true，则返回True
                # 如果最后没有True, 就返回False
                if dfs(i, j, count, word, board):
                    return True
                
        return False
```

### Tips

* 回溯搜索
* 一定要注意回溯的时机，以及返回`False`的时机



## 130.被围绕的区域

![130.被围绕的区域](.\images\130.PNG)

```python
class Solution(object):
    def solve(self, board):
        """
        :type board: List[List[str]]
        :rtype: None Do not return anything, modify board in-place instead.
        """
        if not board:
            return None

        h = len(board)
        w = len(board[0])
        
        direction = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        def dfs(i, j):
            if board[i][j] == 'O':
                board[i][j] = 'P'
                for d in direction:
                    if 0 <= i + d[0] < h and 0 <= j + d[1] < w:
                        dfs(i + d[0], j + d[1])
        
        for i in range(h):
            for j in range(w):
                if i == 0 or i == h - 1 or j == 0 or j == w - 1:
                    dfs(i, j)

        for i in range(h):
            for j in range(w):
                if board[i][j] == 'O':
                    board[i][j] = 'X'
                if board[i][j] == 'P':
                    board[i][j] = 'O'
```

### Tips

* 沿着边沿遍历矩阵，遇到O就进行DFS，标记为P
* 再次遍历整个矩阵，将所有O变为X，所有P变为O



## 54.螺旋矩阵

![54.螺旋矩阵](.\images\54.PNG)

```python
class Solution(object):
    def spiralOrder(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """

        if not matrix:
            return []

        h = len(matrix)
        w = len(matrix[0])
        res = []
        mask = [[0 for _ in range(w)] for _ in range(h)]

        status = 'right'

        movement = {
            'right':(0, 1),
            'left':(0, -1),
            'up':(-1, 0),
            'down':(1, 0)
        }

        next_order = {
            'right':'down',
            'down':'left',
            'left':'up',
            'up':'right'
        }

        count = 0
        i = 0
        j = 0

        while count < h * w:
            res.append(matrix[i][j])
            mask[i][j] = 1
            count += 1
            
            new_i = i + movement[status][0]
            new_j = j + movement[status][1]
			
            # 如果能够进行合法移动，就合法移动
            if 0 <= new_i < h and 0 <= new_j < w and mask[new_i][new_j] == 0:
                i = new_i
                j = new_j
			
            # 否则转向
            else:
                status = next_order[status]
                i += movement[status][0]
                j += movement[status][1]

        return res
```

### Tips

* 灵活使用哈希表，建立状态转移条件

* LC上面的题解都是垃圾



## 55.跳跃游戏

![55.跳跃游戏](.\images\55.PNG)

```python
class Solution(object):
    def canJump(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """

        max_i = 0
        i = 0

        # 逐个元素挨着跳
        while i <= len(nums) - 1:
            # 如果当前已经来到一个理论上根本无法到达的地方，则返回False
            if i > max_i:
                return False
            
            # 更新当前能够跳到的最远距离
            max_i = max(max_i, i + nums[i])
            i += 1
        
        # 说明此时i已经到了最后一位，并且记录在案的理论最大能跳距离是足够的
        return True
```



### Tips

* 贪心解法，如果某一点能够到达，则这一点前面的所有点一定可以到达
* 所以只要逐个遍历元素，动态更新理论上能够到达的最远距离，判断这个距离是否超出最后一位即可



## 322.零钱兑换

![322.零钱兑换](.\images\322.PNG)

```python
class Solution(object):
    def coinChange(self, coins, amount):
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """
        if not amount:
            return 0

        stack = [(amount, 0)]
        cache = {}

        while stack:
            tmp = stack.pop(0)
            for value in coins:
                residual = tmp[0] - value               
                if residual == 0:
                    return tmp[1] + 1
                if residual > 0:
                    if residual not in cache:
                        cache[residual] = True
                        stack.append((residual, tmp[1] + 1))
        
        return -1

```

### Tips

* 使用BFS，注意使用哈希表存储已经访问过的值，防止重复访问相同的值



## 34. 在排序数组中查找元素的第一个和最后一个位置

![34. 在排序数组中查找元素的第一个和最后一个位置](.\images\34.PNG)

```python
class Solution(object):
    def searchRange(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """

        n = len(nums)

        left = 0
        right = n - 1
        res = []

        # 找左边界
        while left <= right:
            mid = left + (right - left) // 2

            if nums[mid] == target:
                right = mid - 1
            elif nums[mid] < target:
                left = mid + 1
            elif nums[mid] > target:
                right = mid - 1
        
        if left > n - 1 or nums[left] != target:
            return [-1, -1]

        else:
            res.append(left)

        # 找右边界
        left = 0
        right = n - 1

        while left <= right:
            mid = left + (right - left) // 2

            if nums[mid] == target:
                left = mid + 1
            elif nums[mid] < target:
                left = mid + 1
            elif nums[mid] > target:
                right = mid - 1

        if right < 0 or nums[right] != target:
            return [-1, -1]

        else:
            res.append(right)

        return res
```

### Tips

* 注意二分查找的模板，找边界与找固定值的区别是每次找到目标值，则将搜索区间的对面边界固定到这个值
* 搜索区间[0, n - 1]
* 用小于等于
* 左动+1， 右动 -1
* 搜索区间边界返回时，注意判断指针是否越界



## 35.搜索插入位置

![35.搜索插入位置](.\images\35.PNG)

```python
class Solution(object):
    def searchInsert(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """

        n = len(nums)
        left = 0
        right = n - 1

        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] > target:
                right = mid - 1
            elif nums[mid] < target:
                left = mid + 1

        return left
```

### Tips

* 套用二分搜索模板，查找目标值索引



## 152.乘积最大子数组

![152.乘积最大子数组](.\images\152.PNG)

```python
class Solution(object):
    def maxProduct(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """

        pre_max_dot = 1
        pre_min_dot = 1
        max_dot = nums[0]

        for i in nums:
            if i < 0:
                pre_max_dot, pre_min_dot = pre_min_dot, pre_max_dot
            
            pre_max_dot = max(i, pre_max_dot * i)
            pre_min_dot = min(i, pre_min_dot * i)
            max_dot = max(pre_max_dot, max_dot)


        return max_dot
```

### Tips

* 类似53.最大子序和，但是需要同时维护一个当前最小连续乘积
* 当遇到负数的时候，最大连续乘积和最小连续乘积互换，再和当前数做相乘



## 19.删除链表的倒数第N个节点

![19.删除链表的倒数第N个节点](.\images\19.PNG)

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """

        if not head.next:
            return None

        record = head

        fast = head
        count = 0
        
        # 让fast先走n步
        while count < n:
            fast = fast.next
            count += 1
		
        # 若链表长度为N，则走了n步后，fast已经来到了None
        if not fast:
            return head.next
        
        slow = head
        while fast.next:
            fast = fast.next
            slow = slow.next
        # 此时fast指向倒数第一个节点，slow指向倒数第n + 1个节点
        slow.next = slow.next.next

        return head
```

### Tips

* 注意链表长度为n的情况



## 334.递增的三元子序列

![334.递增的三元子序列](.\images\334.PNG)

```python
class Solution(object):
    def increasingTriplet(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        small = float('inf')
        mid = float('inf')

        for i in nums:
            if i <= small:
                small = i
            elif i <= mid:
                mid = i
            elif i > mid:
                return True
        return False
```

### Tips

* 类似于贪心的算法

* 遍历数组，维护两个值，一个最小值，一个倒数第二小值，不断更新
* 只要某个值比倒数第二个值大，就说明有递增序列



## 33.搜索旋转排序数组

![33.搜索旋转排序数组](.\images\33.PNG)

```python
class Solution(object):
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        
        n = len(nums)
        left = 0
        right = n - 1

        while left <= right:
            mid = left + (right - left) // 2

            if nums[mid] == target:
                return mid
            
            # 如果左边有序
            if nums[left] <= nums[mid]:
                if nums[left] <= target < nums[mid]:
                    right = mid - 1         
                else:
                    left = mid + 1
            
            # 否则就是右边有序
            else:
                if nums[mid] < target <= nums[right]:
                    left = left + 1
                
                else:
                    right = mid - 1

        return -1
```

### Tips

* 同样是二分搜索，但是需要讨论每次选取`mid`的时候，到底左边是单调区间，还是右边是单调区间，如果想继续查找的区域是单调区间，则按二分策略搜索，否则就换一边搜索



## 2.两数相加

![2.两数相加](.\images\2.PNG)

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """

        pre = ListNode(0)
        cur = pre
        carry = 0
        while l1 or l2:
            val1 = l1.val if l1 else 0
            val2 = l2.val if l2 else 0
            sums = val1 + val2 + carry

            carry = sums // 10
            residual = sums % 10

            cur.next = ListNode(residual)

            if l1:
                l1 = l1.next
            if l2:
                l2 = l2.next
            
            cur = cur.next
        
        if carry:
            cur.next = ListNode(1)

        return pre.next
```

### Tips

* 注意如何使代码更简单易懂
* 一般来说构造一个辅助头部链表`pre`比较舒服



## 179.最大数

![179.最大数](.\images\179.PNG)

```python
class Solution(object):
    def largestNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: str
        """

        nums = map(str, nums)
        nums.sort(lambda x, y:cmp(y + x, x + y))
        res = "".join(nums).lstrip("0")

        return res or "0"
```

### Tips

* 自定义排序规则
* 注意`sort`和`cmp`的用法



## 215.数组中的第k个最大元素

![215.数组中的第k个最大元素](.\images\215.PNG)

```python
class Solution(object):
    def findKthLargest(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        n = len(nums)
        k_s = n - k
        # 将nums[left:right + 1]分一次区
        def partition(left, right):
            # 随机初始化pivot
            random_index = random.choice(range(left, right + 1))
            nums[random_index], nums[left] = nums[left], nums[random_index]

            l = left
            r = right

            pivot = left

            while l < r:
                while l < r and nums[r] > nums[pivot]:
                    r -= 1
                while l < r and nums[l] <= nums[pivot]:
                    l += 1
                nums[l], nums[r] = nums[r], nums[l]
            nums[left], nums[l] = nums[l], nums[pivot]

            return l

        left = 0
        right = n - 1

        while True:
            index = partition(left, right)
            if index == k_s:
                return nums[index]
            if index > k_s:
                right = index - 1
            else:
                left = index + 1
```

### Tips

* 快速选择，每次选择一个pivot将数组分为两个区，判断pivot的最终位置和k的大小
* 还可以使用堆来实现（维护一个长度为k的最小堆）



## 324.摆动排序II

![324.摆动排序II](.\images\324.PNG)

```python
class Solution(object):
    def wiggleSort(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        # 左中位数索引
        mid_index = (len(nums) - 1) // 2

        def partition(left, right):
            random_pivot = random.choice(range(left, right + 1))
            nums[left], nums[random_pivot] = nums[random_pivot], nums[left]

            i = left
            j = right
            pivot = left

            while i < j:
                while i < j and nums[j] > nums[pivot]:
                    j -= 1
                while i < j and nums[i] <= nums[pivot]:
                    i += 1
                nums[i], nums[j] = nums[j], nums[i]
            nums[pivot], nums[i] = nums[i], nums[pivot]

            return i

        left = 0
        right = len(nums) - 1

        while True:
            index = partition(left, right)
            if index == mid_index:
                meddian = nums[index]
                break
            if index > mid_index:
                right = index - 1
            if index < mid_index:
                left = index + 1

        # 此时还有和中位数相同的值散步在数列其他位置
        # 需要让中位数集中在中间，参考荷兰国旗

        i, j, k = 0, 0, len(nums) - 1

        while j < k:
            if nums[j] < meddian:
                nums[j], nums[i] = nums[i], nums[j]
                j += 1
                i += 1
            if nums[j] > meddian:
                nums[k], nums[j] = nums[j], nums[k]
                k -= 1
            else:
                j += 1
        print(nums)
        res = []

        # 以左中位数将数组分为前后两个部分，左中位数在前面部分
        mid = (len(nums)- 1) // 2
        i = mid
        j = len(nums) - 1

        while j > mid:
            res.append(nums[i])
            res.append(nums[j])
            i -= 1
            j -= 1
        if i == 0:
            res.append(nums[0])
        
        for i in range(len(res)):
            nums[i] = res[i]
```

### Tips

* 首先寻找中位数
* 将数组的中位数集中放中间，小于放前面，大于放后面
* 中间切断，两数组逆序后，组合成新数组
* 垃圾题目