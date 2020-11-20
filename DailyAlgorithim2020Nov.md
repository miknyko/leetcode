## 349.两个数组的交集

![349.两个数组的交集](./images/349.png)

```python 
class Solution(object):
    def intersection(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        
        rec1 = collections.defaultdict(int)
        rec2 = collections.defaultdict(int)

        for num in nums1:
            rec1[num] += 1
        
        for num in nums2:
            rec2[num] += 1

        res = []

        for i in rec1.keys():
            if i in rec2:
                res.append(i)

        return res
        
```

### Tips

* 构建哈希表



## 941. 有效得山脉数组

![941. 有效得山脉数组](./images/941.png)

```python
class Solution(object):
    def validMountainArray(self, A):
        """
        :type A: List[int]
        :rtype: bool
        """

        i = 0
        n = len(A)

        while i < n - 1 and A[i] < A[i + 1]:
            i += 1

        # 封顶不能是头尾
        if i == 0 or i == n - 1:
            return False

        while i < n - 1 and A[i] > A[i + 1]:
            i += 1

        if i != n - 1:
            return False

        return True
```

### Tips

* 线性扫描列表，使用while循环
* 峰顶不能是头尾



## 57.插入区间

![57.插入区间](./images/57.png)

```python
class Solution(object):
    def insert(self, intervals, newInterval):
        """
        :type intervals: List[List[int]]
        :type newInterval: List[int]
        :rtype: List[List[int]]
        """
        n = len(intervals)

        # 新加区域是否完全超过了此区域
        def is_after(area1, area2):
            if area1[0] > area2[1]:
                return True
        # 新加区域与该区域是否重合
        def is_overlayed(area1, area2):
            if area1[0] > area2[1] or area2[0] > area1[1]:
                return False
            else:
                return True
        # 合并重合区域
        def merge(area1, area2):
            new_area = []
            new_area.append(min(area1[0], area2[0]))
            new_area.append(max(area1[1], area2[1]))
            
            return new_area

        # 寻找第一个重合区域索引
        i = 0
        while i < n and is_after(newInterval, intervals[i]):
            i += 1
        
        # 寻找第一个脱离重合的区域索引
        j = i
        temp = newInterval
        while j < n and is_overlayed(newInterval, intervals[j]):
            temp = merge(temp, intervals[j])
            j += 1

        # 切片返回
        return intervals[:i] + [temp] + intervals[j:]

```

### Tips

* 遍历，寻找第一个重合区域，记录索引i
* 开始合并，并继续遍历，直到脱离重合
* 切片返回



## 1356.根据数字二进制下1的数目排序

![1356.根据数字二进制下1的数目排序](./images/1356.png)

```python
class Solution(object):
    def sortByBits(self, arr):
        """
        :type arr: List[int]
        :rtype: List[int]
        """
        result = arr[:]

        def count_one(num):
            res = 0
            while num:
                res += num % 2
                num = num / 2
            return res
        
        result.sort()
        result.sort(key=lambda x:count_one(x))

        return result
```

### Tips:

* 利用python两次排序不会改变第一次排序位置的特性，进行双键排序



## 973. 最接近远点的K个点

![973. 最接近远点的K个点](./images/973.png)

```python
class Solution(object):
    def kClosest(self, points, K):
        """
        :type points: List[List[int]]
        :type K: int
        :rtype: List[List[int]]
        """
        n = len(points)
        points_t = points[:]

        def get_distance(point):
            return point[0] ** 2 + point[1] ** 2
        
        # 对points[left, right]根据distance排序
        def partition(left, right):
            l = left
            r = right

            pivot = left

            while l < r:
                while l < r and get_distance(points_t[r]) > get_distance(points_t[pivot]):
                    r -= 1
                while l < r and get_distance(points_t[l]) <= get_distance(points_t[pivot]):
                    l += 1
                points_t[l], points_t[r] = points_t[r], points_t[l]

            points_t[left], points_t[l] = points_t[l], points_t[pivot]

            return l

        
        # 开始二分查找
        left = 0
        right = n - 1

        while True:
            index = partition(left, right)
            # 一定注意这里的逻辑控制！！！！！！！！！不然一直出错，想清楚！
            if index == K - 1:
                return points_t[:K]
            if index > K - 1:
                right = index - 1
            else:
                left = index + 1

```

### Tips

* 类似数组第k大的元素
* 进行二分搜索



## 31.下一个排列

![31.下一个排列](./images/31.png)

```python
class Solution(object):
    def nextPermutation(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """

        n = len(nums)

        i = n - 2

        while i >= 0 and nums[i] >= nums[i + 1]:
            i -= 1
        # 找到第一个上升序列（从前看）
        # 此时记录当前索引i，然后再此后往前开始寻找第一个大于i的数索引j
        # 如果本来数列就是上升数列，此时i为-1, 直接跳过下面步骤

        if i >= 0:
            j = n - 1
            while j >= 0 and nums[j] <= nums[i]:
                j -= 1
            # 进行交换
            nums[i], nums[j] = nums[j], nums[i]

        # 然后将索引i之后的数进行升序排列
        # 因为[i + 1, n - 1]一定是一个降序数列，所以首尾交换即可

        left = i + 1
        right = n - 1

        while left < right:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
            right -= 1
```

### Tips

* 两遍 ***从后往前*** 扫描
* 第一遍找到第一个升序，第二遍找到第一个大于i的j，对换
* 然后将i之后的数升序排列



## 514.自由之路

![514.自由之路](./images/514.png)

```python
class Solution(object):
    def findRotateSteps(self, ring, key):
        """
        :type ring: str
        :type key: str
        :rtype: int
        """
        ring_length = len(ring)
        key_length = len(key)

        # 记录key中每个单词在ring中的所有索引位置
        position = collections.defaultdict(list)
        for c in key:
            for i, w in enumerate(ring):
                if w == c:
                    position[c].append(i)
        
        dp = [[float('inf') for _ in range(ring_length)] for _ in range(key_length)]

        # 遍历(key的首字母在ring中的所有索引)，初始化第一行
        for i in position[key[0]]:
            dp[0][i] = min(i, ring_length - i) + 1

        for i in range(1, key_length):
            for j in position[key[i]]:
                # 遍历上一行
                for k in position[key[i - 1]]:
                    dp[i][j] = min(dp[i][j], dp[i - 1][k] + min(abs(j - k), ring_length - abs(j - k)) + 1)
        
        return min(dp[i])
```

### Tips

* `dp[i][j]`定义为ring的第J个字母对正的时候，完成了KEY的前I个字母的拼写，所需要的最小步数
* ![514.自由之路](C:/Users/42338/leetcode/images/514-2.png)



## 922.按奇偶排序数组II

![922.按奇偶排序数组II](./images/922.png)

```python
class Solution(object):
    def sortArrayByParityII(self, A):
        """
        :type A: List[int]
        :rtype: List[int]
        """

        def is_odd(number):
            return number % 2 == 1

        odd_nums = []
        even_nums = []

        for num in A:
            if is_odd(num):
                odd_nums.append(num)
            else:
                even_nums.append(num)

        res = []
        for i in range(len(A) / 2):
            res.append(even_nums[i])
            res.append(odd_nums[i])

        return res  
```

### Tips

* 没什么tips





## 1122.数组的相对排序

![1122.数组的相对排序](./images/1122.png)

```python
class Solution(object):
    def relativeSortArray(self, arr1, arr2):
        """
        :type arr1: List[int]
        :type arr2: List[int]
        :rtype: List[int]
        """
        
        # 构建哈希表，键为元素值，值为索引
        rank = {}
        for i, number in enumerate(arr2):
            rank[number] = i

        # 构建自定义排序函数
        # 若两个值都在哈希表中，则根据元组第二个元素，即是他们的索引大小来排序
        # 若只有其中一个在哈希表，则没在表中的元素更大
        # 若两个都不在哈希表，则根据他们本身的值来排序
        def order(number):
            return (0, rank[number]) if number in rank else (1, number)

        arr1.sort(key=order)

        return arr1
```



### Tips

* 自定义排序函数，利用内置`sort()`函数排序



## 406. 根据身高重建队列

![406. 根据身高重建队列](./images/406.png)

```python
class Solution(object):
    def reconstructQueue(self, people):
        """
        :type people: List[List[int]]
        :rtype: List[List[int]]
        """
		# 首先按照身高第一序，以及位置第二序排列
        people.sort(key=lambda x: (-x[0], x[1]))
        res = []
        # 依次将排序的结果插入结果数组中
        # 可以这么想，后插入的任何元素，随便X什么位置，对于已经插入的元素来说，都是合规合法的
        # 因为后插入的元素一定比之前的元素小（就算相同的元素，我们也可以认为后来的仍然是小那么一点点）
        # 所以不会影响他的位置序列
        # 所以每个元素只要按照他的序列号插入（元组第二个元素），就可以保证他本身的合法性
        for person in people:
            res.insert(person[1], person)
        return res
```

### Tips

* 有点东西



## 1030.距离顺序排列矩阵单元格

![1030.距离顺序排列矩阵单元格](./images/1030.png)

```python
class Solution(object):
    def allCellsDistOrder(self, R, C, r0, c0):
        """
        :type R: int
        :type C: int
        :type r0: int
        :type c0: int
        :rtype: List[List[int]]
        """

        directions = [(1, 1), (1, -1), (-1, -1), (-1, 1)]
        res = [(r0, c0)]

        # 矩阵中有可能的最远曼哈顿距离
        max_distance = max(r0, R - 1- r0) + max(c0, C - 1 - c0)

        # 起始坐标
        column = c0
        row = r0
        # 按照曼哈顿距离遍历所有点
        for distance in range(1, max_distance + 1):
            row -= 1
            # 从上顶点开始，顺时针遍历相同距离点
            for i, (dr, dc) in enumerate(directions):
                # 沿着这条边一直走，一直把这条边走完
                # 判断条件很关键
                while (i % 2 == 0 and row != r0) or (i % 2 != 0 and column != c0):
                    # 如果在矩阵范围内,则添加此点
                    if 0 <= row < R and 0 <= column < C:
                        res.append([row, column])
                    # 继续沿着这条边走
                    row += dr
                    column += dc
        
        return res
```

### Tips

* 根据距离来遍历所有点

* 主要注意如何判断“沿着这条边一直走”

* 也可以暴力遍历之后按照距离排序




## 147.对链表进行插入排序

![147.对链表进行插入排序](./images/147.png)

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def insertionSortList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """

        if not head:
            return None

        dummy_head = ListNode(float('-inf'))
        dummy_head.next = head

        last_sorted = head
        current = head.next

        while last_sorted.next:
            if current.val >= last_sorted.val:
                last_sorted = last_sorted.next
                current = current.next

            else:
                # 首先将这个小值节点拿出来
                last_sorted.next = current.next

                # 然后从头开始遍历，寻找插入点
                insert = dummy_head
                while current.val >= insert.next.val:
                    insert = insert.next
                current.next = insert.next
                insert.next = current

                # current继续往后走
                # last sorted 不动
                current = last_sorted.next

        return dummy_head.next

```

### Tips

* 每次遇到更小值需要把他插到前面的时候，需要重新从前往后进行遍历，寻找插入位置