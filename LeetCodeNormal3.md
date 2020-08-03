## 73.矩阵置零

![73.矩阵置零](.\images\73.PNG)

```python
class Solution(object):
    def setZeroes(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: None Do not return anything, modify matrix in-place instead.
        """

        h = len(matrix)
        w = len(matrix[0])

        for i in range(h):
            for j in range(w):
                if matrix[i][j] == 0:
                    for p in range(h):
                        if matrix[p][j] != 0:
                            matrix[p][j] = float('inf')
                    for p in range(w):
                        if matrix[i][p] != 0:
                            matrix[i][p] = float('inf')
                
        for i in range(h):
            for j in range(w):
                if matrix[i][j] ==  float('inf'):
                    matrix[i][j] = 0
```

### Tips

* 使用额外标记，将本来不是0，但是因为规则变为0的数记为inf，以免引起连锁反应
* 第二次遍历，将inf改为0



## 454.四数相加II

![454.四数相加II](.\images\454.PNG)

```python
class Solution(object):
    def fourSumCount(self, A, B, C, D):
        """
        :type A: List[int]
        :type B: List[int]
        :type C: List[int]
        :type D: List[int]
        :rtype: int
        """

        withoutA = [0 - i for i in A]
        withoutAB = [[i - j for j in B] for i in withoutA]
        withoutABC = [[[i - j for j in C ]for i in k]for k in withoutAB]

        count = 0 
        
        for i in withoutABC:
            for j in i:
                for k in j:
                    for p in D:
                        if p == k:
                            count += 1

        return count
```

### Tips

* 这种算法时间复杂度为O(n) + O(n^3)，会超时间

```python
class Solution(object):
    def fourSumCount(self, A, B, C, D):
        """
        :type A: List[int]
        :type B: List[int]
        :type C: List[int]
        :type D: List[int]
        :rtype: int
        """

        AB = collections.defaultdict(int)
        CD = collections.defaultdict(int)

        for i in A:
            for j in B:
                AB[str(i + j)] += 1

        for i in C:
            for j in D:
                CD[str(- i - j)] += 1

        count = 0
        for i in AB.keys():
            if i in CD:
                count += AB[i] * CD[i]
        
        return count
```

### Tips

* 将AB和CD的和分别存入两个以和为键，以次数位置的字典
* 如果键相同，值就相乘



## 75.颜色分类

![75.颜色分类](.\images\75.PNG)

```python
class Solution(object):
    def sortColors(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """

        p0 = 0
        current = 0
        p2 = len(nums) - 1

        while current <= p2:
            if nums[current] == 0:
                nums[current], nums[p0] = nums[p0], nums[current]
                current += 1
                p0 += 1

            elif nums[current] == 2:
                nums[current], nums[p2] = nums[p2], nums[current]
                p2 -= 1

            else:
                current += 1
```

### Tips

* 使用三指针，两个从左到右，一个从右到左
* current指针遇到0往前换，遇到2往后换
* 关键是遇0两个指针都动，遇2只有右指针动
* 因为左指针一定会指向非0元素



## 138.复制带随机指针的链表

![138.复制带随机指针的链表](.\images\138.PNG)

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, x, next=None, random=None):
        self.val = int(x)
        self.next = next
        self.random = random
"""

class Solution(object):
    def __init__(self):
        self.visitedNodes = {}

    def copyRandomList(self, head):
        """
        :type head: Node
        :rtype: Node
        """

        if not head:
            return None
		# 若访问过，直接从字典读取
        if head in self.visitedNodes:
            return self.visitedNodes[head]
		# 创建一个新节点
        node = Node(head.val, None, None)
		# 将此节点加入访问过的字典
        self.visitedNodes[head] = node
		# 递归的方式添加next和random指针
        node.next = self.copyRandomList(head.next)
        node.random = self.copyRandomList(head.random)

        return node
```

### Tips

* 采用递归的方式，使用额外字典记录O(n)访问过的节点
* 深拷贝的意思就是有丝分裂



```python
"""
# Definition for a Node.
class Node:
    def __init__(self, x, next=None, random=None):
        self.val = int(x)
        self.next = next
        self.random = random
"""

class Solution(object):
    def copyRandomList(self, head):
        """
        :type head: Node
        :rtype: Node
        """
        if not head:
            return None

        p = head
        # 按照next指针遍历链表，给每个节点创建一个副本，并且串联
        while p:
            new_node = Node(p.val, None, None)
            new_node.next = p.next
            p.next = new_node
            p = p.next.next

        # 再次遍历链表，将每个副本节点的random指针正确连接
        p = head
        while p:
            if not p.random:
                p.next.random = None
            else:
                p.next.random = p.random.next
            p = p.next.next
        
        # 最后一次遍历链表，将原始节点，和新建节点完全分开，有丝分裂？
        p = head
        p_new = head.next
        start = head.next
        while p:
            p.next = p_new.next
            try:
                p_new.next = p.next.next
            except:
                break
            p = p.next
            p_new = p_new.next

        return start


        

        

```


### Tips

* 三次按照next指针遍历链表
* 第一次给每个节点创建副本
* 第二次给每个节点的副本的random指针正确连接
* 第三次把原始节点和副本节点分开
* 有丝分裂



## 103.二叉树的锯齿形层次遍历

![103.二叉树的锯齿形层次遍历](.\images\103.PNG)

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def zigzagLevelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """

        if not root:
            return []
        
        queue = [root]
        res = []
        flag = 0

        while queue:
            tmp = []
            size = len(queue)
            for i in range(size):
                node = queue.pop(0)
                tmp.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            if flag % 2 == 0:
                res.append(tmp)
            else:
                res.append(tmp[::-1])
            flag += 1

        return res
```

### Tips

* 可以使用BFS也可以使用DFS进行层序遍历



## 17.电话号码的字母组合

![17.电话号码的字母组合](.\images\17.PNG)

```python
class Solution(object):
    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """

        phone = {'2': ['a', 'b', 'c'],
                 '3': ['d', 'e', 'f'],
                 '4': ['g', 'h', 'i'],
                 '5': ['j', 'k', 'l'],
                 '6': ['m', 'n', 'o'],
                 '7': ['p', 'q', 'r', 's'],
                 '8': ['t', 'u', 'v'],
                 '9': ['w', 'x', 'y', 'z']}
        if not digits:
            return []

        if len(digits) == 1:
            return phone[digits]
        

        last = self.letterCombinations(digits[:-1])

        res = []
        for letter in phone[digits[-1]]:
            res += [i + letter for i in last]        

        return res
```

### Tips

* 递归
* 也可以使用回溯的写法



## 134.加油站

![134.加油站](.\images\134.PNG)

```python
class Solution(object):
    def canCompleteCircuit(self, gas, cost):
        """
        :type gas: List[int]
        :type cost: List[int]
        :rtype: int
        """

        trip = len(gas)

        for start in range(trip):
            tank = 0
            count = 0
            for i in range(trip):
                stop = (start + i) % trip
                tank += gas[stop]
                tank -= cost[stop]
                if tank < 0:
                    break
                count += 1
            if count == trip:
                return start
            
        return -1
```

### Tips

* 这种题比较简单

* 正向思维，判断每次加油用油之后邮箱是否小于0



## 384. 打乱数组

![384. 打乱数组](.\images\384.PNG)

```python

class Solution(object):

    def __init__(self, nums):
        """
        :type nums: List[int]
        """
        self.data = nums
        self.original = list(nums)

    def reset(self):
        """
        Resets the array to its original configuration and return it.
        :rtype: List[int]
        """
        self.data = self.original
        self.original = list(self.original)
        return self.data
        

    def shuffle(self):
        """
        Returns a random shuffling of the array.
        :rtype: List[int]
        """
        for i in range(0, len(self.data)):
            randomindex = random.randint(i, len(self.data) - 1)
            self.data[i], self.data[randomindex] = self.data[randomindex], self.data[i]

        return self.data
        
```

### Tips

* 将原始数组深度拷贝一份作为备用
* reset的时候同样深拷贝一份备用
* 使用`random.randint()`取随机整数



## 207.课程表

![207.课程表](.\images\207.PNG)

```python
# BFS
class Solution(object):
    def canFinish(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool
        """

        adjacency = [[] for _ in range(numCourses)]
        indegrees = {}

        for i in range(numCourses):
            indegrees[str(i)] = 0

        for combination in prerequisites:
            indegrees[str(combination[0])] += 1
            adjacency[combination[1]].append(combination[0])

        queue = []
        for key, value in indegrees.items():
            if not value:
                queue.append(int(key))

        count = 0
        while queue:
            course = queue.pop(0)
            count += 1
            for next_course in adjacency[course]:
                indegrees[str(next_course)] -= 1
                if not indegrees[str(next_course)]:
                    queue.append(next_course)
        
        return count == numCourses
```

### Tips

* 拓扑图遍历
* 建立入读表`indegrees`，依存表`adjacency`，使用一个队列来记录当前可以选择的课程
* 将最终遍历步数与要求进行比对



```python
# DFS
class Solution(object):
    def canFinish(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool
        """

        adjacency = [[] for _ in range(numCourses)]
        path = [0] * numCourses

        for combination in prerequisites:
            adjacency[combination[1]].append(combination[0])

        def dfs(start, adjacency, path):
            if path[start] == -1:
                return True
            if path[start] == 1:
                return False
            path[start] = 1
            for to in adjacency[start]:
                if not dfs(to, adjacency, path):
                    return False
            path[start] = -1
            return True

        for course in range(numCourses):
            if not dfs(course, adjacency, path):
                return False
        
        return True
```

### Tips

* 也可以使用DFS，建立path，按照规则走遍，如果遇到之前访问过的，说明有环
* 这道题，等价于检测有向图中是否有环

## 210.课程表II

![210.课程表](.\images\210.PNG)

```python
class Solution(object):
    def findOrder(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: List[int]
        """

        adjacency = [[] for _ in range(numCourses)]
        indegrees = [0 for _ in range(numCourses)]

        for combination in prerequisites:
            indegrees[combination[0]] += 1
            adjacency[combination[1]].append(combination[0])

        queue = []
        path = []

        for i in range(numCourses):
            if not indegrees[i]:
                queue.append(i)

        while queue:
            node = queue.pop(0)
            path.append(node)
            for next_course in adjacency[node]:
                indegrees[next_course] -= 1
                if not indegrees[next_course]:
                    queue.append(next_course)
        
        if len(path) == numCourses:
            return path
        else:
            return []
```

### Tips

* 和上题一样，只不过需要记录路径



## 150.逆波兰表达式求值

![150.逆波兰表达式求值](.\images\150.PNG)

```python
class Solution(object):
    
    def div(self,num1,num2):
        sgn = 1
        if num1 < 0:
            num1 = -num1
            sgn *= -1
        if num2 < 0:
            num2 = -num2
            sgn *= -1
        return sgn*(num1//num2)
        
    def evalRPN(self, tokens):
        """
        :type tokens: List[str]
        :rtype: int
        """   
        stack = []
        for c in tokens:
            if c in '+-*':
                tmp1 = stack.pop()
                tmp2 = stack.pop()
                stack.append(str(int(eval(tmp2 + c + tmp1))))
            elif c in '/':          
                tmp1 = stack.pop()
                tmp2 = stack.pop()
                stack.append(str(self.div(int(tmp2), int(tmp1))))
            else:
                stack.append(c)
      
        return int(stack[0])

```

### Tips

* 使用栈

* 这里注意，在python里面除法是个坑，异号相除时，取整有问题



## 146.LRU缓存机制

![146.LRU缓存机制](.\images\146.PNG)

```python
class DlinkedNodes():
    def __init__(self, key, val):
        self.key = key
        self.val = val
        self.next = None
        self.prev = None

class LRUCache(object):

    def __init__(self, capacity):
        """
        :type capacity: int
        """
        self.capacity = capacity
        self.table = {}
        self.vhead = DlinkedNodes(0, 0)
        self.vtail = DlinkedNodes(0, 0)
        self.vhead.next = self.vtail
        self.vtail.prev = self.vhead
        self.size = 0        

    def get(self, key):
        """
        :type key: int
        :rtype: int
        """
        if key not in self.table:
            return -1
        node = self.table[key]   
        self.moveTohead(node)
    
        return node.val



    def put(self, key, value):
        """
        :type key: int
        :type value: int
        :rtype: None
        """
        if key not in self.table:
            node = DlinkedNodes(key, value)
            self.table[key] = node
            self.size += 1
            if self.size > self.capacity:
                self.table.pop(self.vtail.prev.key)
                self.removeNode(self.vtail.prev)
            self.addTohead(node)
            

        else:
            node = self.table[key]
            node.val = value
            self.moveTohead(node)

        return node.val
    
    def moveTohead(self, node):
        self.removeNode(node)
        self.addTohead(node)

    def removeNode(self, node):
        node.next.prev = node.prev
        node.prev.next = node.next
    
    def addTohead(self, node):
        node.next = self.vhead.next
        node.prev = self.vhead
        node.next.prev = node
        self.vhead.next = node
```

### Tips

* 使用哈希表 + 双向链表，哈希表的值指向链表中的节点
* 当使用或者插入或者修改某个值后，将这个节点放置于链表头部



## 200.岛屿数量

![200.岛屿数量](.\images\200.PNG)

```python
class Solution(object):
    def numIslands(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """

        def dfs(grid, r, c):
            grid[r][c] = '0'
            for new_r, new_c in [[(r - 1), c], [r, (c - 1)], [r, (c + 1)], [(r + 1), c]]:
                if 0 <= new_c < len(grid[0]) and 0 <= new_r < len(grid) and grid[new_r][new_c] == '1':
                    dfs(grid, new_r, new_c)

        if not grid:
            return 0

        count = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == '1':
                    dfs(grid, i, j)
                    count += 1
                    
        return count
            
        
```

### Tips

* DFS比较方便
* 遍历所有元素，遇到1就按照规则，计数，并使用递归，把所有相邻的元素全部变为0



## 380.常数时间插入、删除和获取随机元素

![380.常数时间插入、删除和获取随机元素](.\images\380.PNG)

```python
class RandomizedSet(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.data = []
        self.table = {}
        self.length = 0


    def insert(self, val):
        """
        Inserts a value to the set. Returns true if the set did not already contain the specified element.
        :type val: int
        :rtype: bool
        """
        if val not in self.table:
            self.data.append(val)
            self.table[val] = self.length
            self.length += 1
            return True
        else:
            return False


    def remove(self, val):
        """
        Removes a value from the set. Returns true if the set contained the specified element.
        :type val: int
        :rtype: bool
        """
        if val in self.table:
            index = self.table[val]
            self.table[self.data[-1]] = index
            self.data[index], self.data[-1] = self.data[-1], self.data[index]
            self.data.pop()
            self.length -= 1
            self.table.pop(val)
            return True
        
        else:
            return False


    def getRandom(self):
        """
        Get a random element from the set.
        :rtype: int
        """
        i = random.randint(0, self.length - 1)
        return self.data[i]
```

### Tips

* 使用哈希 + 散列表
* 删除元素的时候，将列表最后一位和删除的值位置互换，注意维护一些变量的变化



## 162.寻找峰值

![162.寻找峰值](.\images\162.PNG)

```python
class Solution(object):
    def findPeakElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        
        left = 0
        right = len(nums) - 1

        while left < right:
            mid = left + (right - left) // 2

            if nums[mid] > nums[mid + 1]:
                right = mid
            
            else:
                left = mid + 1

        return left
```

### Tips

* 遇到O(Logn)就肯定和二分查找有关
* 将`mid`和`mid + 1`比较，如果下降，说明找过头了，如果上升，说明小了
* 最终至少能找到一个峰值
* 使用左中位数



## 139.单词拆分

![139.单词拆分](.\images\139.PNG)

```python
class Solution(object):
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: bool
        """

        dp = [False] * (len(s) + 1)
        
        dp[0] = True

        for i in range(1, len(s) + 1):
            for j in range(i):
                if dp[j] and s[j:i] in wordDict:
                    dp[i] = True

        return dp[-1]
```

### Tips

* 动态转移方程 `dp[i] = dp[j] && check(s[j : i])`
* 注意DP数组的初始化，0号元素代表空字符串，需要设定为True
* 代表的意思就是考察此位置的所有字母组合在一起，`s[:i]`是否在字典里



## 300.最长上升子序列

![300.最长上升子序列](.\images\300.PNG)

```python
# 动态规划
class Solution(object):
    def lengthOfLIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return 0

        dp = [1] * len(nums)
        
        for i in range(len(nums)):
            for j in range(0, i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i], dp[j] + 1)
        
        return max(dp)

```

### Tips

* 动态转移方程`dp[i] = max(dp[j]) + 1,其中0 ≤j <i且 num[j] < num[i]`
* `dp[i]`代表以第i个元素结尾的上升子序列最长长度
* 时间复杂度O(n2)

```python
class Solution(object):
    def lengthOfLIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """

        if not nums:
            return 0

        tail = [nums[0]]

        for i in nums[1:]:
            # 判断此数是否大于此单调递增数组的最后一个元素
            if i > tail[-1]:
                tail.append(i)

            # 否则，进行二分查找
            else:
                left = 0
                right = len(tail) - 1

                while left < right:
                    # 取左中位数，详见leetcode 35
                    mid = left + (right - left) // 2

                    if tail[mid] < i:
                        left = mid + 1

                    else:
                        right = mid

                tail[left] = i

        return len(tail)
```

### Tips

* 优化后的动态规划
* `tail[i]`代表长度为`i + 1`的子序列中，结尾元素最小的那个序列的结尾元素，`tail`数组是单调递增的
* 遍历`nums`，如果`nums[i]`小于`tail`的最后一个元素，则把`tail`中从左到右第一个大于`nums[i]`的元素替换为`nums[i]`
* 所以在有序数组中的查找，可以使用二分法
* 时间复杂度O(nlogn)



## 395.至少有K个重复字符的最长子串

![395.至少有K个重复字符的最长子串](.\images\395.PNG)

```python
# 分治递归
class Solution(object):
    def longestSubstring(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: int
        """
        if len(s) < k:
            return 0

        for c in set(s):
            # 如果某个单词的数量少于k，则这个单词一定不能在最终子串之中，则以这个单词分割成子串
            if s.count(c) < k:
                return max(self.longestSubstring(sub, k) for sub in s.split(c))
        
        return len(s)
```

### Tips

* 分治 + 递归



## 127.单词接龙

![127.单词接龙](.\images\127.PNG)

```python
# BFS
class Solution(object):
    def ladderLength(self, beginWord, endWord, wordList):
        """
        :type beginWord: str
        :type endWord: str
        :type wordList: List[str]
        :rtype: int
        """
        if not wordList:
            return 0

        midPhase = collections.defaultdict(list)
		
        # 键为某种中间状态，值为拥有这些中间状态的单词
        for word in wordList:
            for i in range(len(word)):
                midPhase[word[:i] + '*' + word[i + 1:]].append(word)
        
        # 使用队列
        queue = [(beginWord, 1)]
        # 记录已经访问过的单词，防止二次访问
        visited = [beginWord]

        while queue:
            word, level = queue.pop(0)
            for i in range(len(word)):
                midphase = word[:i] + '*' + word[i + 1:]
                # 使用共同的中间状态，在单词直接转换
                for new_word in midPhase[midphase]:
                    if new_word == endWord:
                        return level + 1
                    if new_word not in visited:
                        queue.append((new_word, level + 1))
                        visited.append(new_word)
        
        return 0
```

### Tips

* 使用BFS，建立字典，每个字典的键为某种单词中间状态
* 注意记录已经访问过的单词
* 还可以双向BFS，即从头和尾同时往中间搜索



## 56.合并区间

![56.合并区间](.\images\56.PNG)

```python
class Solution(object):
    def merge(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: List[List[int]]
        """

        if not intervals:
            return []

        intervals.sort(key=lambda x: x[0])

        merged = [intervals[0]]
        for i in range(1, len(intervals)):
            if intervals[i][0] > merged[-1][1]:
                merged.append(intervals[i])
            else:
                merged.append([merged[-1][0], max(merged[-1][1], intervals[i][1])])
                merged.pop(-2)
        
        return merged
```

### Tips

* 首先按照每个元素的第一个元素排序
* 逐个合并区间



