## 236.二叉树的最近公共祖先

![236.二叉树的最近公共祖先](.\images\236.PNG)

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        # 递归终止条件
        if not root or root == p or root == q:
            return root
        
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)

        if not left:
            return right
        
        if not right:
            return left

        # if not left and not right:
        #     return None
        # 可合并至上方

        # if left and right:
        return root
```

### Tips

* 当我们用递归去做这个题时不要被题目误导，应该要明确一点
  **这个函数的功能有三个**：给定两个节点 p 和 q
  1. 如果 p 和 q 都存在，则返回它们的公共祖先；
  2. 如果只存在一个，则返回存在的一个
  3. 如果 p 和 q 都不存在，则返回NULL
* 这个函数从上往下递归，当遇到第一个目标p或者q的时候就不往下递归了，返回
* 可想而知，这样两个值会被一直搬运至某个节点，left或者right都不为空的时候停止



## 341.扁平化嵌套列表迭代器

![341.二叉树的最近公共祖先](.\images\341.PNG)

```python
# """
# This is the interface that allows for creating nested lists.
# You should not implement it, or speculate about its implementation
# """
#class NestedInteger(object):
#    def isInteger(self):
#        """
#        @return True if this NestedInteger holds a single integer, rather than a nested list.
#        :rtype bool
#        """
#
#    def getInteger(self):
#        """
#        @return the single integer that this NestedInteger holds, if it holds a single integer
#        Return None if this NestedInteger holds a nested list
#        :rtype int
#        """
#
#    def getList(self):
#        """
#        @return the nested list that this NestedInteger holds, if it holds a nested list
#        Return None if this NestedInteger holds a single integer
#        :rtype List[NestedInteger]
#        """

class NestedIterator(object):

    def __init__(self, nestedList):
        """
        Initialize your data structure here.
        :type nestedList: List[NestedInteger]
        """
        
        def toBottom(nested):
            res = []
            
            for i in nested:
                if i.isInteger():
                    res.append(i.getInteger())
                else:
                    res += toBottom(i.getList())
            
            return res
            

        self.data = toBottom(nestedList)
        self.length = len(self.data)
        self.count = 0
                   

    def next(self):
        """
        :rtype: int
        """

        if self.hasNext:
            self.count += 1
            return self.data[self.count - 1]
      

    def hasNext(self):
        """
        :rtype: bool
        """
        return self.count < self.length
        

# Your NestedIterator object will be instantiated and called as such:
# i, v = NestedIterator(nestedList), []
# while i.hasNext(): v.append(i.next())
```



### Tips

* 注意`NestedInteger`类的API，测试例里面的所有元素，无论是整数还是列表，都被这个类给封装了



## 11.盛最多水的容器

![11.盛最多水的容器](.\images\11.PNG)

```python
class Solution(object):
    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """

        left = 0 
        right = len(height) - 1

        maxVol = 0

        while left < right:
            maxVol = max(maxVol, (right - left) * min(height[left], height[right]))
            if height[left] <= height[right]:
                left += 1
            else:
                right -= 1
            
        return maxVol

```



### Tips

* 双指针
* 一个从前往后，一个从后往前，记录当前最大容积，然后较小的一边往中间动
* 因为每一步实际上都记录了以这条小边为界的最大可能的容积



## 102.二叉树的层序遍历

![102.二叉数的层序遍历](.\images\102.PNG)



```python
# 使用BFS的迭代
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """

        if not root:
            return []
        
        queue = [root]
        final = []

        while queue:
            # 记录这一层的长度
            size = len(queue)
            res = []
            for i in range(size):
                tmp = queue.pop(0)
                res.append(tmp.val)

                if tmp.left:
                    queue.append(tmp.left)
                if tmp.right:
                    queue.append(tmp.right)
            final.append(res)
        
        return final
            
```

### Tips

* 使用队列，前出后进
* 如果结果需要分层，那需要记录当前层节点个数，也就是队列的长度

```python
# DFS递归
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root:
            return []
        
        res = []

        def dfs(level, root):
            if level > len(res):
                res.append([])
            
            res[level - 1].append(root.val)

            if root.left:
                dfs(level + 1, root.left)
            if root.right:
                dfs(level + 1, root.right)
        
        dfs(1, root)

        return res
```

### Tips

* DFS使用递归
* 每到下一层，如果当前层的结果列表还不存在，结果就增加一个空列表



## 328.奇偶链表

![328.奇偶链表](.\images\328.PNG)



```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def oddEvenList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head:
            return None
        odd = head
        even = head.next
        rec = head.next

        while even and even.next:
            odd.next = even.next
            odd = odd.next
            even.next = odd.next
            even = even.next

        # 此时odd在even之前，要么even是个none, 或者even下一个是none

        odd.next = rec

        return head
```

### Tips

* 两个指针，奇数指针和偶数指针互相攀附前进
* 注意循环结束条件，因为每一个循环结束奇数指针在前，偶数指针在后，所以只要偶数指针或偶数指针的next为空，就结束循环
* 最后注意返回



### 49.字母异位词分组

![49.字母异位词分组](.\images\49.PNG)

```python
class Solution(object):
    def groupAnagrams(self, strs):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """

        record = collections.defaultdict(list)
        for s in strs:
            chash = [0] * 26
            for c in s:
                chash[ord(c) - ord('a')] += 1
            record[tuple(chash)].append(s)

        return record.values()
```

### Tips

* 将每个字符串按照字母出现次数hash之后再存入一个以hash值为键的字典之中



## 378.有序矩阵中第k小的元素

![378.有序矩阵中第k小的元素](.\images\378.PNG)

```python
class Solution(object):
    def kthSmallest(self, matrix, k):
        """
        :type matrix: List[List[int]]
        :type k: int
        :rtype: int
        """
        n = len(matrix[0])

        def check(mid):
            i = n - 1
            j = 0
            count = 0
            while i >= 0 and j <= n - 1:
                # 这里统计的是小于等于，也就是不大于
                if matrix[i][j] <= mid:
                    j += 1
                    count = count + i + 1
                else:
                    i -= 1
            # 当以mid为界限的时候，小于mid的数的数量与k比较
            return count

        left = matrix[0][0]
        right = matrix[-1][-1]

        while left < right:
            # 使用左中位数
            mid = left + (right - left) // 2

            if check(mid) >= k:
                right = mid 

            else:
                left = mid + 1

        return left
```

### TIPs

* 在值域中选择一个数，按照特定的走法，（遇见小的就往右，遇见大的就往上），可以统计出矩阵中小于等于这个数的元素的数量
* 二分搜索，注意此题最好使用左中位数模板，使用右中位数模板可能出错
* 矩阵可从左下角走，可从右上角走



## 116.填充每个节点的下一个右侧节点指针

![116.填充每个节点的下一个右侧节点指针](.\images\116.PNG)

```python
"""
# Definition for a Node.
class Node(object):
    def __init__(self, val=0, left=None, right=None, next=None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
"""

class Solution(object):
    def connect(self, root):
        """
        :type root: Node
        :rtype: Node
        """
        if not root:
            return None

        queue = [root]
        head = root
        while queue:
            size = len(queue)
            tmp = []
            
            for i in range(size):
                if queue[0].left:
                    queue.append(queue[0].left)
                    queue.append(queue[0].right)
                # 如果到了该层最后一个
                if i == size - 1:
                    queue[0].next = None
                else:
                    queue[0].next = queue[1]                   
                queue.pop(0)
        
        return head
```

### Tips

* 使用队列，层序遍历

```python
"""
# Definition for a Node.
class Node(object):
    def __init__(self, val=0, left=None, right=None, next=None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
"""

class Solution(object):
    def connect(self, root):
        """
        :type root: Node
        :rtype: Node
        """
        if not root:
            return None

        current_level_head = root

        while current_level_head.left:
            p = current_level_head

            while p:
                # 第一种链接
                p.left.next = p.right
                # 第二种链接，利用上一层已经搭好的桥
                if p.next:
                    p.right.next = p.next.left
                # 继续在本层向右
                p = p.next
            # 下一层
            current_level_head = current_level_head.left

        return root
```

### Tips

* 逐层迭代，利用上一层搭好的桥
* 时间O(n)，空间O(1)



## 62.不同路径

![62.不同路径](.\images\62.PNG)

```python
class Solution(object):
    def uniquePaths(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """

        # m列,n行
        global res
        res = 0
        
        def dfs(right, down):
            if right == m and down == n:
                global res
                res += 1
                return 

            if right <= m - 1:
                dfs(right + 1, down)
                
                
            if down <= n - 1:
                dfs(right, down + 1)
                

        dfs(1, 1)
        
        return res
```

### Tips

* 使用深度搜索的回溯会超时，回溯的精髓在于全局变量需要回溯，如`paths`，而一些指针是不需要回溯的

```python
class Solution(object):
    def uniquePaths(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """

        # m列,n行
        res = [1] * m

        for i in range(1, n):
            for j in range(1, m):
                res[j] += res[j - 1]
        return res[-1]
```

### Tips

* 动态规划，转移方程`f[i][j] = f[i - 1][j] + f[i][j - 1]`
* 上图写法是优化后的结果



## 36.有效的数独

![36.有效的数独](.\images\36-1.PNG)

![36.有效的数独](.\images\36-2.PNG)

```python
class Solution(object):
    def isValidSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: bool
        """
        row_dict = [{} for i in range(9)]
        column_dict = [{} for i in range(9)]
        box_dict = [{} for i in range(9)]

        for i in range(9):    
            for j in range(9):
                value = board[i][j]
                if value != '.':
                    if value not in row_dict[i] and value not in column_dict[j] and value not in box_dict[(i // 3) * 3 + (j // 3)]:
                        row_dict[i][value] = True
                        column_dict[j][value] = True
                        box_dict[(i // 3) * 3 + (j // 3)][value] = True
                    else:
                        return False

        return True
```

### Tips

* 一次遍历，创建三个散列表，分别储存当前列，当前行，当前方块的数字个数，发现重复的时候返回false



## 279.完全平方数

![279.完全平方数](.\images\279.PNG)

```python
class Solution(object):
    
    def numSquares(self, n):
        """
        :type n: int
        :rtype: int
        """

        smallerSqrts = [i ** 2 for i in range(1, int(n ** 0.5) + 1)]
        print(smallerSqrts)

        level = 0
        queue = [n]

        while queue:
            level += 1
            new_queue = set()
            for num in queue:
                for sqrs in smallerSqrts:
                    residual = num - sqrs
                    if residual == 0:
                        return level
                    # 说明这个平方数比这个残差大，直接去检查下一个残差
                    if residual < 0:
                        break
                    else:
                        new_queue.add(residual)
            queue = new_queue

        return level
```

### Tips

* 枚举小于等于n的所有平方数
* 在这些平方数里面搜索最小组合，构建多叉数，每下一层都等于目标减去某个平方数的差
* 当某层达到0的时候，就是最小的组合



![279.完全平方数](.\images\279-1.PNG)