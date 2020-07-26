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



## 

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



