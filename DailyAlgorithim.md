## 剑指offer 20. 表示数值的字符串

![剑指offer 20. 表示数值的字符串](.\images\jz20.png)

```python
class Solution(object):
    def isNumber(self, s):
        """
        :type s: str
        :rtype: bool
        """

        state = [
            {'.' : 4, 'b' : 0, 'd' : 2, 's' : 1},
            {'.' : 4, 'd' : 2},
            {'d' : 2, 'b' : 8, 'e': 5, '.' : 3},
            {'d' : 3, 'b' : 8, 'e': 5},
            {'d' : 3},
            {'d' : 7, 's' : 6},
            {'d' : 7},
            {'d' : 7, 'b' : 8},
            {'b' : 8},
        ]

        # 初始状态
        p = 0
        for c in s:
            if c in '0123456789':
                t = 'd'
            elif c in '+-':
                t = 's'
            elif c == '.':
                t = '.'
            elif c in 'eE':
                t = 'e'
            elif c == ' ':
                t = 'b'
            else:
                t = 'unknown'
            if t not in state[p]:
                return False
            else:
                p = state[p][t]
        
        # 最终接受状态   
        if p in[2, 3, 7, 8]:
            return True
        else:
            return False
```

### Tips

* 使用有限状态机
* ![剑指offer 20. 表示数值的字符串](.\images\jz20-1.png)





## 51.N皇后

![51.N皇后](.\images\51.png)

```python

class Solution(object):
    def solveNQueens(self, n):
        """
        :type n: int
        :rtype: List[List[str]]
        """
        borad = ['.' * n for _ in range(n)]
        res = []

        def replace_char(string, index, char):
            tmp = list(string)
            tmp[index] = char
            return ''.join(tmp)

        def isValid(row, column):
            for i in range(row):
                for j in range(n):
                    # 判断当前格子的上方，同列/左上/右上是否已经有Q
                    if (borad[i][j] == 'Q') and (j == column or i - j == row - column or i + j == row + column):
                        return False
            return True

        def dfs_withBP(row):
            # 递归返回条件
            if row == n:
                res.append(borad[:])
                return
            
            # 做选择
            for column in range(n):
                if isValid(row, column):
                    borad[row] = replace_char(borad[row], column, 'Q')
                    dfs_withBP(row + 1)
                borad[row] = replace_char(borad[row], column, '.')
        
        dfs_withBP(0)
        return res
```

### Tips

* `(borad[i][j] == 'Q') and (j == column or i - j == row - column or i + j == row + column)`判断某格子上方同列/左上/右上/是否有女王，可以用这种简单的表达式
* 带回溯的搜索



### 257.二叉树的所有路径

![257.二叉树的所有路径](.\images\257.png)

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def binaryTreePaths(self, root):
        """
        :type root: TreeNode
        :rtype: List[str]
        """
        if not root:
            return []

        res = []
        
        def dfs(node, path):
            path.append(str(node.val))
            
            if not node.left and not node.right:
                tmp = '->'.join(path)
                res.append(tmp)

            if node.left:
                dfs(node.left, path)
                path.pop()

            if node.right:
                dfs(node.right, path)
                path.pop()
        
        dfs(root, [])

        return res
```

### Tips

* 带回溯的深度搜索，递归



## 60.第K个排列

![257.二叉树的所有路径](.\images\257.png)

```python
class Solution(object):
    def getPermutation(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: str
        """
        path = []
        used = set()

        # 阶乘数列
        factorial =[1]
        for i in range(2, n + 1):
            factorial.append(factorial[-1] * i)

        def dfs(index, k, path):
            if len(path) == n:
                return
            # 在已确定前面某位的情况下，排列的种数
            count = factorial[n - 2 - index]
            for i in range(1, n + 1):
                if i in used:
                    continue
                if k > count:
                    # 直接跳过所有以i为当前位数的所有排列
                    k -= count
                    continue
                    
                path.append(str(i))
                used.add(i)
                dfs(index + 1, k, path)
                return
        dfs(0, k, path)
        return(''.join(path))
```

### Tips

* 建立阶乘数列，每次直接跳过[n - 2 - index]，节省搜索时间



## 107.二叉树的层序遍历

![107.二叉树的层序遍历](.\images\107.png)

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def levelOrderBottom(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root:
            return []
        res = []
        queue = [root]

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
            res.append(tmp)

        return res[::-1]
```

### Tips

* 从底往上，注意最后反向输出



## 347.前K个高频元素

![347.前K个高频元素](.\images\347.png)

```python
class Solution(object):
    def topKFrequent(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        freq = collections.defaultdict(int)
        for num in nums:
            freq[num] += 1
        
        temp = []
        for key, value in freq.items():
            temp.append((key, value))
        temp.sort(key=lambda x:x[1])

        return [i[0] for i in temp[::-1][:k]]
```

### Tips

* 也可以用小顶堆



## 486.预测赢家

![486.预测赢家](.\images\486.png)

```python
class Solution(object):
    def PredictTheWinner(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        
        def max_score(nums):
            total = sum(nums)
            if len(nums) == 1:
                return nums[0]
                # 代表此时的先手玩家，从头选或者从尾选，所能获得的最大分数
                # total = nums[0] + sum(nums[1:]) = nums[-1] + sum(nums[:-1])
            return max(total - max_score(nums[1:]), total - max_score(nums[:-1]))

        return max_score(nums) >= (sum(nums) - max_score(nums))
```

### Tips

* 此法超时

```python
class Solution(object):
    def PredictTheWinner(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        # dp[i][j]代表当先手玩家面临选择nums[i:j]时，所能取得的最大净胜分（自己的分 - 对手的分）
        n = len(nums)
        dp = [[0 for _ in range(n)] for _ in range(n)]

        # 对角线初始化为0
        for i in range(n):
            dp[i][i] = nums[i]

        for j in range(1, n):
            for i in range(j - 1, -1, -1):
                dp[i][j] = max(nums[i] - dp[i + 1][j], nums[j] - dp[i][j - 1])

        return (dp[0][-1] >= 0)

```

### Tips

* 为啥你每次都不想想二维动态规划
* `dp[i][j]`代表当先手玩家面临选择`nums[i:j]`时，所能取得的最大净胜分（自己的分 - 对手的分）



## 77.组合

![77.组合](.\images\77.png)

```python
class Solution(object):
    def combine(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: List[List[int]]
        """
        res = []
        used = set()
        
        
        def dfs(path):
            if len(path) == k:
                res.append(path[:])
                return 

            for i in range(1, n + 1):
                if len(path) >= 1 and i <= path[-1]:
                    continue
                # if i in used:
                #     continue
                path.append(i)
                used.add(i)
                dfs(path)
                path.pop(-1)
                used.remove(i)

        dfs([])
        return res
```

### Tips

* 带回溯的深度搜索
* 注意利用搜索的单调性，去除重复的组合



## 39.组合总和

![39.组合总和](.\images\39.png)

```python
class Solution(object):
    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        
        candidates.sort()
        res = []
        min_num = candidates[0]

        def dfs(diff, path, start_index):
            if diff == 0:
                res.append(path[:])
                return 
            
            if diff < min_num:
                return
			
            # 每次搜索的时候，只能选择自身或者自身之后的元素，因为自身之前的元素已经被选择过了
            for i in range(start_index, len(candidates)):
                if diff >= candidates[i]:
                    path.append(candidates[i])
                    # 这里用i表示还可以继续选择自己
                    dfs(diff - candidates[i], path, i)
                    path.pop()

        dfs(target, [], 0)
            
        return res
                

                

```

### Tips

* 主要注意搜索的时候， 不能再次搜索自身同层已经选择过的元素



## 40.组合综合II

![40.组合综合II](.\images\40.png)

```python
class Solution(object):
    def combinationSum2(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        res = []
        candidates.sort()

        def dfs(diff, path, start_index):
            if diff == 0:
                res.append(path[:])            
                return
            
            if diff < candidates[0]:
                return

            for i in range(start_index, len(candidates)):
                # 同一层不能选择相同的数
                if candidates[i] in candidates[start_index:i]:
                    continue
                if diff >= candidates[i]:
                    path.append(candidates[i])
                    # 同一分支是可以选择相同的数的，只不过不能是他自己
                    dfs(diff - candidates[i], path, i + 1)
                    path.pop()

        dfs(target, [], 0)
        
        return res
```

### Tips

* 和39一起，全面搞懂搜索时的去重
* 此题同一层不能用相同的重复数，同一分支是可以使用的，但是不能是他自己



## 216.组合总和III

![216.组合总和III](.\images\216.png)

```python
class Solution(object):
    def combinationSum3(self, k, n):
        """
        :type k: int
        :type n: int
        :rtype: List[List[int]]
        """

        res = []

        
        def dfs(diff, start, path):
            if len(path) == k:
                if diff == 0:
                    res.append(path[:])
                return 
            
            for i in range(start, 10):
                path.append(i)
                # 只能在后面的数里面选，不能再选前面的，因为要么已经被其他path选过，要么这个PATH里面已经用过
                dfs(diff - i, i + 1, path)
                path.pop(-1)
            
        dfs(n, 1, [])
        return res

```

### Tips

* 也是注意去重的问题



## 637.二叉树的层平均值

![637.二叉树的层平均值](.\images\637.png)

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def averageOfLevels(self, root):
        """
        :type root: TreeNode
        :rtype: List[float]
        """

        queue = [root]
        res = []
        while queue:
            size = len(queue)
            tmp = 0
            for i in range(size):
                node = queue.pop(0)
                tmp += node.val
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(float(tmp) / float(size))

        return res

```

### Tips

* 层序遍历