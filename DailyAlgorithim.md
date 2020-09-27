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



## 37.解数独

![37.解数独](.\images\37.png)

```python
class Solution(object):
    def solveSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: None Do not return anything, modify board in-place instead.
        """

        row_rec = [[False for _ in range(9)] for _ in range(9)]
        column_rec= [[False for _ in range(9)] for _ in range(9)]
        box_rec = [[False for _ in range(9)] for _ in range(9)]

        for i in range(9):
            for j in range(9):
                if board[i][j] != '.':
                    digit = int(board[i][j])
                    row_rec[i][digit - 1] = True
                    column_rec[j][digit - 1] = True
                    box_rec[(i // 3) * 3 + (j // 3)][digit - 1] = True
        
        def dfs(board, i, j):
            if j == 9:
                return dfs(board, i + 1, 0)
            if i == 9:
                return True 

            if board[i][j] == '.':
                for num in range(1, 10):
                    if not row_rec[i][num - 1] and not column_rec[j][num - 1] and not box_rec[(i // 3) * 3 + (j // 3)][num - 1] :
                        board[i][j] = str(num)
                        row_rec[i][num - 1] = True
                        column_rec[j][num - 1] = True
                        box_rec[(i // 3) * 3 + (j // 3)][num - 1] = True
                        if dfs(board, i, j + 1):
                            return True
                        row_rec[i][num - 1] = False
                        column_rec[j][num - 1] = False
                        box_rec[(i // 3) * 3 + (j // 3)][num - 1] = False
                        board[i][j] = '.'
            else:
                return dfs(board, i, j + 1)
            return False

        
        dfs(board, 0, 0)
                        

                        
```

### Tips

* 回溯，但是这种回溯和我们以前的写法有点不一样，给递归函数添加一个布尔返回值判断是否继续往下搜索
* 中间往后递归的时候，需要判断dfs(n + 1)的返回值



## 226.翻转二叉树

![226.翻转二叉树](.\images\226.png)

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def invertTree(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """

        if not root:
            return []

        queue = [root]

        while queue:
            node = queue.pop()
            tmp = node.left
            node.left = node.right
            node.right = tmp

            if node.left:
                queue.append(node.left)

            if node.right:
                queue.append(node.right)

        return root
```

### Tips

* BFS迭代
* 也可以用递归



## 47. 全排列II

![47. 全排列II](.\images\47.png)

```python
class Solution(object):
    def permuteUnique(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        nums.sort()

        res = []
        n = len(nums)
        used = [0 for _ in range(n)]

        def dfs(used, path):
            if len(path) == n:
                res.append(path[:])
                return 

            for i in range(n):
                
                if used[i] == 1:
                    continue
                # 当待选择的数字和排他前面的数字一样，并且他前面的这个数字此轮没有被选过
                # 说明同一层的和他相同元素已经被选择过
                if i > 0 and nums[i] == nums[i - 1] and used[i - 1] == 0:
                    continue
                path.append(nums[i])
                used[i] = 1
                dfs(used, path)
                used[i] = 0
                path.pop()

        dfs(used, [])

        return res
```

### TIPs

* 同一分支可以选择相同的数
* 同一层不能选择相同的数
* 注意去重



## 404.左叶子之和

![404.左叶子之和](.\images\404.png)

```python
class Solution:
    def sumOfLeftLeaves(self, root: TreeNode) -> int:
        isLeafNode = lambda node: not node.left and not node.right

        def dfs(node: TreeNode) -> int:
            ans = 0
            if node.left:
                ans += node.left.val if isLeafNode(node.left) else dfs(node.left)
            if node.right and not isLeafNode(node.right):
                ans += dfs(node.right)
            return ans
        
        return dfs(root) if root else 0

```

### Tips

* 多种解法
* 定义一个辅助函数判断节点是否是叶子很有用



## 538.把二叉搜索树转换为累加树

![538.把二叉搜索树转换为累加树](.\images\538.png)

```python
class Solution:
    def convertBST(self, root: TreeNode) -> TreeNode:
        cur = 0

        def dfs(root):
            nonlocal cur
            if not root: return 
            dfs(root.right)
            cur += root.val
            root.val = cur
            dfs(root.left)

        dfs(root)
        return root
```

### Tips

* 使用递归，反中序遍历BST
* 使用一个全局变量total， 记录累加值



## 968.监控二叉树

![968.监控二叉树](.\images\968.png)

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def minCameraCover(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        # 信号从底往上返回
        # 每个节点根据子节点的信号，相应的往他的父节点返回对应的信号

        # 0 没装，但是被检测到了
        # 1 没装，也没被检测到
        # 2 装了
        self.result = 0

        def dfs(root):
            # 关于空节点返回值的设定，只要不要和其他的冲突就好
            if not root:
                return 0

            left_state = dfs(root.left)
            right_state = dfs(root.right)
            # 信号从底往上返回
            # 这种情况此节点必须装一个摄像头
            if left_state == 1 or right_state == 1:
                self.result += 1
                return 2
            # 这种情况此节点可以不装
            if left_state == 2 or right_state == 2:
                return 0
            # 节点说“反正我没被检测到，我也不知道该不该装，爸爸你看着办吧”
            else:
                return 1
        
        if dfs(root) == 1:
            self.result += 1

        return self.result
```

### Tips

* 这种深度搜索结果一定是最小的摄像头数量



## 617.合并二叉树

![617.合并二叉树](.\images\617.png)

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def mergeTrees(self, t1, t2):
        """
        :type t1: TreeNode
        :type t2: TreeNode
        :rtype: TreeNode
        """

        if not t1 and not t2:
            return None
      
        if t1 and t2:
            node = TreeNode(t1.val + t2.val)
            node.left = self.mergeTrees(t1.left, t2.left)
            node.right = self.mergeTrees(t1.right, t2.right)
            return node
        
        else:
            node = t1 if t1 else t2
            return node
```

### Tips

* 递归
* 注意只有两棵树的该节点都存在的时候， 才往下继续递归



## 501.二叉搜索树中的众数

![501.二叉搜索树中的众数](.\images\501.png)

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         val = x
#         left = None
#         right = None

class Solution(object):
    def findMode(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if not root:
            return []

        max_count = 0
        recent = float('-inf')
        last = float('-inf')
        count = 0
        res = [root.val]

        stack = []
        p = root

        while stack or p:
            while p:
                stack.append(p)
                p = p.left
            recent = stack.pop(-1)
            if recent.val == last:
                count += 1       
            else:
                count = 1
            
            if count == max_count:
                res.append(recent.val)

            if count > max_count:
                res = []
                max_count = count
                res.append(recent.val)
            
            last = recent.val
            p = recent.right
          
        return res
        
```

### TIPs

* 中序遍历后，BST的相同数一定邻近

* 可以使用Morris遍历，省去中序遍历时的额外空间

  

  

## 106.从中序与后序遍历序列构造二叉树

![106.从中序与后序遍历序列构造二叉树](.\images\106.png)

### Tips:

* 没啥好说的，递归



### 113.路径总和II

![113.路径总和II](.\images\113.png)

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def pathSum(self, root, sum):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: List[List[int]]
        """
        res = []

        def dfs(node, target, path):

            if not node:
                return 

            if not node.left and not node.right and target == node.val:
                path.append(node.val)
                res.append(path[:])
                path.pop()
                return

            path.append(node.val)
            dfs(node.left, target - node.val, path)
            path.pop()

            path.append(node.val)
            dfs(node.right, target - node.val, path)
            path.pop()

        dfs(root, sum, [])
        return res
```

### Tips

* 注意返回时的处理