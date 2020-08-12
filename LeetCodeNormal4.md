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