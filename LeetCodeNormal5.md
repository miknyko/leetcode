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
                    # 注意高度，宽度
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
                # 新元素之前加入队列的较小元素，将永无出头之日，所以不用记录
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



## 329.矩阵中的最长递增路径

![329.矩阵中的最长递增路径](.\images\329.PNG)

```python
class Solution(object):
    def longestIncreasingPath(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: int
        """
        if not matrix:
            return 0

        h = len(matrix)
        w = len(matrix[0])
        
        directions = [(1, 0),(-1, 0),(0, 1),(0, -1)]
        record = [[0 for _ in range(w)] for _ in range(h)]

        def dfs(i, j):
            if record[i][j] != 0:
                return record[i][j]

            record [i][j] = 1
            
            for dx, dy in directions:
                new_i = i + dy
                new_j = j + dx
                if 0 <= new_i < h and 0 <= new_j < w and matrix[new_i][new_j] > matrix[i][j]:
                    record[i][j] = max(record[i][j], dfs(new_i, new_j) + 1)
                                  
            return record[i][j]
        
        res = 0
        for i in range(h):
            for j in range(w):
                res = max(res, dfs(i, j))
        
        return res
```

### Tips

* 递归，带记忆的深度搜索
* 其实有点像动态规划，使用一个二维dp，`dp[i][j]`为以`[i][j]`为最大值的递增序列的长度
* 也可以用深度搜索做



## 218.天际线问题

![218.天际线问题](.\images\218.PNG)



```python
class Solution(object):
    def getSkyline(self, buildings):
        """
        :type buildings: List[List[int]]
        :rtype: List[List[int]]
        """
        class recorder():
            def __init__(self):
                self.data = collections.defaultdict(int)
                self.max_height = 0
                self.data[0] = 1

            def insert(self, num):
                self.data[num] += 1
                if num > self.max_height:
                    self.max_height = num
            
            def pop(self, num):
                self.data[num] -= 1
                if self.data[num] == 0:
                    del self.data[num]
                    self.max_height = max(self.data.keys())



        points = []
        # 将所有左右端点按x坐标顺序排序
        for l, r, h in buildings:
            points.append((l, -h))
            points.append((r, h))
        points.sort()

        current_heights = recorder()
        last = 0
        res = []

        for x, h in points:
            if h < 0:
                current_heights.insert(-h)
            else:
                current_heights.pop(h)
            
            cur_height = current_heights.max_height
            if cur_height != last:
                res.append([x, cur_height])
                last = cur_height
            print(current_heights.max_height)
            
        return res
```

### Tips:

* 线性扫描所有的关键点，动态记录高度发生变化的临界时刻
* 使用哈希表来记录动态高度，如果使用大顶堆，删除某值的时候可能超时



## 124.二叉树中的最大路径和

![124.二叉树中的最大路径和](.\images\124.PNG)

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def maxPathSum(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        
        self.res = float('-inf')

        def maxSum(node):
            if not node:
                return 0
            # 信息汇总
            left = max(maxSum(node.left), 0)
            right = max(maxSum(node.right), 0)
			
            # 有可能在递归过程中遇到惊喜，原来这个节点就是我们两边都可以走的节点，也就是根节点
            self.res = max(self.res, node.val + left + right)
            # 不然，到了一个节点要么走左边，要么走右边，要么不走
            return max(left + node.val, right + node.val)

        maxSum(root)
        return self.res
```



### Tips

* 不要把路径想象为从某节点到某节点，把路径想象为从某个根节点长出的两个分支
* 从这个根节点出发的两个分支，遇到一个新节点，要么止步于该节点，要么继续走左边，要么继续走右边
* 那么如何选择走左边走右边或者不走？则需要递归的从底而上将信息汇总



## 212.单词搜索II

![212.单词搜索II](.\images\212.PNG)

```python
class Solution(object):
    def findWords(self, board, words):
        """
        :type board: List[List[str]]
        :type words: List[str]
        :rtype: List[str]
        """
        # 实现一个Trie
        class Trie():
            def __init__(self):
                self.data = {}

            def insert(self, word):
                word = word + '#'
                
                if word[0] == '#':
                    self.data['#'] = True
                    return 

                elif word[0] not in self.data:
                    self.data[word[0]] = Trie()
                self.data[word[0]].insert(word[1:])

            def search(self, word):
                word = word + '#'
                if word[0] == '#' and self.data['#'] == True:
                    return True
                elif word[0] in self.data:
                    return self.data[word[0]].search(word[1:])
                else:
                    return False

            def startwith(self, c):
                if not c:
                    return True
                
                if c[0] in self.data:
                    return self.data[c[0]].startwith(c[1:])

                else:
                    return False

        record = Trie()
        for word in words:
            record.insert(word)

        res = []
        directions = [(1, 0),(-1, 0),(0, 1),(0, -1)]
        visited = [[0 for _ in range(len(board[0]))] for _ in range(len(board))]

        def dfs(i, j, record, path):
            if board[i][j] not in record.data:
                return 
            record = record.data[board[i][j]]
            path += board[i][j]
            if '#' in record.data:
                res.append(path[:])
            visited[i][j] = 1
            for dx, dy in directions:
                new_i = i + dy
                new_j = j + dx
                if 0 <= new_i < len(board) and 0 <= new_j < len(board[0]) and visited[new_i][new_j] == 0:
                    dfs(new_i, new_j, record, path)
            path = path[:-1] # 这个回溯可加可不加，因为每次从新的元素出发，path都进行了重置
            visited[i][j] = 0 # 这个回溯必须加
            return
                      
 
        for i in range(len(board)):
            for j in range(len(board[0])):
                dfs(i, j, record, "")

        return list(set(res))
```

### Tips

* 回溯的写法，真的好难，主要掌握回溯的时机，一定在深度搜索的主逻辑中回溯
* Trie的实现



## 84.柱状图中的最大矩形

![84.柱状图中的最大矩形](.\images\84.PNG)

```python
class Solution(object):
    def largestRectangleArea(self, heights):
        """
        :type heights: List[int]
        :rtype: int
        """
		# 注意首尾加0
        heights = [0] + heights + [0]
        # 单调递增栈
        stack = [0]
        n = len(heights)

        max_vol = 0
        for i in range(n):
            # 遇到比栈顶A元素小的新元素B，就开始出栈进行结算
            # 结算，遍历以此B元素为右边界（不包含B！）的所有矩形，所以首尾一定要加0
            while heights[stack[-1]] > heights[i]:
                h_index = stack.pop(-1)
                max_vol = max(max_vol, heights[h_index] * (i - stack[-1] - 1))
            stack.append(i)
        
        return max_vol
```

### Tips

* 使用单调递增栈
* 和接雨水一样，只不过一个用递减栈，这个用递增栈



## 315.计算右侧小于当前元素的个数

![315.计算右侧小于当前元素的个数](.\images\315.PNG)

```python
class Solution(object):
    def countSmaller(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        class Node():
            def __init__(self, val):
                self.val = val
                self.left = None
                self.right = None
                # 这个是关键，记录目前该节点的左孩子的数目，也就是比自己小的节点数目
                self.left_count = 0

        class BST():
            def __init__(self, dataset):
                self.root = None
                self.dataset = dataset
        
            def push(self, index):
                count = 0
                if not self.root:
                    self.root = Node(index)
                    return count
                p = self.root
                while p:
                    if self.dataset[index] > self.dataset[p.val]:
                        count = count + p.left_count + 1
                        if p.right:
                            p = p.right
                        else:
                            p.right = Node(index)
                            return count
                    else:
                        p.left_count += 1
                        if p.left:
                            p = p.left
                        else:
                            p.left = Node(index)
                            return count
        
        bst = BST(nums)
        res = [0 for _ in range(len(nums))]
        for i in range(len(nums))[::-1]:
            count = bst.push(i)     
            res[i] = count
        return res
```

### Tips

* 从右往左遍历数组，将数组元素插入二叉搜索树
* 每个节点增加一个属性，记录此节点左孩子的节点个数
* 当一个遍历到一个新值的时候，这个值在插入过程中，只要往右走，就不断记录所经历的节点的左孩子数，只要往左走，就更新这个岔路口节点的左孩子数，在插入完毕的时候，返回统计的值，就是这颗树里，小于这个数的数的个数

```python
class Solution(object):
    def countSmaller(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        count = [0 for _ in range(len(nums))]
        index = [i for i in range(len(nums))]

        def mergeSort(nums):
            if len(nums) <= 1:
                return nums
            mid = len(nums) // 2
            left = mergeSort(nums[:mid])
            right = mergeSort(nums[mid:])
            return merge(left, right)

        def merge(left, right):
            res = []
            i = 0 
            j = 0

            while i < len(left) and j < len(right):
                if nums[left[i]] <= nums[right[j]]:
                    res.append(left[i])
                    count[left[i]] += j
                    i += 1
                else:
                    res.append(right[j])
                    j += 1

            if i == len(left):
                res += right[j:]
            
            if j == len(right):
                while i < len(left):
                    res.append(left[i])
                    count[left[i]] += j
                    i += 1

            return res

        mergeSort(index)

        return count
```

### Tips

* 计算逆序对，归并排序的副产物
* 初始化一个索引数组，排序的时候排的是索引数组，而不是原数组
* 这样可以通过索引数组，方便的映射结果数组，在结果数组的指定索引位置计数