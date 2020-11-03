## 701.二叉搜索树中的插入操作

![剑指offer 20. 表示数值的字符串](.\images\701.png)

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def insertIntoBST(self, root, val):
        """
        :type root: TreeNode
        :type val: int
        :rtype: TreeNode
        """
        
        if not root:
            return TreeNode(val)

        if val > root.val:
            root.right = self.insertIntoBST(root.right, val)
        else:
            root.left = self.insertIntoBST(root.left, val)

        return root
```

### Tips

* 没有Tips



## 18.四数之和

![18.四数之和](.\images\18.png)

```python
class Solution(object):
    def fourSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[List[int]]
        """

        nums.sort()
        n = len(nums)
        res = []

        for i in range(n):
            
            if i > 0 and nums[i] == nums[i - 1]:
                continue

            for j in range(i + 1, n):
                if j > i + 1 and nums[j] == nums[j - 1]:
                    continue

                l = j + 1
                r = n - 1
                
                while l < r:
                    sums = nums[i] + nums[j] + nums[l] + nums[r]
                    if sums == target:
                        res.append([nums[i], nums[j], nums[l], nums[r]])
                        while l < r and nums[l] == nums[l + 1]:
                            l += 1
                        while l < r and nums[r] == nums[r - 1]:
                            r -= 1
                        l += 1
                        r -= 1

                    elif sums < target:
                        l += 1
                    elif sums > target:
                        r -= 1

        return res
                    
```

### TIPS

* 仿照三数之和
* 固定最小的两个指针，然后左右指针往中间搜索



## 834.树中距离之和

![834.树中距离之和](.\images\834.png)

```python
class Solution(object):
    def sumOfDistancesInTree(self, N, edges):
        """
        :type N: int
        :type edges: List[List[int]]
        :rtype: List[int]
        """

        tree = [[] for _ in range(N)]
        child_count = [1 for _ in range(N)] #存储每个节点的子节点数, 包括自身
        distance_sum = [0 for _ in range(N)] # dfs1 :存储每个节点作为根节点时，他在子树中的距离和
                                            # dfs2: 变换成为答案

        # 建立邻接关系表
        for start, to in edges:
            tree[start].append(to)
            tree[to].append(start)

        # 计算根节点的距离和
        def dfs1(node, parent):
            for child in tree[node]:
                # 跳过父亲节点
                if child == parent:
                    continue
                dfs1(child, node)
                child_count[node] += child_count[child]
                distance_sum[node] += distance_sum[child] + child_count[child]
            
            # 递归到底的时候，此时node没有子节点，开始返回

        # 开始从上往下更新每个节点的距离和
        # distanceSUM[i] = distanceSUM[root] - child_count[i] + N - child_count[i]
        def dfs2(node, parent):
            for child in tree[node]:
                # 跳过父亲节点
                if child == parent:
                    continue
                distance_sum[child] = distance_sum[node] - child_count[child] + N - child_count[child]
                dfs2(child, node)

        dfs1(0, -1)
        dfs2(0, -1)

        return distance_sum

```

### Tips

* 第一次dfs从下而上（后序遍历）统计每个节点的子节点，以及以每个节点为根节点的子树距离和

* 第二次dfs从上而下（前序遍历）统计每个节点的所有节点距离和

* 将问题拆分：对于两个相邻节点A和B，将树拆分为两个子树，根节点分别为A和B，A节点到其他所有节点的距离和ans(A) = A子树中所有节点到A节点的距离和sum(A) + B子树中所有节点到B节点的距离和sum(B) + B子树的大小cnt(B);同理，ans(B) = sum(B) + sum(A) + cnt(A);

  由此我们得到：
  ans(A) = sum(A) + sum(B) + cnt(B);
  ans(B) = sum(B) + sum(A) + cnt(A);

  则，两个相邻接点的解之间的关系为：ans(A) = ans(B) - cnt(A) + cnt(B) = ans(B) - cnt(A) + (N - cnt(A));

  因此，对于根节点root的任意子节点child，ans(child) = ans(root) - cnt(child) + N - cnt(child);

  得到root的答案就可以DFS递归得到其他所有节点的答案。（这里需要一个DFS）

  那么，剩下的问题就是解决root的距离和就可以了。

  我们一般想到DFS，根节点的距离和S = Σ s[i] + cnt[i];其中，s[i]为root的某子节点i到其子节点的距离和，cnt[i]为子节点i的大小





## 142.环形链表II

![142.环形链表II](.\images\142.png)

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def detectCycle(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """

        fast = head
        slow = head

        while fast and slow and fast.next and slow.next and fast.next.next:
            fast = fast.next.next
            slow = slow.next

            if fast == slow:
                p = head
                while p != slow:
                    p = p.next
                    slow = slow.next
                return p

        return None


```

### Tips:

* 快慢指针，两指针相遇时派出第三个指针p，p和慢指针再次相遇的时候，就是循环点





## 416. 分割等和子集

![416. 分割等和子集](.\images\416.png)



```python
class Solution(object):
    def canPartition(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """

        if not nums or len(nums) == 1:
            return False

        n = len(nums)
        total = sum(nums)

        if total & 1 == 1:
            return False

        target = total / 2

        dp = [[False for _ in range(target + 1)] for _ in range(n)]

        if target >= nums[0]:
            dp[0][nums[0]] = True
        
        for i in range(1, n):
            for j in range(target + 1):
                if dp[i - 1][j] or j == nums[i]:
                    dp[i][j] = True

                if j > nums[i] and dp[i - 1][j - nums[i]]:
                    dp[i][j] = True

        return dp[-1][-1]

        



```

### Tips

* 等同于在数组中找一群数，他们的和刚好为sum / 2，转化为背包问题
* 动态规划，`dp[i][j]`意义为数组前i个数中是否可以挑一些数，使他们的和刚好为j
* 则`dp[i][j] = dp[i - 1][j] or dp[i - 1][j - nums[i]] or j == nums[i](j > nums[i])`
* 对应要么取当前数，要么不取当前数，或者刚好目标为当前数
* TRIKCY



## 24.两两交换链表中的节点

![24.两两交换链表中的节点](.\images\24.png)

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def swapPairs(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """

        if not head or not head.next:
            return head
        
        start = ListNode(-1)
        start.next = head

        cur = start
        p = cur.next
        
        while cur.next.next:
            cur.next = p.next
            p.next = p.next.next
            cur.next.next = p
            if p.next:
                cur = p
                p = cur.next
            else:
                break
        
        return start.next
            
```

### Tips:

* 使用辅助头部
* 双指针前进
* 把图画出来就完了



## 1002.查找常用字符

![1002.查找常用字符](.\images\1002.png)

```python
class Solution(object):
    def commonChars(self, A):
        """
        :type A: List[str]
        :rtype: List[str]
        """

        record = [0 for _ in range(26)]
        total_record = [0 for _ in range(26)]

        for i, word in enumerate(A):
            for c in word:
                index = ord(c) - ord('a')
                record[index] += 1
            if i != 0:
                for i in range(26):
                    total_record[i] = min(total_record[i], record[i])
            else:
                total_record = record[:]
            record = [0 for _ in range(26)]

        res = []

        for i in range(26):
            for freq in range(total_record[i]):
                res.append(chr(i + ord('a')))

        return res
```

### TIPs

* 逐单词逐字母统计字母个数



### 977. 有序数组的平方

![977. 有序数组的平方](.\images\977.png)

```python
class Solution(object):
    def sortedSquares(self, A):
        """
        :type A: List[int]
        :rtype: List[int]
        """

        n = len(A)
        left = 0 
        right = n - 1
        res = []

        while left <= right:
            if abs(A[left]) >= abs(A[right]):
                res.append(A[left] ** 2)
                left += 1
            else:
                res.append(A[right] ** 2)
                right -= 1

        return res[::-1]
            


        

```





## 52.N皇后II

![52.N皇后II](.\images\52.png)

```python
class Solution(object):
    def totalNQueens(self, n):
        """
        :type n: int
        :rtype: int
        """

        self.board = [['.' for _ in range(n)] for _ in range(n)]
        self.res = 0

        def is_valid(row, column):
            for i in range(row):
                for j in range(n):
                    if (self.board[i][j] == 'Q') and (j == column or (column - row == j - i) or (column + row == i + j)):
                        return False
            return True

        def dfs(row):
            if row == n:
                self.res += 1
                return
            
            for i in range(n):
                if is_valid(row, i):
                    self.board[row][i] = 'Q'
                    dfs(row + 1)
                self.board[row][i] = '.'

        dfs(0)
        return self.res
```

### Tips

* 同51，带回溯的深度搜索



## 844.比较会退格的字符串

![844.比较会退格的字符串](.\images\844.png)

```python
class Solution(object):
    def backspaceCompare(self, S, T):
        """
        :type S: str
        :type T: str
        :rtype: bool
        """

        n_s = len(S)
        n_t = len(T)

        p_s = n_s - 1
        p_t = n_t - 1

        count_s = 0
        count_t = 0

        while p_s >= 0 or p_t >= 0:
            while p_s >= 0:
                if S[p_s] == '#':
                    count_s += 1
                    p_s -= 1
                elif count_s > 0:
                    p_s -= 1
                    count_s -= 1
                else:
                    # 开始比较
                    break
            while p_t >= 0:
                if T[p_t] == '#':
                    count_t += 1
                    p_t -= 1
                elif count_t > 0:
                    p_t -= 1
                    count_t -= 1
                else:
                    # 开始比较
                    break
            if p_s >= 0 and p_t >= 0:
                if S[p_s] != T[p_t]:
                    return False
            # 若其中一个指针已经到-1,另外一个还在没到，也说明不对
            elif p_s >= 0 or p_t >= 0:
                return False
            
            p_s -= 1
            p_t -= 1

        return True

```

### Tips

* 双指针从后往前遍历，逐位比对，注意比较一方指针已到-1 而另外方还在里面的情况



## 763.划分字母区间

![763.划分字母区间](.\images\763.png)

```python
class Solution(object):
    def partitionLabels(self, S):
        """
        :type S: str
        :rtype: List[int]
        """
        rec = {}
        for i in range(26):
            c = chr(ord('a') + i)
            rec[c] = [float('inf'), float('-inf')]

        # 统计26个字母的首次出现索引和最后一次出现的索引
        for i, c in enumerate(S):
            start = rec[c][0]
            end = rec[c][1]

            if i < start:
                start = i

            if i > end:
                end = i

            rec[c] = [start, end]
	 	# 将此索引列表排序
        record = list(rec.values())
        record.sort(key=lambda x :x[0])
        
        res = []
        start = record[0][0]
        end = record[0][1]
        record.append([float('inf'), float('-inf')])
        
        # 合并重合区间
        for part in record[1:]:

            if part[0] < end:
                end = max(end, part[1])
            
            else:
                res.append(end - start + 1)
                if part[0] == float('inf'):
                    break
                else:
                    start = part[0]
                    end = part[1]
            
        return res
```

### Tips

* 没啥好说的



## 1024.视频拼接

![1024.视频拼接](.\images\1024.png)

```python
class Solution(object):
    def videoStitching(self, clips, T):
        """
        :type clips: List[List[int]]
        :type T: int
        :rtype: int
        """

        # 贪心算法
        
        max_distance_from_this = [0 for _ in range(T + 1)]

        # 合并被包含的子区间，记录以每个索引i为左端点的最大子区间右端点
        for a, b in clips:
            if a <= T:
                max_distance_from_this[a] = max(max_distance_from_this[a], b)

        # 遍历每个点
        res = 0
        pre_area_end = 0
        max_reachable_point = 0

        for i in range(T):
            max_reachable_point = max(max_reachable_point, max_distance_from_this[i])

            if i == max_reachable_point:
                return -1

            # 当到达当前子区间的右端点时，说明必须去一个新的区间了，结果+1
            if i == pre_area_end:
                res += 1
                pre_area_end = max_reachable_point

        return res
```

### Tips

* 贪心法，类似跳跃问题
* 1. 记录每个节点能达到的最远节点，
  2. 循环所有节点，更新能达到的最大节点
  3. 如果当循环还没结束的时候，就遇到了极限，说明不能到达终点





## 845.数组中的最长山脉

![845.数组中的最长山脉](.\images\845.png)

```python
class Solution(object):
    def longestMountain(self, A):
        """
        :type A: List[int]
        :rtype: int
        """

       
        n = len(A)
        # left[i] 表示从i点往左能延申多少个位置
        left = [0 for _ in range(n)]    
        for i in range(1, n):
            left[i] = left[i - 1] + 1 if A[i] > A[i - 1] else 0

        # right[i] 表示从i点往右能延申多少个位置
        right = [0 for _ in range(n)]
        for i in range(n - 1)[::-1]:
            right[i] = right[i + 1] + 1 if A[i] > A[i + 1] else 0


        longest = 0
        # 若某个节点能同时向左向右延申，则此点为峰顶
        for i in range(n):
            if left[i] > 0 and right[i] > 0:
                longest = max(longest, left[i] + right[i] + 1)

        return longest
```

### Tips

* DP，枚举所有有可能出现的山顶



## 1365.有多少小于当前数字的数字

![1365.有多少小于当前数字的数字](.\images\1365.png)

```python
class Solution(object):
    def smallerNumbersThanCurrent(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """

        n = len(nums)
		
        index_list = [x for x in range(n)]
		
        # 将索引数列按照原数组值大小排序
        index_list.sort(key=lambda x: nums[x])

        res = [0 for _ in range(n)]
        same = 0

        for count, i in enumerate(index_list):
            if count == 0:
                res[i] = 0

            else:
                if nums[i] == nums[index_list[count - 1]]:
                    res[i] = res[index_list[count - 1]]
                    same += 1
                else:
                    res[i] = res[index_list[count - 1]] + same + 1
                    same = 0

        return res
```

### TIPS

* 建立索引数组，对索引数组进行排序
* 遍历索引数组，计数
* 脑壳都绕晕了



## 463.岛屿周长

![463.岛屿周长](.\images\463.png)

```python
class Solution(object):
    def islandPerimeter(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        
        rows = len(grid)
        columns = len(grid[0])

        res = 0

        for r in range(rows):
            for c in range(columns):
                if grid[r][c] == 1:
                    count = 4
                    
                    for y, x in [(r + 1, c), (r, c + 1), (r - 1, c), (r, c - 1)]:
                        # 此陆地每多一个相邻的陆地，他对总周长的贡献就少1
                        if 0 <= x < columns and 0 <= y < rows and grid[y][x] == 1:
                            count -= 1
                    res += count
        
        return res
                        
```

### Tips

* 遍历每一个陆地，此陆地每多一个相邻的陆地，他对总周长的贡献就少1





## 349.两个数组的交集

![349.两个数组的交集](.\images\349.png)

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

![941. 有效得山脉数组](.\images\941.png)

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

        # 不能是头尾
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