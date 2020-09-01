## 41.缺失的第一个正整数

![41.缺失的第一个正整数](.\images\41.png)

```python
class Solution(object):
    def firstMissingPositive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """

        def swap(index1, index2):
            nums[index1], nums[index2] = nums[index2], nums[index1]


        n = len(nums)

        for i in range(n):
            while 0 < nums[i] < n and nums[nums[i] - 1] != nums[i]:
                swap(nums[i] - 1, i)

        for i in range(n):
            if nums[i] != i + 1:
                return i + 1
        
        return n + 1
```

### Tips

* 自哈希，对于长度为n的数组，只能存储`[1, n]`一共n个连续正整数
* 遍历原始数组中的每一个元素，如果元素i确实是最终数组的一部分，就将数字i放到索引i - 1上
* 然后再次遍历数组，当某个位置不是该有的数字时，返回该索引



## 76.最小覆盖子串

![76.最小覆盖子串](.\images\76.png)

```python
class Solution(object):
    def minWindow(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        n = len(s)
        # 计数字典，大于0表示窗口需要，小于0表示窗口不需要(并不代表这个单词不在T中，只不过数量满足了)
        record = collections.defaultdict(int)
        for c in t:
            record[c] += 1
        
        left = 0
        right = 0
        need = len(t)
        # 初始化为无限字符串，左闭右闭，意思是有可能没有
        res = (0, float('inf'))

        while right < n:
            # 判断新进来的数，是否真是需要的
            if record[s[right]] > 0:
                need -= 1
            record[s[right]] -= 1
            # print(record)
            if need == 0:
                # 说明已经找齐了神龙，开始缩减窗口
                while left <= right:
                    # 说明如果去掉left指着的单词，则会丢失某条神龙，则需要记录
                    if record[s[left]] == 0:
                        break
                    record[s[left]] += 1
                    left += 1
                # 丢失神龙时，需要记录此时的窗口长度，是否更新
                if right - left < res[1] - res[0]:
                    res = (left, right)
                # 正式丢弃这条神龙
                record[s[left]] += 1
                left += 1
                need += 1
            right += 1
        print(res)
        # 如果真存在此子串，（或者子串就是他本身），res[1]
        if res[1] >= n:
            return ""
        
        else:
            # 因为是左闭右闭，所有最后右边索引需要+1
            return s[res[0]:res[1] + 1]
```

### Tips

* 滑动数组，动态收缩两个边界，记录最小值
* 主要初始值和最终边界的处理



## 4.寻找两个（合并）正序数组的中位数

![4.寻找两个（合并）正序数组的中位数](.\images\4.png)

```python
class Solution(object):
    def findMedianSortedArrays(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """

        if len(nums1) > len(nums2):
            return self.findMedianSortedArrays(nums2, nums1)
        
        # 首尾增加inf,便于处理边界条件
        nums1 = [float('-inf')] + nums1 + [float('inf')]
        nums2 = [float('-inf')] + nums2 + [float('inf')]
        m = len(nums1)
        n = len(nums2)

        # 注意二分搜索左右起点
        left = 1
        right = m - 1

        # 在较短数组中进行二分搜索间隔位置
        while left <= right:
            mid = left + (right - left) // 2
            i = mid
            # 为保持左右数量相等，较长数组的分隔必须在这儿
            j = (m + n + 1) // 2 - i

            # 当条件满足
            if max(nums1[i - 1], nums2[j - 1]) <= min(nums1[i], nums2[j]):
                left_max = max(nums1[i - 1], nums2[j - 1])
                right_min = min(nums1[i], nums2[j])
                # 若最终合并数组元素个数为偶数
                if (m + n) % 2 == 0:
                    return float(left_max + right_min) / 2.0
                else:
                    return left_max
            # 找大了
            elif nums1[i - 1] > nums2[j]:
                right = mid - 1
            # 找小了
            elif nums2[j - 1] > nums1[i]:
                left = mid + 1
        
        # 无需最后返回，因为当搜索区间只剩一个元素的时候，就一定是答案
```

### Tips

* 在较短数组中针对分割位置进行二分搜索，为了保证最终分割左右两边数量相等，则可以求出对应的较长数组中的分割位置，在数量相等的情况下，只要分割条件满足`max(nums1[i - 1], nums2[j - 1]) <= min(nums1[i], nums2[j])`，就是我们想找的分割条件，注意，索引`i, j `代表一种切割，这种切割将元素`nums[i]`划到右边，且是右边第一个元素。
* 在原数组的首尾增加`inf`可以避免讨论各种边界情况，简化代码



## 140.单词拆分II

![140.单词拆分II](.\images\140.png)

```python
class Solution(object):
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: List[str]
        """
        # dp[i]代表字符子串s[:i - 1]是否可分
        dp_issplitable = [False] * (len(s) + 1)
        dp_tmp = [[] for _ in range(len(s) + 1)]

        # 初始化dp_is[0]为True，因为空字符默认是可分的, LEETCODE 139
        dp_issplitable[0] = True
        dp_tmp[0] = ['']
        for i in range(1, len(s) + 1):
            for j in range(i):
                if dp_issplitable[j] and s[j:i] in wordDict:
                    dp_issplitable[i] = True
        
        # 面向测试编程，先把一些极端FALSE给返回了
        if not dp_issplitable[-1]:
            return []

        for i in range(1, len(s) + 1):
            if dp_issplitable[i]:
                for j in range(i):
                    if dp_issplitable[j] and s[j:i] in wordDict:
                        # 并且记录
                        if j == 0:
                            # 说明此时s[:i]是一个完整的单词，单独处理是为了不要开头的空格
                            dp_tmp[i].append(s[:i])
                        else:
                            dp_tmp[i].extend([string + " " + s[j:i] for string in dp_tmp[j]])
        
        return dp_tmp[-1]
```

### TIPs

* 参照139
* 动态规划，遍历每一个可以分割的位置
* 为了AC，先把一些极端FALSE测试用例使用139的方法返回了



## 44.通配符匹配

![140.单词拆分II](.\images\44.png)

```python
class Solution(object):
    def isMatch(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: bool
        """

        m = len(p)
        n = len(s)

        dp = [[False for _ in range(n + 1)]for _ in range(m + 1)]
        dp[0][0] = True

        for i in range(1, m + 1):
            for j in range(n + 1):
                # 如果这个格子已经被某种规则变为True了，就不再管他
                if dp[i][j]:
                    continue
                # 如果遇到一刀一路
                if p[i - 1] == '*' and dp[i - 1][j]:
                    for k in range(j, n + 1):
                        dp[i][k] = True
                # 如果遇到狸猫换太子
                elif j > 0 and p[i - 1] == '?' and dp[i - 1][j - 1]:
                    dp[i][j] = True
                elif j > 0 and p[i - 1] == s[j - 1] and dp[i - 1][j - 1]:
                    dp[i][j] = True
        
        return dp[-1][-1]
        
```

### Tips

* 动态规划，`dp[i][j]`为`s[j - 1]`与`p[i - 1]`是否能匹配
* *为   **之后整行为True**，？为**狸猫换太子**
* 注意DP表头部需要添加两个start，表示空串
* `dp[0][0]`初始化为True



## 10.正则表达式的匹配

![10.正则表达式的匹配](.\images\10.png)

```python
class Solution(object):
    def isMatch(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: bool
        """
        m = len(p)
        n = len(s)

        dp = [[False for _ in range(n + 1)] for _ in range(m + 1)]
        dp[0][0] = True

        for i in range(1, m + 1):
            for j in range(n + 1):
                if dp[i][j]:
                    continue
                elif p[i - 1] == '*':
                    for k in range(n + 1):
                        # 当前列的上两行中，取或运算
                        if dp[i - 1][k] or dp[i - 2][k]:
                            dp[i][k] = True
                        # 或者。。当前列的前一个元素为T，并且。
                        elif k > 0 and dp[i][k - 1] and (s[k - 1] == p[i - 2] or p[i - 2] == '.'):
                            dp[i][k] = True

                elif j > 0 and p[i - 1] == '.' and dp[i - 1][j - 1]:
                    dp[i][j] = True
                elif j > 0 and p[i - 1] == s[j - 1] and dp[i - 1][j - 1]:
                    dp[i][j] = True
        
        return dp[-1][-1]

                        
```

### Tips

* 动态规划，`dp[i][j]`为`s[j - 1]`与`p[i - 1]`是否能匹配
* 注意和44.通配符匹配不一样的是，当遇到*时的处理



## 149.直线上最多的点数

![149.直线上最多的点数](.\images\149.png)

```python
class Solution(object):
    def maxPoints(self, points):
        """
        :type points: List[List[int]]
        :rtype: int
        """
        if not points:
            return 0

        def gcd(a, b):
            while b != 0:
                ret = a % b
                a = b
                b = ret
            return a

        
        count = 0
        n = len(points)
        for p1 in range(n):
            record = collections.defaultdict(int)
            # 记录重复点个数
            repeat = 1
            for p2 in range(n):
                if p1 == p2:
                    continue
                cord_p1 = points[p1]
                cord_p2 = points[p2]

                if cord_p1 == cord_p2:
                    repeat += 1
                    continue

                dy = cord_p2[1] - cord_p1[1]
                dx = cord_p2[0] - cord_p1[0]

                cd = gcd(dy, dx)

                k = str(dy / cd) + '/' + str(dx / cd)

                record[k] += 1

            if record.values():
                max_points = max(record.values()) + repeat
            else:
                max_points = repeat

            count = max(count, max_points)

        return count


```

### Tips

* 遍历每一个点，求得此点与其他所有点得斜率，使用斜率为哈希表的键，记录个数
* 求斜率的时候，使用分数表示，使用辗转相除法，将分子分母约到不能再约
* 注意重复点的处理

