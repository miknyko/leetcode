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
        # 计数字典，大于0表示窗口需要，小于0表示窗口不需要
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