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