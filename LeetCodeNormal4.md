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