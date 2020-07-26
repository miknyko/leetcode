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

