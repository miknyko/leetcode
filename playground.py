
def exist(board, word):
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
            print(mask)
        else:
            return False
        
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
                    else:
                        mask[k][l] = 0
        mask[i][j] = 0
        return False
                
    mask = [[0 for _ in range(w)] for _ in range(h)]
    count = 0  
    for i in range(h):
        for j in range(w):
            print((i, j))
            # 对每一个元素进行这种搜索
            # 只要有一个元素返回true，则返回True
            # 如果最后没有True, 就返回False
            if dfs(i, j, count, word, board):
                return True
            
    return False

test = [["C","A","A"],["A","A","A"],["B","C","D"]]

print(exist(test, 'AAB'))