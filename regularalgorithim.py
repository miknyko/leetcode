# 二叉树DFS
# 递归（前后中）

def dfs(root):

    if not root:
        return None

    # pre-order
    return [root.val] + dfs(root.left) + dfs(root.right)
    # in-order
    # return dfs(root.left) + [root.val] + dfs(root.right)
    # post-order
    # return dfs(root.left) + dfs(root.right) + [root.val]


# 栈迭代（前后序）
# pre-order
def dfs(root):
    if not root:
        return None
    
    queue = [root]
    res = []

    while queue:
        node = queue.pop()
        res.append(node.val)
        if node.right:
            queue.append(node.right)
        if node.left:
            queue.append(node.left)

    return res

# post-order
def dfs(root):

    if not root:
        return None

    queue = [root]
    res = []

    while queue:
        node = queue.pop()
        res.append(node.val)

        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)

    return res[::-1]

# 栈回溯(前中后序)
# 前序
def dfs(root):

    if not root:
        return None

    p = root
    stack = []
    res = []

    while p or stack:
        while p:
            res.append(p.val)
            stack.append(p)
            p = p.left
        p = stack.pop()
        p = p.right

    return res

# 中序

def dfs(root):
    
    if not root:
        return None
    
    p = root
    stack = []
    res = []

    while p or stack:
        while p:
            stack.append(p)
            p = p.left
        p = stack.pop()
        res.append(p.val)
        p = p.right

    return res

# post-order

def dfs(root):

    if not root:
        return None

    p = root
    stack = []
    res = []

    while p or stack:
        while p:
            res.append(p.val)
            stack.append(p)
            p = p.right
        p = stack.pop()
        p = p.left
    
    return res[::-1]
            
# 队列层序

def bfs(root):
    if not root:
        return None

    queue = [root]
    res = []

    while queue:
        size = len(queue)
        tmp = []
        for i in range(size):
            node = queue.pop(0)
            tmp.append(node.val)

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
    return res

    




    