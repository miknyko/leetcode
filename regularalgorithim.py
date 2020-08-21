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
        node = queue.pop(-1)
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
        node = queue.pop(-1)
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
        res.append(tmp)
    return res

# dfs层序

def levelOrder(root):

    if not root:
        return []

    res = []

    def dfs(level, root):
        if level > len(res):
            res.append([])

        res[level - 1].append(root.val)

        if root.left:
            dfs(level + 1, root.left)

        if root.right:
            dfs(level + 1, root.right)

    
    dfs(1, root)

    return res


    
# 层序序列化
def serializeBFS(root):
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

# 层序反序列化
def deserializeBFS(data):
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


# 隐式小顶二叉堆（数组实现）

class heap():
    def __init__(self):
        # 注意，数组第一个元素，即索引为0的位置不能存放元素
        self.data = [None]

    def heapify(self, i):
        length = len(self.data)
        while True:
            left = 2 * i
            right = 2 * i + 1
            smallest = i

            # 找到节点，左孩子，右孩子中的最小值
            if (left < n and self.data[left] < self.data[i]):
                smallest = left
            if (right < n and self.data[right] < self.data[i]):
                smallest = right
            # 如果这个最小值不是节点，就把节点和它交换
            if smallest != i:
                self.data[smallest], smallest[i] = self.data[i], self.data[smallest]
                # 指针指向原本被置换的位置，沿着这里往下检测
                i = smallest
            # 超出索引范围，说明当前i没有孩子
            else:
                break
            
        def build(self):
            n = len(self.data)
            # 因为在完全树中，有孩子的节点的索引一定不超过n // 2，所以我从这个索引开始，倒着检查堆序
            for i in range(n // 2, 0, -1):
                heapify(i)

        def top(self):
            return self.data[1]

        def pop(self):
            # 弹出时稍微复杂点
            # 首尾元素交换
            self.data[1], self.data[-1] = self.data[-1], self.data[1]
            result = self.data[-1]
            # 改变数组长度
            self.data = self.data[:-1]
            # 检查此时首元素的堆性质
            heapify(1)
            return result

        def insert(self, num):
            self.data.append(num)
            i = len(self.data) - 1

            # 自下而上检查节点与父节点的关系
            while i > 1 and self.data[i] < self.data[i // 2]:
                self.data[i // 2], self.data[i] = self.data[i], self.data[i // 2]
                i = i // 2



            
            

            
