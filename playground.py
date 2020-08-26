
def countSmaller(nums):
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
                        # p.right.left_count = p.left_count + 1
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

test = [5, 2, 6, 1]

print(countSmaller(test))