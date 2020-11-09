points = [[68,97],[34,-84],[60,100],[2,31],[-27,-38],[-73,-74],[-55,-39],[62,91],[62,92],[-57,-67]]

K = 5
n = len(points)

distances = [point[0] ** 2 + point[1] ** 2 for point in points]
indexs = [i for i in range(len(points))]

n = len(points)
points_t = points[:]

def get_distance(point):
    return point[0] ** 2 + point[1] ** 2

def partition(left, right):
    l = left
    r = right

    pivot = left

    while l < r:
        while l < r and get_distance(points_t[r]) > get_distance(points_t[pivot]):
            print('right')
            r -= 1
        while l < r and get_distance(points_t[l]) <= get_distance(points_t[pivot]):
            print('left')
            l += 1
        points_t[l], points_t[r] = points_t[r], points_t[l]

    points_t[left], points_t[l] = points_t[l], points_t[pivot]

    return l


# 开始二分查找
left = 0
right = n - 1

while True:
    print((left, right))

    index = partition(left, right)
    print(index)
    if index > K - 1:
        right = index - 1
    elif index < K - 1:
        left = index + 1
    else:
        print(points_t[:K])
        break
        # return points_t[:K]