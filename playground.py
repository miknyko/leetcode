
def findMedianSortedArrays(nums1, nums2):
    """
    :type nums1: List[int]
    :type nums2: List[int]
    :rtype: float
    """
    if len(nums1) > len(nums2):
        return findMedianSortedArrays(nums2, nums1)

    nums1 = [float('-inf')] + nums1 + [float('inf')]
    nums2 = [float('-inf')] + nums2 + [float('inf')]
    m = len(nums1)
    n = len(nums2)

    # print(nums1)
    # print(nums2)

    left = 1
    right = m - 2

    while left <= right:
        mid = left + (right - left) // 2
        i = mid
        j = (m + n + 1) // 2 - i

        # 满足条件
        if max(nums1[i - 1], nums2[j - 1]) <= min(nums1[i], nums2[j]):
            # print((i, j))
            left_max = max(nums1[i - 1], nums2[j - 1])
            right_min = min(num1[i], nums2[j])
            print(left_max)
            print(right_min)
            if (m + n) // 2 == 0:
                return float(left_max + right_min) / 2.0
            else:
                return left_max

        # 若mid大了
        elif nums1[i] > nums2[j]:
            right = mid - 1
        # 若mid小了
        elif nums2[j - 1] > nums1[i]:
            left = mid + 1

    
                
res = findMedianSortedArrays([2],[1,3])           
print(res)
test = [float('-inf'), 2]
# print(min(test))