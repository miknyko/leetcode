# 双指针快排
def quickSort(nums):
    n = len(nums)

    def quick(left, right):
        if left >= right:
            return
        pivot = left

        i = left
        j = right

        while i < j:
            while i < j and nums[j] > nums[pivot]:
                j -= 1
            while i < j and nums[i] <= nums[pivot]:
                i += 1
            nums[i], nums[j] = nums[j], nums[i]
        nums[j], nums[pivot] = nums[pivot], nums[j]
        quick(left, j - 1)
        quick(j + 1, right)

    quick(0, n - 1)

    return nums

# 归并排序
def merge_sort(nums):
    if len(nums) <= 1:
        return nums
    mid = len(nums) // 2
    # 分
    left = merge_sort(nums[:mid])
    right = merge_sort(nums[mid:])
    # 合并
    return merge(left, right)


def merge(left, right):
    res = []
    i = 0
    j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            res.append(left[i])
            i += 1
        else:
            res.append(right[j])
            j += 1
    res += left[i:]
    res += right[j:]
    return res





def main():
    test = [-1,48,-20,2,10,-84,-5,-9,11,-24,-91,2,-71,64,63,80,28,-30,-58,-11,-44,-87,-22,54,-74,-10,-55,-28,-46,29,10,50,-72,34,26,25,8,51,13,30,35,-8,50,65,-6,16,-2,21,-78,35,-13,14,23,-3,26,-90,86,25,-56,91,-13,92,-25,37,57,-20,-69,98,95,45,47,29,86,-28,73,-44,-46,65,-84,-96,-24,-12,72,-68,93,57,92,52,-45,-2,85,-63,56,55,12,-85,77,-39]
    quickSort(test)
    print(test)

    

if __name__ == "__main__":
    main()
    