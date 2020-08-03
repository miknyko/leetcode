def partition(nums, l, r):
    pivot = nums[l]
    left = l
    right = l

    while right <= r:
        if nums[right] < pivot:
            left += 1
            nums[left], nums[right] = nums[right], nums[left]
        right += 1
    nums[left], nums[l] = nums[l], nums[left]
    return left + 1

    test = [3,2,1,5]

def quickSort(nums, l, r):
    if l < r:
        start = partition(nums, l, r)
        quickSort(nums, 0, start - 1)
        quickSort(nums, start, r)
        
def main():
    test = [-74,48,-20,2,10,-84,-5,-9,11,-24,-91,2,-71,64,63,80,28,-30,-58,-11,-44,-87,-22,54,-74,-10,-55,-28,-46,29,10,50,-72,34,26,25,8,51,13,30,35,-8,50,65,-6,16,-2,21,-78,35,-13,14,23,-3,26,-90,86,25,-56,91,-13,92,-25,37,57,-20,-69,98,95,45,47,29,86,-28,73,-44,-46,65,-84,-96,-24,-12,72,-68,93,57,92,52,-45,-2,85,-63,56,55,12,-85,77,-39]
    quickSort(test, 0, len(test) - 1)
    print(test)

    

if __name__ == "__main__":
    main()
    