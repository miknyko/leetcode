def LargestAverage(nums, k):
  n = len(nums)
  #建立滚动和数组，第i个元素为数组nums前i + 1个元素之和
  sums = [0 for _ in range(n)]
  sums[0] = nums[0]
  for i in range(1, n):
    sums[i] = sums[i] + nums[i]
  
  max_avg = 0
  # 遍历所有可能的k 
  for t in range(k, n + 1):
    # 初始值设为连续数组的起点为索引为0的元素
    max_sum = sums[t - 1]
    # 遍历连续数组所有可能的起点，
    for i in range(t, n):
      max_sum = max(max_sum, sums[i] - sums[i - t])
    avg = max_sum / t
    max_avg = max(max_avg, avg)

  return max_avg


    