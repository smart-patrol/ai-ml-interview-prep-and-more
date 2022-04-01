from typing import List


def maximumProduct(nums:List[int]) -> int:
    """Given an integer array nums, find three numbers whose product is maximum and return the maximum product."""
    nums.sort()
    return max(nums[0] * nums[1] * nums[-1], nums[-1] * nums[-2] * nums[-3])


def maximumProduct2(nums:List[int]) -> int:
    """Given an integer array nums, find three numbers how product is maximum and return the maximum product."""
    max1,max2,max3,min1,min2 = float('-Inf'),float('-Inf'),float('-Inf'),float('Inf'),float('Inf')

    for n in nums:

        if n >= max1:
            max3 = max2
            max2 = max1
            max1 = n
        elif n >= max2: 
            max3 = max2
            max2 = n
        elif n > max3:
            max3  = n

        if n <= min1:
            min1 = n
            min2 = min1
        elif n < min2:
            min2 = n
        
    return max(max1 * max2 * max3, max1 * min1 * min2)

nums = [1,2,3]
assert maximumProduct(nums) == 6
nums = [1,2,3,4]
assert maximumProduct(nums) == 24

nums = [1,2,3]
assert maximumProduct2(nums) == 6
nums = [1,2,3,4]
assert maximumProduct2(nums) == 24


        