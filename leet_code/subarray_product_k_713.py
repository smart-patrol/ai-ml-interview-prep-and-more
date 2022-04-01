
def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
    """
    Given an array of integers nums and an integer k, return the number of contiguous subarrays where the product of all the elements in the subarray is strictly less than k.

    For each j, let opt(j) be the smallest i so that nums[i] * nums[i+1] * ... * nums[j] is less than k. opt is an increasing function.
    """
    if k <= 1:
        return 0
    ans = left = 0
    prod = 1
    for right, val in enumerate(nums):
        prod *= val
        while prod >= k:
            prod //= nums[left]
            left += 1
        ans += right - left + 1
    return ans


assert numSubarrayProductLessThanK([10,5,2,6], 100) == 8
assert numSubarrayProductLessThanK([1,2,3],0) == 0
