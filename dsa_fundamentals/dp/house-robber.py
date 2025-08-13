def rob(nums):
    rob1, rob2 = 0, 0
    for num in nums:
        temp = max(num+rob1, rob2)
        rob1=rob2
        rob2=temp
    return rob2


def rob_dp_bottom(nums):
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    dp = [0] * len(nums)
    dp[0] = nums[0]
    dp[1] = max(nums[0], nums[1])
    for i in range(2, len(nums)):
        dp[i] = max(dp[i-1], dp[i-2] + dp[i])
    return dp[-1]
