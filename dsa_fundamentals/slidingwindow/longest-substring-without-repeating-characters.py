class Solution:
    # Brute Force
    # def lengthOfLongestSubstring(self, s: str) -> int:
    #     res = 0
    #     for i in range(len(s)):
    #         char_set = set()
    #         for j in range(i, len(s)):
    #             if s[j] in char_set:
    #                 break
    #         char_set.add(s[j])
    #         res = max(res, len(char_set))
    #     return res
    # sliding window
    def lengthOfLongestSubstring(self, s: str) -> int:
        charSet = set()
        l = 0
        res = 0
        for r in range(len(s)):
            while s[r] in charSet:
                charSet.remove(s[l])
                l += 1
            charSet.add(s[r])
            res = max(res, r-l+1)
        return res