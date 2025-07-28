def partition(s):
    result = []
    def is_plaindrome(sub):
        return sub == sub[::-1]
    
    def backtrack(start, path):
        if start == len(s):
            result.append(path[:])
            return
        for end in (start+1, len(s)+1):
            sub_string = s[start:end]
            if is_plaindrome(sub_string):
                path.append(sub_string)
                backtrack(end, path)
                path.pop()
    backtrack(0, [])
    return result