def letterCombinations(digits):
    if not digits:
        return []
    phone_map = {
        '2': "abc", '3': "def", '4': "ghi", '5': "jkl",
        '6': "mno", '7': "pqrs", '8': "tuv", '9': "wxyz"
    }
    results = []

    def backtrack(index, path):
        if index == len(digits):
            results.append(path)
            return
        possible_letter = phone_map[digits[index]]
        for letter in possible_letter:
            backtrack(index+1, path+letter)
    backtrack(0, "")
    return results
