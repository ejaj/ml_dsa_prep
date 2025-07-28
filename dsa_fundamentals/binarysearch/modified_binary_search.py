def binary_search(arr, target):
    low = 0
    high = len(arr) - 1
    while low<=high:
        mid = (low+high)//2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid+1
        else:
            high = mid -1
    return -1

arr = [1, 2, 4, 4, 4, 5, 6]
target = 4
print(binary_search(arr, target))


def find_first_occurrence(arr, target):
    low = 0
    high = len(arr) - 1
    result = -1
    while low <= high:
        mid = (low+high) // 2
        if arr[mid] == target:
            result =  mid
            high = mid - 1
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return result

def find_last_occurrence(arr, target):
    low = 0
    high = len(arr) - 1
    result = -1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            result = mid
            low = mid + 1 
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return result

def search_rotated_array(arr, target):
    low = 0
    high = len(arr) - 1
    while low <= high:
        mid = (low+high) // 2
        if arr[mid] == target:
            return mid
        if arr[low] <= arr[mid]:
            if arr[low] <= target < arr[mid]:
                high = mid - 1
            else:
                low = mid+1
        else:
            if arr[mid] < target <= arr[high]:
                low = mid+1
            else:
                high = mid-1
    return -1
