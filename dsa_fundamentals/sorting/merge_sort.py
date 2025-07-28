def merge_sort(arr, start, end):
    if end - start + 1 <= 1:
        return arr

    m = (start + end) // 2  # middle index of the array
    # sort the left half
    merge_sort(arr, start, m)
    # sort the right half
    merge_sort(arr, m + 1, end)

    # merge sorted halfs
    merge(arr, start, m, end)
    return arr


def merge(arr, start, m, end):
    L = arr[start:m + 1]
    R = arr[m + 1:end + 1]
    i = j = 0
    k = start
    while i < len(L) and j < len(R):
        if L[i] <= R[j]:
            arr[k] = L[i]
            i += 1
        else:
            arr[k] = R[j]
            j += 1
        k += 1

    while i < len(L):
        arr[k] = L[i]
        i += 1
        k += 1
    while j < len(R):
        arr[k] = R[j]
        j += 1
        k += 1


arr = [12, 11, 13, 5, 6, 7]

# Calling mergeSort to sort the array
sorted_arr = merge_sort(arr, 0, len(arr) - 1)

print(sorted_arr)
