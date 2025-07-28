def bucketSort(arr):
    counts = [0,0,0]

    for n in arr:
        counts[n] += 1
    
    i = 0
    for n in range(len(counts)):
        for j in range(counts[n]):
            arr[i] = n
            i +=1
    return arr