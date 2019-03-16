def merge(left,right):
    ans = []
    i = 0
    j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            ans.append(left[i])
            i += 1
        else:
            ans.append(right[j])
            j += 1
    ans += left[i:] + right[j:]
    return ans
def mergeSort(tmpList):
    size = len(tmpList)
    if size < 2:
        return tmpList
    mid = size // 2
    left = mergeSort(tmpList[:mid])
    right = mergeSort(tmpList[mid:])
    return merge(left,right)
print(mergeSort([56,82,199,36,59,45,153,20,101]))
