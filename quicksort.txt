def subSort(tmpList,left,right):
    key = tmpList[left]
    while left < right:
        while left < right and tmpList[right] >= key:
            right -= 1
        tmpList[left] = tmpList[right]
        while left < right and tmpList[left] <= key:
            left += 1
        tmpList[right] = tmpList[left]
    tmpList[left] = key
    return left
 
def quickSort(tmpList,left,right):
    if left < right:
        keyIndex = subSort(tmpList,left,right)
        quickSort(tmpList,left,keyIndex)
        quickSort(tmpList,keyIndex+1,right)
 
tmpList = [5,1,9,3,7,4,8,6,2]
quickSort(tmpList,0,len(tmpList)-1)
print(tmpList)
