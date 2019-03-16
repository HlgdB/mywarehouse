def insertsort(tmpList):
    for i in range(1,len(tmpList)):
        key = tmpList[i]
        a = i
        while key < tmpList[a-1]:
            tmpList[a] = tmpList[a-1]
            a = a-1
            if a - 1 < 0:
                break
        tmpList[a] = key
    print(tmpList)
 
insertsort([23,56,89,25,14,99,123,56,55,27])
