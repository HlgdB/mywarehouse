def bucketSort(tmpList): # 待排序数组， 最大的筒子，数组的长度
    max_num = max(tmpList) # 选择一个最大数
    bucket = [0]*(max_num+1)  # 创建一个元素全为0的列表当桶
    for i in tmpList: # 把所有元素放入桶中，计算该数出现次数
        bucket[i] += 1
    ans = []
    for i in range(len(bucket)):
        if bucket[i] != 0: # 如果该数出现的次数大于0
            for j in range(bucket[i]): # 出现多少次都将数放入桶中
                ans.append(i)
    return ans
 
tmpList = [5,6,3,2,1,65,2,0,8,0]
print(bucketSort(tmpList))
