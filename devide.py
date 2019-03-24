def hold_out(dataSet, train_size):
    """
    留出法
    dataSet:数据集
    train_size:留出法中训练集所占得比例
    """
    totalLen = dataSet.shape[0]
    train_len = int(totalLen * train_size)
    index = range(totalLen)
    index_1 = np.random.choice(index, train_len, replace=False)     # 训练集的索引
    index_2 = np.delete(index, index_1)                      # 测试集的索引
    train = dataSet[index_1]
    test = dataSet[index_2]
    return train, test


def cross_validation(dataSet, k):
    """
    交叉验证法
    dataSet:数据集
    k：交叉验证的次数
    return : datasets：shape(k,num,feature)  list类型
    """
    datasets = []
    num = int(dataSet.shape[0]/k)
    start = 0
    end = num
    for i in range(k):
        datasets.append(dataSet[start:end, :].tolist())
        start += num
        end += num
    return datasets


def BootStrapping(dataSet):
    """
    自助法
    :param dataSet:数据集
    :return: train训练集,test测试集
    """
    m = dataSet.shape[0]
    index1 = []
    index2 = []
    for i in range(m):
        index1.append(np.random.randint(m))
    index2 = np.delete(range(m), index1)
    train = dataSet[index1]
    test = dataSet[index2]
    return train, test