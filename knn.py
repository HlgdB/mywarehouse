import numpy as np 
import pandas as pd 
from sklearn import preprocessing

min_max = preprocessing.MaxAbsScaler()

data = pd.read_csv('diabetes.csv')

def creatdata():                  #数据初始化
    data_1 = data[:700]
    train_data = data_1.drop('Outcome',axis = 1)    
    train_array = np.array(train_data)
    train_array = min_max.fit_transform(train_array)    #将值都集中到0到1区间中，减少误差
    train_outcome = data_1.pop('Outcome')
    outcome_array = np.array(train_outcome)

    group = train_array
    labels = outcome_array 
    return group,labels

def classify(input,data,label,k):                  #knn算法具体实现
    datasize = data.shape[0]                       #计算训练集行长度
    diff = np.tile(input,(datasize,1)) - data      #计算预测目标与训练集个组数据的元素差
    sqdiff = diff ** 2
    squaredist = np.sum(sqdiff,axis = 1)           #计算每一行的和
    dist= squaredist ** 0.5

    sortdist = np.argsort(dist)                    #按从小到大排序

    classcount = {}                       

    for i in range(k): 
        votel = label[sortdist[i]]                 #用label产生新的键

        classcount[votel] = classcount.get(votel,0) + 1    #get的作用：若字典中有votel存在，则返回其值；若没有则初始一个并且初始值设置为0

    maxcount = 0
    for key,value in classcount.items():           #遍历字典查询最大值对应的种类
        if value > maxcount:
            maxcount = value
            classes = key

    return classes

dataset,labels = creatdata()

data_2 = data[700:768]                  #初始化测试集
test_data = data_2.drop('Outcome',axis = 1)
test_array = np.array(test_data)
test_array = min_max.fit_transform(test_array)

output = []

true_output = data_2.pop('Outcome')
true_array = np.array(true_output)

maxcorrect = 0
maxj = 0

for j in range(50,200):                #设置循环寻找最优K值

    for i in range(68):           
        input = test_array[i]
        k = j
        output.append(classify(input,dataset,labels,k))

#print("the output is:",output,"the true output is:",true_array)

    correct_array = true_array - output
    correct_array = correct_array **2
    sf = np.sum(correct_array,axis = 0)
    #print(sf)
    st = 68 - sf
    correct = st/68
    #print(correct)

    del output[:]                     #每循环一次要将列表清零不然会导致列表越来越长

    if correct > maxcorrect:
        maxcorrect = correct
        maxj = j
      
print("正确率最高为",maxcorrect)
print("此时所取的K值为",maxj)