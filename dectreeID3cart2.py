import numpy as np 
import pandas as pd 
import operator
from math import log
from sklearn.model_selection import train_test_split


def CreatDataSet():
    data = pd.read_csv('diabetes.csv')
    dataset = data.iloc[0:20, :].values
    dataset = np.array(dataset)
    labels = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
    return dataset,labels


def CreatDataSet_1():
    dataSet=[[0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 1, 0, 1, 1],
            [0, 1, 1, 0, 1],
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 1, 0],
            [1, 1, 1, 1, 1],
            [1, 0, 1, 2, 1],
            [1, 0, 1, 2, 1],
            [2, 0, 1, 2, 1],
            [2, 0, 1, 1, 1],
            [2, 1, 0, 1, 1],
            [2, 1, 0, 2, 1],
            [2, 0, 0, 0, 0]]
    dataSet=np.array(dataSet)        
    labels=['年龄','有工作','有自己的房子','信贷情况']
    return dataSet,labels


def Inentropy(dataset):
    current = []
    for featVec in dataset:
        current.append(featVec[-1])         #记录类别的向量
    nument = len(dataset)
    labelcounts = {}
    for i in range(nument):       #记录正负值出现次数 
        votel = current[i]
        labelcounts[votel] = labelcounts.get(votel, 0) + 1
          
    inentropy = 0.0

    for key in labelcounts:      
        prob = float(labelcounts[key])/nument
        inentropy -= prob*log(prob, 2) 

    return inentropy


def SplitDataSet(dataSet, axis, value):           #按照制定的特征和特征值分割矩阵
    dataSet = dataSet.tolist()
    retdataset=[] 
    for featVec in dataSet:
        if featVec[axis]==value:
           reducedFeatVec=featVec[:axis]
           reducedFeatVec.extend(featVec[axis+1:]) 
           retdataset.append(reducedFeatVec)
    
    retdataset = np.array(retdataset)
    return retdataset     
    

def Caculateent(dataset, axis, value):      #计算某一个特征的某一个特征值的信息熵的函数
    sum = len(dataset)
    n = 0.0                           #这一类特征值所占的个数
    n_y = 0.0
    n_n = 0.0
    ent = 0.0
    for i in range(sum):
        if dataset[i, axis] == value:
            n = n+1
            if dataset[i, -1] == 'yes':
                n_y += 1
            else:
                n_n += 1 
    if n_n == n or n_y == n:
        ent = 0
    else:
        ent = -n/sum*(n_y/n*log((n_y/n), 2) + n_n/n*log((n_n/n), 2))

    return ent         

def Caculateent_classify(dataset, axis, feature, value):      #计算某一个特征的某一个特征值的信息熵的函数
    sum = len(feature)
    n = 0.0                           
    n_y = 0.0
    n_n = 0.0
    ent = 0.0
    for i in range(sum):
        if feature[i] == value:
            n = n+1
            if dataset[i, -1] == 1:
                n_y += 1
            else:
                n_n += 1 
    if n_n == n or n_y == n:
        ent = 0
    else:
        ent = -n/sum*(n_y/n*log((n_y/n), 2) + n_n/n*log((n_n/n), 2))

    return ent         


def Classifypoint(feature, mid, axis, dataset):    #计算最佳分割点
    summid = len(mid)
    sumlist = len(feature)
    minent = 100.0
    bestpoint = 0
    for i in range(summid):
        feature_s = feature[:]                     #使用另一个列表替代feature以免feature被改变,但是直接用等于会改变一个的同时改变另一个，应该加一个[:]
        point = mid[i]
        for j in range(sumlist):
            if feature_s[j] >= mid[i]:
                feature_s[j] = 1
            else:
                feature_s[j] = 0     
        sument = Caculateent_classify(dataset, axis, feature_s, 0) + Caculateent_classify(dataset, axis, feature_s, 1)
        if sument < minent:
            minent = sument
            bestpoint = point  
    return bestpoint 

def Classify(dataset):                             #连续值变为离散值
    countdata = len(dataset)
    countfeatures = len(dataset[0]) - 1
    for i in range(countfeatures):
        feature = list(dataset[:, i])
        sortfeature = sorted(feature)
        mid = []
        for j in range(len(sortfeature) - 1):
            mid.append((sortfeature[j]+sortfeature[j+1])/2)
        bestpoint = Classifypoint(feature, mid, i, dataset)
        for k in range(countdata):
            if dataset[k, i]>=bestpoint:
                dataset[k, i] = 1
            else:
                dataset[k, i] = 0

    return dataset   


def Caculategini(dataset, axis, value):           #计算某个特征的某个特征值的基尼系数
    sum = len(dataset)
    n=0.0
    n_n=0.0
    n_y=0.0
    gini = 0.0
    for i in range(sum):
        if dataset[i, axis] == value:
            n = n+1
            if dataset[i, -1] == 1:
                n_y += 1
            else:
                n_n += 1 
    gini = n/sum * (1- ((n_y/n) **2) - ((n_n/n) **2))
    return gini   


# def ChooseBestFeature(dataset):      #通过计算信息增益选择最优划分特征
#     baseent = Inentropy(dataset)
#     numfea = len(dataset[0]) - 1
#     numdata = len(dataset)
#     maxgain = 0.0
#     bestfeature = -1
#     for i in range(numfea):
#         feaent = 0.0
#         feautures_dist = {}
#         for j in range(numdata):
#             votel = dataset[j, i]
#             feautures_dist[votel] = feautures_dist.get(votel, 0) + 1   
#         for key in feautures_dist.keys():
#             feaent = feaent + Caculateent(dataset, i, key)   
#         gain = baseent - feaent
#         print("第"+str(i)+"个索引值的特征的信息增益是"+str(gain))   
#         if gain > maxgain:
#             maxgain = gain
#             bestfeature = i    

#     return bestfeature 


def ChooseBestFeature(dataset):      #通过计算基尼系数选择最优划分特征
    numfea = len(dataset[0]) - 1
    numdata = len(dataset)
    mingini = 100.0
    bestfeature = -1
    for i in range(numfea):
        feagini = 0.0
        feautures_dist = {}
        for j in range(numdata):
            votel = dataset[j, i]
            feautures_dist[votel] = feautures_dist.get(votel, 0) + 1   
        for key in feautures_dist.keys():
            feagini = feagini + Caculategini(dataset, i, key)     
        if feagini < mingini:
            mingini = feagini
            bestfeature = i   
    return bestfeature 


def Maxcnt(classlist):
    classcount = {}
    numcl = len(classlist)
    for i in range(numcl):
        votel = classlist[i]
        classcount[votel] = classcount.get(votel, 0) +1
    sortclasscount = sorted(classcount.items(), key = operator.itemgetter(1), reverse = True)   #降序排列字典并生成一个元组列表
    return sortclasscount[0][0]    


def Createtree(dataset, labels, featlabels):
    classlist = dataset[:, -1]
    cl = list(classlist)
    if cl.count(cl[0])==len(cl):
        return cl[0]
    if len(labels)==1:
        return Maxcnt(classlist)
    bestfeature = ChooseBestFeature(dataset)
    bestfeatlabel = labels[bestfeature]      #最优特征的标签
    featlabels.append(bestfeatlabel)
    mytree = {bestfeatlabel:{}}
    del(labels[bestfeature])             #删除最优索引
    featvalues = list(dataset[:, bestfeature])
    uniquevalues = set(featvalues)
    for value in uniquevalues:
        mytree[bestfeatlabel][value] = Createtree(SplitDataSet(dataset, bestfeature, value), labels, featlabels)
       
    return mytree         

       
if __name__ == "__main__":
    dataset,labels = CreatDataSet()
    dataset = Classify(dataset)
    #print(dataset)
    featlabels = []
    mytree = Createtree(dataset, labels, featlabels)
    print(mytree)


