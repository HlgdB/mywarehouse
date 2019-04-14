import numpy as np 
import pandas as pd 
import operator
from math import log
from sklearn.model_selection import train_test_split


# def Creatdataset():
#     data  = pd.read_csv('diabetes.csv')
#     x = data.iloc[0:10, : ].values
#     x_train, x_test = train_test_split(x, test_size=0.25, random_state=0)
#     y_test = x_test[:, -1]
#     x_test = x_test[:, 0:8]

#     return x_train


def CreatDataSet_1():
    dataSet=[[0, 0, 0, 0, 'no'],
            [0, 0, 0, 1, 'no'],
            [0, 1, 0, 1, 'yes'],
            [0, 1, 1, 0, 'yes'],
            [0, 0, 0, 0, 'no'],
            [1, 0, 0, 0, 'no'],
            [1, 0, 0, 1, 'no'],
            [1, 1, 1, 1, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [2, 0, 1, 2, 'yes'],
            [2, 0, 1, 1, 'yes'],
            [2, 1, 0, 1, 'yes'],
            [2, 1, 0, 2, 'yes'],
            [2, 0, 0, 0, 'no']]
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

def SplitDataSet(dataSet,axis,value):           #按照制定的特征和特征值分割矩阵
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


def ChooseBestFeature(dataset):      #通过计算信息增益（，率，基尼系数）选择最优划分特征
    baseent = Inentropy(dataset)
    numfea = len(dataset[0]) - 1
    numdata = len(dataset)
    maxgain = 0.0
    bestfeature = -1
    for i in range(numfea):
        feaent = 0.0
        feautures_dist = {}
        for j in range(numdata):
            votel = dataset[j, i]
            feautures_dist[votel] = feautures_dist.get(votel, 0) + 1   
        for key in feautures_dist.keys():
            feaent = feaent + Caculateent(dataset, i, key)   
        gain = baseent - feaent
        print("第"+str(i)+"个索引值的特征的信息增益是"+str(gain))   
        if gain > maxgain:
            maxgain = gain
            bestfeature = i    

    return bestfeature 


def Maxcnt(classlist):
    classcount = {}
    numcl = len(classlist)
    for i in range(classlist):
        votel = classlist[i]
        classcount[votel] = classcount.get(votel, 0) +1
    sortclasscount = sorted(classcount.items(), key = operator.itemgetter(1), reverse = True)   #降序排列字典并生成一个元组列表
    return sortclasscount[0][0]    


def Createtree(dataset, labels, featlabels):
    classlist = dataset[:, -1]
    cl = list(classlist)
    if cl.count(cl[0])==len(cl):
        return cl[0]
    if len(dataset[0])==1:
        return Maxcnt(classlist)
    bestfeature = ChooseBestFeature(dataset)
    bestfeatlabel = labels[bestfeature]      #最优特征的标签
    featlabels.append(bestfeatlabel)
    mytree = {bestfeatlabel:{}}
    del(labels[bestfeature])             #删除最优索引
    uniquevalues = {}                    #获得最优特征的种类数
    for i in range(len(dataset)):
        vote = dataset[i, bestfeature]
        uniquevalues[vote] = uniquevalues.get(vote, 0) + 1   
    for value in uniquevalues:
        mytree[bestfeatlabel][value] = Createtree(SplitDataSet(dataset, bestfeature, value), labels, featlabels)
    return mytree         

       
if __name__ == "__main__":
    dataset,labels = CreatDataSet_1()
    featlabels = []
    mytree = Createtree(dataset, labels, featlabels)
    print(mytree)


