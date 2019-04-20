import pandas as pd 
import numpy as np
from math import log


def Creatdataset():
    data = pd.read_csv('diabetes.csv')
    dataset = data.iloc[0:20, :].values
    labels = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
    return dataset,labels


def Caculateent(dataset, axis, feature, value):      #计算某一个特征的某一个特征值的信息熵的函数
    sum = len(feature)
    n = 0.0                           #这一类特征值所占的个数
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


def Classifypoint(feature, mid, axis, dataset):  
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
        sument = Caculateent(dataset, axis, feature_s, 0) + Caculateent(dataset, axis, feature_s, 1)
        print(sument)
        if sument < minent:
            minent = sument
            bestpoint = point  
    return bestpoint        


if __name__ == "__main__":
    dataset,labels = Creatdataset()
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

    print(dataset)                    