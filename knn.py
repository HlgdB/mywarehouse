import numpy as np 
import pandas as pd 
from sklearn import preprocessing

min_max = preprocessing.MaxAbsScaler()

data = pd.read_csv('diabetes.csv')

def creatdata():
    data_1 = data[:700]
    train_data = data_1.drop('Outcome',axis = 1)
    train_array = np.array(train_data)
    train_array = min_max.fit_transform(train_array)
    train_outcome = data_1.pop('Outcome')
    outcome_array = np.array(train_outcome)

    group = train_array
    labels = outcome_array 
    return group,labels

def classify(input,data,label,k):
    datasize = data.shape[0]
    diff = np.tile(input,(datasize,1)) - data
    sqdiff = diff ** 2
    squaredist = np.sum(sqdiff,axis = 1)
    dist= squaredist ** 0.5

    sortdist = np.argsort(dist)

    classcount = {}

    for i in range(k):
        votel = label[sortdist[i]]

        classcount[votel] = classcount.get(votel,0) + 1

    maxcount = 0
    for key,value in classcount.items():
        if value > maxcount:
            maxcount = value
            classes = key

    return classes

dataset,labels = creatdata()

data_2 = data[700:768]
test_data = data_2.drop('Outcome',axis = 1)
test_array = np.array(test_data)
test_array = min_max.fit_transform(test_array)

output = []

true_output = data_2.pop('Outcome')
true_array = np.array(true_output)

maxcorrect = 0
maxj = 0

for j in range(50,200):

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

    del output[:]    

    if correct > maxcorrect:
        maxcorrect = correct
        maxj = j
      
print("正确率最高为",maxcorrect)
print("此时所取的K值为",maxj)