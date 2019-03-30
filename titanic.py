import pandas as pd
import numpy as np

train_data = pd.read_csv('train.csv')  #读取数据集

age = train_data['Age']
#print(age)

mean_age = age.mean()   #计算年龄均值
#print(mean_age)

sur_for_pclass = train_data.pivot_table(index="Pclass",values="Survived",aggfunc=np.mean)     #计算每个船票等级的生存平均率
#print(sur_for_pclass)

fare = train_data['Fare']    #求船票价格的最大最小值
#print(max(fare))
#print(min(fare))

embarked = train_data['Embarked']
#print(embarked.value_counts())    #求各个上船地点出现次数

#print(len(age))
age_null = pd.isnull(age)
#print(age_null)
age_null_true = age[age_null]
#print(age_null_true)
len_true_age = len(age) - len(age_null_true)    #计算存在的年龄个数
#print(len_true_age)
len_age = len(age)

age.fillna(mean_age , inplace = True)   #用平均值填充缺失值
#print(age)

a = 0

for i in range(0,len_age):
    a = a+((age[i]-mean_age)*(age[i]-mean_age)) #遍历全部年龄计算差值平方和，缺失值用平均值填充所以都是0，不会对计算产生影响

#print(a)

age_variance = a/len_true_age   #计算方差时不需要将缺失值计算进去，因此除以存在的年龄值即可
#print(age_variance)