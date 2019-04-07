import numpy as np 
import pandas as pd 

data = pd.read_csv('diabetes.csv')

x = data.iloc[:,[0,1,2,3,4,5,6,7]].values
y = data.iloc[:,8].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state = 0)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

from sklearn.linear_model import LogisticRegression
classify = LogisticRegression()
classify.fit(x_train,y_train)

y_pred = classify.predict(x_test)
print("预测结果是:",y_pred)

diff = (y_pred - y_test) **2
sumdiff = np.sum(diff)
sumtrue = 192 - sumdiff

right = sumtrue/192
print("正确率为:",right)
