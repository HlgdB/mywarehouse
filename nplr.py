import numpy as np 
import pandas as pd 

data = pd.read_csv('diabetes.csv')

x = data.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7]].values
y = data.iloc[:, 8].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler() 
x_train = mm.fit_transform(x_train)
x_test = mm.fit_transform(x_test)             
y_train = y_train.reshape((576,1))
y_test = y_test.reshape((192,1))                # 数据预处理

def sigmoid(z):                  # sigmoid函数的实现
    return 1/(1+np.exp(-z))    


def model(X, theta):                      # 计算每一组X的预测值     
    return sigmoid(np.dot(X, theta.T))    

 
def cost(x, y, theta):                    # 计算代价函数,len(x)表示X的组数即矩阵的列数
    l = -np.multiply(y, np.log(model(x, theta))) 
    r = -np.multiply(1-y, np.log(1-(model(x, theta))))
    return np.sum((l+r)/len(x))              


def gradient(x, y, theta):                   # 计算梯度函数中的导数部分
    grad = np.zeros(theta.shape)    
    error = (model(x, theta)-y).ravel()     # 将矩阵转换成行向量,把负号移到了括号里面
    for j in range(len(theta.ravel())):      # 遍历矩阵
        term = np.multiply(error, x[:, j])     # x[:,j]生成的是行一维矩阵
        grad[0, j] = np.sum(term)/len(x)

    return grad                              


def descent(x, y, theta, thresh, alpha):
    grad = np.zeros(theta.shape)
    k = 0
    i = 0

    while True:
        grad = gradient(x,y,theta)            # 一次性计算所有系数再进行更新
        theta = theta - alpha*grad

        i += 1
        value = i
        if value > thresh:
            break

    return theta


theta = np.zeros([1,8])
good_theta = descent(x_train, y_train, theta, thresh=50000, alpha=0.05)

print(good_theta)

def predict(X, theta):                    # 设置i函数，阀值为0.5，代入最优系数，计算预测值对应0还是1,注意这里返回的是一个列表而不是一个矩阵
    return[1 if x>= 0.5 else 0 for x in model(X, theta)]          # 不能直接用x_test矩阵乘以good_theta，要代入model进行运算不然无法得出0到1之间的值！！！    

# print(predict(x_test, good_theta))

predict_array = np.array(predict(x_test, good_theta)).reshape((192,1))   # 把预测值转换成二维矩阵

correct = (192 - np.sum((predict_array - y_test) ** 2)) / 192       # 计算正确率
print(correct)