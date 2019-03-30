import numpy as np
def cost(X,y,theta):
    m = len(y)
    J = 0
    
    j_1 = np.dot(X,theta) - y
    j_2 = np.transpose(j_1)
    j_3 = np.dot(j_2,j_1)
    J = j_3[0,0]/(2*m)
    
    return J



X = np.array([[1,1],[1,2],[1,3]])

y = np.array([[2],[4],[6]])

theta = np.array([0],[1.5])

print("The result of the cost function is:" + str(cost(X,y,theta)))
