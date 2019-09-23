# XOR problem using RBF function (RBNN)

import numpy as np
import matplotlib.pyplot as plt 

class Perceptron(object):  
    def __init__(self,eta=0.01,n_iter=20):
        self.eta = eta
        self.n_iter = n_iter
        self.weights  = np.zeros(x.shape[1])
        self.bias = 0
    
    def train(self,x,y):
        while(self.n_iter):
            for x_train,y_train in zip(x,y):
                    error = y_train-self.predict(x_train)
                    if(error != 0):
                        self.weights += self.eta*error*x_train
                        self.bias += self.eta*error
            self.n_iter -= 1
            
    def predict(self,x):
        return 1/(1+np.exp(-(np.dot(x,self.weights)+self.bias)))
    
    def rbf(self,x):
        return np.column_stack((np.exp(-(np.square(np.sum(x-[0,0],axis=1)))),np.exp(-(np.square(np.sum(x-[1,1],axis=1))))))

x = np.array([[0,0],[1,0],[0,1],[1,1]])
y = np.array([0,1,1,0])

pnn = Perceptron(eta=0.01,n_iter=200000)
x = pnn.rbf(x)
pnn.train(x,y)
print("output : ",pnn.predict(x))

#plot
x_line = np.arange(0,1,0.01)
y_line = -(pnn.weights[0]*x_line+pnn.bias)/pnn.weights[0]
plt.plot(np.array([1,0.37,0.37,0.18]),np.array([0.018,0.37,0.37,1]),'o--')
plt.plot(x_line,y_line,linewidth=0.5)
plt.show()