
import numpy as np
import pandas as pd
import sklearn.datasets
import sklearn
from sklearn.model_selection import KFold,train_test_split
from sklearn.datasets import load_boston,load_digits
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

class NNetwork:
    
    def __init__(self,X,y,hidden=[3,4,5],typ=[ 'sigmoid' ,'sigmoid','sigmoid'],n_classes=1):
        self.hidden=hidden
        self.typ=typ
        self.n_classes=n_classes
        n_out=self.n_classes
        tot=2+len(hidden)
        layers=[X.shape[1]]+hidden+[n_out]
        self.errors=[]
        
    
        
        self.weights=[]
        for i in range(tot-1):
            w=np.random.rand(layers[i],layers[i+1])
            self.weights.append(w)
        

        self.activations=[]
        for i in range(tot):
            temp=np.zeros(layers[i])
            self.activations.append(temp)

        self.derivatives=[]
        for i in range(len(layers)-1):
            temp=np.zeros((layers[i],layers[i+1]))
            self.derivatives.append(temp)

    def sigmoid(self,x):
        return 1.0/(1.0+np.exp(-x))

    def forward_pass(self,X):
        self.activations[0]=np.array(X)
        activations=np.array(X)

        for i,w in enumerate(self.weights):
            net_inputs=np.dot(activations,w)
            activations=self.sigmoid(net_inputs)
            self.activations[i+1]=activations

        return activations

    

    def sigmoid_derivative(self,x):
        return x*(1.0-x)

    def backward_pass(self,error):
        
        for i in reversed(range(len(self.derivatives))):
            activations=self.activations[i+1]
            delta= error*self.sigmoid_derivative(activations)
            delta_reshaped=delta.reshape(delta.shape[0],-1).T
            current_activations=self.activations[i]
            current_activations_reshaped=current_activations.reshape(current_activations.shape[0],-1)

            self.derivatives[i]=np.dot(current_activations_reshaped,delta_reshaped)
            error=np.dot(delta,self.weights[i].T)
        return error

    def update(self,learning_rate):

        for i in range(len(self.weights)):
            weights=self.weights[i]
            derivatives=self.derivatives[i]
            weights+= derivatives*learning_rate


    def train(self,X,y,epochs,learning_rate):
        for _ in range(epochs):
            tot_error=0
            for j,(X_,y_) in enumerate(zip(X,y)):
                output=self.forward_pass(X_)
                error=y_-output
                self.backward_pass(error)
                self.update(learning_rate)
                tot_error+=self.mse(y_,output)
                self.errors.append(tot_error/(len(X[0])))
        if self.n_classes==1:
          print("MSE ",tot_error/len(X[0]))

    def mse(self,y,output):
        return np.average((y-output)**2)

data = load_boston()
X = data.data
y = data.target
X = sklearn.preprocessing.normalize(X)
X, y = shuffle(X, y)

ratio=0.3
train=int(ratio*len(X[0]))
X_train=X[:train,:]
X_test=X[train:,:]
y_train=y[:train]
y_test=y[train:]
MLP = NNetwork(X_train,y_train,[4,4,4,5,6,8], ['sigmoid','sigmoid','sigmoid'],1)
MLP.train(X_train,y_train,200,0.01)


output=MLP.forward_pass(X_test)

data = load_digits()
X = data.data
y = data.target
X = sklearn.preprocessing.normalize(X)
X, y = shuffle(X, y)

ratio=0.3
train=int(ratio*len(X[0]))
X_train=X[:train,:]
X_test=X[train:,:]
y_train=y[:train]
y_test=y[train:]
MLP = NNetwork(X_train,y_train,[6,6,6,6,6], ['sigmoid','sigmoid','sigmoid'],10)
MLP.train(X_train,y_train,4000,0.05)


output=MLP.forward_pass(X_test)

y_final=[]
for i in range(len(output)):
  max_ind=0
  for j in range(len(output[0])):
      if output[i,j]>output[i,max_ind]:
        max_ind=j
  y_final.append(max_ind)

temp=0
for i in range(len(y_test)):
  if y_final[i]==y_test[i]:
    temp+=1

print('Accuracy', temp*100/len(y_test))
