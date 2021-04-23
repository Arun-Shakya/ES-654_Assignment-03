
import autograd.numpy as np
import pandas as pd
from autograd import grad
import matplotlib.pyplot as plt
import seaborn as sns
from autograd import grad
from sklearn import datasets
from sklearn.model_selection import train_test_split,KFold
from sklearn.utils import shuffle
import sklearn
from sklearn.decomposition import PCA

class MultiLogisticRegression():
    
    def __init__(self,epochs=50,lr=0.02,n_classes=2):
        self.coef=None
        self.classes=n_classes
        self.lr=lr
        self.epochs=epochs

    def sigmoid(z):
      return 1/(1+np.exp(-z))

    def fit_autograd(self,X,y):
        X=sklearn.preprocessing.normalize(X)
        self.coef=np.ones(len(X[0])*self.classes).reshape(len(X[0]),self.classes)

        def cost(coef):
          X_coef=np.matmul(X_,coef)
          delta=np.exp(X_coef)
          delta_=delta.sum(axis=1)
          delta=delta/delta_[:,np.newaxis]
          ans=0
          for i in range(len(y_)):
            ans+=np.log(delta[i,y_[i]])
          return -1*ans

        gradient=grad(cost)
        
        X_=X
        y_=y

        for i in range(self.epochs):
          self.coef-=self.lr*gradient(self.coef)

    def predict(self,X):
      X_test=X[:]
      X_test=sklearn.preprocessing.normalize(X_test)
      X_coef=np.matmul(X_test,self.coef)
      d=np.exp(X_coef)
      d_=d.sum(axis=1)
      d=d/d_[:,np.newaxis]
      return d
    
    def Accuracy(self,y,y_hat):
      y_final=[]
      TP,TN,FP,FN=[0]*10,[0]*10,[0]*10,[0]*10
      for i in range(len(y_hat)):
        max_ind=0
        for j in range(len(y_hat[0])):
          if y_hat[i,j]>y_hat[i,max_ind]:
            max_ind=j
        y_final.append(max_ind)
      
      temp=0
      for i in range(len(y)):
        if y_final[i]==y[i]:
          temp+=1
        for j in range(10):
          if y[i]==j and y_final[i]==j:
            TP[j]+=1
          elif y_final[i]==j:
            FP[j]+=1
          elif y[i]==j:
            FN[j]+=1
          else:
            TN[j]+=1
      print("TP ",TP)
      print("TN ",TN)
      print("FP ",FP)
      print("FN ",FN)
      return temp*100/len(y)

data = datasets.load_digits()
X = data.data
y = data.target
X, y = shuffle(X, y)
folds = KFold(n_splits=4)
folds.get_n_splits(X)

fold_id=1
tot = 0
MLR = MultiLogisticRegression(n_classes=10)
for train_index, test_index in folds.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    MLR.fit_autograd(X_train,y_train) 
    y_hat = MLR.predict(X_test)
    print("accuracy for fold:",fold_id,MLR.Accuracy(y_test,y_hat))
    tot += MLR.Accuracy(y_test,y_hat)
    fold_id+=1
print("Overall accuracy for model" ,tot/4)

pca = PCA(n_components=2)
Components = pca.fit_transform(X)
Df = pd.DataFrame(data = Components, columns = ['principal component 1', 'principal component 2'])
final_Df = pd.concat([Df, pd.Series(data = y)], axis = 1)
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
pred = [i for i in range(10)]
colors = ['0.0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9']
for pred, color in zip(targets,colors):
    ind = final_Df[0] == pred
    ax.scatter(final_Df.loc[ind, 'principal component 1'], final_Df.loc[ind, 'principal component 2'], c = color, s = 50)
ax.legend(targets)
ax.grid()
plt.show()
