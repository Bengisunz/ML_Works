
# region 1:Import libraries &  knowing the data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

np.seterr(all='warn')

os.chdir("C:/Users/bengisu.oniz/Documents\datasets")

data1 = pd.read_csv('ex2data1.txt', header = None) #read from dataset
data2= pd.read_csv('ex2data2.txt', header = None) #read from dataset

def know_your_data(data):
    print('Head: {}'.format(data.ndim))
    print('Shape: {}'.format(data.shape))
    print('Size: {}'.format(data.size))
    print('Columns: {}'.format(data.columns))
    print("-----------------------------------")
    print('Types: {}'.format(data.dtypes))
    print('Head: {}'.format(data.head(3)))

know_your_data(data1)

plt.scatter(data1[0],data1[1], alpha=0.5,c=data1[2])
plt.xlabel('Exam 1 Score')
plt.ylabel('Exam 2 Score')
plt.title("Scatter plot of training data")
plt.legend(["Not admitted"],loc='upper left')
plt.show()

# endregion

#region 2: Data Prep

dn=data1.values
m=len(dn[:,-1])
bias_term=np.ones((m, 1))
xf=dn[:,0:2]
print("shapes",xf.shape,bias_term.shape)

X=np.concatenate((bias_term,xf),axis=1)

y=dn[:,2].reshape(m,1)
theta=np.zeros((3,1))
print("X SHAPE:", X.shape, "theta SHAPE:",theta.shape,"y SHAPE:", y.shape)

#endregion


#region 3: Logistic Regression

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def reg(x,theta):
    pre=x.dot(theta)
    return pre

def log_reg(x,theta):
    return sigmoid(reg(x,theta))

def prob(y_that):
    lt=[]
    for i in range(0,len(y_that)):
        if y_that[i]>0.5:
            lt.append(1)
        else:
            lt.append(0)
    return lt

def plot_sigmoid(x,res_sig):
    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(x, res_sig, 'r')
    plt.show()

res=prob(log_reg(X,theta))
plot_sigmoid(X,res)


#endregion

#region 3: Cost & Gradient Descent

def cost_log(x,y,theta):
    m = x.shape[0]
    pred = sigmoid(x.dot(theta))
    j= -(1/m)*np.sum((y*np.log(pred))+(1-y)*np.log(1-pred))

    return j

cost_log(X,y,theta)

Js=[]
def gradient_des(x,y,theta,alpha,iterations):
    no_of_rows=len(y)
    for _ in range(iterations):
        pred = sigmoid(x.dot(theta))
        loss=y-pred
        J=cost_log(X,y,theta)
        Js.append(J)
        gradient=np.dot(x.T,loss)
        theta=theta-(alpha/no_of_rows)* gradient
    return theta,Js


#endregion


r,J_hist = gradient_des(X, y, theta, alpha=0.5, iterations=3)
print("parameters", r)
res = prob(log_reg(X, r))

print(res)


plt.plot(J_hist)
plt.xlabel("# of Iterationa")
plt.ylabel("Theta")
plt.title("Cost function & Gradient Descent")

