
# region 1:Import libraries & data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

os.chdir("C:/Users/bengisu.oniz/Documents\datasets")

data = pd.read_csv('ex1data1.txt', header = None) #read from dataset
datax = pd.read_csv('ex1data1.txt', header = None) #read from dataset


data_n=data.values
m=len(data_n[:,-1])
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))
#print("X", X.shape,"THETA", theta.shape,"Y", y.shape)

 #endregion

# region 2: Cost



def computeCost(x,y,theta):
    """linear reg"""
    m=len(y)
    pre=x.dot(theta)
    error=((pre-y)**2)

    return 1/(2*m) * np.sum(error)


#NEDEN THETA 0,0?0,1 OLMASI GEREKMİYOR MU?

print(computeCost(X,y,theta))




 #endregion

# region 3: GradientDescent

def gradientDescent(X, y, theta, alpha, iterations):
    no_of_rows = len(y)
    for _ in range(iterations):
        loss = np.dot(X, theta) - y
        gradient_v = np.dot(X.T, loss)
        theta = theta - (alpha/no_of_rows) * gradient_v
    return theta

theta = gradientDescent(X, y, theta, alpha=0.5, iterations=2)
print(theta)

####################################################################

def computeCost(X, y, theta):
    """
    Take in a numpy array X,y, theta and generate the cost function     of using theta as parameter in a linear regression model
    """
    m = len(y)
    predictions = X.dot(theta)
    square_err = (predictions - y) ** 2

    return 1 / (2 * m) * np.sum(square_err)

def gradientDescent(X, y, theta, alpha, num_iters):
    """
    Take in numpy array X, y and theta and update theta by taking   num_iters gradient steps
    with learning rate of alpha

    return theta and the list of the cost of theta during each  iteration
    """

    m = len(y)
    J_history = []

    for i in range(num_iters):
        predictions = X.dot(theta)
        error = np.dot(X.transpose(), (predictions - y))
        descent = alpha * 1 / m * error
        theta -= descent
        J_history.append(computeCost(X, y, theta))

    return theta, J_history

theta_1, J_history = gradientDescent(X, y, theta, 0.01, 5)
print("h(x) =" + str(round(theta_1[0, 0], 2)) + " + " + str(round(theta_1[1, 0], 2)) + "x1")

# endregion

#region 4: Vis of the linear reg

prediction=X.dot(theta_1)

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(X[:,1], prediction, 'r', label='Prediction')
ax.scatter(X[:,1], y, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')

print(computeCost(X,y,theta))

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

#endregion



#192 848 99











