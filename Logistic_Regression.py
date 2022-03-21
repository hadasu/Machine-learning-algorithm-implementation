# -*- coding: utf-8 -*-
"""

@author: hadas
Logistic Regression
"""
import numpy as np
import matplotlib.pyplot as plt

def print_regression(i, loss, point):
    print('epoch {}, loss {}, new t {}'.format(i,loss,point))

EPS = 1e-7

#4
#sigmoid function p=1/(1+e^(-y)).
def sigmoid(y):
    return 1.0 / (1 + np.exp(-y))
  
#5   
def predict(W, x):
    return sigmoid(np.dot(x, W)) > 0.5

#6
#likelihood function f = - ( y * log(sigmoid + EPS) + (1+y)*log(1 - sigmoid + EPS )/N
def negative_likelihood(p_y, y):
    sigmoidp = p_y
    return -(y*np.log(sigmoidp+EPS)+(1-y)*np.log(1-sigmoidp+EPS)).mean()

def grad_likelihood(p_y, y, x):
    sigmoidp= p_y
    return np.dot((sigmoidp-y),x)/len(x)

#8
#def train_logistic_regression(W, x, y ,learning_rate, iterations):
def train_logistic_regression(points,y,start_point, iterations, learning_rates, loss_function):
    point = start_point.copy() 
    for i in range(iterations):
        params_grad, loss = evaluate_gradient(loss_function, points, point, y)
        point = point - learning_rates * params_grad
        print_regression(i, loss, point)
    return point

def evaluate_gradient(loss_function, X, point, y):
    res=sigmoid(np.dot(X, point))
    loss= loss_function(res, y)
    gradient=np.dot((res-y),x)/len(x)
    return gradient, loss
       
            
def split_train_test(X,y,percentage_test):
    per_index=int(len(y)*(1-percentage_test))
    return X[:per_index,:],X[per_index:,:],y[:per_index],y[per_index:]
        
#1
n_points_in_cluster = 50
n_clusters = 2
std = 3
main_x1 = 4
main_y1 = 4
main_x2 = 1
main_y2 = 1

data1 = np.random.multivariate_normal([main_x1, main_y1], np.diag([std, std]), n_points_in_cluster)
data2 = np.random.multivariate_normal([main_x2, main_y2], np.diag([std, std]), n_points_in_cluster)

#2
data_point = np.r_[data1, data2]
data_point = np.c_[data_point, np.ones(n_points_in_cluster* n_clusters)]
#print(data_point)

plt.scatter(data1[:,0],data1[:,1])
plt.scatter(data2[:,0],data2[:,1])
plt.show()

#3
x = data_point
y = np.r_[np.ones([n_points_in_cluster]), np.zeros([n_points_in_cluster])]
W = np.random.rand(3)


percentage_test = 0.2
indices= np.array(range(n_points_in_cluster*n_clusters))
np.random.shuffle(indices)

#9
x_train, x_test, y_train, y_test = split_train_test(x[indices,:],y[indices],percentage_test)

point_t = train_logistic_regression(x,y,W,100,0.1, negative_likelihood)

train_precision = (predict(point_t,x_train)==y_train).sum()/len(y_train)
test_precision = (predict(point_t,x_test)==y_test).sum()/len(y_test)
print('Train precision: {} Test precision: {}, t: {}'.format(train_precision, test_precision, point_t))

#10**
#phi = np.random.uniform(0, 2*np.pi, n_points_in_cluster)
#r1 = np.random.uniform(2, 3.5, n_points_in_cluster)
#x1 = r1 * np.cos(phi)
#y1 = r1 * np.sin(phi)
#
#r2 = np.random.uniform(1.5, 2, n_points_in_cluster)
#x2 = r2 * np.cos(phi)
#y2 = r2 * np.sin(phi)
# 
#
#data1 = np.c_[x1, y1]
#data2 = np.c_[x2, y2]
#data_point = np.r_[data1, data2]
#data_point = np.c_[data_point, np.ones(n_points_in_cluster* n_clusters)]
#
#plt.scatter(x1,y1, color='red')
#plt.scatter(x2,y2, color='green')
#plt.show() 
#
#x = data_point
#y = np.r_[np.ones([n_points_in_cluster]), np.zeros([n_points_in_cluster])]
#W = np.random.rand(3)
#
#
#percentage_test = 0.2
#indices= np.array(range(n_points_in_cluster*n_clusters))
#np.random.shuffle(indices)
#
#x_train, x_test, y_train, y_test = split_train_test(x[indices,:],y[indices],percentage_test)
#
#point_t = train_logistic_regression(x,y,W,100,1, negative_likelihood)
#
#train_precision = (predict(point_t,x_train)==y_train).sum()/len(y_train)
#test_precision = (predict(point_t,x_test)==y_test).sum()/len(y_test)
#print('Train precision: {} Test precision: {}, t: {}'.format(train_precision, test_precision, point_t))




















