# -*- coding: utf-8 -*-
"""

@author: Hadas
"""
import numpy as np

#      t     -1   t
#h = (x  * x)  * x  *y  
def linear_regression(x,y):
    assert len(x.shape)==2
    return np.dot(np.dot(np.linalg.inv(np.dot(x.transpose(),x)),x.transpose()),y)

def print_regression(X,y):
    res=linear_regression(X,y)
    print('Regression result: ',res)
    print('predicted y:', np.dot(X,res))
    print('loss :', ((np.dot(X,res)-y)**2).mean())
 
    
#A
data_older_sibling = np.array([31,22,40,26])
data_younger_sibling = np.array([22,21,37,25])
data = np.c_[data_older_sibling, data_younger_sibling]
data_y = np.array([2,3,8,12])
print_regression(data,data_y)

#B
data_sub_age = data[:,0]-data[:,1]
data = np.c_[data_older_sibling, data_younger_sibling, data_sub_age]
print_regression(data,data_y)

#C
data_sub_age = data[:,0]-data[:,1]
data = np.c_[data_older_sibling, data_younger_sibling, data_sub_age**2]
print_regression(data,data_y)

#D 
data = np.c_[data_older_sibling, data_younger_sibling, np.ones(4)]
print_regression(data,data_y)

#E ? order
data = np.c_[data_older_sibling, data_younger_sibling, np.ones(4), data_sub_age**2]
print_regression(data,data_y)



#Gradient descent
def print_regression(i, loss, point):
    print('epoch {}, loss {}, new t {}'.format(i,loss,point))


def loss_function_MSE(res, y):
    return ((res-y)**2).mean()

#1
#θ=θ − η⋅∇θJ(θ)
def batch_gradient_descent(points,y,start_point, iterations, learning_rates, loss_function):
    point = start_point.copy() 
    for i in range(iterations):
        params_grad, loss = evaluate_gradient(loss_function, points, point, y)
        point = point - learning_rates * params_grad
        print_regression(i, loss, point)
    return point
        
# ∇θJ(θ)       
def evaluate_gradient(loss_function, X, point, y):
    res=np.dot(X,point)
    loss= loss_function(res, y)
    gradient=(np.dot(res-y,X))/len(X)
    return gradient, loss


#C
#v = 1
#v=γv+η∇θJ(θ)
#θ=θ−v
def SGD_gradient_descent(points,y,start_point, iterations, learning_rates, loss_function, momentum):
    point = start_point.copy() 
    v=np.zeros_like(start_point)
    for i in range(iterations):
        params_grad, loss = evaluate_gradient(loss_function, points, point, y)
        v = momentum*v + (learning_rates * params_grad)
        point = point - v
        print_regression(i, loss, point)
    return point
        
#D
#v = 1
#v=γv+η∇θJ(θ- γv)
#θ=θ−v
def nesterov_accelerated_gradient_descent(points,y,start_point, iterations, learning_rates, loss_function, momentum):
    point = start_point.copy() 
    v=np.zeros_like(start_point)
    for i in range(iterations):
        params_grad, loss = evaluate_gradient(loss_function, points, point - momentum*v, y)
        v = momentum*v + (learning_rates * params_grad)
        point = point - v
        print_regression(i, loss, point)
    return point
       


x = np.array([0,1,2])
y = np.array([1,3,7])
points =np.c_[np.ones_like(x),x,x**2]
batch_gradient_descent(points,y,[2,2,0],100,0.1, loss_function_MSE)
SGD_gradient_descent(points,y,[2,2,0],100,0.1,loss_function_MSE, 0.5)
nesterov_accelerated_gradient_descent(points,y,[2,2,0],100,0.1,loss_function_MSE, 0.5)











