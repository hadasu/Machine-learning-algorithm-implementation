# -*- coding: utf-8 -*-
"""

@author: Hadas
linear regression 
"""

import numpy as np
import matplotlib.pyplot as plt

#def fib_recursive(n):
#    if n == 0:
#        return 0
#    elif n == 1:
#        return 1
#    else:
#        return fib_recursive(n-1) + fib_recursive(n-2)
    
def fib(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b
        
        
#      t     -1   t
#h = (x  * x)  * x  *y       
def linear_regression(x,y):
    assert len(x.shape)==2
    return np.dot(np.dot(np.linalg.inv(np.dot(x.transpose(),x)),x.transpose()),y)

#1.a
a = np.random.randint(10,size=10)
print(a)

#1.b 
b = np.random.rand(10)
print(b)

#1.c
c = 3*np.random.randint(10,size=10)
print(c)

#1.d
#print(random.choice([fib_recursive(i) for i in range(10)])) 
print(list(fib(10)))

#2.a
x_points = np.random.rand(10)
first_array = 2*x_points

#2.b
gaussian_noise = np.random.normal(0,0.01,10)
first_array= first_array + gaussian_noise

#2.c
second_array = 3*x_points +3
second_array = second_array + gaussian_noise

#2.d
third_array = 2*(x_points**2) + 1*x_points + 2

#3.a
m1 = np.random.rand(4,4)
m2 = np.random.rand(4,4)

#3.b
m = np.dot(m1,2)

#3.c
m_t= m.transpose()
m_i = np.linalg.inv(m)
print(m_i)

#4
x = x_points.reshape(10,1)
linear_regression_h = linear_regression(x,first_array)
print("first_array Y=X*", linear_regression_h)

plt.figure()
plt.scatter(x,first_array)
points= np.linspace(0,1,11)
plt.plot(points, points*linear_regression_h)
plt.scatter(x, first_array,color='green')
plt.show()

#5
#[1,x]
#[1,x]
#[1,x].. = x2 
x2=np.hstack((np.ones((10,1)),x))
linear_regression_h2=linear_regression(x2,second_array)
print("second_array Y=X*" + str(linear_regression_h2[1]) + "+" + str(linear_regression_h2[0]))

plt.figure()
plt.scatter(x,second_array)
points= np.linspace(0,1,11)
plt.plot(points, points*linear_regression_h2[1]+linear_regression_h2[0], )
plt.scatter(x, second_array, color='green')
plt.show()

#7
x3 = np.hstack((np.ones((10,1)),x, x**2))
linear_regression_h3 =linear_regression(x3,third_array)
print('third_array Y=X^2*'+ str(linear_regression_h3[2])+ "+ X*" + 
      str(linear_regression_h3[1])+"+"+ str(linear_regression_h3[0]))

plt.figure()
plt.scatter(x,third_array)
plt.plot(points, (points**2)*linear_regression_h3[2]+points*linear_regression_h3[1]+linear_regression_h3[0])
plt.scatter(x, third_array, color='green')
plt.show()


#8
x=[0.08750722,0.01433097,0.30701415,0.35099786,0.80772547,0.16525226,0.46913072,0.69021229,0.84444625,0.2393042,0.37570761,0.28601187,0.26468939,0.54419358,0.89099501,0.9591165,0.9496439 ,0.82249202,0.99367066,0.50628823]
x=(np.array(x)).reshape(len(x),1)
y=[4.43317755,4.05940367,6.56546859,7.26952699,33.07774456,4.98365345,9.93031648,20.68259753,38.74181668,5.69809299,7.72386118,6.27084933,5.99607266,12.46321171,47.70487443,65.70793999,62.7767844 ,35.22558438,77.84563303,11.08106882]
y=np.array(y)
y_new=np.log(y)

x4=np.hstack((np.ones((20,1)),x, x**2))
reg_res = linear_regression(x4,y_new)
print(np.exp(reg_res[0]),reg_res[1],reg_res[2])













