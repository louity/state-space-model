
import math
import matplotlib.pyplot as plt
import numpy as np

def rbf_1d(x, h, c, sigma):
    return h * math.exp(-0.5 * math.pow((x - c) / sigma, 2))

def function_1(x):
    a = 0.4 / 3.4
    b = 1.5 / 3.4
    h_1 = -1.6 / 3.4
    h_2 = -h_1
    c_1 = 5.0 / 11
    c_2 = 6.0 / 11
    sigma_1 = 0.5 / 11
    sigma_2 = sigma_1
    return a * x + b + rbf_1d(x, h_1, c_1, sigma_1) + rbf_1d(x, h_2, c_2, sigma_2)

def function_2(x):
    a = -0.4 / 3.4
    b = 1.9 / 3.4
    h_1 = 1.6 / 3.4
    h_2 = -h_1
    c_1 = 5.0 / 11
    c_2 = 6.0 / 11
    sigma_1 = 0.6 / 11
    sigma_2 = sigma_1
    return a * x + b + rbf_1d(x, h_1, c_1, sigma_1) + rbf_1d(x, h_2, c_2, sigma_2)

def function_3(x):
    a = -0.4 / 3.4
    b = 1.9 / 3.4
    h_1 = 0.5 / 3.4
    h_2 = -1.2 / 3.4
    h_3 = -h_2
    h_4 = -h_1
    c_1 = 4.0 / 11
    c_2 = 5.0 / 11
    c_3 = 6.0 / 11
    c_4 = 7.0 / 11
    sigma = 0.3 / 11
    return a * x + b + rbf_1d(x, h_1, c_1, sigma) + rbf_1d(x, h_2, c_2, sigma) + rbf_1d(x, h_3, c_3, sigma) + rbf_1d(x, h_4, c_4, sigma)

def function_4(x):
    a = 3.3 / 3.4
    b = 0.2 / 3.4
    h_1 = -6.0 / 11
    c_1 = 6.0 / 11
    sigma = 0.1 / 11
    return a * x + b + rbf_1d(x, h_1, c_1, sigma)

def function_5(x):
    a = 2.4 / 3.4
    b = 0.5 / 3.4
    h_1 = -0.7 / 3.4
    h_2 = -h_1
    c_1 = 4.0 / 11
    c_2 = 7.0 / 11
    sigma = 1.0 / 11
    return a * x + b + rbf_1d(x, h_1, c_1, sigma) + rbf_1d(x, h_2, c_2, sigma)


x = np.linspace(0, 1, 200)
f1 = np.vectorize(function_1)
f2 = np.vectorize(function_2)
f3 = np.vectorize(function_3)
f4 = np.vectorize(function_4)
f5 = np.vectorize(function_5)

plt.figure(1)
plt.plot(x, f1(x))
plt.plot(x, x, 'r--')
plt.axis('equal')
plt.xlabel('x_t')
plt.ylabel('x_t+1')

plt.figure(2)
plt.plot(x, f2(x))
plt.plot(x, x, 'r--')
plt.axis('equal')
plt.xlabel('x_t')
plt.ylabel('x_t+1')

plt.figure(3)
plt.plot(x, f3(x))
plt.plot(x, x, 'r--')
plt.axis('equal')
plt.xlabel('x_t')
plt.ylabel('x_t+1')

plt.figure(4)
plt.plot(x, f4(x))
plt.plot(x, x, 'r--')
plt.axis('equal')
plt.xlabel('x_t')
plt.ylabel('x_t+1')

plt.figure(5)
plt.plot(x, f5(x))
plt.plot(x, x, 'r--')
plt.axis('equal')
plt.xlabel('x_t')
plt.ylabel('x_t+1')

plt.show()
