import testFunctions as testF
import numpy as np
import math
import matplotlib.pyplot as plt
import random

n_sample = 1000
x = np.linspace(0, 1, n_sample)
f1 = np.vectorize(testF.function_1)
f2 = np.vectorize(testF.function_2)
f3 = np.vectorize(testF.function_3)
# set noise variance
Q = 0.1
R = 0.1
v = np.random.normal(0, Q, size=n_sample)
w = np.random.normal(0, R, size=n_sample)


X=np.zeros(n_sample)
X[0] = x[0]

for i in range(n_sample-1):
    X[i+1] = f3(X[i]) # add noise with : + v[i]

plt.figure(1)
plt.scatter(X[0:-1], X[1:]) # plot le data set dans le plan (X[t], X[t+1]) mais le rendu est bof...
#plt.xlim(-5,5)
#plt.ylim(-5,5)
plt.show()