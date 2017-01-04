# -*- coding: utf-8 -*-
import testFunctions as testF
import numpy as np
import math
import matplotlib.pyplot as plt
import random

# script qui construit nos exemples 1-D où la dynamique est non-lineaire mais de la forme
# RBF et où la relation state-output est linaire
# avec un input lineaire pour les states mais pas pour les autres...


#n_sample est donc le nbr de pas de tps.
n_sample = 1000
#le vrai state est dans [0,1], le noisy sera plus ou moins dans ces eaux là
x = np.linspace(0, 1, n_sample)
f1 = np.vectorize(testF.function_1)
f2 = np.vectorize(testF.function_2)
f3 = np.vectorize(testF.function_3)
f4 = np.vectorize(testF.function_4)
# set noise variance
Q1 = np.array([[0.03]])
R1 = np.array([[0.01,0],[0,0.01]])
Q2 = np.array([[0.03]])
R2 = np.array([[0.01,0],[0,0.01]])
Q3 = np.array([[0.03]])
R3 = np.array([[0.01,0],[0,0.01]])
Q4 = np.array([[0.03]])
R4 = np.array([[0.01,0],[0,0.01]])
#on va fixer l'alea
random.seed(10)
v = np.random.normal(0, Q1, size=n_sample)
w = np.random.multivariate_normal([0,0], R1, size=n_sample)
#on va prendre un input qui est croissant
U1=np.zeros((n_sample,1))
U1[:,0]=0*np.sin(np.linspace(0, 1, n_sample))



########PREMIER CAS#############
###On va prendre Y de dimension 2
X1_noisy= np.zeros((n_sample,1))
#les outputs sont de dimensions 2
Y1_noisy = np.zeros((n_sample,2))
#on fait de l'initialisation
X1_noisy[0] = 0

#Les minuscules pour les vecteurs et les majuscules pour les matrices
A1 =  np.array([[0.4 / 3.4]]) #deja inclu dans f1
B1 =  np.array([[0]])
b1 =  np.array([1.5 / 3.4]) #deja inclu dans f1
C1 =  np.array([[1],[2]])
D1 =  np.array([[0],[0]])
d1 =  np.array([3,4]) 


A2 =  np.array([[-0.4 / 3.4]]) #deja inclu dans f1
B2 =  np.array([[0]])
b2 =  np.array([1.9 / 3.4]) #deja inclu dans f1
C2 =  np.array([[1],[2]])
D2 =  np.array([[0],[0]])
d2 =  np.array([3,4]) 


A3 =  np.array([[-0.4 / 3.4]]) #deja inclu dans f1
B3 =  np.array([[0]])
b3 =  np.array([1.9 / 3.4]) #deja inclu dans f1
C3 =  np.array([[1],[2]])
D3 =  np.array([[0],[0]])
d3 =  np.array([3,4]) 


A4 =  np.array([[3.3 / 3.4]]) #deja inclu dans f1
B4 =  np.array([[0]])
b4 =  np.array([0.2 / 3.4]) #deja inclu dans f1
C4 =  np.array([[1],[2]])
D4 =  np.array([[0],[0]])
d4 =  np.array([3,4]) 

for i in range(n_sample-1):
    X1_noisy[i+1,0] = f1(X1_noisy[i])+B1[0,0]*U1[i,0] + v[i+1]
    Y1_noisy[i,:]= C1.dot(X1_noisy[i,:])+d1+w[i,:]

Y1_noisy[n_sample-1,:]= C1.dot(X1_noisy[n_sample-1,:])+d1+w[n_sample-1,:]



#plt.figure(1)
#plt.scatter(X1_noisy[0:-1], X1_noisy[1:]) # plot le data set dans le plan (X[t], X[t+1]) mais le rendu est bof...
#plt.plot(x,f1(x),'r-')
#plt.plot(x,x)
#plt.title('True state VS noisy with noise = ' + str(Q))
#plt.show()







