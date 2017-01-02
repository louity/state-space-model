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
# set noise variance
Q = 0.01
R = 0.1
#on va fixer l'alea
random.seed(10)
v = np.random.normal(0, Q, size=n_sample)
w = np.random.normal(0, R, size=n_sample)


########PREMIER CAS#############
###Nous allons nous contenter de cela...
#construisons un output
U1=0.1*np.sin(np.linspace(0, 1, n_sample))
# X1 represente donc le true state, construit avec 2 RBF
X1=np.zeros(n_sample)
X1_noisy= np.zeros(n_sample)
Y1=np.zeros(n_sample)
Y1_noisy = np.zeros(n_sample)
#pourquoi cette initialisation
X1[0] = 0#x[0]
X1_noisy[0] = 0#x[0]+v[0]

#Les parametres a faire apprendre
#Pour f ils se lisent dans testFunction.py
#Pour "g" on les choisit ici:
C=1.5
d=2
A=1 #c'est pour les inputs
b=0.01

#IL n'y aurait pas besoin de faire de boucle...
for i in range(n_sample-1):
    X1[i+1] = f1(X1[i])+A*U1[i]+b # add noise with : + v[i]
    X1_noisy[i+1] = f1(X1[i])+A*U1[i]+b + v[i+1]
    Y1[i] = C*X1[i]+d
    Y1_noisy[i]= C*X1[i]+d+w[i]


Y1[n_sample-1]= C*X1[n_sample-1]+d
Y1_noisy[n_sample-1]= C*X1[n_sample-1]+d+w[n_sample-1]


#plt.figure(1)
#plt.scatter(X1_noisy[0:-1], X1_noisy[1:]) # plot le data set dans le plan (X[t], X[t+1]) mais le rendu est bof...
#plt.plot(x,f1(x),'r-')
#plt.plot(x,x)
#plt.title('True state VS noisy with noise = ' + str(Q))
#plt.show()
#
#plt.figure(2)
#plt.plot(x,U1)
#plt.title('The value of the input')
#plt.show()

#POur le main il faut donc recuperer X1, X1_noisy, U1, Y1, Y1_noisy

