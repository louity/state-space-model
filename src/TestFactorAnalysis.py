# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from numpy import power, exp, sqrt, log
from numpy.linalg import inv, det
from numpy.random import multivariate_normal as mv_norm
import random
import math
import utils
import kalman
'''
On va simuler le modele x_{k+1}~N(0,1) et y_k=Cx_k+d+v_k avec x de dim 1, y de dim 2, T=100, d=(1 3) C= [1 2]^T et R=diag(0.1 0.4)
'''
T=1000
C_true =  np.array([[1],[2]])
d_true =  np.array([1, 3])
R_true =  np.array([[0.1 ,0] ,[0 ,0.4]])

#x_true = np.random.normal(0, 1, size=(T,1))

x_true = np.random.multivariate_normal([0], np.array([[1]]), size=T)#matrice Tx1

random.seed(10)
w = np.random.multivariate_normal([0,0], R_true, size=T)
print(w[0,0])

y_output = x_true.dot(np.transpose(C_true))+d_true[np.newaxis,:]+w



ssm=kalman.StateSpaceModel(
	is_f_linear= True,
	is_g_linear= True,
	state_dim=1,
	input_dim=0,
	output_dim=2,
	Sigma_0=np.ones((1,1)),# je ne sais pas quoi en faire pour l'instant
	Q=np.ones((1,1)),
	C=np.array([[10.1],[2.1]]),
	d=np.array([10, 3]),
	R=np.array([[0.1 , 2],[7 , 4]])
	)

ssm.output_sequence=y_output

ssm.initialize_f_with_factor_analysis()

x_learn=ssm.state_sequence

print(ssm.C)

plt.figure(1)
plt.plot(range(T),x_learn[:,0])
plt.figure(2)
plt.plot(range(T),x_true[:,0])

#la likelihood est croissante
likeli=ssm.Factor_likelihood
plot(range(30),likeli[:,0])
