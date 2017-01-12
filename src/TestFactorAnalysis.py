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
T=100
C_true =  np.array([[1],[2]])
d_true =  np.array([1, 3])
R_true =  np.array([[0.1 ,0] ,[0 ,0.4]])

#x_true = np.random.normal(0, 1, size=(T,1))

x_true = np.random.multivariate_normal([0], np.array([[1]]), size=T)#matrice Tx1

random.seed(10)
w = np.random.multivariate_normal([0,0], R_true, size=T)


y_output = x_true.dot(np.transpose(C_true))+d_true[np.newaxis,:]+w



ssm=kalman.StateSpaceModel(
	is_f_linear= True,
	is_g_linear= True,
	state_dim=1,
	input_dim=0,
	output_dim=2,
	Sigma_0=np.ones((1,1)),# je ne sais pas quoi en faire pour l'instant
	Q=np.ones((1,1)),
	C=np.array([[1],[2]]),
	d=np.array([1, 3]),
	R=np.array([[0.1 , 0],[0 , 0.4]])
	)

ssm.output_sequence=y_output
E_x_true = np.zeros((T,1))
sigma_x_true = inv(np.identity(1) + C_true.transpose().dot(inv(R_true)).dot(C_true))
for t in range(T):    
    E_x_true[t]=sigma_x_true.dot(C_true.transpose()).dot(inv(R_true)).dot(y_output[t] - d_true)
ssm.E_x = E_x_true
print(ssm.Expected_complete_log_likelihood_factor_analysis())
ssm.initialize_f_with_factor_analysis_Bis(30)
print(ssm.Expected_complete_log_likelihood_factor_analysis())
#ssm.Expected_complete_log_likelihood_factor_analysis()

#ssm.E_step_factor_Analysis(2)

#Ex=ssm.E_x
#plt.figure(1)
#plt.plot(range(T),Ex[:,0])
#plt.plot(range(T),x_true[:,0])


#x_learn=ssm.estimated_state_sequence_with_FA
#
#ssm.C
#ssm.estimated_state_variance_with_FA 
#
##plottons le vrai <X_n> que l'on devrait obtenir
##Il s'agit de C^T(CC^T+R)^{-1}(y_n-d)
#sigma_x_true = inv(np.identity(1) + C_true.transpose().dot(inv(R_true)).dot(C_true))
#E_x_true = np.zeros((T,1))
#
#for t in range(T):    
#    E_x_true[t]=sigma_x_true.dot(C_true.transpose()).dot(inv(R_true)).dot(y_output[t] - d_true)
#
#plt.figure(1)
#plt.plot(range(T),x_learn[:,0])
#
#plt.figure(2)
#plt.plot(range(T),x_true[:,0])
#plt.plot(range(T),E_x_true[:,0])
##cette figure montre que <X_n> et x_true devrait etre globalement la meme chose...
#
##la likelihood est croissante
##likeli=ssm.Factor_likelihood
##plot(range(30),likeli[:,0])
##x_rapport= x_true / x_learn
#plt.figure(3)
#plt.plot(range(T),E_x_true[:,0])
##plt.plot(range(T),x_rapport[:,0])
##print(T)
##print(np.mean(x_rapport))
