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

y_output = np.zeros((T,2))
x_true = np.random.normal(0, 1, size=(T,1))
#x_true = np.random.multivariate_normal([0], np.array([[1]]), size=(T,1))
print(x_true.shape())
w = np.random.multivariate_normal([0,0], R_true, size=(T,2))
y_output = C_true.dot(x_true)+d_true[:,np.newaxis]+w


ssm=kalman.StateSpaceModel(
	is_f_linear= True,
	is_g_linear= True,
	state_dim=1,
	input_dim=0,
	output_dim=2,
	output_sequence=y_output,
	Sigma_0=np.ones((1,1)),# je ne sais pas quoi en faire pour l'instant
	Q=np.ones((1,1)),
	C=np.array([[12],[21]]),
	d=np.array([0, 2]),
	R=np.array([[1 , 0],[0 , 1]])
	)

ssm.initialize_f_with_factor_analysis()

print(ssm.C)

