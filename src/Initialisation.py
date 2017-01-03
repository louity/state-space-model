# - * - coding: utf - 8 - * -
"""
Created on Mon Jan  2 03:00:38 2017

@author: thomas
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import Simu as Simu
from scipy.stats import norm
import kalman


def Ini_RBF_state(X_estimated, Nbr_step_center):
    """
    Ici on fixe les valeurs des center et variance des RBF en fct des données des states.
    Techniquement il faudrait estimer des states valables (avec un model lineaire)... Ici on trichera en prenant les vrais states
    c'est a dire ce que l'on a appelé X1_noisy dans Simu.py et ici nommé X_estimated
    On va espacer les center de manière uniforme et les states sont forcement de dimension 1
    """
    M = max(X_estimated)
    m = min(X_estimated)
    center= np.zeros((Nbr_step_center,1))
    center[:,0] = np.linspace(m, M, Nbr_step_center+2)[1:(Nbr_step_center+2 - 1)]
    eta = center[0,0] - m
    #var = eta**2 / (8 * np.log(2)) * np.ones(Nbr_step_center)
    var = np.array( [eta**2 / (8 * np.log(2))*np.ones((1,1)) for i in range(Nbr_step_center)] )
    return {
        'n_rbf': Nbr_step_center,
        'centers':center,
        'width':var
    }

#verification de la fonction Ini_RBF_state

#print(Ini_RBF_state([1, 2, 3], 10)['centers'])
#print(Ini_RBF_state([1, 2, 3], 10)['width'])
#
#X_estimated = Simu.X1_noisy
#m = min(X_estimated)
#M = max(X_estimated)
#n_sample = Simu.n_sample
#Nbr_step_center = 10
#time = np.linspace(m, M, n_sample)
#dico = Ini_RBF_state(X_estimated, Nbr_step_center)
#center = dico['centers']
#var = dico['width']
#
#plt.clf
#plt.figure(3)
#for i in range(Nbr_step_center):
#    plt.plot(time, norm.pdf(time, loc=center[i,0], scale=np.sqrt(var[i,0])))
#    plt.xlim(m, M)
#plt.show()

# ok c'est bon on est content...


def Ini_Parameters_1(Nbr_step_center, Size_noise, X_estimated):
    '''
    Ici on initialise tous les parameters du RBF, dans l'ordre:
    h1;...;h_Nbr;A;B;b;Q;C;d;R
    Cette fonction permet d'initialiser le cas ou le state est de dimension 1,
    avec les vrais parametres pour A,B,b,Q,C,d,R
    auxquels on a rajouté un bruit de taille Size_Noise.
    f est non-linear, g est lineaire, il y a un input de dimension 1.
    les h_i de f sont initializes a zero.
    '''
    f_rbf_parameters = Ini_RBF_state(X_estimated, Nbr_step_center)
    f_rbf_coeffs = np.zeros(Nbr_step_center)
    Noise1=np.random.normal(0,Size_noise)
    Noise2=np.random.multivariate_normal([0,0],np.diag([Size_noise,Size_noise]))
    
    
    ssm=kalman.StateSpaceModel(
        is_f_linear= False,
        is_g_linear= True,
        state_dim=1,
        input_dim=0,
        output_dim=2,
        Sigma_0=None,# je ne sais pas quoi en faire pour l'instant
        A=(Simu.A1+Noise1)*np.ones((1,1)),
        B=(Simu.B1+Noise1)*np.ones(1),
        b=(Simu.b1+Noise1)*np.ones(1),
        Q=(Simu.Q+Noise1)*np.ones((1,1)),
        C=Simu.C1+Noise2[:,np.newaxis],
        d=Simu.d1+Noise2,
        R=Simu.R+Noise2[:,np.newaxis],
        f_rbf_parameters =f_rbf_parameters,
        f_rbf_coeffs =f_rbf_coeffs)
        
    ssm.input_sequence=Simu.U1
    return(ssm)
    

ssm=Ini_Parameters_1(10,0.01, Simu.X1_noisy)