# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from numpy import power, exp, sqrt, log
from numpy.linalg import inv, det
from numpy.random import multivariate_normal as mv_norm
import random
import math
import utils


def E_step_factor_Analysis(self,Nbr_iteration):
    '''
    Ici on fait une etape du E-step a la "Nbr_iteartion" iteration.
    '''
    T = len(self.output_sequence)
    
    #on recupere les parametres
    C = self.C
    CT = C.transpose()
    RInv = inv(self.R)
    p = self.state_dim

    for t in range(0, T):
        y_t = self.output_sequence[t]
        sigma_x_t = inv(np.identity(p) + CT.dot(RInv).dot(C))
        
        #c'est le resultat du E-step
        self.E_x[t] = sigma_x_t.dot(CT).dot(RInv).dot(y_t - self.d)
        self.E_xxT[t] = sigma_x_t + self.E_x[t][t][:, np.newaxis].dot(self.E_x[t][np.newaxis, :])
        
def M_step_factor_Analysis(self,Nbr_iteration):
    '''
    Ici on fait une etape du M-step a la "Nbr_iteartion" iteration.
    '''
    #on recupere la taille des donnees
    T = len(self.output_sequence)
    p = self.state_dim
    n = self.output_dim

    #variables intermediaires
    yxT = np.zeros((n, p))
    yyT = np.zeros((n, n))
    xxT = np.sum(self.E_xxT, axis=0)

    for t in range(0, T):
        
        y_t = self.output_sequence[t][:, np.newaxis]
        x_t = self.E_x[t][:, np.newaxis]

        yxT = yxT + y_t.dot(x_t.transpose())
        yyT = yyT + y_t.dot(y_t.transpose())
        

    xyT = yxT.transpose()
    #c'est le resultat du M-step
    self.C = yxT.dot(inv(xxT))
    self.R = np.diag(np.diag(yyT - C.dot(xyT)) / T)
    
def initialize_f_with_factor_analysis_Bis(self,n_EM_iterations):
    '''
    '''
    T = len(self.output_sequence)
    p = self.state_dim
    n = self.output_dim
    
    #on cree les outputs du E-step
    self.E_x = np.zeros((T, p))
    self.E_xxT = np.zeros((T, p, p))
    #le parametres renvoyés par le M-step sont des attributs de self
    
    for i in range(n_EM_iterations):
        self.E_step_factor_Analysis(i)
        self.M_step_factor_Analysis(i)
    
    #on va renvoyer une state sequence comme moyenne des E_x[t]
    self.estimated_state_sequence_with_FA = E_x
    # mais la variance des state sach
    self.estimated_state_variance_with_FA = inv(np.identity(p) + CT.dot(RInv).dot(C))

            
















            #M-step
            yxT = np.zeros((n, p))
            yyT = np.zeros((n, n))
            xxT = np.sum(E_xxT, axis=0)

            for t in range(0, T):
                
                y_t = ssm.output_sequence[t][:, np.newaxis]
                x_t = E_x[t][:, np.newaxis]

                yxT = yxT + y_t.dot(x_t.transpose())
                yyT = yyT + y_t.dot(y_t.transpose())
                
                Expected_complete_likelihood+= -0.5*np.trace(inv(R).dot((y_t-C.dot(x_t)).dot((y_t-C.dot(x_t)).transpose()) ))

            xyT = yxT.transpose()
            C = yxT.dot(inv(xxT))
            R = np.diag(np.diag(yyT - C.dot(xyT)) / T)
            
            
            CT = C.transpose()
            RInv = inv(R)

