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


def Ini_RBF_state(X_estimated, Nbr_step_center):
    """
    Ici on fixe les valeurs des center et variance des RBF en fct des données des states.
    Techniquement il faudrait estimer des states valables (avec un model lineaire)... Ici on trichera en prenant les vrais states
    c'est a dire ce que l'on a appelé X1_noisy dans Simu.py et ici nommé X_estimated
    On va espacer les center de manière uniforme et les states sont forcement de dimension 1
    """
    M = max(X_estimated)
    m = min(X_estimated)
    center = np.linspace(m, M, Nbr_step_center+2)
    center = center[1:(np.size(center) - 1)]
    eta = center[0] - m
    var = eta**2 / (8 * np.log(2))
    return {
        'centers':center,
        'variance':var
    }

#verification de la fonction Ini_RBF_state
print(Ini_RBF_state([1, 2, 3], 2)['centers'])
print(Ini_RBF_state([1, 2, 3], 2)['variance'])

X_estimated = Simu.X1_noisy
m = min(X_estimated)
M = max(X_estimated)
n_sample = Simu.n_sample
Nbr_step_center = 10
time = np.linspace(m, M, n_sample)
dico = Ini_RBF_state(X_estimated, Nbr_step_center)
center = dico['centers']
var = dico['variance']

plt.clf
plt.figure(3)
for i in range(Nbr_step_center):
    plt.plot(time, norm.pdf(time, loc=center[i], scale=np.sqrt(var)))
    plt.xlim(m, M)
plt.show()
# ok c'est bon on est content...


def Ini_Parameters(Nbr_step_center, Size_noise, True_Parameters=None):
    """
    Ici on initialise tous les parameters du RBF, dans l'ordre:
    h1;...;h_Nbr;A;B;b;Q;C;d;R
    Eventuellement pour l'instant on se limite a considerer les vrais parametres auxquel on ajoute un bruit
    """
    return()
