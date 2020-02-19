#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 15:28:34 2017

@author: Fay
"""

import numpy as np
import scipy as sp

from CommonFun.CommonFun import Readin_CovMeansProba
from Fuzzy.APrioriFuzzyLaw_Series1    import LoiAPrioriSeries1
from Fuzzy.APrioriFuzzyLaw_Series2    import LoiAPrioriSeries2
from Fuzzy.APrioriFuzzyLaw_Series2bis import LoiAPrioriSeries2bis
from Fuzzy.APrioriFuzzyLaw_Series3    import LoiAPrioriSeries3
from Fuzzy.APrioriFuzzyLaw_Series4    import LoiAPrioriSeries4
from Fuzzy.APrioriFuzzyLaw_Series4bis import LoiAPrioriSeries4bis


def simulateFuzzy(filenameParam, FSParameters, N):

    n_r, A, B, Q, Cov, Mean_X, Mean_Y = Readin_CovMeansProba(filenameParam)
    n_x = np.shape(Mean_X)[1]
    n_y = np.shape(Mean_Y)[1]
    n_z = n_x+n_y
    s_xz     = slice(n_x, n_z)
    s_0x     = slice(0,   n_x)
    s_0z     = slice(0,   n_z)


    # Le bruit
    W = np.random.normal(loc=0., scale=1., size=(n_z, N))

    # Le modèle Flou
    FS = None
    if FSParameters[0] == '1':
        FS = LoiAPrioriSeries1(alpha=float(FSParameters[1]), gamma=float(FSParameters[2]))
    elif FSParameters[0] == '2':
        FS = LoiAPrioriSeries2(alpha=float(FSParameters[1]), eta=float(FSParameters[2]), delta=float(FSParameters[3]))
    elif FSParameters[0] == '2bis':
        FS = LoiAPrioriSeries2bis(alpha=float(FSParameters[1]), eta=float(FSParameters[2]), delta=float(FSParameters[3]), lamb=float(FSParameters[4]))
    elif FSParameters[0] == '3':
        FS = LoiAPrioriSeries3(alpha=float(FSParameters[1]), delta=float(FSParameters[2]))
    elif FSParameters[0] == '4':
        FS = LoiAPrioriSeries4(alpha=float(FSParameters[1]), gamma=float(FSParameters[2]), delta_d=float(FSParameters[3]), delta_u=float(FSParameters[4]))
    elif FSParameters[0] == '4bis':
        FS = LoiAPrioriSeries4bis(alpha=float(FSParameters[1]), gamma=float(FSParameters[2]), delta_d=float(FSParameters[3]), delta_u=float(FSParameters[4]), lamb=float(FSParameters[5]))
    else:
        input('Impossible')

    # La chainee de Markov
    R       = np.zeros(shape=(1, N))
    R[0, 0] = FS.tirageR1()
    for np1 in range(1, N):
        R[0, np1] = FS.tirageRnp1CondRn(R[0, np1-1])

    # Les moyennes
    Mean_Z = np.zeros(shape=(n_z, N))
    for np1 in range(N):
        alpha = R[0, np1]
        Mean_Z[:, np1] =  np.hstack((InterLineaire_Vector(Mean_X, alpha), InterLineaire_Vector(Mean_Y, alpha)))

    # Simulation de Z = [[X],[Y]]
    #######################################@
    Z    = np.zeros(shape=(n_z, N)) # Z = [[X],[Y]]
    N_xy = np.zeros(shape=(n_z, N))

    i = 0
    N_xy[:, i]     = Mean_Z[:, 0]
    Cov_alpha_beta = InterBiLineaire_Matrix(Cov, R[0, i], R[0, i+1])
    Z[:, i]        = np.random.multivariate_normal(mean=N_xy[:, i], cov=Cov_alpha_beta[s_0z, s_0z], size=1)
    for i in range(1, N):
        alpha = R[0, i-1]
        beta  = R[0, i]
        A_alpha_beta = InterBiLineaire_Matrix(A, alpha, beta)
        B_alpha_beta = InterBiLineaire_Matrix(B, alpha, beta)

        N_xy[:, i] = Mean_Z[:, i] - np.dot(A_alpha_beta, Mean_Z[:, i-1])
        Z[:, i]    = np.dot(A_alpha_beta, Z[:, i-1]) + np.dot(B_alpha_beta, W[:, i]) + N_xy[:, i]

    X = Z[s_0x, :]
    Y = Z[s_xz, :]
    return n_r, X, R, Y

def InterLineaire_Vector(Vect, alpha):
    """ Return the Vect_alpha by linear interpolation of vectors in Vect."""
    assert alpha >= 0 and alpha <= 1.
    assert np.shape(Vect)[0] == 2, print('The number of matrices should be 2') 
    return (1. - alpha) * Vect[0, :] + alpha * Vect[1, :]

def InterLineaire_Matrix(Matrix, alpha):
    """ Return the Matrix_alpha by linear interpolation of matrices in Matrix."""
    assert alpha >= 0 and alpha <= 1.
    assert np.shape(Matrix)[0] == 2, print('The number of matrices should be 2') 
    return (1. - alpha) * Matrix[0, :] + alpha * Matrix[1, :]

def InterBiLineaire_Matrix(Matrix, alpha, beta):
    """Return Matrix_alpha_beta obtained by bilinear interpolation of Matrices in Matrix."""
    assert alpha >= 0 and alpha <= 1.
    assert beta  >= 0 and beta  <= 1.
    assert np.shape(Matrix)[0] == 4, print('The number of Matrixs should be 4')
    return (1. - alpha) * (1. - beta) * Matrix[0, :, :] \
        + alpha * beta * Matrix[3, :, :] + alpha * (1. - beta) * Matrix[2, :, :] \
        + beta * (1. - alpha) * Matrix[1, :, :]
