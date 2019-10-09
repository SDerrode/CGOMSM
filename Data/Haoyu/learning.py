#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import clipboard

def main():

    Prefix='../../../Data_CGPMSM/Haoyu/'

    hard, filt, smooth = 1, 1, 0
    chWork  = str(hard) + ',' + str(filt) + ',' + str(smooth)
    steps   = '1,3,5'
    verbose = 2
    plot    = 1

    # reading the data
    name         = Prefix + 'Data/DFBSmoothData.txt'
    haoyuData_df = pd.read_csv(name, parse_dates=[0], dtype = 'float')
    haoyuData_df.AccelX = haoyuData_df.AccelX.astype(float)
    # print(haoyuData_df.head(5))
    # for y in haoyuData_df.columns:
    #   print(haoyuData_df[y].dtype, end=', ')
    # listHeader = list(haoyuData_df)
    # print(listHeader)

    n_x, n_y = 2, 6
    n_z      = n_x + n_y
    n_r      = int(haoyuData_df['Steps'].nunique())


    ##################################
    # Apprentissage des paramètres X,Y cond R --> enregistrement du fichier de paramètres
    filanemaneParam = 'SignalHaoyu.param'

    # # Learning parameters mean
    meanX, meanY, cpt = getMean(haoyuData_df, n_x, n_y, n_r)
    # meanX = np.around(meanX, decimals=4)
    # meanY = np.around(meanY, decimals=4)
    # print('meanX=', meanX)
    # print('meanY=', meanY)
    # print('cpt=', cpt)

    # Leanrnig parameters Cov
    Cov        = getCov(haoyuData_df, n_r, n_x, n_y, meanX, meanY)
    Cov_CGPMSM = convertCov_CGPMSM(Cov, n_r, n_x, n_y)

    # Learning of jump proba
    JointProba = getJProba(haoyuData_df, n_r)

    # Enregistrement du fichier de paramètres
    printfileParam('./../../Parameters/Signal/' + filanemaneParam, Cov_CGPMSM, JointProba, meanX, meanY, n_r)

def getMean(df, n_x, n_y, n_r):

    meanX = np.zeros((n_r, n_x))
    meanY = np.zeros((n_r, n_y))
    cpt   = np.zeros((n_r))
    df_X = df.iloc[:,7:9].values
    df_Y = df.iloc[:,0:6].values
    for i, jump in enumerate(df['Steps']):
        k = int(jump)-1
        cpt[k]     += 1.
        meanX[k, :] = meanX[k, :] + df_X[i, :]
        meanY[k, :] = meanY[k, :] + df_Y[i, :]

    for r in range(n_r):
        meanX[r, :] /= cpt[r]
        meanY[r, :] /= cpt[r]

    return meanX, meanY, cpt


def getCov(df, n_r, n_x, n_y, meanX, meanY):

    df_X = df.iloc[:,7:9].values
    df_Y = df.iloc[:,0:6].values

    n_z = n_x + n_y
    Cov = np.zeros((n_r*n_r, n_z*2, n_z*2))
    cpt = np.zeros((n_r*n_r))

    cptCouples = 0
    for i, jump in enumerate(df['Steps'][:-1]):
        j = int(df['Steps'][i]-1)
        k = int(df['Steps'][i+1]-1)

        l = j*n_r+k

        cpt[l] += 1.
        A = (meanX[j, :] - df_X[i,   :])[np.newaxis]
        B = (meanY[j, :] - df_Y[i,   :])[np.newaxis]
        C = (meanX[k, :] - df_X[i+1, :])[np.newaxis]
        D = (meanY[k, :] - df_Y[i+1, :])[np.newaxis]

        Cov[l, 0:n_x, 0:n_x]         += np.transpose(np.dot(np.transpose(A), A))
        Cov[l, 0:n_x, n_x:n_z]       += np.transpose(np.dot(np.transpose(B), A))
        Cov[l, 0:n_x, n_z:n_z+n_x]   += np.transpose(np.dot(np.transpose(C), A))
        Cov[l, 0:n_x, n_z+n_x:2*n_z] += np.transpose(np.dot(np.transpose(D), A))

        Cov[l, n_x:n_z, 0:n_x]         +=  np.transpose(np.dot(np.transpose(A), B))
        Cov[l, n_x:n_z, n_x:n_z]       +=  np.transpose(np.dot(np.transpose(B), B))
        Cov[l, n_x:n_z, n_z:n_z+n_x]   +=  np.transpose(np.dot(np.transpose(C), B))
        Cov[l, n_x:n_z, n_z+n_x:2*n_z] +=  np.transpose(np.dot(np.transpose(D), B))

        Cov[l, n_z:n_z+n_x, 0:n_x]         +=  np.transpose(np.dot(np.transpose(A), C))
        Cov[l, n_z:n_z+n_x, n_x:n_z]       +=  np.transpose(np.dot(np.transpose(B), C))
        Cov[l, n_z:n_z+n_x, n_z:n_z+n_x]   +=  np.transpose(np.dot(np.transpose(C), C))
        Cov[l, n_z:n_z+n_x, n_z+n_x:2*n_z] +=  np.transpose(np.dot(np.transpose(D), C))

        Cov[l, n_z+n_x:2*n_z, 0:n_x]         +=  np.transpose(np.dot(np.transpose(A), D))
        Cov[l, n_z+n_x:2*n_z, n_x:n_z]       +=  np.transpose(np.dot(np.transpose(B), D))
        Cov[l, n_z+n_x:2*n_z, n_z:n_z+n_x]   +=  np.transpose(np.dot(np.transpose(C), D))
        Cov[l, n_z+n_x:2*n_z, n_z+n_x:2*n_z] +=  np.transpose(np.dot(np.transpose(D), D))

    for l in range(n_r**2):
        if cpt[l] != 0:
            Cov[l, :, :] /= cpt[l]

    return Cov


def convertCov_CGPMSM(Cov, n_r, n_x, n_y):
    n_z = n_x+ n_y
    Cov_CGPMSM = Cov.copy()

    Sigma = np.zeros((n_r, n_z, n_z))
    # Recuperation of the Sigma matrices
    for r in range(n_r):
        l = r*n_r+r
        Sigma[r, :, :] = Cov[l, 0:n_z, 0:n_z]

    OK = True
    for l in range(n_r**2):
        j = l//n_r
        k = l%n_r
        Cov_CGPMSM[l, 0:n_z,     0:n_z]     = Sigma[j, :, :]
        Cov_CGPMSM[l, n_z:2*n_z, n_z:2*n_z] = Sigma[k, :, :]

    return Cov_CGPMSM

def getJProba(df, n_r):

    N = np.shape(df)[0]
    df1 = df.iloc[:,6].values

    JointProba = np.zeros(shape=(n_r, n_r))
    for i, classe in enumerate(range(N-1)):
        JointProba[int(df1[i]-1), int(df1[i+1])-1] += 1.
    JointProba /= (N-1.)

    return JointProba


def printfileParam(filanemane, Cov, JointProba, meanX, meanY, n_r):

    # L'entete
    f = open(filanemane, 'w')
    f.write('#=====================================#\n# parameters for CGPMSM            # \n#=====================================# \n# \n# \n# matrix Cov_XY \n# ===============================================#\n# \n')
    f.close()

    f = open(filanemane, 'ab')
    
    # Les covariances
    for j in range(n_r):
        for k in range(n_r):
            l = j*n_r+k
            np.savetxt(f, Cov[l, :, :], delimiter=" ", header='Cov_xy'+str(j)+str(k)+'\n----------------------------', footer='\n', fmt='%.10g')

    # Les moyennes
    np.savetxt(f, JointProba, delimiter=" ", header='joint jump matrix'+'\n================================', footer='\n', fmt='%.4f')

    # Les moyennes
    np.savetxt(f, meanX, delimiter=" ", header='mean of X'+'\n================================', footer='\n', fmt='%.10f')
    np.savetxt(f, meanY, delimiter=" ", header='mean of Y'+'\n================================', footer='\n', fmt='%.10f')

    f.close()

if __name__ == '__main__':
    main()