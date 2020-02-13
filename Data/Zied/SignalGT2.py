#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import PlotSignals as PS
import pandas as pd
import datetime as dt


Prefix     = './inputs/'
pathToSave = './results/'
listFiles  = ['Xtrain_extract.out', 'Ytrain_extract.out', 'Rtrain_extract.out']

def main():

    # first file ##################################
    
    # The observations
    filenameorig = pathToSave+listFiles[0]
    X = np.loadtxt(filenameorig)
    N = X.shape[0]
    print('number of data : ', N)

    i=1
    #PS.draw(Y, 1, pathToSave, filename, listFiles[0], 0, N, save=False)

    # The jumps ##################################
    filenameorig = pathToSave+listFiles[2]
    basename = os.path.basename(filenameorig)
    filename, file_extension = os.path.splitext(basename)
    print(filename, file_extension)
    R = np.zeros(shape=(N))+0.5
    R[20:45]   = 0.
    R[115:144] = 0.
    R[144:155] = 1.
    R[216:243] = 0.
    R[243:254] = 1.
    R[312:338] = 0.
    R[338:349] = 1.
    R[405:432] = 0.
    R[432:443] = 1.

    #i=2
    #PS.draw(R, i, pathToSave, filename, listFiles[1], 0, N, save=False)
    np.savetxt(filenameorig, R)

    # draw both of them
    PS.draw1 (X, R, 'Xtrain_extract', 0, 'Rtrain_extract', 2, pathToSave, "XRtrain", 100, 250, save=True)

    # save as csv file
    filenameorig = pathToSave+listFiles[1]
    Y = np.loadtxt(filenameorig)
    XYR = np.zeros(shape=(3, N))
    XYR[0,:] = X
    XYR[1,:] = Y
    XYR[2,:] = R
    pd1= pd.DataFrame(np.transpose(XYR))
    pd1.insert(0, 'TimeStamp', pd.date_range('20131212',freq='H', periods=500))
    pd1.columns = ['Timestamp','X','Y','R_GT']
    pd1.to_csv(pathToSave + 'XYR.csv', index=False)

if __name__ == '__main__':
    main()
