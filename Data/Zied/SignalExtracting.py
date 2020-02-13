#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import PlotSignals as PS

Prefix     = './inputs/'
pathToSave = './results/'
listFiles  = ['Xtrain.txt', 'Ytrain.txt']


def main():

    N1=1800
    N2=2300

    # first file ##################################
    filenameorig = Prefix+listFiles[0]
    basename = os.path.basename(filenameorig)
    filename, file_extension = os.path.splitext(basename)
    print(filename, file_extension)

    # The data (simulated)
    X = np.loadtxt(filenameorig)
    N = X.shape[0]
    print('number of data : ', N)

    i = 0
    PS.draw (X, i, pathToSave, filename, listFiles[i], N1, N2)
    np.savetxt(pathToSave+filename+'_extract.out', X[N1:N2])

    # second file ################################
    filenameorig = Prefix+listFiles[1]
    basename = os.path.basename(filenameorig)
    filename, file_extension = os.path.splitext(basename)
    print(filename, file_extension)

    # The data (simulated)
    Y = np.loadtxt(filenameorig)
    Nprimer = Y.shape[0]
    if N != Nprimer:
        print('pb with number of data, Nprimer= ', Nprimer)

    i = 1
    PS.draw (Y, i, pathToSave, filename, listFiles[i], N1, N2)
    np.savetxt(pathToSave+filename+'_extract.out', Y[N1:N2])

    # draw both of them
    PS.draw1 (X, Y, 'Xtrain', 0, 'Ytrain', 1, pathToSave, "global", N1, N2)



if __name__ == '__main__':
    main()
