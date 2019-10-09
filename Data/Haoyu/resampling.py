#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.interpolate import interp1d


def main():

    Prefix='../../../Data_CGPMSM/Haoyu/'
    
    # Les données à interpoler
    Long_Lat_GT = np.genfromtxt(Prefix + 'Data/ArroundITECH_360m_GroundTruth.txt', delimiter=',')
    Long_Lat_GT = Long_Lat_GT[1:] # on enleve l'entete
    Norig = np.shape(Long_Lat_GT)[0]
    print('Norig=', Norig)

    # La longueur target
    stepData = np.genfromtxt(Prefix + 'Data/step.txt', delimiter=',')
    stepData = stepData[1:] # on enleve l'entete
    Ntarget = np.shape(stepData)[0]
    print('Ntarget=', Ntarget)

    xorig = np.linspace(0, Norig, num=Norig,   endpoint=True)
    xnew  = np.linspace(0, Norig, num=Ntarget, endpoint=True)
    f = interp1d(xorig, Long_Lat_GT, axis=0)
    Long_Lat_GT_new = f(xnew)
    print('np.shape(Long_Lat_GT_new)=', np.shape(Long_Lat_GT_new))

    np.savetxt(Prefix + 'Data/ArroundITECH_360m_GroundTruth_interp.txt', Long_Lat_GT_new, delimiter=',')

if __name__ == '__main__':
    main()