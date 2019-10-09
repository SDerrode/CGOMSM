# Traces sur les traitements

Il s'agit de filtrer les observations (6 DOF) de Haoyu (Accel and velocity) to reconstruct the trajectory (2 DOF: longitude and latitude)

## Matlab preprocessing of data given by Haoyu

	- Enregistrement des steps pahses dans un fichier csv
> csvwrite('step.txt', stepPhase)

    - Construction d'une matrice matlab avec les 3 observations
> Y(1:3, :) = motionAccelSeries

> Y(4:6, :) = motionVelocitySeries

> csvwrite('Y.txt', Y)

       Editing the file 'Y.txt', add the following header: "AccelX,AccelY,AccelZ,VelX,VelY,VelZ"

       Editing the file 'step.txt', add the following header: "stepPhases"

## Resampling data

The idea is to have as many samples for *Y.txt* or *step.txt* as for *ArroundITECH_360m_GroundTruth.txt*. 

> python resampling.py

Edit the generated file *ArroundITECH_360m_GroundTruth_interp.txt* to had the same header as the original file.

## Assembling all data to 1 file only

The resulting file is named *data/DFBSmoothData.txt*
    
> python assambling.py


