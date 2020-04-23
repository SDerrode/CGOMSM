#!/usr/bin/env zsh
# -*- coding: utf-8 -*-

listeF="0 1"
for F in $listeF
do
    python3 CGOFMSM_SemiSupervLearn.py ./Data/OpenEI/BuildingTemp/input/building1retail_June_Week_672.csv FS2ter 0 3 $F 0 1 & python3 CGOFMSM_SignalRest.py ./Parameters/Fuzzy/building1retail_June_Week_672_F=$F\_direct.param2 -1 0,1,1,1 ./Data/OpenEI/BuildingTemp/input/building1retail_June_Week_672.csv -1 0 1
    python3 CGOFMSM_SemiSupervLearn.py ./Data/Traffic/TMUSite5509-2/TMUSite5509-2_train.csv FS2ter 0 3 $F 0 1 & python3 CGOFMSM_SignalRest.py ./Parameters/Fuzzy/TMUSite5509-2_train_F=$F\_direct.param2 -1 0,1,1,1 ./Data/Traffic/TMUSite5509-2/TMUSite5509-2_train.csv -1 0 1
done

