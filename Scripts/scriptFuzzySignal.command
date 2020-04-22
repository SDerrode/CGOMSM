
#!/usr/bin/env zsh
# -*- coding: utf-8 -*-

listeF="1"
for F in $listeF
do
    python3 CGOFMSM_SemiSupervLearn.py ./Data/OpenEI/BuildingTemp/input/building1retail_June_Week_672.csv FS2ter 0 1 $F 0 1 & python3 CGOFMSM_SignalRest.py ./Parameters/Fuzzy/building1retail_June_Week_672_F=$F\_direct.param2 2ter:0.3592:0.2757:0.0022 0,1,1,1 ./Data/OpenEI/BuildingTemp/input/building1retail_June_Week_672.csv -1 0 1
done

# python3 CGOFMSM_SemiSupervLearn.py ./../Data_CGPMSM/OpenEI/BuildingTemp/input/building1retail_June_Week_672.csv FS2ter 0 1 1 0 1
# python3 CGOFMSM_SignalRest.py ./Parameters/Fuzzy/building1retail_June_Week_672_F=1_direct.param2 2ter:0.3592:0.2757:0.0022 0,1,1,1 ./../Data_CGPMSM/OpenEI/BuildingTemp/input/building1retail_June_Week_672.csv -1 0 1
