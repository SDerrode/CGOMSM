##################################
# ATTENTION : LIBERER LES RANDOM SINON TOUTES LES EXP SERONT LES MEMES !!!!!
##################################

#xpra start ssh:sderrode@156.18.90.100 --start=xterm
scp sderrode@156.18.90.100:CGPMSM/{serie2_percen1.out,serie2_percen2.out,serie2_percen3.out,serie2_percen4.out,serie2_percen5.out} .
scp sderrode@156.18.90.100:CGPMSM/{serie4_percen1.out,serie4_percen2.out,serie4_percen3.out,serie4_percen4.out,serie4_percen5.out} .
scp sderrode@156.18.90.100:CGPMSM/courbe.out .
scp sderrode@156.18.90.100:CGPMSM/Result/Fuzzy/Figures/FS4_pH_0_624625_pHNum_0_585_MSE.png .

nohup python3 
CGOFMSM_SimRest.py Parameters/Fuzzy/SP2018.param 2:0.07:0.275:0.05     1,1,0 500  4 20 1 0 > serie2_F.out & #--> 48%

python3 
CGOFMSM_SimRest.py Parameters/Fuzzy/SP2018.param 2:0.07:0.16:0.05      1,1,0 500 0,1,2,3,4,5 5 1 0    > serie2_F.out &
python3 
CGOFMSM_SimRest.py Parameters/Fuzzy/SP2018.param 4:0.15:0.65:0.15 1,1,0 500 2,5,7,10,14 5 1 0    > serie4_F.out &
python3 
CGOFMSM_SimRest.py Parameters/Fuzzy/SP2018.param 4:0.15:0.65:0.15 1,1,0 500 0,1,2,3,5,7,10 5 1 0 > serie4_F.out &


# Fig 2 et 3 du papier 
#_________________
# Aller dans le répertoire fuzzy et lancer python python3 APrioriFuzzyLaw_Series2.py
# et python3 APrioriFuzzyLaw_Series4.py en modifiantra les paramètres directement dans
# les programmes principaux

# Fig 4 du papier 
#_________________
# ne pas oublier 
#   - de mettre seed(1)
#   - i_min = 6     # index min for plot
#   - i_max = 60     # index max for plot
#   - verififer le dpi == 300
rm ./Result/Fuzzy/Figures/*.png && python3 
CGOFMSM_SimRest.py Parameters/Fuzzy/SP2018.param 4:0.15:0.60:0.20 1,1,0 81 5 1 2 1


# Fig 5 du papier 
#_________________
# ne pas oublier 
#   - de mettre seed(None)
#   - verififer le dpi == 300
#nohup python3 
CGOFMSM_SimRest.py Parameters/Fuzzy/SP2018.param 4:0.15:0.60:0.20 1,1,0 300 1,2,3,5,7,9 50 1 0 > courbe.out &
nohup python3 
CGOFMSM_SimRest.py Parameters/Fuzzy/SP2018.param 2:0.07:0.16:0.05 1,1,0 300 0,1,2,3,5,7,9 50 1 0 > courbe.out &

# Table 1 du papier
# ne pas oublier 
#   - de mettre seed(None)
#   - verififer le dpi == 300

nohup python3 
CGOFMSM_SimRest.py Parameters/Fuzzy/SP2018.param 2:0.07:0.275:0.05     1,1,0 1000  4 20 1 0 > serie2_percen1.out & #--> 48%
nohup python3 
CGOFMSM_SimRest.py Parameters/Fuzzy/SP2018.param 2:0.07:0.21:0.05      1,1,0 1000  4 20 1 0 > serie2_percen2.out & #--> 58%
nohup python3 
CGOFMSM_SimRest.py Parameters/Fuzzy/SP2018.param 2:0.07:0.16:0.05      1,1,0 1000  4 20 1 0 > serie2_percen3.out & #--> 67%
nohup python3 
CGOFMSM_SimRest.py Parameters/Fuzzy/SP2018.param 2:0.07:0.108:0.05     1,1,0 1000  4 20 1 0 > serie2_percen4.out & #--> 75%
nohup python3 
CGOFMSM_SimRest.py Parameters/Fuzzy/SP2018.param 2:0.07:0.005:0.05     1,1,0 1000  4 20 1 0 > serie2_percen5.out & #--> 93%

nohup python3 
CGOFMSM_SimRest.py Parameters/Fuzzy/SP2018.param 4:0.10:0.55:0.30 1,1,0 1000 10 20 1 0 > serie4_percen1.out & #--> 48%
nohup python3 
CGOFMSM_SimRest.py Parameters/Fuzzy/SP2018.param 4:0.15:0.65:0.15 1,1,0 1000 10 20 1 0 > serie4_percen2.out & #--> 62%
nohup python3 
CGOFMSM_SimRest.py Parameters/Fuzzy/SP2018.param 4:0.15:0.65:0.10 1,1,0 1000 10 20 1 0 > serie4_percen3.out & #--> 75%
nohup python3 
CGOFMSM_SimRest.py Parameters/Fuzzy/SP2018.param 4:0.15:0.55:0.06 1,1,0 1000 10 20 1 0 > serie4_percen4.out & #--> 87%
nohup python3 
CGOFMSM_SimRest.py Parameters/Fuzzy/SP2018.param 4:0.15:0.65:0.00 1,1,0 1000 10 20 1 0 > serie4_percen5.out & #--> 100%
