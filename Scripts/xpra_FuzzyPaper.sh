
nohup python3 Test_CGOFMSM.py Parameters/Fuzzy/SP2018.param 4:0.15:0.60:0.20:0.20 1,1,0 300 0,1,2,3,5,7,9 50 1 0 > courbe4.out &
nohup python3 Test_CGOFMSM.py Parameters/Fuzzy/SP2018.param 2:0.07:0.16:0.05      1,1,0 300 0,1,2,3,5,7,9 50 1 0 > courbe2.out &

nohup python3 Test_CGOFMSM.py Parameters/Fuzzy/SP2018.param 2:0.07:0.275:0.05     1,1,0 1000  4 20 1 0 > serie2_percen1.out & #--> 48%
nohup python3 Test_CGOFMSM.py Parameters/Fuzzy/SP2018.param 2:0.07:0.21:0.05      1,1,0 1000  4 20 1 0 > serie2_percen2.out & #--> 58%
nohup python3 Test_CGOFMSM.py Parameters/Fuzzy/SP2018.param 2:0.07:0.16:0.05      1,1,0 1000  4 20 1 0 > serie2_percen3.out & #--> 67%
nohup python3 Test_CGOFMSM.py Parameters/Fuzzy/SP2018.param 2:0.07:0.108:0.05     1,1,0 1000  4 20 1 0 > serie2_percen4.out & #--> 75%
nohup python3 Test_CGOFMSM.py Parameters/Fuzzy/SP2018.param 2:0.07:0.005:0.05     1,1,0 1000  4 20 1 0 > serie2_percen5.out & #--> 93%

nohup python3 Test_CGOFMSM.py Parameters/Fuzzy/SP2018.param 4:0.10:0.55:0.30:0.30 1,1,0 1000 10 20 1 0 > serie4_percen1.out & #--> 48%
nohup python3 Test_CGOFMSM.py Parameters/Fuzzy/SP2018.param 4:0.15:0.65:0.15:0.15 1,1,0 1000 10 20 1 0 > serie4_percen2.out & #--> 62%
nohup python3 Test_CGOFMSM.py Parameters/Fuzzy/SP2018.param 4:0.15:0.65:0.10:0.10 1,1,0 1000 10 20 1 0 > serie4_percen3.out & #--> 75%
nohup python3 Test_CGOFMSM.py Parameters/Fuzzy/SP2018.param 4:0.15:0.55:0.06:0.06 1,1,0 1000 10 20 1 0 > serie4_percen4.out & #--> 87%
nohup python3 Test_CGOFMSM.py Parameters/Fuzzy/SP2018.param 4:0.15:0.65:0.00:0.00 1,1,0 1000 10 20 1 0 > serie4_percen5.out & #--> 100%
