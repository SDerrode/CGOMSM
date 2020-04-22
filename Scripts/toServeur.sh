
git add -A && git commit -a -m "avant archivage" && git archive --format=zip HEAD > CGPMSM.zip
scp CGPMSM.zip  sderrode@156.18.90.100:~/
ssh sderrode@156.18.90.100 
#rm -r -f CGPMSM && mkdir CGPMSM && cd CGPMSM && mv ../CGPMSM.zip . && unzip CGPMSM.zip && rm CGPMSM.zip


#xpra start ssh:sderrode@156.18.90.100 --start=xterm
#nohup python3 
# CGOFMSM_SimRest.py Parameters/Fuzzy/SP2018.param 1000 1,2,3,5   10 0 1 0 1 2:0.07:0.27:0.05   > serie2.out &
#nohup python3 
# CGOFMSM_SimRest.py Parameters/Fuzzy/SP2018.param 1000 5,7,10,15 10 0 1 0 1 4:0.01:0.01:0.2:0.2 > serie4.out &

