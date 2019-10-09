##################################
# ATTENTION : LIBERER LES RANDOM SINON TOUTES LES EXP SERONT LES MEMES !!!!!
##################################

#xpra start ssh:sderrode@156.18.90.100 --start=xterm
#nohup python3 Test_DBFSmoothing.py ...

# python3 Test_DFBSmoothing.py ./Parameters/DFBSmoothing/test5.param 100 2 1 0
# python3 Test_DFBSmoothing.py ./Parameters/DFBSmoothing/test5.param 100 2000 1 0

# nohup python3 Test_DFBSmoothing.py ./Parameters/DFBSmoothing/test5.param 100 50000 1 0 > smooth_test5.out &
# nohup python3 Test_DFBSmoothing.py ./Parameters/DFBSmoothing/test8.param 100 50000 1 0 > smooth_test8.out &


nohup python3 Test_DFBSmoothing.py ./Parameters/DFBSmoothing/Cov_CGPMSM_0.0.param          15 1000000 1 0 &
nohup python3 Test_DFBSmoothing.py ./Parameters/DFBSmoothing/Cov_CGPMSM_1.5.param          15 1000000 1 0 &
nohup python3 Test_DFBSmoothing.py ./Parameters/DFBSmoothing/Cov_CGPMSM_5.0.param          15 1000000 1 0 &

nohup python3 Test_DFBSmoothing.py ./Parameters/DFBSmoothing/Cov_CGPMSM2_0.0.param         15 1000000 1 0 &
nohup python3 Test_DFBSmoothing.py ./Parameters/DFBSmoothing/Cov_CGPMSM2_1.5.param         15 1000000 1 0 &
nohup python3 Test_DFBSmoothing.py ./Parameters/DFBSmoothing/Cov_CGPMSM2_5.0.param         15 1000000 1 0 &

nohup python3 Test_DFBSmoothing.py ./Parameters/DFBSmoothing/Cov_CGPMSM3_0.0.param         15 1000000 1 0 &
nohup python3 Test_DFBSmoothing.py ./Parameters/DFBSmoothing/Cov_CGPMSM3_1.5.param         15 1000000 1 0 &
nohup python3 Test_DFBSmoothing.py ./Parameters/DFBSmoothing/Cov_CGPMSM3_5.0.param         15 1000000 1 0 &

nohup python3 Test_DFBSmoothing.py ./Parameters/DFBSmoothing/test_nr3_0.0.param            15 1000000 1 0 &
nohup python3 Test_DFBSmoothing.py ./Parameters/DFBSmoothing/test_nr3_1.5.param            15 1000000 1 0 &
nohup python3 Test_DFBSmoothing.py ./Parameters/DFBSmoothing/test_nr3_5.0.param            15 1000000 1 0 &

nohup python3 Test_DFBSmoothing.py ./Parameters/DFBSmoothing/Cov_CGPMSM_e==d_0.0.param     15 1000000 1 0 &
nohup python3 Test_DFBSmoothing.py ./Parameters/DFBSmoothing/Cov_CGPMSM_e==d_1.5.param     15 1000000 1 0 &
nohup python3 Test_DFBSmoothing.py ./Parameters/DFBSmoothing/Cov_CGPMSM_e==d_5.0.param     15 1000000 1 0 &

nohup python3 Test_DFBSmoothing.py ./Parameters/DFBSmoothing/Cov_CGPMSM_bj_sigma_j_cst_0.0 15 1000000 1 0 &


