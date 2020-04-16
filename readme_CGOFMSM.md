# CGOFMSM

Project to simulate and restore CGOFMSM : Conditionnally Gaussian Observed Fuzzy Markov Switching Model 

      >>  python3 CGOFMSM_Simulation.py [arguments]

## Arguments

      argv[1] : Parameters filename (file examples in Parameters repository)
                  Default value: Parameters/SP2018.param
      argv[2] : Fuzzy joint law model and parameters. e.g. 2:0.07:0.24:0.09, or 4:0.15:0.15:0.:0.1
                  Default value: 2:0.07:0.24:0.09
      argv[3] : Compute hard filter and smoother, fuzzy filter, fuzzy smoother? 0/1,0/1,0/1
                  Default value: 1,1,0
      argv[4] : Sample size to simulate and then to restore
                  Default value: 300
      argv[5] : Discretization step. e.g. 3 or 3,5,10 (one or several values separeted by a coma)
                  Default value: 3,7
      argv[6] : Number of experiments (for getting mean results)
                  Default value: 1
      argv[7] : Verbose: debug (3), high (2), medium (1) ou low (0)
                  Default value: 2
      argv[8] : Generate graphic plots? yes -> 1, no -> 0
                  Default value: 1

## Running Examples

      >> python3 CGOFMSM_Simulation.py Parameters/Fuzzy/SP2018.param 2:0.07:0.24:0.09 1,1,1 72 3 5 2 1
      >> nohup python3 CGOFMSM.py Parameters/SP2018.param 4:0.15:0.15:0.:0.1 1,1,0 1000 1,2,3,5,7,10 10 1 0 1 > serie2.out &

# CGOFMSM for Seattle Temperature data

There is now also a program for testing the fuzzy model on real data (temperture form the open data web site of the town of Seattle)

     >>  python3 Test_CGOFMSM_Signals.py [arguments]

More information on how to generate the command can be found in file *./Data/Kaggle/Traces.md*.

## Arguments

      argv[1] : Parameters filename (file examples in Parameters repository)
                  Default value: Parameters/Signal.param
      argv[2] : Fuzzy joint law model and parameters. e.g. 2:0.07:0.24:0.09, or 4:0.15:0.15:0.:0.1
                  Default value: 2:0.07:0.24:0.09
      argv[3] : Compute hard filter and smoother, fuzzy filter, fuzzy smoother? 0/1,0/1,0/1
                  Default value: 1,1,0
      argv[4] : Signal filename
                  Default value: './Data/Kaggle/input/JoseRizalBridgeNorth_all_resample_1303_GT.csv
      argv[5] : Discretization step. e.g. 3 or 3,5,10 (one or several values separeted by a coma)
                  Default value: 3,7
      argv[6] : Verbose: debug (3), high (2), medium (1) ou low (0)
                  Default value: 2
      argv[7] : Generate graphic plots? yes -> 1, no -> 0
                  Default value: 1

## Running Example

      >> python3 Test_CGOFMSM_Signals.py ./Parameters/Signal.param 4:0.10:1.05:0.13:0.13 1,1,0 ./Data/Kaggle/input/JoseRizalBridgeNorth_all_resample_1303_GT.csv 1,3,5,7,9 2 1

