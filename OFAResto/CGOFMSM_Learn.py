import numpy as np
import scipy as sp
import copy
import clipboard
import warnings
import pathlib

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates  as md
from matplotlib.ticker import MaxNLocator
from sklearn.cluster import KMeans

years    = md.YearLocator()   # every year
months   = md.MonthLocator()  # every month
days     = md.DayLocator()    # every day
yearsFmt = md.DateFormatter('%Y      ')
monthFmt = md.DateFormatter('%B %Y')
dayFmt   = md.DateFormatter('%d')

from OFAResto.LoiDiscreteFuzzy_TMC    import calcF, calcB
from OFAResto.LoiDiscreteFuzzy_TMC    import Loi1DDiscreteFuzzy_TMC, Loi2DDiscreteFuzzy_TMC
from OFAResto.TabDiscreteFuzzy        import Tab1DDiscreteFuzzy, Tab2DDiscreteFuzzy

from CommonFun.CommonFun              import From_FQ_to_Cov_Lyapunov, Test_if_CGPMSM, Test_isCGOMSM_from_Cov, is_pos_def, From_Cov_to_FQ
from CGPMSMs.CGPMSMs                  import GetParamNearestCGO_cov

from Fuzzy.APrioriFuzzyLaw_Series1    import LoiAPrioriSeries1
from Fuzzy.APrioriFuzzyLaw_Series2    import LoiAPrioriSeries2
from Fuzzy.APrioriFuzzyLaw_Series2bis import LoiAPrioriSeries2bis
from Fuzzy.APrioriFuzzyLaw_Series2ter import LoiAPrioriSeries2ter
from Fuzzy.APrioriFuzzyLaw_Series3    import LoiAPrioriSeries3
from Fuzzy.APrioriFuzzyLaw_Series4    import LoiAPrioriSeries4
from Fuzzy.APrioriFuzzyLaw_Series4bis import LoiAPrioriSeries4bis


fontS = 13     # font size
DPI   = 150    # graphic resolution



def Check_CovMatrix(Mat, verbose=False):
    w, v = np.linalg.eig(Mat)
    if verbose == True:
        print('w=', w)
        print('v=', v)
        print('Mat=', Mat)
        print('det Mat=', np.linalg.det(Mat))
        corr = np.zeros(shape = np.shape(Mat))
        for i in range(np.shape(corr)[0]):
            for j in range(np.shape(corr)[1]):
                corr[i, j] = Mat[i, j]/np.sqrt(Mat[i,i] * Mat[j,j])
        print('corr=', corr)

    if np.all(np.logical_not(np.iscomplex(w))) == False or np.all(w>0.) == False:
        return False
    return True

###################################################################################################
class CGOFMSM_Learn:
    def __init__(self, STEPS, nbIterSEM, nbRealSEM, Datatrain, fileTrain, FSstring, verbose, graphics):

        self.__n_r          = 2
        self.__nbIterSEM    = nbIterSEM
        self.__nbRealSEM    = nbRealSEM
        self.__verbose      = verbose
        self.__graphics     = graphics
        self.__STEPS        = STEPS
        self.__STEPSp1      = STEPS+1
        self.__STEPSp2      = STEPS+2
        self.__EPS          = 1E-8
        self.__fileTrain    = fileTrain
        self.__filestem     = pathlib.Path(fileTrain).stem
        self.__Datatrain    = Datatrain
        self.__FSstring     = FSstring
        self._interpolation = False

        self.__Datatrain.set_index(list(self.__Datatrain)[0], inplace=True)
        self.__listeHeader = list(self.__Datatrain)
        
        # dimensions des données
        self.__n_y, self.__n_x = 1, 1
        self.__n_z   = self.__n_x + self.__n_y
        len_y, len_x = self.__Datatrain[self.__listeHeader[0]].count(), self.__Datatrain[self.__listeHeader[1]].count()
        if len_x != len_y:
            print('The number of values in X and Y are differents!!!\n')
            exit(1)
        self.__N = len_y

        # les données 
        self.__Ztrain = np.zeros(shape=(self.__N, self.__n_z))
        self.__Ztrain[:, 0] = self.__Datatrain[self.__listeHeader[1]].values
        self.__Ztrain[:, 1] = self.__Datatrain[self.__listeHeader[0]].values

        if self.__STEPS != 0:
            self.__Rcentres = np.linspace(start=1./(2.*self.__STEPS), stop=1.0-1./(2.*self.__STEPS), num=self.__STEPS, endpoint=True)
        else:
            self.__Rcentres = np.empty(shape=(0,))

        # used to print the evolution of FS parameters
        self.__Tab_ParamFS   = np.zeros(shape=(self.__nbIterSEM+1, 4)) 
        self.__Tab_M_00      = np.zeros(shape=(self.__nbIterSEM+1, 4)) 
        self.__Tab_Lambda_00 = np.zeros(shape=(self.__nbIterSEM+1, 1)) 
        self.__Tab_P_00      = np.zeros(shape=(self.__nbIterSEM+1, 2)) 
        self.__Tab_Pi_00     = np.zeros(shape=(self.__nbIterSEM+1, 1))
        
        # plage graphique pour les plots
        self.__graph_mini = 0
        self.__graph_maxi = min(1000, self.__N) # maxi=self.__N, maxi=min(500, self.__N)
        self.__graphRep   = './Result/Fuzzy/SimulatedR/'

        # Detect the weekend days switches
        self.__weekend_indices = []
        BoolWE = False
        for i in range(self.__graph_mini,self.__graph_maxi):
            if self.__Datatrain.index[i].weekday() >= 5 and BoolWE == False:
                self.__weekend_indices.append(i) 
                BoolWE = True
            if self.__Datatrain.index[i].weekday() < 5 and BoolWE == True:
                self.__weekend_indices.append(i) 
                BoolWE = False
        # refermer si ouvert
        if BoolWE==True:
            self.__weekend_indices.append(i)
        # print('self.__weekend_indices=', self.__weekend_indices)
        # input('weekday')

        if self.__graphics >= 1:

            s_Graph = slice(self.__graph_mini, self.__graph_maxi)

            # Plot of the original data
            fig, ax1 = plt.subplots()

            color = 'tab:red'
            ax1.set_ylabel(self.__listeHeader[0], color=color, fontsize=fontS)
            ax1.plot(self.__Datatrain.index[s_Graph], self.__Datatrain[self.__listeHeader[0]].iloc[s_Graph], color=color)
            ax1.tick_params(axis='y', labelcolor=color, labelsize=fontS-2)
            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            color = 'tab:blue'
            ax2.set_ylabel(self.__listeHeader[1], color=color, fontsize=fontS)
            ax2.plot(self.__Datatrain.index[s_Graph], self.__Datatrain[self.__listeHeader[1]].iloc[s_Graph], color=color)
            ax2.tick_params(axis='y', labelcolor=color, labelsize=fontS-2)

            # surlignage des jours de WE
            i = 0
            while i < len(self.__weekend_indices)-1:
                ax1.axvspan(self.__Datatrain.index[self.__weekend_indices[i]], self.__Datatrain.index[self.__weekend_indices[i+1]], facecolor='gray', edgecolor='none', alpha=.25, zorder=-100)
                i += 2

            # format the ticks
            ax1.xaxis.set_major_locator(months)
            ax1.xaxis.set_major_formatter(monthFmt)
            ax1.xaxis.set_minor_locator(days)
            ax1.xaxis.set_minor_formatter(dayFmt)
            #ax1.format_xdata = md.DateFormatter('%m-%d')
            ax1.grid(True, which='major', axis='both')
            ax1.set_title('Data: ' + self.__filestem, fontsize=fontS+2)
            ax1.tick_params(axis='x', which='both', labelsize=fontS-2)
            ax1.set_xlim(xmin=self.__Datatrain.index[self.__graph_mini], xmax=self.__Datatrain.index[self.__graph_maxi-1])

            fig.autofmt_xdate()
            plt.xticks(rotation=35)
            fig.tight_layout()  # otherwise the right y-label is slightly clipped
            plt.savefig(self.__graphRep+self.__filestem+'_origData.png', bbox_inches='tight', dpi=DPI)    
            plt.close()


    def run_several(self):

        ########## initialisation by kmeans 
        iter = 0
        print('SEM ITERATION', iter, 'over', self.__nbIterSEM, '(kmeans)')

        self.__FS, self.__M, self.__Lambda2, self.__P, self.__Pi2, self.__aMeanCovFuzzy, Rsimul = \
            self.updateParamFromRsimul(numIterSEM=iter, kmeans=True, Plot=False)

        # Save of parameters for parametrization 2
        filenameParam = './Parameters/Fuzzy/' + self.__filestem + '_F=' + str(self.__STEPS) + '_direct.param2'
        CovZ_CGO, MeanX, MeanY = self.GenerateParameters_2(Rsimul)
        if len(np.shape(CovZ_CGO)) != 0:
            self.SaveParameters_2(filenameParam, CovZ_CGO, MeanX, MeanY)
            chWork = str(0) + ',' + str(1) + ',' + str(1) + ',' + str(1)
            self.GenerateCommandline(chWork, self.__fileTrain, filenameParam, -1, clipboardcopy=True)

        # To plot the evolution of some parameters
        self.__Tab_ParamFS  [iter,:] = self.__FS.getParam()
        self.__Tab_M_00     [iter,:] = self.__M[0,0,:]
        self.__Tab_Lambda_00[iter,0] = np.sqrt(self.__Lambda2[0,0])
        self.__Tab_P_00     [iter,:] = self.__P[0,0,:]
        self.__Tab_Pi_00    [iter,0] = np.sqrt(self.__Pi2[0,0])

        # Convert parametrization 3 to parametrization 2 (by param 1)
        # filenameParam = self.__graphRep+'parametrization2_F=' + str(self.__STEPS) + '_Iter_0.param'
        # Cov, MeanX, MeanY = self.ConvertParameters()
        # self.SaveParameters(filenameParam, Cov, MeanX, MeanY)
        # input('saved')

        # Print des paramètres
        if self.__verbose >= 1: self.printParam()
        #input('pause')

        ########## the SEM loop
        Plot=False
        for iter in range(1, self.__nbIterSEM+1):
            print('SEM ITERATION', iter, 'over', self.__nbIterSEM)
            if iter==self.__nbIterSEM: Plot=True
            self.run_one(iter, Plot)

            # Convert parametrization 3 to parametrization 2 (by param 1) 
            # filenameParam = self.__graphRep+'parametrization2_F=' + str(self.__STEPS) + '_Iter_' +str(iter) + '.param'
            # Cov, MeanX, MeanY = self.ConvertParameters()
            # self.SaveParameters(filenameParam, Cov, MeanX, MeanY)
            # input('saved')

        # on dessine la dernire simulation
        if self.__graphics>=1 and self.__nbIterSEM>0:
            self.PlotConvSEM()


    def run_one(self, iter, Plot=False):

        # MAJ des proba sur la base des paramètres courants
        Tab_GaussXY                         = self.compute_tab_GaussXY()
        ProbaForwardNorm, Tab_Normalis      = self.compute_fuzzyjumps_forward(Tab_GaussXY)
        ProbaBackwardNorm                   = self.compute_fuzzyjumps_backward(ProbaForwardNorm, Tab_GaussXY, Tab_Normalis)
        ProbaGamma, ProbaPsi, ProbaJumpCond = self.compute_fuzzyjumps_gammapsicond(ProbaForwardNorm, ProbaBackwardNorm, Tab_GaussXY)

        # Update of param from some simulated R
        if self.__verbose >= 2: print('         update parameters')
        
        self.__FS, self.__M, self.__Lambda2, self.__P, self.__Pi2, self.__aMeanCovFuzzy, Rsimul = \
            self.updateParamFromRsimul(numIterSEM=iter, kmeans=False, Plot=Plot, ProbaGamma_0=ProbaGamma[0], ProbaJumpCond=ProbaJumpCond, ProbaPsi=ProbaPsi)
         
        # To plot the evolution of some parameters
        self.__Tab_ParamFS  [iter, :] = self.__FS.getParam()
        self.__Tab_M_00     [iter, :] = self.__M[0,0,:]
        self.__Tab_Lambda_00[iter, 0] = np.sqrt(self.__Lambda2[0,0])
        self.__Tab_P_00     [iter, :] = self.__P[0,0,:]
        self.__Tab_Pi_00    [iter, 0] = np.sqrt(self.__Pi2[0,0])

        # Print des paramètres
        if self.__verbose >= 1: self.printParam()
        #input('pause')

    def ConvertParameters_3to2by1(self):

        # Convert from Param 3 to Param 1 (using equations from Zied) ############################################################@
        MeanX = np.zeros(shape=(self.__STEPSp2, self.__n_x))
        MeanY = np.zeros(shape=(self.__STEPSp2, self.__n_y))
        for indrn in range(self.__STEPSp2):
            MeanY[indrn] = self.__P[indrn, indrn, 1] / (1.- self.__P[indrn, indrn, 0])
            MeanX[indrn] = (self.__M[indrn, indrn, 3] + MeanY[indrn] * (self.__M[indrn, indrn, 1]+self.__M[indrn, indrn, 2])) / (1. - self.__M[indrn, indrn, 0])
            print('indrn = ', indrn)
            print('self.__P[indrn, indrn] = ', self.__P[indrn, indrn, :])
            print('self.__M[indrn, indrn] = ', self.__M[indrn, indrn, :])
            print('MeanX[indrn]=', MeanX[indrn])
            print('MeanY[indrn]=', MeanY[indrn])
            print('MeanX[indrn] approche=', self.__aMeanCovFuzzy.getMean(indrn)[0])
            print('MeanY[indrn] approche=', self.__aMeanCovFuzzy.getMean(indrn)[1])
            #input('pause')

        F = np.zeros(((self.__STEPSp2)**2, self.__n_z, self.__n_z))
        Q = np.zeros(((self.__STEPSp2)**2, self.__n_z, self.__n_z))
        # B = np.zeros(((self.__STEPSp2)**2, self.__n_z, self.__n_z))
        for indrn in range(self.__STEPSp2):
            for indrnp1 in range(self.__STEPSp2):
                ind = indrn*(self.__STEPSp2) + indrnp1

                ### F 
                F[ind, 0, 0] = self.__M[indrn, indrnp1, 0]
                F[ind, 1, 1] = self.__P[indrn, indrnp1, 0]
                F[ind, 1, 0] = 0.
                F[ind, 0, 1] = self.__M[indrn, indrnp1, 1] + F[ind, 1, 1] * self.__M[indrn, indrnp1, 2]

                ### Q
                Q[ind, 1, 1] = self.__Pi2[indrn, indrnp1]
                Q[ind, 0, 1] = self.__M[indrn, indrnp1, 2] * Q[ind, 1, 1]
                Q[ind, 1, 0] = Q[ind, 0, 1]
                Q[ind, 0, 0] = self.__Lambda2[indrn, indrnp1] + self.__M[indrn, indrnp1, 2] * Q[ind, 1, 0]
                
                # print('ind=', ind)
                if is_pos_def(Q[ind,:,:]) == False:
                    print('ind=', ind, ' --> PROBLEM with Q matrix in parameter file!!')
                    input('pause Q')
                
                # Non requis
                # with warnings.catch_warnings():
                #     warnings.simplefilter('error')
                #     B[ind,:, :] = sp.linalg.sqrtm(Q[ind,:, :])

        print('F:', F)
        print('Q:', Q)
        # print('B:', B)
        # input('attente')

        # Convert from Param 1 to Param 2 (usig method from Fei) ##################################################@
        #print('########## CONVERSION VERS COV ##########')
        Cov = From_FQ_to_Cov_Lyapunov(F, Q, self.__n_x)
        #print('########## CONVERSION VERS A,Q ##########')
        # Fbis, Qbis = From_Cov_to_FQ(Cov)
        # print('########## VERIF VERIF VERIF ##########')
        # for ind in range((self.__STEPSp2)**2):
        #     print('ind=', ind)
        #     print(np.around(F[ind, :,:] - Fbis[ind, :,:], decimals=2))
        #     print(np.around(Q[ind, :,:] - Qbis[ind, :,:], decimals=2))
        #     #input('pause')
        #input('fin verif lyapunov')

        # for ind in range((self.__STEPSp2)**2):
        #     if is_pos_def(Cov[ind,:,:]) == False:
        #         print('ind=', ind, ' --> PROBLEM with Cov matrix in parameter file!!')
        #         print(Cov[ind,:,:])
        #         input('pause Cov')

        return Cov, MeanX, MeanY


    def GenerateCommandline(self, chWork, fileTrain, filenameParam, steps, clipboardcopy=False):
 
        A  = 'python3 CGOFMSM_SignalRest.py ' + filenameParam + ' ' + self.__FS.stringName() + ' '
        A += chWork + ' ' + fileTrain + ' ' + str(steps) + ' 2 1'
        print('  -> Command line for data restoration:')
        print('        ', A, '\n')

        if clipboardcopy == True:
            clipboard.copy(A.strip()) # mise en mémoire de la commande à exécuter


    def GenerateParameters_2(self, Rsimul):

        # Objet permettant d'estimert les paramètres
        aMeanCovZ1Z2Fuzzy=MeanCovZ1Z2Fuzzy(self.__Ztrain, self.__n_z, self.__STEPS, self.__verbose)
        OK = aMeanCovZ1Z2Fuzzy.update(Rsimul)

        # Les moyennes 
        meanZ = aMeanCovZ1Z2Fuzzy.getMeanZAll()
        meanX = np.around(meanZ[:, 0:self.__n_x], decimals=4)
        meanY = np.around(meanZ[:, self.__n_x: ], decimals=4)

        if OK == False:
            CovZ_CGO = None
        else:
            # Les covarainces
            CovZ  = aMeanCovZ1Z2Fuzzy.getCovZ1Z2All()

            CovZ_CGO = GetParamNearestCGO_cov(CovZ, n_x=self.__n_x)
            if Test_isCGOMSM_from_Cov(CovZ_CGO, self.__n_x) == False:
                print('ATTENTION: Le modele directe n''est pas un CGOMSM!!!')
                print('self.__CovZ1Z2_Fuzzy=\n', self.__CovZ1Z2_Fuzzy)
                input('pause')
            # print('CovZ_CGO=', CovZ_CGO)

        return CovZ_CGO, meanX, meanY


    def SaveParameters_2(self, filenameParam, Cov, MeanX, MeanY):

        # Save the parameters for the parametrization 2 #####################################################@
        
        # L'entete
        f = open(filenameParam, 'w')
        f.write(  '#==========================================================#')
        f.write('\n# CGOFMSM parameters (parametrization 2)                   #')
        f.write('\n#==========================================================#\n#\n')
        f.close()

        f = open(filenameParam, 'ab')
        
        # the number of fuzzy steps
        np.savetxt(f, np.array([self.__STEPS], dtype=int), delimiter=" ", header='number of fuzzy steps'+'\n================================', footer='\n')
      
        # Les covariances
        for j in range(self.__STEPSp2):
            for k in range(self.__STEPSp2):
                ind = j*(self.__STEPSp2) + k
                np.savetxt(f, Cov[ind,:,:], delimiter=" ", header='Cov_xy'+str(j)+str(k)+'\n----------------------------', footer='\n', fmt='%.4f')
        
        # Les moyennes
        np.savetxt(f, MeanX, delimiter=" ", header='mean of X'+'\n================================', footer='\n', fmt='%.4f')
        np.savetxt(f, MeanY, delimiter=" ", header='mean of Y'+'\n================================', footer='\n', fmt='%.4f')
        f.close()


    def SaveParameters_2Interpolation(self, filenameParam, Cov, MeanX, MeanY):

        # Save the CGOFMSM file ##################################################################################################@
        
        # L'entete
        f = open(filenameParam, 'w')
        f.write(  '#==========================================================#')
        f.write('\n# CGOFMSM parameters (parametrization 2 for interpolation) #')
        f.write('\n#==========================================================#\n#\n')
        f.close()

        f = open(filenameParam, 'ab')

        # Les covariances
        j, k = 0, 0
        ind = j*(self.__STEPSp2) + k
        np.savetxt(f, Cov[ind,:,:], delimiter=" ", header='Cov_xy'+str(0)+str(0)+'\n----------------------------', footer='\n', fmt='%.4f')
        j, k = 0, self.__STEPSp1
        ind = j*(self.__STEPSp2) + k
        np.savetxt(f, Cov[ind,:,:], delimiter=" ", header='Cov_xy'+str(0)+str(1)+'\n----------------------------', footer='\n', fmt='%.4f')
        j, k = self.__STEPSp1, 0
        ind = j*(self.__STEPSp2) + k
        np.savetxt(f, Cov[ind,:,:], delimiter=" ", header='Cov_xy'+str(1)+str(0)+'\n----------------------------', footer='\n', fmt='%.4f')
        j, k = self.__STEPSp1, self.__STEPSp1
        ind = j*(self.__STEPSp2) + k
        np.savetxt(f, Cov[ind,:,:], delimiter=" ", header='Cov_xy'+str(1)+str(1)+'\n----------------------------', footer='\n', fmt='%.4f')
        
        # Les moyennes
        np.savetxt(f, MeanX[0], delimiter=" ", header='mean of X'+'\n================================', fmt='%.4f')
        np.savetxt(f, MeanX[self.__STEPSp1], delimiter=" ", footer='\n', fmt='%.4f')
        np.savetxt(f, MeanY[0], delimiter=" ", header='mean of Y'+'\n================================', fmt='%.4f')
        np.savetxt(f, MeanY[self.__STEPSp1], delimiter=" ", footer='\n', fmt='%.4f')
        f.close()

    def SaveParameters_3(self, filenameParam):

        # Save the parameter for the parametrization 3 #####################################################@
        
        # L'entete
        f = open(filenameParam, 'w')
        f.write(  '#==========================================================#')
        f.write('\n# CGOFMSM parameters (parametrization 3)                   #')
        f.write('\n#==========================================================#\n#\n')
        f.close()

        f = open(filenameParam, 'ab')
        
        # the number of fuzzy steps
        np.savetxt(f, np.array([self.__STEPS], dtype=int), delimiter=" ", header='number of fuzzy steps'+'\n================================', footer='\n')
      
        # Les paramètres M, Lambda2, P, Pi2
        for j in range(self.__STEPSp2):
            np.savetxt(f, self.__M[j],       delimiter=" ", header='M_'+str(j)+'x\n----------------------------', footer='\n', fmt='%.4f')
            np.savetxt(f, self.__Lambda2[j], delimiter=" ", header='Lambda2_'+str(j)+'x\n----------------------------', footer='\n', fmt='%.4f')
            np.savetxt(f, self.__P[j],       delimiter=" ", header='P_'+str(j)+'x\n----------------------------', footer='\n', fmt='%.4f')
            np.savetxt(f, self.__Pi2[j],     delimiter=" ", header='Pi2_'+str(j)+'x\n----------------------------', footer='\n', fmt='%.4f')

        # Les paramètres de maoyenneet covaraince pour le premier
        for j in range(self.__STEPSp2):
            np.savetxt(f, self.__aMeanCovFuzzy.getMean(j), delimiter=" ", header='Mean_'+str(j)+'x\n----------------------------', footer='\n', fmt='%.4f')
            np.savetxt(f, self.__aMeanCovFuzzy.getCov(j),  delimiter=" ", header='Cov_'+str(j)+'x\n----------------------------', footer='\n', fmt='%.4f')
        
        f.close()   
    

    def simulRealization(self, Rsimul, ProbaGamma_0, ProbaJumpCond):
        
        for real in range(self.__nbRealSEM):
            Nreal = real*self.__N
            
            # sampling of the first from gamma
            np1=0
            Rsimul[Nreal + np1] = ProbaGamma_0.getSample()
            
            # next ones according to conditional law
            for np1 in range(1, self.__N):
                Rsimul[Nreal + np1] = ProbaJumpCond[np1-1][Rsimul[Nreal + np1 - 1]].getSample()


    def updateParamFromRsimul(self, numIterSEM, kmeans, Plot, ProbaGamma_0=None, ProbaJumpCond=None, ProbaPsi=None):

        # used for simulation of discretized fuzzy r (in [0..self.__STEPSp1])
        Rsimul = np.zeros(shape=(self.__nbRealSEM*self.__N), dtype=int)

        ################### Simulation of R
        if kmeans == False:
            # Simuler une chaîne de Markov a posteriori sur la base de la valeur des paramètres courants
            if self.__verbose >= 2: print('         simul realization')
            self.simulRealization(Rsimul, ProbaGamma_0, ProbaJumpCond)

            # Parameters of parametrization 3 (M=[A, B, C, D], Lambda**2, and P=[F, G], Pi**2)
            M, Lambda2, P, Pi2 = self.EstimParam2ter(ProbaPsi, numIterSEM)

            if self.__graphics >= 2:
                fname = self.__graphRep+self.__filestem +'Rsimul_Iter_' + str(numIterSEM) + '_cl' + str(self.__STEPSp2)
                title = 'Simulated R - Iter ' + str(numIterSEM)
                self.plotRsimul(Rsimul, fname=fname, title=title)
        else:
            for real in range(self.__nbRealSEM):
                #kmeans = KMeans(n_clusters=2+self.__STEPS, random_state=4, init='random', n_init=1).fit(self.__Ztrain)
                kmeans = KMeans(n_clusters=2+self.__STEPS, random_state=None, init='random', n_init=1).fit(self.__Ztrain)
                #kmeans = KMeans(n_clusters=2+self.__STEPS, n_init=1).fit(self.__Ztrain)
                # print(kmeans.inertia_, kmeans.n_iter_)
                # print('kmeans.cluster_centers_=', kmeans.cluster_centers_)
                # Sorting of labels according to the X-coordinate of cluster centers
                sortedlabel = np.argsort(kmeans.cluster_centers_[:, 1])
                # print('sortedlabel=', sortedlabel)
                # input('attente')

                for n in range(self.__N):
                    Rsimul[real*self.__N + n] = np.where(sortedlabel == kmeans.labels_[n])[0][0]

                # Parameters of parametrization 3 (M=[A, B, C, D], Lambda**2, and P=[F, G], Pi**2)
                M, Lambda2, P, Pi2 = self.EstimParam2terInit(Rsimul)

            if self.__graphics >= 1:
                fname = self.__graphRep+self.__filestem + 'Rsimul_Kmeans_cl' + str(self.__STEPSp2)
                title = 'Simulated R - Iter ' + str(numIterSEM)
                self.plotRsimul(Rsimul, fname=fname, title=title)


        # Parameter for fuzzy Markov model called APrioriFuzzyLaw_serie2ter.py
        FS = None
        if   self.__FSstring == '1':
            FS = LoiAPrioriSeries1(alpha=0., gamma=0.)
        elif self.__FSstring == '2':
            FS = LoiAPrioriSeries2(alpha=0., eta=0., delta=0.)
        elif self.__FSstring == '2bis':
            FS = LoiAPrioriSeries2bis(alpha=0., eta=0., delta=0., lamb=0.)
        elif self.__FSstring == '2ter':
            FS = LoiAPrioriSeries2ter(alpha0=0., alpha1=0., beta=0.)
        elif self.__FSstring == '3':
            FS = LoiAPrioriSeries3(alpha=0., delta=0.)
        elif self.__FSstring == '4':
            FS = LoiAPrioriSeries4(alpha=0., gamma=0., delta_d=0., delta_u=0.)
        elif self.__FSstring == '4bis':
            FS = LoiAPrioriSeries4bis(alpha=0., gamma=0., delta_d=0., delta_u=0., lamb=0.)
        else:
            input('Impossible')
            exit(1)
        FS.setParametersFromSimul(Rsimul, self.__STEPSp2)

        # if self.__graphics>=2:
        #     fig = plt.figure()
        #     ax = fig.add_subplot(1, 1, 1, projection='3d')
        #     FS.plotR1R2(self.__graphRep+self.__filestem + 'FS_2D_iter' + str(numIterSEM) + '.png', ax, dpi=DPI)
        #     FS.plotR1  (self.__graphRep+self.__filestem + 'FS_1D_iter' + str(numIterSEM) + '.png', dpi=DPI)
       
        # Parameters for the first p(r_1 | z_1) - Nécessaire pour le forward n=1
        aMeanCovFuzzy = MeanCovFuzzy(self.__Ztrain, self.__n_z, self.__STEPS, self.__verbose)
        aMeanCovFuzzy.update(Rsimul)

        # Plot de la dernière réalisation
        if Plot == True and ((self.__graphics==0) or (self.__graphics==1)): 
            fname = self.__graphRep+self.__filestem + 'Rsimul_Iter_' + str(self.__nbIterSEM) + '_cl' + str(self.__STEPSp2)
            title = 'Simulated R - Iter ' + str(self.__nbIterSEM)
            self.plotRsimul(Rsimul, fname=fname, title=title)

        return FS, M, Lambda2, P, Pi2, aMeanCovFuzzy, Rsimul


    def EstimParam2ter(self, ProbaPsi, numIterSEM):

        # M = [A, B, C, D] ##################################################################################
        MNum   = np.zeros(shape=(self.__STEPSp2, self.__STEPSp2, 4))
        MDenom = np.zeros(shape=(self.__STEPSp2, self.__STEPSp2, 4, 4))
        # P = [F, G] ##################################################################################
        PNum   = np.zeros(shape=(self.__STEPSp2, self.__STEPSp2, 2))
        PDenom = np.zeros(shape=(self.__STEPSp2, self.__STEPSp2, 2, 2))

        for n in range(self.__N-1):
            yn    = (self.__Ztrain[n,   self.__n_x:self.__n_z]).item()
            ynpun = (self.__Ztrain[n+1, self.__n_x:self.__n_z]).item()
            xn    = (self.__Ztrain[n,   0:self.__n_x])         .item()
            xnpun = (self.__Ztrain[n+1, 0:self.__n_x])         .item()
            
            vect1      = np.array([xn, yn, ynpun, 1.]) 
            vect1vect1 = np.outer(vect1, vect1)
            vect1      = xnpun * vect1
            vect2      = np.array([yn, 1.])
            vect2vect2 = np.outer(vect2, vect2)
            vect2      = ynpun * vect2

            for indrn in range(self.__STEPSp2):
                for indrnp1 in range(self.__STEPSp2):
                    probaPsi = ProbaPsi[n].getindr(indrn, indrnp1)

                    ###################### M ########################
                    MDenom[indrn, indrnp1, :, :] += probaPsi * vect1vect1
                    MNum  [indrn, indrnp1, :]    += probaPsi * vect1

                    ###################### P ########################
                    PDenom[indrn, indrnp1, :, :] += probaPsi * vect2vect2
                    PNum  [indrn, indrnp1, :]    += probaPsi * vect2


        M = np.zeros(shape=(self.__STEPSp2, self.__STEPSp2, 4))
        P = np.zeros(shape=(self.__STEPSp2, self.__STEPSp2, 2))
        for indrn in range(self.__STEPSp2):
            for indrnp1 in range(self.__STEPSp2):
                try:
                    # print(MNum[indrn, indrnp1,:])
                    # print(np.linalg.inv(MDenom[indrn, indrnp1,:,:]))
                    # print(np.dot(MNum[indrn, indrnp1,:], np.linalg.inv(MDenom[indrn, indrnp1,:,:])))
                    M[indrn, indrnp1,:] = np.dot(MNum[indrn, indrnp1,:], np.linalg.inv(MDenom[indrn, indrnp1,:,:]))
                    # print('M[indrn, indrnp1,:]=', M[indrn, indrnp1,:])
                    # input('xvxv;dk')
                except np.linalg.LinAlgError:
                    if self.__verbose>2:
                        print('indrn, indrnp1 = ', indrn, indrnp1)
                        print("det pour M=", np.linalg.det(MDenom[indrn, indrnp1,:,:]), ' --> pseudo inverse')
                    M[indrn, indrnp1,:] = np.dot(MNum[indrn, indrnp1,:], np.linalg.pinv(MDenom[indrn, indrnp1,:,:], rcond=1e-15, hermitian=True))
                    # input('pause')
                try:
                    P[indrn, indrnp1,:] = np.dot(PNum[indrn, indrnp1,:], np.linalg.inv(PDenom[indrn, indrnp1,:,:]))
                except np.linalg.LinAlgError:
                    if self.__verbose>2:
                        print('indrn, indrnp1 = ', indrn, indrnp1)
                        print("det pour P=", np.linalg.det(PDenom[indrn, indrnp1,:,:]), ' --> pseudo inverse')
                    P[indrn, indrnp1,:] = np.dot(PNum[indrn, indrnp1,:], np.linalg.pinv(PDenom[indrn, indrnp1,:,:], rcond=1e-15, hermitian=True))

        # Pi2 and Lambda2 ##################################################################################
        Pi2     = np.zeros(shape=(self.__STEPSp2, self.__STEPSp2))
        Lambda2 = np.zeros(shape=(self.__STEPSp2, self.__STEPSp2))
        Denom   = np.zeros(shape=(self.__STEPSp2, self.__STEPSp2))
        for n in range(self.__N-1):
            yn    = (self.__Ztrain[n,   self.__n_x:self.__n_z]).item()
            ynpun = (self.__Ztrain[n+1, self.__n_x:self.__n_z]).item()
            xn    = (self.__Ztrain[n,   0:self.__n_x])         .item()
            xnpun = (self.__Ztrain[n+1, 0:self.__n_x])         .item()

            for indrn in range(self.__STEPSp2):
                for indrnp1 in range(self.__STEPSp2):
                    probaPsi = ProbaPsi[n].getindr(indrn, indrnp1)

                    Lambda2[indrn, indrnp1] += probaPsi * (xnpun - (xn * M[indrn, indrnp1, 0] + yn * M[indrn, indrnp1, 1] + ynpun * M[indrn, indrnp1, 2] + M[indrn, indrnp1, 3]))**2
                    Pi2    [indrn, indrnp1] += probaPsi * (ynpun - (yn * P[indrn, indrnp1, 0] + P[indrn, indrnp1, 1]))**2
                    Denom  [indrn, indrnp1] += probaPsi

        for indrn in range(self.__STEPSp2):
            for indrnp1 in range(self.__STEPSp2):
                if Lambda2[indrn, indrnp1] != 0.: 
                    Lambda2[indrn, indrnp1] /= Denom[indrn, indrnp1]
                if Pi2[indrn, indrnp1]     != 0.: 
                    Pi2    [indrn, indrnp1] /= Denom[indrn, indrnp1]

        # print('Lambda2=', Lambda2)
        #print('Pi2=', Pi2)
        # input('LES MEMES ?')
        return M, Lambda2, P, Pi2


    def EstimParam2terInit(self, Rsimul):

        # M = [A, B, C, D] ##################################################################################
        MNum      = np.zeros(shape=(self.__STEPSp2, self.__STEPSp2, 4))
        MDenom    = np.zeros(shape=(self.__STEPSp2, self.__STEPSp2, 4, 4))
        # P = [F, G] ##################################################################################
        PNum      = np.zeros(shape=(self.__STEPSp2, self.__STEPSp2, 2))
        PDenom    = np.zeros(shape=(self.__STEPSp2, self.__STEPSp2, 2, 2))

        for real in range(self.__nbRealSEM):

            for n in range(self.__N-1):
                yn    = (self.__Ztrain[n,   self.__n_x:self.__n_z]).item()
                ynpun = (self.__Ztrain[n+1, self.__n_x:self.__n_z]).item()
                xn    = (self.__Ztrain[n,   0:self.__n_x])         .item()
                xnpun = (self.__Ztrain[n+1, 0:self.__n_x])         .item()
                
                vect1      = np.array([xn, yn, ynpun, 1.]) 
                vect1vect1 = np.outer(vect1, vect1)
                vect2      = np.array([yn, 1.])
                vect2vect2 = np.outer(vect2, vect2)

                indrn   = Rsimul[real*self.__N + n]
                indrnp1 = Rsimul[real*self.__N + n+1]

                ###################### M ########################
                MDenom[indrn, indrnp1, :, :] += vect1vect1
                MNum  [indrn, indrnp1, :]    += xnpun * vect1

                ###################### P ########################
                PDenom[indrn, indrnp1, :, :] += vect2vect2
                PNum  [indrn, indrnp1, :]    += ynpun * vect2

        M = np.zeros(shape=(self.__STEPSp2, self.__STEPSp2, 4))
        P = np.zeros(shape=(self.__STEPSp2, self.__STEPSp2, 2))
        for indrn in range(self.__STEPSp2):
            for indrnp1 in range(self.__STEPSp2):
                try:
                    # print(MNum[indrn, indrnp1,:])
                    # print(np.linalg.inv(MDenom[indrn, indrnp1,:,:]))
                    # print(np.dot(MNum[indrn, indrnp1,:], np.linalg.inv(MDenom[indrn, indrnp1,:,:])))
                    M[indrn, indrnp1,:] = np.dot(MNum[indrn, indrnp1,:], np.linalg.inv(MDenom[indrn, indrnp1,:,:]))
                    # print('M[indrn, indrnp1,:]=', M[indrn, indrnp1,:])
                    # input('xvxv;dk')
                except np.linalg.LinAlgError:
                    if self.__verbose>2:
                        print('indrn, indrnp1 = ', indrn, indrnp1)
                        print("det pour M (kmeans)=", np.linalg.det(MDenom[indrn, indrnp1,:,:]), ' --> pseudo inverse')
                    M[indrn, indrnp1,:] = np.dot(MNum[indrn, indrnp1,:], np.linalg.pinv(MDenom[indrn, indrnp1,:,:], rcond=1e-15, hermitian=True))
                    # input('pause')
                try:
                    P[indrn, indrnp1,:] = np.dot(PNum[indrn, indrnp1,:], np.linalg.inv(PDenom[indrn, indrnp1,:,:]))
                except np.linalg.LinAlgError:
                    if self.__verbose>2:
                        print('indrn, indrnp1 = ', indrn, indrnp1)
                        print("det pour P (kmeans)=", np.linalg.det(PDenom[indrn, indrnp1,:,:]), ' --> pseudo inverse')
                    P[indrn, indrnp1,:] = np.dot(PNum[indrn, indrnp1,:], np.linalg.pinv(PDenom[indrn, indrnp1,:,:], rcond=1e-15, hermitian=True))

        # Pi2 and Lambda2 ##################################################################################
        Pi2     = np.zeros(shape=(self.__STEPSp2, self.__STEPSp2))
        Lambda2 = np.zeros(shape=(self.__STEPSp2, self.__STEPSp2))
        Denom   = np.zeros(shape=(self.__STEPSp2, self.__STEPSp2))
        
        for real in range(self.__nbRealSEM):
            for n in range(self.__N-1):
                yn    = (self.__Ztrain[n,   self.__n_x:self.__n_z]).item()
                ynpun = (self.__Ztrain[n+1, self.__n_x:self.__n_z]).item()
                xn    = (self.__Ztrain[n,   0:self.__n_x])         .item()
                xnpun = (self.__Ztrain[n+1, 0:self.__n_x])         .item()
                
                vect1      = np.array([xn, yn, ynpun, 1.]) 
                vect1vect1 = np.outer(vect1, vect1)
                vect2      = np.array([yn, 1.])
                vect2vect2 = np.outer(vect2, vect2)

                indrn   = Rsimul[real*self.__N + n]
                indrnp1 = Rsimul[real*self.__N + n+1]

                Lambda2[indrn, indrnp1] += (xnpun - (xn * M[indrn, indrnp1, 0] + yn * M[indrn, indrnp1, 1] + ynpun * M[indrn, indrnp1, 2] + M[indrn, indrnp1, 3]))**2
                Pi2    [indrn, indrnp1] += (ynpun - (yn * P[indrn, indrnp1, 0] + P[indrn, indrnp1, 1]))**2
                Denom  [indrn, indrnp1] += 1.

        for indrn in range(self.__STEPSp2):
            for indrnp1 in range(self.__STEPSp2):
                if Lambda2[indrn, indrnp1] != 0.: 
                    Lambda2[indrn, indrnp1] /= Denom[indrn, indrnp1]
                if Pi2[indrn, indrnp1]     != 0.: 
                    Pi2    [indrn, indrnp1] /= Denom[indrn, indrnp1]

        # print('Lambda2=', Lambda2)
        # print('Pi2=', Pi2)
        # input('LES MEMES ?')
        return M, Lambda2, P, Pi2

    def compute_tab_GaussXY(self):

        tab_GaussXY = []

        # La premiere ne sert à rien, uniquementà a synchroniser les indices
        tab_GaussXY.append(Tab2DDiscreteFuzzy(self.__EPS, self.__STEPS, self._interpolation, self.__Rcentres, (1, 1)))

        znp1=self.__Ztrain[0, :]

        # Les suivantes
        for np1 in range(1, self.__N):
            if self.__verbose >= 2:
                print('\r         proba tnp1 condit. to tn, np1=', np1, ' sur N=', self.__N, end='   ', flush = True)

            zn   = znp1
            znp1 = self.__Ztrain[np1, :]

            tab_GaussXY.append(Tab2DDiscreteFuzzy(self.__EPS, self.__STEPS, self._interpolation, self.__Rcentres, (1, 1)))
            tab_GaussXY[np1].Calc_GaussXY(self.__M, self.__Lambda2, self.__P, self.__Pi2, zn, znp1, self.__n_x)
            # tab_GaussXY[np1].print()
            # input('tab gauss')

        if self.__verbose >= 2: print(' ')
        return tab_GaussXY


    def compute_fuzzyjumps_forward(self, Tab_GaussXY):
        
        ProbaForward = []
        Tab_Normalis = np.zeros(shape=(self.__N))

        ######################
        # Initialisation
        np1  = 0
        ProbaForward.append(Loi1DDiscreteFuzzy_TMC(self.__EPS, self.__STEPS, self._interpolation, self.__Rcentres))
        ProbaForward[np1].CalcForw1(self.__FS, self.__Ztrain[np1, :], self.__aMeanCovFuzzy)
        Tab_Normalis[np1] = ProbaForward[np1].Integ()
        # normalisation (devijver)
        ProbaForward[np1].normalisation(Tab_Normalis[np1])

        # ProbaForward[np1].print()
        # input('forward')

        ###############################
        # Boucle
        for np1 in range(1, self.__N):
            if self.__verbose >= 2:
                print('\r         forward np1=', np1, ' sur N=', self.__N, end='', flush = True)

            ProbaForward.append(Loi1DDiscreteFuzzy_TMC(self.__EPS, self.__STEPS, self._interpolation, self.__Rcentres))
            ProbaForward[np1].CalcForB(calcF, ProbaForward[np1-1], self.__FS, Tab_GaussXY[np1])
            Tab_Normalis[np1] = ProbaForward[np1].Integ()
            #print('Tab_Normalis[np1]=', Tab_Normalis[np1])
            # normalisation (devijver)
            ProbaForward[np1].normalisation(Tab_Normalis[np1])
            if abs(1.-ProbaForward[np1].Integ()) > 1E-3: # 0.1 % d'erreur toléré
                ProbaForward[np1].print()
                print('1.-ProbaForward[np1].Integ()=', 1.-ProbaForward[np1].Integ())
                input('forward, ca ne va pas!')

            #input('fin forward np1')

        if self.__verbose >= 2: print(' ')
        return ProbaForward, Tab_Normalis


    def compute_fuzzyjumps_backward(self, ProbaForwardNorm, Tab_GaussXY, Tab_Normalis):

        ProbaBackward = []

        loicorrective = Loi1DDiscreteFuzzy_TMC(self.__EPS, self.__STEPS, self._interpolation, self.__Rcentres)

        # on créé la liste de tous les lois discrétisées
        for n in range(self.__N):
            ProbaBackward.append(Loi1DDiscreteFuzzy_TMC(self.__EPS, self.__STEPS, self._interpolation, self.__Rcentres))

        ##############################
        # initialisation de beta
        n = self.__N-1
        ProbaBackward[n].setValCste(1.0)

        ###############################
        # Boucle pour backward
        for n in range(self.__N-2, -1, -1):
            if self.__verbose >= 2:
                print('\r         backward n=', n, ' sur N=', self.__N, end='             ', flush = True)
        
            ProbaBackward[n].CalcForB(calcB, ProbaBackward[n+1], self.__FS, Tab_GaussXY[n+1])
            
            # normalisation (devijver)
            ProbaBackward[n].normalisation(Tab_Normalis[n+1])
            # ProbaBackward[n].print()
            # input('pause')

            # Cette normalisation n'est implémentée que pour contre-carrer la dérive suite à l'intégration numérique
            # Important lorsque F est faible
            loicorrective.ProductFB(ProbaForwardNorm[n], ProbaBackward[n])
            # loicorrective.print()
            # print('loicorrective.Integ()=', loicorrective.Integ())
            # input('attente')
            ProbaBackward[n].normalisation(loicorrective.Integ())
            
        if self.__verbose >= 2: print(' ')
        return ProbaBackward


    def compute_fuzzyjumps_gammapsicond(self, ProbaForwardNorm, ProbaBackwardNorm, Tab_GaussXY):

        tab_gamma = []
        tab_psi   = []
        tab_cond  = []

        ###############################
        # Boucle sur gamma et psi
        for n in range(self.__N-1):
            if self.__verbose >= 2:
                print('\r         proba gamma psi cond n=', n, ' sur N=', self.__N, end='   ', flush = True)

            # print('AAAAAAAAA1')

            # calcul de gamma = produit forward norm * backward norm ****************************************************************
            tab_gamma.append(Loi1DDiscreteFuzzy_TMC(self.__EPS, self.__STEPS, self._interpolation, self.__Rcentres))
            tab_gamma[n].ProductFB(ProbaForwardNorm[n], ProbaBackwardNorm[n])
    
            # normalisation : uniquement due pour compenser des pb liés aux approximations numeriques de forward et de backward
            # Si F= 20, on voit que la normalisation n'est pas necessaire (deja la somme == 1.)
            integ = tab_gamma[n].Integ()
            if abs(1.-integ) > 1E-2: # we stop only if more than 1% error
                print('\ntab_gamma[n].Integ()=', integ)
                print(np.fabs(1.-integ))
                print(np.fabs(1.-integ) > 1E-2)
                ProbaForward[n].print()
                ProbaBackward[n].print()
                input('PB PB PB Gamma')
            tab_gamma[n].normalisation(integ)

            # print('AAAAAAAAA2')


            # if tab_gamma[n].getindr(2)>tab_gamma[n].getindr(0) and tab_gamma[n].getindr(2)>tab_gamma[n].getindr(1):
            #     tab_gamma[n].print()
            #     input('pause tab-gamma')

            # calcul de psi (loi jointe a posteriori) ****************************************************************
            tab_psi.append(Loi2DDiscreteFuzzy_TMC(self.__EPS, self.__STEPS, self._interpolation, self.__Rcentres))
            tab_psi[n].CalcPsi(ProbaForwardNorm[n], ProbaBackwardNorm[n+1], self.__FS, Tab_GaussXY[n+1])
            # normalisation
            integ = tab_psi[n].Integ()
            if integ == 0.:
                print("Tab Spi normalisation")
                print('tab_psi[n].Integ()=', integ)
                tab_psi[n].print()
                input('pause')
            tab_psi[n].normalisation(integ)

            # print('AAAAAAAAA3')


            # integ = tab_psi[n].Integ()
            # if abs(1.-integ) > 1E-2: # we stop only if more than 5% error
            #     print('integ tab_psi=', integ)
            #     input('Attente dans tab_psi')

            # calcul de p(rnp1 | rn, z_1^M)  ****************************************************************
            # on créé une liste, pour chaque valeur de rn, de lois 1D
            Liste = []
            for indrn in range(self.__STEPSp2):
                
                Liste.append(Loi1DDiscreteFuzzy_TMC(self.__EPS, self.__STEPS, self._interpolation, self.__Rcentres))
                if tab_gamma[n].getindr(indrn) != 0.:
                    Liste[indrn].CalcCond(indrn, tab_gamma[n].getindr(indrn), tab_psi[n], self.__verbose)
                    #Liste[indrn].print()
                else:
                    if self.__verbose>2:
                        print('\nPBPB tab_gamma[n].getindr(indrn) == 0. when indrn=', indrn)
                        #input('ATTENTE ERROR')

                    # loi uniforme
                    Liste[indrn].setValCste(1.)
                    Liste[indrn].normalisation(Liste[indrn].Integ())
                
            tab_cond.append(Liste)
        
        # le dernier gamma : on n'en a pas besoin

        if self.__verbose >= 2: print(' ')
        return tab_gamma, tab_psi, tab_cond


    def PlotConvSEM(self):

        ax = plt.figure().gca()
        ax.ticklabel_format(useOffset=False)
        plt.plot(self.__Tab_ParamFS[:, 0], color='g', label=r'$\alpha_0$')
        plt.plot(self.__Tab_ParamFS[:, 1], color='r', label=r'$\alpha_1$')
        plt.plot(self.__Tab_ParamFS[:, 2], color='b', label=r'$\beta$')
        plt.plot(self.__Tab_ParamFS[:, 3], color='k', label=r'$\eta$')
        plt.ylim(ymax=1.05, ymin=-0.05)
        plt.xlim(xmax=self.__nbIterSEM, xmin=0)
        plt.xlabel('SEM iteration')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        plt.title('Evolution of FS param - Fuzzy steps (F)=' + str(self.__STEPS)+ ', mean of ' + str(self.__nbRealSEM) + ' realizations')
        plt.savefig(self.__graphRep+self.__filestem + 'ParamFS_STEPS_' + str(self.__STEPS)+ '_NbReal_' + str(self.__nbRealSEM) + '.png', bbox_inches='tight', dpi=DPI)    
        plt.close()

        ax = plt.figure().gca()
        ax.ticklabel_format(useOffset=False)
        plt.plot(self.__Tab_M_00[:, 0], color='g', label=r'$\mathcal{A}_{0}^{0}$')
        plt.plot(self.__Tab_M_00[:, 1], color='r', label=r'$\mathcal{B}_{0}^{0}$')
        plt.plot(self.__Tab_M_00[:, 2], color='b', label=r'$\mathcal{C}_{0}^{0}$')
        #plt.plot(self.__Tab_M_00[:, 3], color='k', label=r'$\mathcal{D}_{0}^{0}$')
        plt.xlim(xmax=self.__nbIterSEM, xmin=0)
        plt.xlabel('SEM iteration')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        plt.title(r'Evolution of $\mathcal{M}_{0}^{0}$ param - Fuzzy steps (F)=' + str(self.__STEPS)+ ', mean of ' + str(self.__nbRealSEM) + ' realizations')
        plt.savefig(self.__graphRep+self.__filestem + 'MOO_STEPS_' + str(self.__STEPS)+ '_NbReal_' + str(self.__nbRealSEM) + '.png', bbox_inches='tight', dpi=DPI)    
        plt.close()

        ax = plt.figure().gca()
        ax.ticklabel_format(useOffset=False)
        plt.plot(self.__Tab_P_00[:, 0], color='g', label=r'$\mathcal{F}_{0}^{0}$')
        #plt.plot(self.__Tab_P_00[:, 1], color='r', label=r'$\mathcal{G}_{0}^{0}$')
        plt.xlim(xmax=self.__nbIterSEM, xmin=0)
        plt.xlabel('SEM iteration')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        plt.title(r'Evolution of $\mathcal{P}_{0}^{0}$ param - Fuzzy steps (F)=' + str(self.__STEPS)+ ', mean of ' + str(self.__nbRealSEM) + ' realizations')
        plt.savefig(self.__graphRep+self.__filestem + 'POO_STEPS_' + str(self.__STEPS)+ '_NbReal_' + str(self.__nbRealSEM) + '.png', bbox_inches='tight', dpi=DPI)    
        plt.close()

        ax = plt.figure().gca()
        ax.ticklabel_format(useOffset=False)
        plt.plot(self.__Tab_Lambda_00, color='g', label=r'$\lambda_{0}^{0}$')
        plt.xlim(xmax=self.__nbIterSEM, xmin=0)
        plt.xlabel('SEM iteration')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        plt.title(r'Evolution of $\lambda_{0}^{0}$ param - Fuzzy steps (F)=' + str(self.__STEPS)+ ', mean of ' + str(self.__nbRealSEM) + ' realizations')
        plt.savefig(self.__graphRep+self.__filestem + 'Lambda_STEPS_' + str(self.__STEPS)+ '_NbReal_' + str(self.__nbRealSEM) + '.png', bbox_inches='tight', dpi=DPI)    
        plt.close()

        ax = plt.figure().gca()
        ax.ticklabel_format(useOffset=False)
        plt.plot(self.__Tab_Pi_00, color='g', label=r'$\pi_{0}^{0}$')
        plt.xlim(xmax=self.__nbIterSEM, xmin=0)
        plt.xlabel('SEM iteration')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        plt.title(r'Evolution of $\pi_{0}^{0}$ param - Fuzzy steps (F)=' + str(self.__STEPS)+ ', mean of ' + str(self.__nbRealSEM) + ' realizations')
        plt.savefig(self.__graphRep+self.__filestem + 'Pi_STEPS_' + str(self.__STEPS)+ '_NbReal_' + str(self.__nbRealSEM) + '.png', bbox_inches='tight', dpi=DPI)    
        plt.close()


    def printParam(self):
        #print('  Estimated parameters')
        print('    --> loi jointe à priori (2ter)=', self.__FS)
        # print('    --> param 3 du modele\n    M=', self.__M)
        # print('    --> param 3 du modele\n    Lambda**2=', self.__Lambda2)
        # print('    --> param 3 du modele\n    P=', self.__P)
        # print('    --> param 3 du modele\n    Pi**2=', self.__Pi2)


    def plotRsimul(self, Rsimul, fname, title):

        RsimulFuzzy = np.zeros(shape=(self.__N))
        s_Graph = slice(self.__graph_mini, self.__graph_maxi)

        for real in range(self.__nbRealSEM):
            for n in range(self.__N):

                indrn = Rsimul[real*self.__N + n]
                if indrn == 0: 
                    rn = 0.
                elif indrn == self.__STEPSp1: 
                    rn= 1.
                else:
                    rn = self.__Rcentres[indrn-1]
                RsimulFuzzy[n] = rn


            fig, ax1 = plt.subplots()

            Centres = np.zeros(shape=(self.__STEPSp2))
            Centres[0] = 0.
            Centres[self.__STEPSp1] = 1.
            Centres[1:self.__STEPSp1] = self.__Rcentres
            
            color = 'tab:green'
            ax1.set_ylabel('Discrete fuzzy jumps', color=color, fontsize=fontS)
            ax1.plot(self.__Datatrain.index[s_Graph], RsimulFuzzy[s_Graph], color=color)
            ax1.tick_params(axis='y', labelcolor=color, labelsize=fontS-2)
            ax1.tick_params(axis='x', which='both', labelsize=fontS-2)
            
            ax1.set_ylim(ymax=1.05, ymin=-0.05)
            ax1.set_xlim(xmin=self.__Datatrain.index[self.__graph_mini], xmax=self.__Datatrain.index[self.__graph_maxi-1])

            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            color = 'tab:olive'
            ax2.hlines (Centres, xmin=self.__Datatrain.index[self.__graph_mini], xmax=self.__Datatrain.index[self.__graph_maxi-1], color=color, linestyle='dashed')
            ax2.tick_params(axis='y', labelcolor=color, labelsize=fontS-2)
            ax2.set_yticks(ticks=Centres)

            # format the ticks
            ax2.xaxis.set_major_locator(months)
            ax2.xaxis.set_major_formatter(monthFmt)
            ax2.xaxis.set_minor_locator(days)
            ax2.xaxis.set_minor_formatter(dayFmt)
            # ax1.format_xdata = md.DateFormatter('%m%Y-%d')

            plt.setp(ax1.xaxis.get_minorticklabels(), rotation=90)
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=35)
            #ax1.grid(True, which='major', axis='both')
            ax1.set_title('Data: ' + self.__filestem + title + ', Realization ' + str(real), fontsize=fontS-4)
            
            i = 0
            while i < len(self.__weekend_indices)-1:
                ax1.axvspan(self.__Datatrain.index[self.__weekend_indices[i]], self.__Datatrain.index[self.__weekend_indices[i+1]], facecolor='gray', edgecolor='none', alpha=.25, zorder=-100)
                i += 2

            fig.autofmt_xdate()
            #fig.tight_layout()  # otherwise the right y-label is slightly clipped
            plt.savefig(fname + '_Real_'  + str(real)+'.png', bbox_inches='tight', dpi=DPI)    
            plt.close()


###################################################################################################
class MeanCovFuzzy:

    def __init__(self, Ztrain, n_z, STEPS, verbose):
        self.__verbose  = verbose
        self.__STEPS    = STEPS
        self.__STEPSp1  = STEPS+1
        self.__STEPSp2  = STEPS+2
        self.__Ztrain   = Ztrain
        self.__n_z      = n_z

        self.__N = np.shape(Ztrain)[0]

        self.__Mean_Zf  = np.zeros(shape=(self.__STEPSp2, self.__n_z))
        self.__Cov_Zf   = np.zeros(shape=(self.__STEPSp2, self.__n_z, self.__n_z))
        self.__cpt      = np.zeros(shape=(self.__STEPSp2), dtype=int)

    # def __del__(self):
    #     if self.__verbose >= 2: 
    #         print('\nMeanCovFuzzy deleted')

    def update(self, Rlabels):

        # nombre de realisation
        nbreal = int(len(Rlabels)/self.__N)
        
        # Remise à 0.
        self.__Mean_Zf.fill(0.)
        self.__Cov_Zf.fill(0.)
        self.__cpt.fill(0)
        
        # The means
        for real in range(nbreal):
            for n in range(self.__N):
                label = Rlabels[real*self.__N+n]
                self.__Mean_Zf[label,:] += self.__Ztrain[n, :]
                self.__cpt[label] += 1
        
        for indrn in range(self.__STEPSp2):
            if self.__cpt[indrn] != 0:
                self.__Mean_Zf[indrn, :] /= self.__cpt[indrn]
        # print(self.__Mean_Zf)
        # input('hard mean')
        
        # The variances
        VectZ = np.zeros(shape=(self.__n_z, 1))
        for real in range(nbreal):
            for n in range(self.__N):
                label = Rlabels[real*self.__N+n]
                VectZ = (np.transpose(self.__Ztrain[n, :]) - self.__Mean_Zf[label,:]).reshape(self.__n_z, 1)
                self.__Cov_Zf[label,:,:] += np.outer(VectZ, VectZ)
        for indrn in range(self.__STEPSp2):
            if self.__cpt[indrn] != 0:
                self.__Cov_Zf[indrn,:,:] /= self.__cpt[indrn]
                # check if cov matrix
                if Check_CovMatrix( self.__Cov_Zf[indrn,:,:]) == False:
                    print('cpt=', self.__cpt, ', sum=', np.sum(self.__cpt))
                    print(self.__Mean_Zf[indrn, :])
                    print(self.__Cov_Zf[indrn,:,:])
                    input('update - This is not a cov matrix!!')

        # print(self.__Cov_Zf)
        # input('hardcov')

    def getMean(self, indrn):
        if indrn>=0 and indrn<self.__STEPSp2:
            return self.__Mean_Zf[indrn, :]
        else:
            return None

    def getCov(self, indrn):
        if indrn>=0 and indrn<self.__STEPSp2:
            return self.__Cov_Zf[indrn, :]
        else:
            return None
        


###################################################################################################
class MeanCovZ1Z2Fuzzy:

    def __init__(self, Ztrain, n_z, STEPS, verbose):
        self.__verbose  = verbose
        self.__STEPS    = STEPS
        self.__STEPSp1  = STEPS+1
        self.__STEPSp2  = STEPS+2
        self.__Ztrain   = Ztrain
        self.__n_z      = n_z

        self.__N = np.shape(Ztrain)[0]

        self.__Mean_Fuzzy    = np.zeros(shape=(self.__STEPSp2, self.__n_z))
        self.__CovZ1Z2_Fuzzy = np.zeros(shape=(self.__STEPSp2**2, self.__n_z*2, self.__n_z*2))


    def update(self, Rlabels):

        # nombre de realisation
        nbreal = int(len(Rlabels)/self.__N)
        
        # Remise à 0.
        self.__Mean_Fuzzy.fill(0.)
        self.__CovZ1Z2_Fuzzy.fill(0.)

        VectZ1 = np.zeros(shape=(self.__n_z*2))

         # The means MEANZ
        ####################################################
        cpt0 = np.zeros((self.__STEPSp2), dtype=int)
        for real in range(nbreal):
            for n in range(self.__N):
                label = Rlabels[real*self.__N+n]
                self.__Mean_Fuzzy[label,:] += self.__Ztrain[n, :]
                cpt0[label] += 1
        
        for j in range(self.__STEPSp2):
            if cpt0[j] != 0:
                self.__Mean_Fuzzy[j, :] /= cpt0[j]
        # print(self.__Mean_Fuzzy)
        # print(cpt0)
        # input('MEANZ')
        
        # The means MeanZ1Z2
        ####################################################
        MeanZ1Z2 = np.zeros((self.__STEPSp2**2, self.__n_z*2))
        cpt = np.zeros((self.__STEPSp2**2), dtype=int)
        for real in range(nbreal):
            for n in range(self.__N-1):
                ln   = Rlabels[real*self.__N+n]
                lnp1 = Rlabels[real*self.__N+n+1]
                l    = ln*self.__STEPSp2+lnp1
                MeanZ1Z2[l, 0:self.__n_z] += self.__Ztrain[n,   :]
                MeanZ1Z2[l, self.__n_z:]  += self.__Ztrain[n+1, :]
                cpt[l] += 1
        
        for l in range(self.__STEPSp2**2):
            if cpt[l] != 0:
                MeanZ1Z2[l, :] /= cpt[l]
        # print(MeanZ1Z2)
        # print(cpt)
        # input('MEANZ')

        # The covariances CovZ1Z2
        ####################################################
        TabIsCov = np.zeros(shape=(self.__STEPSp2**2), dtype=bool)
        TabIsCov.fill(True)
        
        for real in range(nbreal):
            for n in range(self.__N-1):
                ln   = Rlabels[real*self.__N+n]
                lnp1 = Rlabels[real*self.__N+n+1]
                l    = ln*self.__STEPSp2+lnp1
                VectZ1[0:self.__n_z] = self.__Ztrain[n,   :]
                VectZ1[self.__n_z:]  = self.__Ztrain[n+1, :]
                VectZ1 -=  MeanZ1Z2[l, :]
                self.__CovZ1Z2_Fuzzy[l, :, :] += np.outer(VectZ1, VectZ1)

        for l in range(self.__STEPSp2**2):
            if cpt[l] > 1: # on évite 0 et 1 (auquel cas la matrice de cov vaut 0 partout)
                self.__CovZ1Z2_Fuzzy[l, :, :] /= cpt[l]
            if Check_CovMatrix(self.__CovZ1Z2_Fuzzy[l, :, :], verbose=False) == False:
                TabIsCov[l] = False
                # print('l=', l, ', cpt[l]=', cpt[l], ', sum=', np.sum(cpt))
                # print('self.__CovZ1Z2_Fuzzy[l, :, :] = ', self.__CovZ1Z2_Fuzzy[l, :, :])
                # Check_CovMatrix(self.__CovZ1Z2_Fuzzy[l, :, :], verbose=True)
                    # input('update - self.__CovZ1Z2_Fuzzy[l, :, :] This is not a cov matrix!!')
        # print('self.__CovZ1Z2_Fuzzy=', self.__CovZ1Z2_Fuzzy)
        # print('TabIsCov=', TabIsCov)
        # input('COVZ1Z2')

        # Correction des matrices Gamma par moyennage
        ####################################################
        Gamma = np.zeros((self.__STEPSp2, self.__n_z, self.__n_z))
        cpt1  = np.zeros((self.__STEPSp2), dtype=int)
        for j in range(self.__STEPSp2):
            l = j*self.__STEPSp2+j
            Gamma[j, :, :] += (self.__CovZ1Z2_Fuzzy[l, 0:self.__n_z, 0:self.__n_z]+self.__CovZ1Z2_Fuzzy[l, self.__n_z:,  self.__n_z:])/2.
            if Check_CovMatrix(Gamma[j, :, :], verbose=False) == False:
                if self.__verbose>1:
                    print('j=', j)
                    print('Gamma[j, :, :]=', Gamma[j, :, :])
                    print('update - This is not a cov matrix!! --> parameters not saved')
                return False
        # print('Gamma=', Gamma)
        # input('Corrections Gamma')

        # implantation dans self.__CovZ1Z2_Fuzzy
        for l in range(self.__STEPSp2**2):
            j = l//self.__STEPSp2
            k = l%self.__STEPSp2
            # print("j=", j, ", k=", k)
            self.__CovZ1Z2_Fuzzy[l, 0:self.__n_z, 0:self.__n_z] = Gamma[j, :, :]
            self.__CovZ1Z2_Fuzzy[l, self.__n_z:,  self.__n_z:]  = Gamma[k, :, :]
            if Check_CovMatrix(self.__CovZ1Z2_Fuzzy[l, :, :], verbose=False) == False:
                #print('TabIsCov[l]=', TabIsCov[l])
                #Check_CovMatrix(self.__CovZ1Z2_Fuzzy[l, :, :], verbose=True)
                #input('Nous avons un pb - les matrices de cov ne sont pas definies positives')
                TabIsCov[l] = False
    
        # Pour les matrices fausse, on remet les matrices sigma à 0
        for l in range(self.__STEPSp2**2):
            if TabIsCov[l] == False:
                self.__CovZ1Z2_Fuzzy[l, 0:self.__n_z, self.__n_z:] = 0.
                self.__CovZ1Z2_Fuzzy[l, self.__n_z:, 0:self.__n_z] = 0.
                if Check_CovMatrix(self.__CovZ1Z2_Fuzzy[l, :, :], verbose=False) == False:
                    if self.__verbose>1:
                        print('l=', l)
                        print('TabIsCov[l]=', TabIsCov[l])
                        Check_CovMatrix(self.__CovZ1Z2_Fuzzy[l, :, :], verbose=True)
                        print('update - This matrix is not pos. def. !')
                    return False
        # print('TabIsCov=', TabIsCov)
        # input('COVZ1Z2')
        return True

    def getMeanZAll(self):
        return self.__Mean_Fuzzy
    
    def getCovZ1Z2All(self):
        return self.__CovZ1Z2_Fuzzy

    def getMean(self, indrn):
        if indrn>=0 and indrn<self.__STEPSp2:
            return self.__Mean_Fuzzy[indrn, :]
        else:
            return None

    def getCovZ1Z2(self, indrn):
        if indrn>=0 and indrn<self.__STEPSp2:
            return self.__CovZ1Z2_Fuzzy[indrn, :]
        else:
            return None
        
