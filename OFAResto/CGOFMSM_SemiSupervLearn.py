import numpy as np
import scipy as sp
import copy
import clipboard
import warnings

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import rc
rc('text', usetex=True)

from sklearn.cluster import KMeans

from OFAResto.LoiDiscreteFuzzy_TMC    import Loi1DDiscreteFuzzy_TMC, Loi2DDiscreteFuzzy_TMC, calcF, calcB
from Fuzzy.APrioriFuzzyLaw_Series2ter import LoiAPrioriSeries2ter
from CommonFun.CommonFun              import From_FQ_to_Cov_Lyapunov, Test_if_CGPMSM, is_pos_def, From_Cov_to_FQ


def Check_CovMatrix(Mat):
    w, v = np.linalg.eig(Mat)
    if np.all(np.logical_not(np.iscomplex(w))) == False or np.all(w>0.) == False:
        return False
    return True

def getrnFromindrn(Rcentres, indrn):
    if indrn == 0:               return 0.
    if indrn == len(Rcentres)+1: return 1.
    return Rcentres[indrn-1]

###################################################################################################
class CGOFMSM_SemiSupervLearn:
    def __init__(self, STEPS, nbIterSEM, nbRealSEM, Ztrain, n_x, n_y, verbose, graphics):

        self.__n_r       = 2
        self.__nbIterSEM = nbIterSEM
        self.__nbRealSEM = nbRealSEM
        self.__N         = np.shape(Ztrain)[1]
        self.__Ztrain    = Ztrain
        self.__verbose   = verbose
        self.__graphics  = graphics
        self.__STEPS     = STEPS
        self.__EPS       = 1E-8
        
        # dimensions
        self.__n_x       = n_x
        self.__n_y       = n_y
        self.__n_z       = self.__n_x + self.__n_y

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

        # Update of param for the simulated R (by kmeans here for init)
        iter = 0
        self.__FS, self.__M, self.__Lambda2, self.__P, self.__Pi2, self.__aMeanCovFuzzy = self.updateParamFromRsimul(numIterSEM=iter, kmeans=True, Plot=False)
                
        # To plot the evolution of some parameters
        self.__Tab_ParamFS  [iter,:] = copy.deepcopy(self.__FS.getParam())
        self.__Tab_M_00     [iter,:] = copy.deepcopy(self.__M[0,0,:])
        self.__Tab_Lambda_00[iter,0] = np.sqrt(self.__Lambda2[0,0])
        self.__Tab_P_00     [iter,:] = copy.deepcopy(self.__P[0,0,:])
        self.__Tab_Pi_00    [iter,0] = np.sqrt(self.__Pi2[0,0])

        # Convert parametrization 3 to parametrization 2 (by param 1)
        # filenameParam = './Result/Fuzzy/SimulatedR/parametrization2_F=' + str(self.__STEPS) + '_Iter_0.param'
        # Cov, MeanX, MeanY = self.ConvertParameters()
        # self.SaveParameters(filenameParam, Cov, MeanX, MeanY)
        # input('saved')

        # Print des paramètres
        if self.__verbose >= 1: self.printParam()

    def __del__(self):
        if self.__verbose >= 2: 
            print('\nCGOFMSM_SemiSupervLearn deleted')

    def run_several(self):

        Plot=False
        for iter in range(1, self.__nbIterSEM+1):
            print('ITERATION ', iter, ' over ', self.__nbIterSEM)
            if iter==self.__nbIterSEM: Plot=True
            self.run_one(iter, Plot)

            # Convert parametrization 3 to parametrization 2 (by param 1) 
            # filenameParam = './Result/Fuzzy/SimulatedR/parametrization2_F=' + str(self.__STEPS) + '_Iter_' +str(iter) + '.param'
            # Cov, MeanX, MeanY = self.ConvertParameters()
            # self.SaveParameters(filenameParam, Cov, MeanX, MeanY)
            # input('saved')

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
        
        self.__FS, self.__M, self.__Lambda2, self.__P, self.__Pi2, self.__aMeanCovFuzzy = self.updateParamFromRsimul(numIterSEM=iter, kmeans=False, Plot=Plot, ProbaGamma_0=ProbaGamma[0], ProbaJumpCond=ProbaJumpCond)

        # To plot the evolution of some parameters
        self.__Tab_ParamFS  [iter,:] = copy.deepcopy(self.__FS.getParam())
        self.__Tab_M_00     [iter,:] = copy.deepcopy(self.__M[0,0,:])
        self.__Tab_Lambda_00[iter,0] = np.sqrt(self.__Lambda2[0,0])
        self.__Tab_P_00     [iter,:] = copy.deepcopy(self.__P[0,0,:])
        self.__Tab_Pi_00    [iter,0] = np.sqrt(self.__Pi2[0,0])

        # Print des paramètres
        if self.__verbose >= 1: self.printParam()


    def ConvertParameters(self):

        # Convert from Param 3 to Param 1 (using equations from Zied) ############################################################@
        MeanX = np.zeros(shape=(self.__STEPS+2, self.__n_x))
        MeanY = np.zeros(shape=(self.__STEPS+2, self.__n_y))
        for indrn in range(self.__STEPS+2):
            MeanY[indrn] = self.__P[indrn, indrn, 1] / (1.- self.__P[indrn, indrn, 0])
            MeanX[indrn] = (self.__M[indrn, indrn, 3] + MeanY[indrn] * (self.__M[indrn, indrn, 1]+self.__M[indrn, indrn, 2])) / (1. - self.__M[indrn, indrn, 0])
            print('indrn = ', indrn)
            print('self.__P[indrn, indrn] = ', self.__P[indrn, indrn, :])
            print('self.__M[indrn, indrn] = ', self.__M[indrn, indrn, :])
            print('MeanX[indrn]=', MeanX[indrn])
            print('MeanY[indrn]=', MeanY[indrn])
            print('MeanX[indrn] approche=', self.__aMeanCovFuzzy.getMean(indrn)[0])
            print('MeanY[indrn] approche=', self.__aMeanCovFuzzy.getMean(indrn)[1])
            input('pause')

        F = np.zeros(((self.__STEPS+2)**2, self.__n_z, self.__n_z))
        Q = np.zeros(((self.__STEPS+2)**2, self.__n_z, self.__n_z))
        # B = np.zeros(((self.__STEPS+2)**2, self.__n_z, self.__n_z))
        for indrn in range(self.__STEPS+2):
            for indrnp1 in range(self.__STEPS+2):
                ind = indrn*(self.__STEPS+2) + indrnp1

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

        # print('F:', F)
        # print('Q:', Q)
        # print('B:', B)


        # Convert from Param 1 to Param 2 (usig method from Fei) ##################################################@
        #print('########## CONVERSION VERS COV ##########')
        Cov = From_FQ_to_Cov_Lyapunov(F, Q, self.__n_x)
        #print('########## CONVERSION VERS A,Q ##########')
        # Fbis, Qbis = From_Cov_to_FQ(Cov)
        # print('########## VERIF VERIF VERIF ##########')
        # for ind in range((self.__STEPS+2)**2):
        #     print('ind=', ind)
        #     print(np.around(F[ind, :,:] - Fbis[ind, :,:], decimals=2))
        #     print(np.around(Q[ind, :,:] - Qbis[ind, :,:], decimals=2))
        #     #input('pause')
        #input('fin verif lyapunov')

        # for ind in range((self.__STEPS+2)**2):
        #     if is_pos_def(Cov[ind,:,:]) == False:
        #         print('ind=', ind, ' --> PROBLEM with Cov matrix in parameter file!!')
        #         print(Cov[ind,:,:])
        #         input('pause Cov')

        return Cov, MeanX, MeanY

    def SaveParameters(self, filenameParam, Cov, MeanX, MeanY):

        # Save the CGOFMSM file ##################################################################################################@
        
        # L'entete
        f = open(filenameParam, 'w')
        f.write('#=====================================#\n# parameters for CGOFMSM with F discrete classes # \n#=====================================# \n# \n# \n# matrix Cov_XY \n# ===============================================#\n# \n')
        f.close()

        f = open(filenameParam, 'ab')
        
        # the number of fuzzy steps
        np.savetxt(f, np.array([self.__STEPS], dtype=int), delimiter=" ", header='number of fuzzy steps'+'\n================================', footer='\n')
      
        # Les covariances
        for j in range(self.__STEPS+2):
            for k in range(self.__STEPS+2):
                ind = j*(self.__STEPS+2) + k
                np.savetxt(f, Cov[ind,:,:], delimiter=" ", header='Cov_xy'+str(j)+str(k)+'\n----------------------------', footer='\n', fmt='%.4f')
        
        # Les moyennes
        np.savetxt(f, MeanX, delimiter=" ", header='mean of X'+'\n================================', footer='\n', fmt='%.4f')
        np.savetxt(f, MeanY, delimiter=" ", header='mean of Y'+'\n================================', footer='\n', fmt='%.4f')

        f.close()

        # Generate the command to run the predictor 
        #############################################################################@
        hard, filt, smooth, predic = 0, 1, 0, 1
        chWork = str(hard) + ',' + str(filt) + ',' + str(smooth) + ',' + str(predic)
        param = self.__FS.getParam()
        nameY = './Data/Traffic/TMU5509/generated/TMU5509_train.csv'
        A  = 'python3 Test_CGOFMSM_Signals.py ' + filenameParam + ' 2ter:' + str('%.4f'%param[0]) + ':' + str('%.4f'%param[1]) + ':' + str('%.4f'%param[2]) + ' '
        A += chWork + ' ' + nameY + ' -1 2 0'

        clipboard.copy(A.strip()) # mise en mémoire de la commande à exécuter
        print('pour restaurer le signal:')
        print('\n', A, '\n')


    def EmpiricalFuzzyJointMatrix(self, Rsimul):
        
        Nsimul = len(Rsimul)

        alpha0, alpha1, beta = 0., 0., 0.
        for n in range(1, Nsimul):
            if Rsimul[n-1] == 0              and Rsimul[n] == 0:              alpha0 += 1.
            if Rsimul[n-1] == self.__STEPS+1 and Rsimul[n] == self.__STEPS+1: alpha1 += 1.
            if Rsimul[n-1] == 0              and Rsimul[n] == self.__STEPS+1: beta   += 1.
            if Rsimul[n-1] == self.__STEPS+1 and Rsimul[n] == 0:              beta   += 1.
        beta /= 2. # ca compte les transitions 0-1, 1-0, donc on divise par deux
        alpha0 /= (Nsimul-1.)
        alpha1 /= (Nsimul-1.)
        beta   /= (Nsimul-1.)

        return alpha0, alpha1, beta
    

    def simulRealization(self, Rsimul, ProbaGamma_0, ProbaJumpCond):
        
        for real in range(self.__nbRealSEM):
            
            # sampling of the first from gamma
            np1=0
            Rsimul[real*self.__N + np1] = ProbaGamma_0.getSample()
            
            # next ones according to conditional law
            for np1 in range(1, self.__N):
                Rsimul[real*self.__N + np1] = ProbaJumpCond[np1-1][Rsimul[real*self.__N + np1 - 1]].getSample()


    def updateParamFromRsimul(self, numIterSEM, kmeans, Plot, ProbaGamma_0=None, ProbaJumpCond=None):

        # used for simulation of discretized fuzzy r (in [0..self.__STEPS+1])
        Rsimul = np.zeros(shape=(self.__nbRealSEM*self.__N), dtype=int)

        ################### Simulation of R
        if kmeans == False:
            # Simuler une chaîne de Markov a posteriori sur la base de la valeur des paramètres courants
            if self.__verbose >= 2: print('         simul realization')
            self.simulRealization(Rsimul, ProbaGamma_0, ProbaJumpCond)

            # Parameters of parametrization 3 (M=[A, B, C, D], Lambda**2, and P=[F, G], Pi**2)
            M, Lambda2, P, Pi2 = self.EstimParam2ter(ProbaJumpCond)

            if self.__graphics >= 2:
                fname = './Result/Fuzzy/SimulatedR/Rsimul_Iter_' + str(numIterSEM) + '_cl' + str(self.__STEPS+2)
                title = 'Simulated R - Iter ' + str(numIterSEM)
                self.plotRsimul(Rsimul, fname=fname, title=title)
        else:
            for real in range(self.__nbRealSEM):
                #kmeans = KMeans(n_clusters=2+self.__STEPS, random_state=4, init='random', n_init=1).fit(np.transpose(self.__Ztrain))
                kmeans = KMeans(n_clusters=2+self.__STEPS, random_state=None, init='random', n_init=1).fit(np.transpose(self.__Ztrain))
                #kmeans = KMeans(n_clusters=2+self.__STEPS, n_init=1).fit(np.transpose(self.__Ztrain))
                # print(kmeans.inertia_, kmeans.n_iter_)
                # print('kmeans.cluster_centers_=', kmeans.cluster_centers_)
                # Sorting of labels according to the X-coordinate of cluster centers
                sortedlabel = np.argsort(kmeans.cluster_centers_[:, 0])
                # print('sortedlabel=', sortedlabel)

                for n in range(self.__N):
                    Rsimul[real*self.__N + n] = np.where(sortedlabel == kmeans.labels_[n])[0][0]

                # Parameters of parametrization 3 (M=[A, B, C, D], Lambda**2, and P=[F, G], Pi**2)
                M, Lambda2, P, Pi2 = self.EstimParam2terInit(Rsimul)

            if self.__graphics >= 1:
                fname = './Result/Fuzzy/SimulatedR/Rsimul_Kmeans_cl' + str(self.__STEPS+2)
                title = 'Simulated R - Kmeans - Iter ' + str(numIterSEM)
                self.plotRsimul(Rsimul, fname=fname, title=title)


        # Parameter for fuzzy Markov model called APrioriFuzzyLaw_serie2ter.py
        alpha0, alpha1, beta = self.EmpiricalFuzzyJointMatrix(Rsimul)
        FS = LoiAPrioriSeries2ter(self.__EPS, 0, alpha0, alpha1, beta)
       
        # Parameters for the first p(r_1 | z_1) - Nécessaire pour le forward n=1
        aMeanCovFuzzy = MeanCovFuzzy(self.__Ztrain, self.__n_z, self.__STEPS, self.__verbose)
        aMeanCovFuzzy.update(Rsimul)

        # Plot de la dernière réalisation
        if Plot == True and ((self.__graphics==0) or (self.__graphics==1)): 
            fname = './Result/Fuzzy/SimulatedR/Rsimul_Iter_' + str(self.__nbIterSEM) + '_cl' + str(self.__STEPS+2)
            title = 'Simulated R - Iter ' + str(self.__nbIterSEM)
            self.plotRsimul(Rsimul, fname=fname, title=title)

        return FS, M, Lambda2, P, Pi2, aMeanCovFuzzy


    def EstimParam2ter(self, ProbaJumpCond):

        # M = [A, B, C, D] ##################################################################################
        MNum      = np.zeros(shape=(self.__STEPS+2, self.__STEPS+2, 4))
        MDenom    = np.zeros(shape=(self.__STEPS+2, self.__STEPS+2, 4, 4))
        # P = [F, G] ##################################################################################
        PNum      = np.zeros(shape=(self.__STEPS+2, self.__STEPS+2, 2))
        PDenom    = np.zeros(shape=(self.__STEPS+2, self.__STEPS+2, 2, 2))

        for n in range(self.__N-1):
            yn    = (self.__Ztrain[self.__n_x:self.__n_z, n])  .item()
            ynpun = (self.__Ztrain[self.__n_x:self.__n_z, n+1]).item()
            xn    = (self.__Ztrain[0:self.__n_x, n])           .item()
            xnpun = (self.__Ztrain[0:self.__n_x, n+1])         .item()
            
            vect1      = np.array([xn, yn, ynpun, 1.]) 
            vect1vect1 = np.outer(vect1, vect1)
            vect1      = xnpun * vect1
            vect2      = np.array([yn, 1.])
            vect2vect2 = np.outer(vect2, vect2)
            vect2      = ynpun * vect2

            for indrn in range(self.__STEPS+2):
                rn = getrnFromindrn(self.__Rcentres, indrn)

                for indrnp1 in range(self.__STEPS+2):
                    rnp1      = getrnFromindrn(self.__Rcentres, indrnp1)
                    probacond = ProbaJumpCond[n][indrn].get(rnp1)

                    ###################### M ########################
                    MDenom[indrn, indrnp1, :, :] += probacond * vect1vect1
                    MNum  [indrn, indrnp1, :]    += probacond * vect1

                    ###################### P ########################
                    PDenom[indrn, indrnp1, :, :] += probacond * vect2vect2
                    PNum  [indrn, indrnp1, :]    += probacond * vect2

        M = np.zeros(shape=(self.__STEPS+2, self.__STEPS+2, 4))
        P = np.zeros(shape=(self.__STEPS+2, self.__STEPS+2, 2))
        for indrn in range(self.__STEPS+2):
            for indrnp1 in range(self.__STEPS+2):
                try:
                    # print(MNum[indrn, indrnp1,:])
                    # print(np.linalg.inv(MDenom[indrn, indrnp1,:,:]))
                    # print(np.dot(MNum[indrn, indrnp1,:], np.linalg.inv(MDenom[indrn, indrnp1,:,:])))
                    M[indrn, indrnp1,:] = np.dot(MNum[indrn, indrnp1,:], np.linalg.inv(MDenom[indrn, indrnp1,:,:]))
                    # print('M[indrn, indrnp1,:]=', M[indrn, indrnp1,:])
                    # input('xvxv;dk')
                except np.linalg.LinAlgError:
                    M[indrn, indrnp1,:] = 0.
                try:
                    P[indrn, indrnp1,:] = np.dot(PNum[indrn, indrnp1,:], np.linalg.inv(PDenom[indrn, indrnp1,:,:]))
                except np.linalg.LinAlgError:
                    P[indrn, indrnp1,:] = 0.

        # Pi2 and Lambda2 ##################################################################################
        Pi2     = np.zeros(shape=(self.__STEPS+2, self.__STEPS+2))
        Lambda2 = np.zeros(shape=(self.__STEPS+2, self.__STEPS+2))
        Denom   = np.zeros(shape=(self.__STEPS+2, self.__STEPS+2))
        for n in range(self.__N-1):
            yn    = (self.__Ztrain[self.__n_x:self.__n_z, n])  .item()
            ynpun = (self.__Ztrain[self.__n_x:self.__n_z, n+1]).item()
            xn    = (self.__Ztrain[0:self.__n_x, n])           .item()
            xnpun = (self.__Ztrain[0:self.__n_x, n+1])         .item()

            for indrn in range(self.__STEPS+2):
                rn = getrnFromindrn(self.__Rcentres, indrn)
                
                for indrnp1 in range(self.__STEPS+2):
                    rnp1 = getrnFromindrn(self.__Rcentres, indrnp1)

                    probacond = ProbaJumpCond[n][indrn].get(rnp1)

                    Lambda2[indrn, indrnp1] += probacond * (xnpun - (xn * M[indrn, indrnp1, 0] + yn * M[indrn, indrnp1, 1] + ynpun * M[indrn, indrnp1, 2] + M[indrn, indrnp1, 3]))**2
                    Pi2    [indrn, indrnp1] += probacond * (ynpun - (yn * P[indrn, indrnp1, 0] + P[indrn, indrnp1, 1]))**2
                    Denom  [indrn, indrnp1] += probacond

        for indrn in range(self.__STEPS+2):
            for indrnp1 in range(self.__STEPS+2):
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
        MNum      = np.zeros(shape=(self.__STEPS+2, self.__STEPS+2, 4))
        MDenom    = np.zeros(shape=(self.__STEPS+2, self.__STEPS+2, 4, 4))
        # P = [F, G] ##################################################################################
        PNum      = np.zeros(shape=(self.__STEPS+2, self.__STEPS+2, 2))
        PDenom    = np.zeros(shape=(self.__STEPS+2, self.__STEPS+2, 2, 2))

        for real in range(self.__nbRealSEM):

            for n in range(self.__N-1):
                yn    = (self.__Ztrain[self.__n_x:self.__n_z, n])  .item()
                ynpun = (self.__Ztrain[self.__n_x:self.__n_z, n+1]).item()
                xn    = (self.__Ztrain[0:self.__n_x, n])           .item()
                xnpun = (self.__Ztrain[0:self.__n_x, n+1])         .item()
                
                vect1      = np.array([xn, yn, ynpun, 1.]) 
                vect1vect1 = np.outer(vect1, vect1)
                vect2      = np.array([yn, 1.])
                vect2vect2 = np.outer(vect2, vect2)

                indrn   = Rsimul[real*self.__N + n]
                indrnp1 = Rsimul[real*self.__N + n+1]
                rn      = getrnFromindrn(self.__Rcentres, indrn)
                rnp1    = getrnFromindrn(self.__Rcentres, indrnp1)

                ###################### M ########################
                MDenom[indrn, indrnp1, :, :] += vect1vect1
                MNum  [indrn, indrnp1, :]    += xnpun * vect1

                ###################### P ########################
                PDenom[indrn, indrnp1, :, :] += vect2vect2
                PNum  [indrn, indrnp1, :]    += ynpun * vect2

        M = np.zeros(shape=(self.__STEPS+2, self.__STEPS+2, 4))
        P = np.zeros(shape=(self.__STEPS+2, self.__STEPS+2, 2))
        for indrn in range(self.__STEPS+2):
            for indrnp1 in range(self.__STEPS+2):
                try:
                    # print(MNum[indrn, indrnp1,:])
                    # print(np.linalg.inv(MDenom[indrn, indrnp1,:,:]))
                    # print(np.dot(MNum[indrn, indrnp1,:], np.linalg.inv(MDenom[indrn, indrnp1,:,:])))
                    M[indrn, indrnp1,:] = np.dot(MNum[indrn, indrnp1,:], np.linalg.inv(MDenom[indrn, indrnp1,:,:]))
                    # print('M[indrn, indrnp1,:]=', M[indrn, indrnp1,:])
                    # input('xvxv;dk')
                except np.linalg.LinAlgError:
                    M[indrn, indrnp1,:] = 0.
                try:
                    P[indrn, indrnp1,:] = np.dot(PNum[indrn, indrnp1,:], np.linalg.inv(PDenom[indrn, indrnp1,:,:]))
                except np.linalg.LinAlgError:
                    P[indrn, indrnp1,:] = 0.

        # Pi2 and Lambda2 ##################################################################################
        Pi2     = np.zeros(shape=(self.__STEPS+2, self.__STEPS+2))
        Lambda2 = np.zeros(shape=(self.__STEPS+2, self.__STEPS+2))
        Denom   = np.zeros(shape=(self.__STEPS+2, self.__STEPS+2))
        
        for real in range(self.__nbRealSEM):

            for n in range(self.__N-1):
            
                yn    = (self.__Ztrain[self.__n_x:self.__n_z, n])  .item()
                ynpun = (self.__Ztrain[self.__n_x:self.__n_z, n+1]).item()
                xn    = (self.__Ztrain[0:self.__n_x, n])           .item()
                xnpun = (self.__Ztrain[0:self.__n_x, n+1])         .item()
                
                vect1      = np.array([xn, yn, ynpun, 1.]) 
                vect1vect1 = np.outer(vect1, vect1)
                vect2      = np.array([yn, 1.])
                vect2vect2 = np.outer(vect2, vect2)

                indrn   = Rsimul[real*self.__N + n]
                indrnp1 = Rsimul[real*self.__N + n+1]
                rn      = getrnFromindrn(self.__Rcentres, indrn)
                rnp1    = getrnFromindrn(self.__Rcentres, indrnp1)

                Lambda2[indrn, indrnp1] += (xnpun - (xn * M[indrn, indrnp1, 0] + yn * M[indrn, indrnp1, 1] + ynpun * M[indrn, indrnp1, 2] + M[indrn, indrnp1, 3]))**2
                Pi2    [indrn, indrnp1] += (ynpun - (yn * P[indrn, indrnp1, 0] + P[indrn, indrnp1, 1]))**2
                Denom  [indrn, indrnp1] += 1.

        for indrn in range(self.__STEPS+2):
            for indrnp1 in range(self.__STEPS+2):
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
        tab_GaussXY.append(Loi2DDiscreteFuzzy_TMC(self.__EPS, self.__STEPS, self.__Rcentres))

        # Les suivantes
        for np1 in range(1, self.__N):
            if self.__verbose >= 2:
                print('\r         proba tnp1 condit. to tn, np1=', np1, ' sur N=', self.__N, end='   ', flush = True)

            tab_GaussXY.append(Loi2DDiscreteFuzzy_TMC(self.__EPS, self.__STEPS, self.__Rcentres))
            tab_GaussXY[np1].Calc_GaussXY(self.__M, self.__Lambda2, self.__P, self.__Pi2, self.__Ztrain[:, np1-1], self.__Ztrain[:, np1])

        if self.__verbose >= 2: print(' ')
        return tab_GaussXY


    def compute_fuzzyjumps_forward(self, Tab_GaussXY):
        
        ProbaForward = []
        Tab_Normalis = np.zeros(shape=(self.__N))

        ######################
        # Initialisation
        np1  = 0
        ProbaForward.append(Loi1DDiscreteFuzzy_TMC(self.__EPS, self.__STEPS, self.__Rcentres))
        ProbaForward[np1].CalcForw1(self.__FS, self.__Ztrain[:, np1], self.__aMeanCovFuzzy)
        Tab_Normalis[np1] = ProbaForward[np1].Integ()
        # normalisation (devijver)
        ProbaForward[np1].normalisation(Tab_Normalis[np1])

        ###############################
        # Boucle
        for np1 in range(1, self.__N):
            if self.__verbose >= 2:
                print('\r         forward np1=', np1, ' sur N=', self.__N, end='', flush = True)

            ProbaForward.append(Loi1DDiscreteFuzzy_TMC(self.__EPS, self.__STEPS, self.__Rcentres))
            ProbaForward[np1].CalcForB(calcF, ProbaForward[np1-1], self.__FS, Tab_GaussXY[np1])
            Tab_Normalis[np1] = ProbaForward[np1].Integ()
            # normalisation (devijver)
            ProbaForward[np1].normalisation(Tab_Normalis[np1])

        if self.__verbose >= 2: print(' ')
        return ProbaForward, Tab_Normalis


    def compute_fuzzyjumps_backward(self, ProbaForwardNorm, Tab_GaussXY, Tab_Normalis):

        # Proba backward
        ProbaBackward = []

        # on créé la liste de tous les lois discrétisées
        for n in range(self.__N):
            ProbaBackward.append(Loi1DDiscreteFuzzy_TMC(self.__EPS, self.__STEPS, self.__Rcentres))

        # REMARQUE: la normalisation loi norm (thèse de Abassi), et la même que utiliser Tab_Normalis

        loicorrective = Loi1DDiscreteFuzzy_TMC(self.__EPS, self.__STEPS, self.__Rcentres)

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

            # Cette normalisation n'est implémentée que pour contre-carrer la dérive suite à l'intégration numérique
            # input('VERIFIER CETTE NORMALISATION')
            loicorrective.ProductFB(ProbaForwardNorm[n], ProbaBackward[n])
            ProbaBackward[n].normalisation(loicorrective.Integ())
            
        if self.__verbose >= 2: print(' ')
        return ProbaBackward


    def compute_fuzzyjumps_gammapsicond(self, ProbaForward, ProbaBackward, Tab_GaussXY):

        tab_gamma = []
        tab_psi   = []
        tab_cond  = []

        ###############################
        # Boucle sur gamma et psi
        for n in range(self.__N-1):
            if self.__verbose >= 2:
                print('\r         proba gamma psi cond n=', n, ' sur N=', self.__N, end='   ', flush = True)

            # calcul de gamma = produit forward norm * backward norm ****************************************************************
            tab_gamma.append(Loi1DDiscreteFuzzy_TMC(self.__EPS, self.__STEPS, self.__Rcentres))
            tab_gamma[n].ProductFB(ProbaForward[n], ProbaBackward[n])
    
            # normalisation : uniquement due pour compenser des pb liés aux approximations numeriques de forward et de backward
            # Si F= 20, on voit que la normalisation n'est pas necessaire (deja la somme == 1.)
            integ = tab_gamma[n].Integ()
            if np.fabs(1.-integ) > 5.E-2: # we stop only if more than 5% error
                print('\ntab_gamma[n].Integ()=', integ)
                print(np.fabs(1.-integ))
                print(np.fabs(1.-integ) > 5.E-2)
                ProbaForward[n].print()
                ProbaBackward[n].print()
                input('PB PB PB Gamma')
            tab_gamma[n].normalisation(integ)

            # calcul de psi (loi jointe a posteriori) ****************************************************************
            tab_psi.append(Loi2DDiscreteFuzzy_TMC(self.__EPS, self.__STEPS, self.__Rcentres))
            tab_psi[n].CalcPsi(ProbaForward[n], ProbaBackward[n+1], self.__FS, Tab_GaussXY[n+1])
            # normalisation
            integ = tab_psi[n].Integ()
            if integ == 0.:
                print("Tab Spi normalisation")
                ptrin('tab_psi[n].Integ()=', integ)
                tab_psi[n].print()
                input('pause')
            tab_psi[n].normalisation(integ)

            # calcul de p(rnp1 | rn, z_1^M)  ****************************************************************
            # on créé une liste, pour chaque valeur de rn, de lois 1D
            Liste = []
            for indrn in range(self.__STEPS+2):
                rn = getrnFromindrn(self.__Rcentres, indrn)
                
                Liste.append(Loi1DDiscreteFuzzy_TMC(self.__EPS, self.__STEPS, self.__Rcentres))
                if tab_gamma[n].get(rn) != 0.:
                    Liste[indrn].CalcCond(rn, tab_gamma[n].get(rn), tab_psi[n], self.__verbose)
                else:
                    if self.__verbose>2:
                        print('\nPBPB tab_gamma[n].get(rn) == 0. when rn=', rn)
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
        fname = './Result/Fuzzy/SimulatedR/'
        plt.savefig(fname + 'ParamFS_STEPS_' + str(self.__STEPS)+ '_NbReal_' + str(self.__nbRealSEM) + '.png', bbox_inches='tight', dpi=150)    
        plt.close()

        ax = plt.figure().gca()
        plt.plot(self.__Tab_M_00[:, 0], color='g', label=r'$\mathcal{A}_{0}^{0}$')
        plt.plot(self.__Tab_M_00[:, 1], color='r', label=r'$\mathcal{B}_{0}^{0}$')
        plt.plot(self.__Tab_M_00[:, 2], color='b', label=r'$\mathcal{C}_{0}^{0}$')
        #plt.plot(self.__Tab_M_00[:, 3], color='k', label=r'$\mathcal{D}_{0}^{0}$')
        plt.xlim(xmax=self.__nbIterSEM, xmin=0)
        plt.xlabel('SEM iteration')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        plt.title(r'Evolution of $\mathcal{M}_{0}^{0}$ param - Fuzzy steps (F)=' + str(self.__STEPS)+ ', mean of ' + str(self.__nbRealSEM) + ' realizations')
        fname = './Result/Fuzzy/SimulatedR/'
        plt.savefig(fname + 'MOO_STEPS_' + str(self.__STEPS)+ '_NbReal_' + str(self.__nbRealSEM) + '.png', bbox_inches='tight', dpi=150)    
        plt.close()

        ax = plt.figure().gca()
        plt.plot(self.__Tab_P_00[:, 0], color='g', label=r'$\mathcal{F}_{0}^{0}$')
        #plt.plot(self.__Tab_P_00[:, 1], color='r', label=r'$\mathcal{G}_{0}^{0}$')
        plt.xlim(xmax=self.__nbIterSEM, xmin=0)
        plt.xlabel('SEM iteration')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        plt.title(r'Evolution of $\mathcal{P}_{0}^{0}$ param - Fuzzy steps (F)=' + str(self.__STEPS)+ ', mean of ' + str(self.__nbRealSEM) + ' realizations')
        fname = './Result/Fuzzy/SimulatedR/'
        plt.savefig(fname + 'POO_STEPS_' + str(self.__STEPS)+ '_NbReal_' + str(self.__nbRealSEM) + '.png', bbox_inches='tight', dpi=150)    
        plt.close()

        ax = plt.figure().gca()
        plt.plot(self.__Tab_Lambda_00[:, 0], color='g', label=r'$\lambda_{0}^{0}$')
        plt.xlim(xmax=self.__nbIterSEM, xmin=0)
        plt.xlabel('SEM iteration')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        plt.title(r'Evolution of $\lambda_{0}^{0}$ param - Fuzzy steps (F)=' + str(self.__STEPS)+ ', mean of ' + str(self.__nbRealSEM) + ' realizations')
        fname = './Result/Fuzzy/SimulatedR/'
        plt.savefig(fname + 'Lambda_STEPS_' + str(self.__STEPS)+ '_NbReal_' + str(self.__nbRealSEM) + '.png', bbox_inches='tight', dpi=150)    
        plt.close()

        ax = plt.figure().gca()
        plt.plot(self.__Tab_Pi_00[:, 0],     color='g', label=r'$\pi_{0}^{0}$')
        plt.xlim(xmax=self.__nbIterSEM, xmin=0)
        plt.xlabel('SEM iteration')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        plt.title(r'Evolution of $\pi_{0}^{0}$ param - Fuzzy steps (F)=' + str(self.__STEPS)+ ', mean of ' + str(self.__nbRealSEM) + ' realizations')
        fname = './Result/Fuzzy/SimulatedR/'
        plt.savefig(fname + 'Pi_STEPS_' + str(self.__STEPS)+ '_NbReal_' + str(self.__nbRealSEM) + '.png', bbox_inches='tight', dpi=150)    
        plt.close()


    def printParam(self):
        print('\n  Estimated parameters')
        print('    --> loi jointe à priori=', self.__FS)
        # print('    --> param du modele (param 3)\n    M=', self.__M)
        # print('    --> param du modele (param 3)\n    Lambda**2=', self.__Lambda2)
        # print('    --> param du modele (param 3)\n    P=', self.__P)
        # print('    --> param du modele (param 3)\n    Pi**2=', self.__Pi2)


    def plotRsimul(self, Rsimul, fname, title):

        RsimulFuzzy = np.zeros(shape=(self.__N))

        for real in range(self.__nbRealSEM):
            for n in range(self.__N):
                RsimulFuzzy[n] = getrnFromindrn(self.__Rcentres, Rsimul[real*self.__N + n])

            fnamereal = fname + '_Real_'  + str(real) + '.png'
            titlereal = title + ', Real ' + str(real)

            maxi=500 # maxi=self.__N-1, maxi=500
            plt.figure()
            plt.plot(RsimulFuzzy, color='g')
            plt.ylim(ymax = 1.05, ymin = -0.05)
            plt.xlim(xmax = maxi, xmin = 0)
            plt.title(titlereal)
            plt.savefig(fnamereal, bbox_inches='tight', dpi=150)    
            plt.close()


###################################################################################################
class MeanCovFuzzy:

    def __init__(self, Ztrain, n_z, STEPS, verbose):
        self.__verbose  = verbose
        self.__STEPS    = STEPS
        self.__Ztrain   = Ztrain
        self.__n_z      = n_z

        self.__N = np.shape(Ztrain)[1]

        self.__Mean_Zf  = np.zeros(shape=(self.__STEPS+2, self.__n_z))
        self.__Cov_Zf   = np.zeros(shape=(self.__STEPS+2, self.__n_z, self.__n_z))
        self.__cpt      = np.zeros(shape=(self.__STEPS+2), dtype=int)

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
                self.__Mean_Zf[label,:] += self.__Ztrain[:, n]
                self.__cpt[label] += 1
        
        for indrn in range(self.__STEPS+2):
            if self.__cpt[indrn] != 0:
                self.__Mean_Zf[indrn, :] /= self.__cpt[indrn]
        # print(self.__Mean_Zf)
        # input('hard mean')
        
        # The variances
        VectZ = np.zeros(shape=(self.__n_z, 1))
        for real in range(nbreal):
            for n in range(self.__N):
                label = Rlabels[real*self.__N+n]
                VectZ = (np.transpose(self.__Ztrain[:, n]) - self.__Mean_Zf[label,:]).reshape(self.__n_z, 1)
                self.__Cov_Zf[label,:,:] += np.outer(VectZ, VectZ)
        for indrn in range(self.__STEPS+2):
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
        return self.__Mean_Zf[indrn, :]

    def getCov(self, indrn):
        return self.__Cov_Zf[indrn, :]
