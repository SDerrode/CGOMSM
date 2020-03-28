import numpy as np
import warnings
import scipy as sp
import clipboard

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import rc
rc('text', usetex=True)

from sklearn.cluster import KMeans
# from sklearn.cluster import AgglomerativeClustering

from OFAResto.LoiDiscreteFuzzy_TMC    import Loi1DDiscreteFuzzy_TMC, Loi2DDiscreteFuzzy_TMC, calcF, calcB
from Fuzzy.APrioriFuzzyLaw_Series2ter import LoiAPrioriSeries2ter
from CommonFun.CommonFun import From_FQ_to_Cov_Lyapunov, Test_if_CGPMSM, is_pos_def, From_Cov_to_FQ


def Check_CovMatrix(Mat):
    w, v = np.linalg.eig(Mat)
    # print("eig value:", w)
    # print(np.all(w>0.))
    # print(np.all(np.logical_not(np.iscomplex(w))))
    # input('pause')
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

        self.__n_x       = n_x
        self.__n_y       = n_y
        self.__n_z       = self.__n_x + self.__n_y

        if self.__STEPS != 0:
            self.__Rcentres = np.linspace(start=1./(2.*self.__STEPS), stop=1.0-1./(2.*self.__STEPS), num=self.__STEPS, endpoint=True)
        else:
            self.__Rcentres = np.empty(shape=(0,))

        # used for simualtion of discretized fuzzy r (in [0..self.__STEPS+1])
        self.__Rsimul = np.zeros(shape=(self.__N), dtype=int)

        # used to print the evolution of FS parameters
        self.__Tab_ParamFS   = np.zeros(shape=(self.__nbIterSEM+1, 4)) 
        self.__Tab_M_00      = np.zeros(shape=(self.__nbIterSEM+1, 4)) 
        self.__Tab_Lambda_00 = np.zeros(shape=(self.__nbIterSEM+1, 1)) 
        self.__Tab_P_00      = np.zeros(shape=(self.__nbIterSEM+1, 2)) 
        self.__Tab_Pi_00     = np.zeros(shape=(self.__nbIterSEM+1, 1)) 

        # Init of parameters : kmeans with 2+STEPS classes
        ###################################################
        #kmeans = KMeans(n_clusters=2+self.__STEPS, random_state=4, init='random', n_init=200).fit(np.transpose(Ztrain))
        kmeans = KMeans(n_clusters=2+self.__STEPS, random_state=None, init='random', n_init=200).fit(np.transpose(Ztrain))
        #kmeans = KMeans(n_clusters=2+self.__STEPS, n_init=200).fit(np.transpose(Ztrain))
        # print(kmeans.inertia_, kmeans.n_iter_)
        # print('kmeans.cluster_centers_=', kmeans.cluster_centers_)
        # Sorting of labels according to the X-coordinate of cluster centers
        sortedlabel = np.argsort(kmeans.cluster_centers_[:, 0])
        # print('sortedlabel=', sortedlabel)
        for n in range(self.__N):
            self.__Rsimul[n] =  np.where(sortedlabel == kmeans.labels_[n])[0][0]

        if self.__graphics >= 2:
            fname = './Result/Fuzzy/SimulatedR/Rsimul_Kmeans_cl' + str(self.__STEPS+2) + '.png'
            title = 'Simulated R - Kmeans'
            self.plotRsimul(fname=fname, title=title)

        # Update of param for the simulated R (by kmeans here for init)
        self.__FS, self.__M, self.__Lambda2, self.__P, self.__Pi2, self.__aMeanCovFuzzy = \
            self.updateParamFromRsimul(ProbaJumpCond=None)
        self.__Tab_ParamFS[0,:]   = self.__FS.getParam()
        self.__Tab_M_00[0,:]      = self.__M[0,0,0,:]
        self.__Tab_Lambda_00[0,0] = np.sqrt(self.__Lambda2[0,0])
        self.__Tab_P_00[0, :]     = self.__P[0,0,0,:]
        self.__Tab_Pi_00[0,0]     = np.sqrt(self.__Pi2[0,0])

        # Print des paramètres
        if self.__verbose >= 1: self.printParam()


    def getParams(self):
        return self.__alpha0, self.__alpha1, self.__beta, self.__FS.getEta()

    def EmpiricalFuzzyJointMatrix(self):
        
        alpha0, alpha1, beta = 0., 0., 0.
        for n in range(1, self.__N):
            if self.__Rsimul[n-1] == 0 and self.__Rsimul[n] == 0:
                alpha0 += 1.
            if self.__Rsimul[n-1] == self.__STEPS+1 and self.__Rsimul[n] == self.__STEPS+1:
                alpha1 += 1.
            if (self.__Rsimul[n-1] == 0 and self.__Rsimul[n] == self.__STEPS+1) or (self.__Rsimul[n-1] == self.__STEPS+1 and self.__Rsimul[n] == 0):
                beta   += 1.
        beta /= 2. # ca compte les transitions 0-1, 1-0, donc on divise par deux

        alpha0 /= (self.__N-1.)
        alpha1 /= (self.__N-1.)
        beta   /= (self.__N-1.)

        return alpha0, alpha1, beta


    def run_several(self, ):

        for i in range(1, self.__nbIterSEM+1):
            print('ITERATION ', i)
            self.__Tab_ParamFS[i,:], self.__Tab_M_00[i,:], self.__Tab_Lambda_00[i,0], self.__Tab_P_00[i,:], self.__Tab_Pi_00[i,0] = self.run_one(i)

        if self.__graphics >= 1 and self.__nbIterSEM>0:
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


    def run_one(self, numIterSEM):

        # MAJ des proba sur la base des paramètres courants
        Tab_GaussXY                         = self.compute_tab_GaussXY ()
        ProbaForwardNorm, Tab_Normalis      = self.compute_fuzzyjumps_forward (Tab_GaussXY)
        ProbaBackwardNorm                   = self.compute_fuzzyjumps_backward(ProbaForwardNorm, Tab_GaussXY, Tab_Normalis)
        ProbaGamma, ProbaPsi, ProbaJumpCond = self.compute_fuzzyjumps_gammapsicond(ProbaForwardNorm, ProbaBackwardNorm, Tab_GaussXY)

        listeFS, listeM, listeLambda2, listeP, listePi2, listeMeanCovFuzzy = [], [], [], [], [], []
        for r in range(self.__nbRealSEM):

            if self.__verbose >= 2:
                print('\r         realization r=', r, ' sur ', self.__nbRealSEM, end='', flush = True)

            # Simuler une chaîne de Markov a posteriori sur la base de la valeur des paramètres courants
            self.simulRealization(ProbaGamma[0], ProbaJumpCond)
            
            if self.__graphics >= 2:
                fname = './Result/Fuzzy/SimulatedR/Rsimul_Iter_' + str(numIterSEM) + '_Real_'+ str(r) + '_cl' + str(self.__STEPS+2) + '.png'
                title = 'Simulated R - Iter ' + str(numIterSEM) + ', Real ' + str(r)
                self.plotRsimul(fname=fname, title=title)
                #input('Dessin de rsimul')
    
            # Update of param for the simulated R (by kmeans here for init)
            FS, M, Lambda2, P, Pi2, aMeanCovFuzzy = self.updateParamFromRsimul(ProbaJumpCond=ProbaJumpCond)
            listeFS.append(FS) 
            listeM.append(M) 
            listeLambda2.append(Lambda2) 
            listeP.append(P) 
            listePi2.append(Pi2) 
            listeMeanCovFuzzy.append(aMeanCovFuzzy)

        # calcul des moyennes des paramètres
        self.__FS, self.__M, self.__Lambda2, self.__P, self.__Pi2, self.__aMeanCovFuzzy = \
            self.setMeanParamRealization(listeFS, listeM, listeLambda2, listeP, listePi2, listeMeanCovFuzzy)

        # Print des paramètres
        if self.__verbose >= 1: self.printParam()
    
        # input('fin de run_one mean')
        return self.__FS.getParam(), self.__M[0,0,0,:], np.sqrt(self.__Lambda2[0,0]), self.__P[0,0,0,:], np.sqrt(self.__Pi2[0,0])


    def ConvertandSaveParameters(self, filenameParam):

        # Convert from Param 3 to Param 1 (using equations from Zied) ############################################################@
        MeanX = np.zeros(shape=(self.__STEPS+2, self.__n_x))
        MeanY = np.zeros(shape=(self.__STEPS+2, self.__n_y))
        for indrn in range(self.__STEPS+2):
            MeanY[indrn] = self.__P[indrn, indrn, 0, 1] / (1.- self.__P[indrn, indrn, 0, 0])
            MeanX[indrn] = (self.__M[indrn, indrn, 0, 3] + MeanY[indrn] * (self.__M[indrn, indrn, 0, 1]+self.__M[indrn, indrn, 0, 2])) / (1. - self.__M[indrn, indrn, 0, 0])
            # print('indrn = ', indrn)
            # print('self.__P[indrn, indrn] = ', self.__P[indrn, indrn, 0, :])
            # print('self.__M[indrn, indrn] = ', self.__M[indrn, indrn, 0, :])

        # print('MeanX=', MeanX)
        # print('MeanY=', MeanY)
        # input('pause')

        A = np.zeros(((self.__STEPS+2)**2, self.__n_z, self.__n_z))
        Q = np.zeros(((self.__STEPS+2)**2, self.__n_z, self.__n_z))
        for indrn in range(self.__STEPS+2):
            for indrnp1 in range(self.__STEPS+2):
                ind = indrn*(self.__STEPS+2) + indrnp1

                ### A 
                A[ind, 0, 0] = self.__M[indrn, indrnp1, 0, 0]
                A[ind, 1, 1] = self.__P[indrn, indrnp1, 0, 0]
                A[ind, 1, 0] = 0.
                A[ind, 0, 1] = self.__M[indrn, indrnp1, 0, 1] + self.__P[indrn, indrnp1, 0, 0] * self.__M[indrn, indrnp1, 0, 2]

                ### Q
                Q[ind, 1, 1] = self.__Pi2[indrn, indrnp1]
                Q[ind, 0, 1] = self.__M[indrn, indrnp1, 0, 2] * Q[ind, 1, 1]
                Q[ind, 1, 0] = Q[ind, 0, 1]
                Q[ind, 0, 0] = self.__Lambda2[indrn, indrnp1] + self.__M[indrn, indrnp1, 0, 2] * Q[ind, 1, 0]
                
                # print('ind=', ind)
                if is_pos_def(Q[ind,:,:]) == False:
                    print('ind=', ind, ' --> PROBLEM with Q matrix in parameter file!!')
                    print(Q[ind,:,:])
                    input('pause Q')
                
                
        # print('Q:', Q)
        # print('A:', A)

        # B = np.zeros(((self.__STEPS+2)**2, self.__n_z, self.__n_z))
        # for i in range((self.__STEPS+2)**2):
        #     with warnings.catch_warnings():
        #         warnings.simplefilter('error')
        #         B[i,:, :] = sp.linalg.sqrtm(Q[i,:, :])

        # Convert from Param 1 to Param 2 ########################################################################################@
        print('########## CONVERSION VERS COV')
        Cov = From_FQ_to_Cov_Lyapunov(A, Q, self.__n_x)
        print('########## CONVERSION VERS INV')
        Fbis, Qbis = From_Cov_to_FQ(Cov)
        print('########## VERIF')
        print(A-Fbis)
        print(Q-Qbis)
        input('fin verif lyapunov')
        for ind in range((self.__STEPS+2)**2):
            if is_pos_def(Cov[ind,:,:]) == False:
                print('ind=', ind, ' --> PROBLEM with Cov matrix in parameter file!!')
                print(Cov[ind,:,:])
                input('pause Cov')

        # Test if the matrices form a CGPMSM
        if Test_if_CGPMSM(Cov) == False:
            print('The cov matrices in the parameter file doest not respect shape for CGPMSM!!')
            exit(1)

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

        # Generate the command to run the predictor #
        #############################################################################@

        #################################
        # Commande d'appel au programme
        hard, filt, smooth, predic = 0, 1, 0, 1
        chWork = str(hard) + ',' + str(filt) + ',' + str(smooth) + ',' + str(predic)
        param = self.__FS.getParam()
        nameY = './Data/Traffic/TMU5509/generated/TMU5509_train.csv'
        A  = 'python3 Test_CGOFMSM_Signals.py ' + filenameParam + ' 2ter:' + str('%.4f'%param[0]) + ':' + str('%.4f'%param[1]) + ':' + str('%.4f'%param[2]) + ' '
        A += chWork + ' ' + nameY + ' -1 2 0'

        clipboard.copy(A.strip()) # mise en mémoire de la commande à exécuter
        print('pour restaurer le signal:')
        print('\n', A, '\n')


    def printParam(self):
        print('\n  Estimated parameters')
        print('    --> loi jointe à priori=', self.__FS)
        # print('    --> param du modele (param 3)\n    M=', self.__M)
        # print('    --> param du modele (param 3)\n    Lambda**2=', self.__Lambda2)
        # print('    --> param du modele (param 3)\n    P=', self.__P)
        # print('    --> param du modele (param 3)\n    Pi**2=', self.__Pi2)
        # if self.__verbose >= 2:
        #     input('pause')

    def plotRsimul(self, fname, title):

        RsimulFuzzy = np.zeros(shape=(len(self.__Rsimul)))
        for n in range(self.__N):
            RsimulFuzzy[n] = getrnFromindrn(self.__Rcentres, self.__Rsimul[n])

        plt.figure()
        plt.plot(RsimulFuzzy, color='g')
        plt.ylim(ymax = 1.05, ymin = -0.05)
        plt.xlim(xmax = self.__N-1, xmin = 0)
        plt.title(title)
        plt.savefig(fname, bbox_inches='tight', dpi=150)    
        plt.close()

    def setMeanParamRealization(self, listeFS, listeM, listeLambda2, listeP, listePi2, listeMeanCovFuzzy):

        nbRealSEM = len(listeFS)

        # Moyennes des FS ######################################
        Tab_ParamFS = np.zeros(shape=(3, nbRealSEM))
        for r in range(nbRealSEM):
            L = listeFS[r].getParam()
            Tab_ParamFS[0][r] += L[0]
            Tab_ParamFS[1][r] += L[1]
            Tab_ParamFS[2][r] += L[2]
        Means = np.mean(Tab_ParamFS, axis=1)
        FS = LoiAPrioriSeries2ter(EPS=self.__EPS, discretization=0, alpha0=Means[0], alpha1=Means[1], beta=Means[2])

            
        # Moyennes des M ######################################
        M       = np.zeros(shape=(self.__STEPS+2, self.__STEPS+2, 1, 4))
        P       = np.zeros(shape=(self.__STEPS+2, self.__STEPS+2, 1, 2))
        Lambda2 = np.zeros(shape=(self.__STEPS+2, self.__STEPS+2))
        Pi2     = np.zeros(shape=(self.__STEPS+2, self.__STEPS+2))
        for r in range(nbRealSEM):
            M       += listeM[r]
            P       += listeP[r]
            Lambda2 += listeLambda2[r]
            Pi2     += listePi2[r]
        M       /= nbRealSEM
        P       /= nbRealSEM
        Lambda2 /= nbRealSEM
        Pi2     /= nbRealSEM

        # Moyennes des MeanCovFuzzy ######################################
        Mean_Zf = np.zeros(shape=(self.__STEPS+2, self.__n_z))
        Cov_Zf  = np.zeros(shape=(self.__STEPS+2, self.__n_z, self.__n_z))
        for r in range(nbRealSEM):
            Mean_Zf += listeMeanCovFuzzy[r].getMeanAll()
            Cov_Zf  += listeMeanCovFuzzy[r].getCovAll()
        Mean_Zf /= nbRealSEM
        Cov_Zf  /= nbRealSEM

        aMeanCovFuzzy = MeanCovFuzzy(self.__Ztrain, self.__n_z, self.__STEPS, self.__verbose)
        aMeanCovFuzzy.setMeanAll(Mean_Zf)
        aMeanCovFuzzy.setCovAll(Cov_Zf)
        
        return FS, M, Lambda2, P, Pi2, aMeanCovFuzzy


    def simulRealization(self, ProbaGamma_0, ProbaJumpCond):

        # sampling of the first from gamma
        np1=0
        # print('np1=', np1)
        # ProbaGamma_0.print()
        self.__Rsimul[np1] = ProbaGamma_0.getSample()
        # print('  --> self.__Rsimul[np1]=', self.__Rsimul[np1])
        # input('pause simul realization')

        # next ones according to conditional law
        for np1 in range(1, self.__N):
            # print('np1=', np1, ', self.__Rsimul[np1-1]=', self.__Rsimul[np1-1])
            # ProbaJumpCond[np1-1][self.__Rsimul[np1-1]].print()
            # print('sum ProbaJumpCond = ', ProbaJumpCond[np1-1][self.__Rsimul[np1-1]].Integ())
            self.__Rsimul[np1] = ProbaJumpCond[np1-1][self.__Rsimul[np1-1]].getSample()
            # print('  --> self.__Rsimul[np1]=', self.__Rsimul[np1])
            # input('pause simul realization')


    def updateParamFromRsimul(self, ProbaJumpCond=None):

        # Parameter for fuzzy Markov model called APrioriFuzzyLaw_serie2ter.py
        alpha0, alpha1, beta = self.EmpiricalFuzzyJointMatrix()
        FS = LoiAPrioriSeries2ter(EPS=self.__EPS, discretization=0, alpha0=alpha0, alpha1=alpha1, beta=beta)
        #print('Fin EmpiricalFuzzyJointMatrix')
        
        # Parameters of parametrization 3 (M=[A, B, C, D], Lambda**2, and P=[F, G], Pi**2)
        if ProbaJumpCond == None:
            #input('ProbaJumpCond == None')
            M, Lambda2, P, Pi2 = self.EstimParam2ter(FSprobaR2CondR1=FS.probaR2CondR1, ProbaJumpCond=None)
        else:
            #input('ProbaJumpCond')
            M, Lambda2, P, Pi2 = self.EstimParam2ter(FSprobaR2CondR1=None, ProbaJumpCond=ProbaJumpCond)

        # print('Fin de EstimParam2ter')

        # for n in range(self.__N):
        #     print('n=', n ,', self.__Rsimul[n]=', self.__Rsimul[n])
        # input('pause')
        # Parameters for the first p(r_1 | z_1)
        # necessaire pour le forward n=1
        aMeanCovFuzzy = MeanCovFuzzy(self.__Ztrain, self.__n_z, self.__STEPS, self.__verbose)
        aMeanCovFuzzy.update(self.__Rsimul)
        # print('Fin update MeanCovFuzzy')

        return FS, M, Lambda2, P, Pi2, aMeanCovFuzzy

    def EstimParam2ter(self, FSprobaR2CondR1=None, ProbaJumpCond=None):

        # M = [A, B, C, D] ##################################################################################
        MNum      = np.zeros(shape=(self.__STEPS+2, self.__STEPS+2, 1, 4))
        MDenom    = np.zeros(shape=(self.__STEPS+2, self.__STEPS+2, 4, 4))
        # P = [F, G] ##################################################################################
        PNum      = np.zeros(shape=(self.__STEPS+2, self.__STEPS+2, 1, 2))
        PDenom    = np.zeros(shape=(self.__STEPS+2, self.__STEPS+2, 2, 2))

        for n in range(self.__N-1):
            indrn    = self.__Rsimul[n]
            indrnp1 = self.__Rsimul[n+1]
            rn       = getrnFromindrn(self.__Rcentres, indrn)
            rnp1    = getrnFromindrn(self.__Rcentres, indrnp1)
            #print('indrn=', indrn, ', indrnp1=', indrnp1, ', rn=', rn, ', rnp1=', rnp1)
            # input('attente')

            zn    = self.__Ztrain[:, n].reshape((2, 1))
            yn    = (self.__Ztrain[self.__n_x:self.__n_z, n])  .item()
            ynpun = (self.__Ztrain[self.__n_x:self.__n_z, n+1]).item()
            xn    = (self.__Ztrain[0:self.__n_x, n])           .item()
            xnpun = (self.__Ztrain[0:self.__n_x, n+1])         .item()

            if FSprobaR2CondR1 != None:
                probacond = FSprobaR2CondR1(rn, rnp1)
            else:
                probacond = ProbaJumpCond[n][indrn].get(rnp1)

            ################## M ########################
            vect     = np.array([xn, yn, ynpun, 1]) 
            MDenom[indrn, indrnp1,:,:] += probacond * np.outer(vect, vect)
            MNum  [indrn, indrnp1, 0, :] += probacond * xnpun * vect
            # print('MNum=', MNum)
            # input('Estim Param - Parametrization 2ter - M')

            ################## P ########################
            vect     = np.array([yn, 1])
            PDenom[indrn, indrnp1,:,:] += probacond * np.outer(vect, vect)
            PNum  [indrn, indrnp1, 0, :] += probacond * ynpun * vect
            #print('PNum=', PNum)
            #input('Estim Param - Parametrization 2ter - P')
        
        M = np.zeros(shape=(self.__STEPS+2, self.__STEPS+2, 1, 4))
        P = np.zeros(shape=(self.__STEPS+2, self.__STEPS+2, 1, 2))
        for indrn in range(self.__STEPS+2):
            for indrnp1 in range(self.__STEPS+2):
                try:
                    M[indrn, indrnp1,:,:] = np.dot(MNum[indrn, indrnp1,:,:], np.linalg.inv(MDenom[indrn, indrnp1,:,:]))
                except np.linalg.LinAlgError:
                    M[indrn, indrnp1,:,:] = 0.
                    #print('\nindrn=', indrn, 'indrnp1=', indrnp1)
                    # print('MNum=', MNum[indrn, indrnp1,:,:])
                    # print('MDenom', MDenom[indrn, indrnp1,:,:])
                    #input('pause M ')

                try:
                    P[indrn, indrnp1,:,:] = np.dot(PNum[indrn, indrnp1,:,:], np.linalg.inv(PDenom[indrn, indrnp1,:,:]))
                except np.linalg.LinAlgError:
                    P[indrn, indrnp1,:,:] = 0.
                    #print('\nindrn=', indrn, 'indrnp1=', indrnp1)
                    #input('pause P ')


        # Pi2 and Lambda2 ##################################################################################
        Pi2     = np.zeros(shape=(self.__STEPS+2, self.__STEPS+2))
        Lambda2 = np.zeros(shape=(self.__STEPS+2, self.__STEPS+2))
        Denom   = np.zeros(shape=(self.__STEPS+2, self.__STEPS+2))
        for n in range(self.__N-1):
            indrn    = self.__Rsimul[n]
            indrnp1 = self.__Rsimul[n+1]
            rn       = getrnFromindrn(self.__Rcentres, indrn)
            rnp1    = getrnFromindrn(self.__Rcentres, indrnp1)
            #print('indrn=', indrn, ', indrnp1=', indrnp1, ', rn=', rn, ', rnp1=', rnp1)
            # input('attente')

            yn    = (self.__Ztrain[self.__n_x:self.__n_z, n])  .item()
            ynpun = (self.__Ztrain[self.__n_x:self.__n_z, n+1]).item()
            xn    = (self.__Ztrain[0:self.__n_x, n])           .item()
            xnpun = (self.__Ztrain[0:self.__n_x, n+1])         .item()

            if FSprobaR2CondR1 != None:
                probacond = FSprobaR2CondR1(rn, rnp1)
            else:
                probacond = ProbaJumpCond[n][indrn].get(rnp1)
            
            Lambda2[indrn, indrnp1] += probacond * (xnpun - (xn * M[indrn, indrnp1, 0, 0] + yn * M[indrn, indrnp1, 0, 1] + ynpun * M[indrn, indrnp1, 0, 2] + M[indrn, indrnp1, 0, 3]))**2
            Pi2[indrn, indrnp1]     += probacond * (ynpun - (yn * P[indrn, indrnp1, 0, 0] + P[indrn, indrnp1, 0, 1]))**2
            Denom[indrn, indrnp1]   += probacond

        for indrn in range(self.__STEPS+2):
            for indrnp1 in range(self.__STEPS+2):
                if Lambda2[indrn, indrnp1] != 0.:
                    Lambda2[indrn, indrnp1] /= Denom[indrn, indrnp1]
                if Pi2[indrn, indrnp1] != 0.:
                    Pi2    [indrn, indrnp1] /= Denom[indrn, indrnp1]
        # print('Lambda2=', Lambda2)
        # print('Pi2=', Pi2)
        # input('pause')

        return M, Lambda2, P, Pi2

    def compute_tab_GaussXY(self):

        tab_GaussXY = []

        # La premiere ne sert à rien, uniquementà a synchroniser les indices
        tab_GaussXY.append(Loi2DDiscreteFuzzy_TMC(self.__EPS, self.__STEPS, self.__Rcentres))

        for np1 in range(1, self.__N):
            if self.__verbose >= 2:
                print('\r         proba tnp1 condit. to tn np1=', np1, ' sur N=', self.__N, end='   ', flush = True)

            tab_GaussXY.append(Loi2DDiscreteFuzzy_TMC(self.__EPS, self.__STEPS, self.__Rcentres))
            tab_GaussXY[np1].Calc_GaussXY(self.__M, self.__Lambda2, self.__P, self.__Pi2, self.__Ztrain[:, np1-1], self.__Ztrain[:, np1])

            # if tab_GaussXY[np1].Integ()<1E-10:
            #     print('np1 = ', np1)
            #     tab_GaussXY[np1].print()
            #     input('Tab Gauss')

        if self.__verbose >= 2:
            print(' ')

        return tab_GaussXY


    def compute_fuzzyjumps_forward(self, Tab_GaussXY):
        
        ProbaForward = []
        Tab_Normalis = np.zeros(shape=(self.__N))

        ######################
        # Initialisation
        np1  = 0
        znp1 = self.__Ztrain[:, np1]
        ProbaForward.append(Loi1DDiscreteFuzzy_TMC(self.__EPS, self.__STEPS, self.__Rcentres))

        ProbaForward[np1].CalcForw1(self.__FS, znp1, self.__aMeanCovFuzzy)
        Tab_Normalis[np1] = ProbaForward[np1].Integ()
        # ProbaForward[np1].print()
        # normalisation (devijver)
        ProbaForward[np1].normalisation(Tab_Normalis[np1])
        #ProbaForward[np1].plot('$p(r_n | y_1^n)$')
        # input('attente forward n=' + str(np1))

        ###############################
        # Boucle
        for np1 in range(1, self.__N):
            if self.__verbose >= 2:
                print('\r         forward np1=', np1, ' sur N=', self.__N, end='', flush = True)

            ProbaForward.append(Loi1DDiscreteFuzzy_TMC(self.__EPS, self.__STEPS, self.__Rcentres))
            # print('self.__Rcentres=', self.__Rcentres)
            ProbaForward[np1].CalcForB(calcF, ProbaForward[np1-1], self.__FS, Tab_GaussXY[np1])
            # ProbaForward[np1].nextAfterZeros() # on evite des proba de zero
            # ProbaForward[np1].print()
            # if ProbaForward[np1].get(0.) < ProbaForward[np1].get(1.):
            #     ProbaForward[np1].print()
            #     input('attente forward n=' + str(np1))
            # print('sum forw=', ProbaForward[np1].Integ())
            Tab_Normalis[np1] = ProbaForward[np1].Integ()
            # normalisation (devijver)
            ProbaForward[np1].normalisation(Tab_Normalis[np1])
            # print('sum forw=', ProbaForward[np1].Integ())
            # ProbaForward[np1].print()
            # input('attente forward n=' + str(np1))

        if self.__verbose >= 2:
            print(' ')

        return ProbaForward, Tab_Normalis


    def compute_fuzzyjumps_backward(self, ProbaForwardNorm, Tab_GaussXY, Tab_Normalis):

        # Proba backward
        ProbaBackward = []

        # on créé la liste de tous les proba
        for n in range(self.__N):
            ProbaBackward.append(Loi1DDiscreteFuzzy_TMC(self.__EPS, self.__STEPS, self.__Rcentres))

        # REMARQUE ; la normalisation loi norm (thèse de Abassi), et la même que utiliser Tab_Normalis
        # Donc j''utilise ci-dfessous Tab_Normalis
        # loinorm = Loi1DDiscreteFuzzy_TMC(self.__EPS, self.__STEPS, self.__Rcentres)

        loicorrective = Loi1DDiscreteFuzzy_TMC(self.__EPS, self.__STEPS, self.__Rcentres)

        ##############################
        # initialisation de beta
        n = self.__N-1
        # print('n=', n)
        # input('temp')
        ProbaBackward[n].setValCste(1.0)
        # normalisation (devijver)
        # loinorm.ProductFB(ProbaBackward[n], ProbaForwardNorm[n])
        # loinorm.nextAfterZeros() # on evite des proba de zero
        # ProbaBackward[n].normalisation(loinorm.Integ())
        # print('  -->loinorm.Integ()=', loinorm.Integ())
        # ProbaBackward[n].plot('$p(r_{n+1} | y_1^{n+1})$')
        # ProbaBackward[n].print()
        if ProbaBackward[n].get(0.) < ProbaBackward[n].get(1.):
            ProbaBackward[n].print()
            print('  -->sum backw=', ProbaBackward[n].Integ())
            input('attente backward n=' + str(n))

        ###############################
        # Boucle pour backward
        for n in range(self.__N-2, -1, -1):
            if self.__verbose >= 2:
                print('\r         backward n=', n, ' sur N=', self.__N, end='             ', flush = True)
        
            ProbaBackward[n].CalcForB(calcB, ProbaBackward[n+1], self.__FS, Tab_GaussXY[n+1])
            
            # normalisation (devijver)
            ProbaBackward[n].normalisation(Tab_Normalis[n+1])

            # loinorm.ProductFB(ProbaForwardNorm[n], ProbaBackward[n])
            # loinorm.nextAfterZeros() # on evite des proba de zero
            # if loinorm.Integ() == 0.:
            #     print('loinorm.Integ()=', loinorm.Integ())
            #     print('ProbaForwardNorm[n]=', ProbaForwardNorm[n])
            #     print('ProbaBackward[n]=', ProbaBackward[n])
            #     input('pb loinorm.Integ() == 0.')
            # ProbaBackward[n].normalisation(loinorm.Integ())
            # ProbaBackward[n].print()
            # print('sum back=', ProbaBackward[n].Integ())
            # ProbaBackward[n].nextAfterZeros() # on evite des proba de zero
            # print('\n  -->sum backw=', ProbaBackward[n].Integ())
            # print('  -->loinorm.Integ()=', loinorm.Integ())
            # if abs(Tab_Normalis[n+1]-loinorm.Integ()) > 1E-14:
            #     print('  -->Tab_Normalis[n+1]-oinorm.Integ()=', Tab_Normalis[n+1]-loinorm.Integ())
            #     input('backward')

            # Cette normalisation n'est implémentée que pour contre-carrer la dérive suite à l'intégration numérique
            loicorrective.ProductFB(ProbaForwardNorm[n], ProbaBackward[n])
            ProbaBackward[n].normalisation(loicorrective.Integ())
            
        if self.__verbose >= 2:
            print(' ')

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
            #tab_gamma[n].nextAfterZeros() # on evite des proba de zero
            #tab_gamma[n].print()
            
            # if n<10 or n>self.__N-10:
            #     integ = tab_gamma[n].Integ()
            #     print('\ntab_gamma[n].Integ()=', integ)
            #     input('pause')

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
                    if self.__verbose>1:
                        print('\nPBPB tab_gamma[n].get(rn) == 0. when rn=', rn)
                        #input('ATTENTE ERROR')

                    # loi uniforme
                    Liste[indrn].setValCste(1.)
                    Liste[indrn].normalisation(Liste[indrn].Integ())
                
            tab_cond.append(Liste)
            #input('pause sdf')

        # for n in range(self.__N-1):
        #     print('n=', n)
        #     tab_gamma[n]  .print()
        #     tab_psi[n]    .print()
        #     tab_cond[n][0].print()
        #     tab_cond[n][1].print()
        #     input('tempo')
        

        # le dernier pour gamma
        # n = self.__N-1
        # tab_gamma.append(Loi1DDiscreteFuzzy_TMC(self.__EPS, self.__STEPS, self.__Rcentres))
        # tab_gamma[n].ProductFB(ProbaForward[n], ProbaBackward[n])
        # integ = tab_gamma[n].Integ()
        # print('\ntab_gamma[n].Integ()=', integ)
        # input('pause')
        # #tab_gamma[n].nextAfterZeros() # on evite des proba de zero
        # if abs(1.-tab_gamma[n].Integ()) > 1E-3:
        #     print('tab_gamma[n].Integ()=', tab_gamma[n].Integ())
        #     input('PB PB PB Gamma')
        #     tab_gamma[n].normalisation(tab_gamma[n].Integ())

        if self.__verbose >= 2:
            print(' ')

        return tab_gamma, tab_psi, tab_cond



###################################################################################################
class MeanCovFuzzy:

    def __init__(self, Ztrain, n_z, STEPS, verbose):
        self.__verbose  = verbose
        self.__STEPS    = STEPS
        self.__Ztrain   = Ztrain
        self.__n_z      = n_z

        self.__Mean_Zf  = np.zeros(shape=(self.__STEPS+2, self.__n_z))
        self.__Cov_Zf   = np.zeros(shape=(self.__STEPS+2, self.__n_z, self.__n_z))
        self.__cpt      = np.zeros(shape=(self.__STEPS+2), dtype=int)


    def update(self, Rlabels):

        N = np.shape(Rlabels)[0]
        # print('N=', N)

        # Remise à 0.
        self.__Mean_Zf.fill(0.)
        self.__Cov_Zf.fill(0.)
        self.__cpt.fill(0)
        # print('self.__Mean_Zf=', self.__Mean_Zf)
    
        # The means
        for n in range(N):
            self.__Mean_Zf[Rlabels[n], :] += self.__Ztrain[:, n]
            self.__cpt[Rlabels[n]] += 1
        for indrn in range(self.__STEPS+2):
            self.__Mean_Zf[indrn, :] /= self.__cpt[indrn]
        print('cpt=', self.__cpt, ', sum=', np.sum(self.__cpt))
        # print(self.__Mean_Zf)
        # input('hard mean')
        
        # The variances
        VectZ = np.zeros(shape=(self.__n_z, 1))
        for n in range(N):
            VectZ = (np.transpose(self.__Ztrain[:, n]) - self.__Mean_Zf[Rlabels[n], :]).reshape(self.__n_z, 1)
            self.__Cov_Zf[Rlabels[n],:,:] += np.outer(VectZ, VectZ)
        for indrn in range(self.__STEPS+2):
            self.__Cov_Zf[indrn,:,:] /= self.__cpt[indrn]
            # check if cov matrix
            if Check_CovMatrix( self.__Cov_Zf[indrn,:,:]) == False:
                print('cpt=', self.__cpt, ', sum=', np.sum(self.__cpt))
                print(self.__Mean_Zf[indrn, :])
                print(self.__Cov_Zf[indrn,:,:])
                input('This is not a cov matrix!!')

        # print(self.__Cov_Zf)
        # input('hardcov')

    def getMean(self, indrn):
        return self.__Mean_Zf[indrn, :]

    def getCov(self, indrn):
        return self.__Cov_Zf[indrn, :]

    def getMeanAll(self):
        return self.__Mean_Zf

    def getCovAll(self):
        return self.__Cov_Zf

    def setMeanAll(self, Mean_Zf):
        self.__Mean_Zf = Mean_Zf

    def setCovAll(self, Cov_Zf):
        self.__Cov_Zf = Cov_Zf

