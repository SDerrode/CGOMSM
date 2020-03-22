import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
# from sklearn.cluster import AgglomerativeClustering

from OFAResto.LoiDiscreteFuzzy_TMC    import Loi1DDiscreteFuzzy_TMC, Loi2DDiscreteFuzzy_TMC, calcF, calcB
from Fuzzy.APrioriFuzzyLaw_Series2ter import LoiAPrioriSeries2ter

def Check_CovMatrix(Mat):
    w, v = np.linalg.eig(Mat)
    # print("eig value:", w)
    # print(np.all(w>0.))
    # print(np.all(np.logical_not(np.iscomplex(w))))
    # input('pause')
    if np.all(np.logical_not(np.iscomplex(w))) == False or np.all(w>0.) == False:
        return False
    return True


###################################################################################################
class CGOFMSM_SemiSupervLearn:
    def __init__(self, STEPS, Ztrain, n_x, n_y, verbose, graphics):
        
        self.__n_r      = 2
        self.__N        = np.shape(Ztrain)[1]
        self.__Ztrain   = Ztrain
        self.__verbose  = verbose
        self.__graphics = graphics
        self.__STEPS    = STEPS
        self.__EPS      = 1E-8

        self.__n_x      = n_x
        self.__n_y      = n_y
        self.__n_z      = self.__n_x + self.__n_y

        if self.__STEPS != 0:
            self.__Rcentres = np.linspace(start=1./(2.*self.__STEPS), stop=1.0-1./(2.*self.__STEPS), num=self.__STEPS, endpoint=True)
        else:
            self.__Rcentres = np.empty(shape=(0,))

        # used for simualtion of discretized fuzzy r (in [0..self.__STEPS+1])
        self.__Rsimul = np.zeros(shape=(self.__N), dtype=int)

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

        if self.__graphics == 1:
            fname = './Result/Fuzzy/SimulatedR/Rsimul_Kmeans_cl' + str(self.__STEPS+2) + '.png'
            title = 'Simulated R - Kmeans'
            self.plotRsimul(fname=fname, title=title)

        # Update of param for the simulated R (by kmeans here for init)
        self.__FS, self.__M, self.__Lambda2, self.__P, self.__Pi2, self.__aMeanCovFuzzy = \
            self.updateParamFromRsimul(ProbaJumpCond = None)

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


    def run_several(self, nbIterSEM, nbRealSEM):

        ok = True
        for i in range(nbIterSEM):
            print('ITERATION ', i)
            ok &= self.run_one(i, nbRealSEM)
        
        return ok

    def run_one(self, numIterSEM, nbRealSEM):

        # MAJ des proba sur la base des paramètres courants
        Tab_GaussXY                         = self.compute_tab_GaussXY ()
        ProbaForwardNorm                    = self.compute_fuzzyjumps_forward (Tab_GaussXY)
        ProbaBackwardNorm                   = self.compute_fuzzyjumps_backward(ProbaForwardNorm, Tab_GaussXY)
        ProbaGamma, ProbaPsi, ProbaJumpCond = self.compute_fuzzyjumps_gammapsicond(ProbaForwardNorm, ProbaBackwardNorm, Tab_GaussXY)

        listeFS, listeM, listeLambda2, listeP, listePi2, listeMeanCovFuzzy = [], [], [], [], [], []
        for r in range(nbRealSEM):

            if self.__verbose >= 2:
                print('\r         realization r=', r, ' sur ', nbRealSEM, end='', flush = True)

            # Simuler une chaines de Markov a posteriori sur la base de la valeur des paramètres courants
            self.simulRealization(ProbaGamma[0], ProbaJumpCond)
            
            if self.__graphics == 1:
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
        return True


    def ConvertandSaveParameters(self, filename):

        print(" TO BE DONE...")

        # Convert from Param 3 to Param 1 (using equations from Zied)

        # Convert from Param 1 to Param 2 

        # Save the CGOFMSM file



    def printParam(self):
        print('\n  Estimated parameters')
        print('    --> loi jointe à priori=', self.__FS)
        # print('  --> param du modele (param 3)\n    M=', self.__M)
        # print('  --> param du modele (param 3)\n    Lambda**2=', self.__Lambda2)
        # print('  --> param du modele (param 3)\n    P=', self.__P)
        # print('  --> param du modele (param 3)\n    Pi**2=', self.__Pi2)
        # if self.__verbose >= 2:
            # input('pause')

    def plotRsimul(self, fname, title):

        RsimulFuzzy = np.zeros(shape=(len(self.__Rsimul)))
        for n in range(self.__N):
            if   self.__Rsimul[n] == 0:              rn = 0.
            elif self.__Rsimul[n] == self.__STEPS+1: rn = 1.
            else:                                    rn = self.__Rcentres[self.__Rsimul[n]-1]
            RsimulFuzzy[n] = rn

        plt.figure()
        plt.plot(RsimulFuzzy, color='g')
        plt.ylim(ymax = 1., ymin = 0)
        plt.xlim(xmax = self.__N-1, xmin = 0)
        plt.title(title)
        plt.savefig(fname, bbox_inches='tight', dpi=150)    
        plt.close()

    def setMeanParamRealization(self, listeFS, listeM, listeLambda2, listeP, listePi2, listeMeanCovFuzzy):

        nbRealSEM = len(listeFS)

        # Moyennes des FS ######################################
        alpha0, alpha1, beta = 0., 0., 0.
        for r in range(nbRealSEM):
            L = listeFS[r].getParam()
            alpha0 += L[0]
            alpha1 += L[1]
            beta   += L[2]
        alpha0 /= nbRealSEM
        alpha1 /= nbRealSEM
        beta   /= nbRealSEM
        FS = LoiAPrioriSeries2ter(EPS=self.__EPS, discretization=0, alpha0=alpha0, alpha1=alpha1, beta=beta)

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
        self.__Rsimul[np1] = ProbaGamma_0.getSample()

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
        MatNum    = np.zeros(shape=(1, 4))
        MDenom    = np.zeros(shape=(self.__STEPS+2, self.__STEPS+2, 4, 4))
        MatDenom  = np.zeros(shape=(4, 4))
        # P = [F, G] ##################################################################################
        PNum      = np.zeros(shape=(self.__STEPS+2, self.__STEPS+2, 1, 2))
        PMatNum   = np.zeros(shape=(1, 2))
        PDenom    = np.zeros(shape=(self.__STEPS+2, self.__STEPS+2, 2, 2))
        PMatDenom = np.zeros(shape=(2, 2))

        for n in range(self.__N-1):
            qn    = self.__Rsimul[n]
            qnpun = self.__Rsimul[n+1]
            if   qn == 0:                 rn=0.
            elif qn == self.__STEPS+1:    rn=1.
            else:                         rn=self.__Rcentres[qn-1]
            if   qnpun == 0:              rnpun=0.
            elif qnpun == self.__STEPS+1: rnpun=1.
            else:                         rnpun=self.__Rcentres[qnpun-1]
            #print('qn=', qn, ', qn+1=', qnpun, ', rn=', rn, ', rnpun=', rnpun)
            # input('attente')

            zn    = self.__Ztrain[:, n].reshape((2, 1))
            yn    = (self.__Ztrain[self.__n_x:self.__n_z, n])  .item()
            ynpun = (self.__Ztrain[self.__n_x:self.__n_z, n+1]).item()
            xn    = (self.__Ztrain[0:self.__n_x, n])           .item()
            xnpun = (self.__Ztrain[0:self.__n_x, n+1])         .item()

            if FSprobaR2CondR1 != None:
                probacond = FSprobaR2CondR1(rn, rnpun)
            else:
                probacond = ProbaJumpCond[n][qn].get(rnpun)

            ################## M ########################
            MatDenom[0:2, 0:2] = np.dot(zn, np.transpose(zn))
            MatDenom[0:2, 2:3] = zn * ynpun
            MatDenom[2:3, 0:2] = np.transpose(MatDenom[0:2, 2:3])
            MatDenom[0:2, 3:4] = zn
            MatDenom[3:4, 0:2] = np.transpose(MatDenom[0:2, 3:4])
            MatDenom[2:3, 2:3] = ynpun*ynpun
            MatDenom[2:3, 3:4] = ynpun
            MatDenom[3:4, 2:3] = np.transpose(MatDenom[2:3, 3:4])
            MatDenom[3:4, 3:4] = 1.
            #print('MatDenom=', MatDenom)
            MDenom[qn, qnpun, :, :] += probacond * MatDenom
            MatNum[0, 0] = xn 
            MatNum[0, 1] = yn 
            MatNum[0, 2] = ynpun 
            MatNum[0, 3] = 1. 
            #print(np.shape(MatNum))
            MNum[qn, qnpun, :, :] += probacond * xnpun * MatNum
            #print('MNum=', MNum)
            #input('Estim Param - Parametrization 2ter - M')

            ################## P ########################
            PMatDenom[0, 0] = yn*yn
            PMatDenom[0, 1] = yn
            PMatDenom[1, 0] = np.transpose(PMatDenom[0, 1])
            PMatDenom[1, 1] = 1.
            #print('PMatDenom=', PMatDenom)
            PDenom[qn, qnpun, :, :] += probacond * PMatDenom

            PMatNum[0, 0] = yn 
            PMatNum[0, 1] = 1. 
            PNum[qn, qnpun, :, :] += probacond * ynpun * PMatNum

            #print('PNum=', PNum)
            #input('Estim Param - Parametrization 2ter - P')
        
        M = np.zeros(shape=(self.__STEPS+2, self.__STEPS+2, 1, 4))
        P = np.zeros(shape=(self.__STEPS+2, self.__STEPS+2, 1, 2))
        for q1 in range(self.__STEPS+2):
            for q2 in range(self.__STEPS+2):
                try:
                    numinverse = np.linalg.inv(MDenom[q1, q2, :, :])
                    M[q1, q2, :, :] = np.dot(MNum[q1, q2, :, :], numinverse)
                except np.linalg.LinAlgError:
                    M[q1, q2, :, :] = 0.
                    # print('\nq1=', q1, 'q2=', q2)
                    # print('MNum=', MNum[q1, q2, :, :])
                    # print('MDenom', MDenom[q1, q2, :, :])
                    # input('pause M ')

                try:
                    numinverse = np.linalg.inv(PDenom[q1, q2, :, :])
                    P[q1, q2, :, :] = np.dot(PNum[q1, q2, :, :], numinverse)
                except np.linalg.LinAlgError:
                    P[q1, q2, :, :] = 0.
                    # print('\nq1=', q1, 'q2=', q2)
                    # input('pause P ')


        # Pi2 and Lambda2 ##################################################################################
        Pi2     = np.zeros(shape=(self.__STEPS+2, self.__STEPS+2))
        Lambda2 = np.zeros(shape=(self.__STEPS+2, self.__STEPS+2))
        Denom   = np.zeros(shape=(self.__STEPS+2, self.__STEPS+2))
        for n in range(self.__N-1):
            qn    = self.__Rsimul[n]
            qnpun = self.__Rsimul[n+1]
            if   qn == 0:                 rn=0.
            elif qn == self.__STEPS+1:    rn=1.
            else:                         rn=self.__Rcentres[qn-1]
            if   qnpun == 0:              rnpun=0.
            elif qnpun == self.__STEPS+1: rnpun=1.
            else:                         rnpun=self.__Rcentres[qnpun-1]
            #print('qn=', qn, ', qn+1=', qnpun, ', rn=', rn, ', rnpun=', rnpun)
            # input('attente')

            yn    = (self.__Ztrain[self.__n_x:self.__n_z, n])  .item()
            ynpun = (self.__Ztrain[self.__n_x:self.__n_z, n+1]).item()
            xn    = (self.__Ztrain[0:self.__n_x, n])           .item()
            xnpun = (self.__Ztrain[0:self.__n_x, n+1])         .item()

            if FSprobaR2CondR1 != None:
                probacond = FSprobaR2CondR1(rn, rnpun)
            else:
                probacond = ProbaJumpCond[n][qn].get(rnpun)
            Lambda2[qn, qnpun] += probacond * (xnpun - (xn * M[q1, q2, 0, 0] + yn * M[q1, q2, 0, 1] + ynpun * M[q1, q2, 0, 2] + M[q1, q2, 0, 3]))**2
            Pi2[qn, qnpun]     += probacond * (ynpun - (yn * P[q1, q2, 0, 0] + P[q1, q2, 0, 1]))**2
            Denom[qn, qnpun]   += probacond

        for q1 in range(self.__STEPS+2):
            for q2 in range(self.__STEPS+2):
                if Lambda2[q1, q2] != 0.:
                    Lambda2[q1, q2] /= Denom[q1, q2]
                if Pi2[q1, q2] != 0.:
                    Pi2    [q1, q2] /= Denom[q1, q2]
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
                print('\r         proba tnp1 con tn np1=', np1, ' sur N=', self.__N, end='   ', flush = True)

            tab_GaussXY.append(Loi2DDiscreteFuzzy_TMC(self.__EPS, self.__STEPS, self.__Rcentres))
            tab_GaussXY[np1].Calc_GaussXY(self.__M, self.__Lambda2, self.__P, self.__Pi2, self.__Ztrain[:, np1-1], self.__Ztrain[:, np1])

        if self.__verbose >= 2:
            print(' ')

        return tab_GaussXY


    def compute_fuzzyjumps_forward(self, Tab_GaussXY):
        
        ProbaForward = []
        #tab_normalis = np.zeros(shape=(self.__N))

        ######################
        # Initialisation
        np1  = 0
        znp1 = self.__Ztrain[:, np1]
        ProbaForward.append(Loi1DDiscreteFuzzy_TMC(self.__EPS, self.__STEPS, self.__Rcentres))

        ProbaForward[np1].CalcForw1(self.__FS, znp1, self.__aMeanCovFuzzy)
        #tab_normalis[np1] = ProbaForward[np1].Integ()
        # normalisation (devijver)
        ProbaForward[np1].normalisation(ProbaForward[np1].Integ())
        #ProbaForward[np1].print()
        #ProbaForward[np1].plot('$p(r_n | y_1^n)$')
        #input('attente forward n=' + str(np1))

        ###############################
        # Boucle
        for np1 in range(1, self.__N):
            if self.__verbose >= 2:
                print('\r         forward np1=', np1, ' sur N=', self.__N, end='', flush = True)

            ProbaForward.append(Loi1DDiscreteFuzzy_TMC(self.__EPS, self.__STEPS, self.__Rcentres))
            #print('self.__Rcentres=', self.__Rcentres)
            ProbaForward[np1].CalcForB(calcF, ProbaForward[np1-1], self.__FS, Tab_GaussXY[np1])
            #ProbaForward[np1].nextAfterZeros() # on evite des proba de zero
            #ProbaForward[np1].print()
            # print('sum forw=', ProbaForward[np1].Integ())
            # tab_normalis[np1] = ProbaForward[np1].Integ()
            # normalisation (devijver)
            ProbaForward[np1].normalisation(ProbaForward[np1].Integ())
            # print('sum forw=', ProbaForward[np1].Integ())
            # ProbaForward[np1].print()
            # input('attente forward n=' + str(np1))

        if self.__verbose >= 2:
            print(' ')

        return ProbaForward#, tab_normalis


    def compute_fuzzyjumps_backward(self, ProbaForwardNorm, Tab_GaussXY):

        # Proba backward
        ProbaBackward = []

        loinorm = Loi1DDiscreteFuzzy_TMC(self.__EPS, self.__STEPS, self.__Rcentres)

        # on créé la liste de tous les proba
        for n in range(self.__N):
            ProbaBackward.append(Loi1DDiscreteFuzzy_TMC(self.__EPS, self.__STEPS, self.__Rcentres))

        ##############################
        # initialisation de beta
        n = self.__N-1
        # print('n=', n)
        # input('temp')
        ProbaBackward[n].setValCste(1.0)
        # normalisation (devijver)
        loinorm.ProductFB(ProbaBackward[n], ProbaForwardNorm[n])
        #loinorm.nextAfterZeros() # on evite des proba de zero
        ProbaBackward[n].normalisation(loinorm.Integ())
        # ProbaBackward[n].print()
        # input('backward')

        ###############################
        # Boucle pour backward
        for n in range(self.__N-2, -1, -1):
            if self.__verbose >= 2:
                print('\r         backward n=', n, ' sur N=', self.__N, end='             ', flush = True)
        
            try:
                ProbaBackward[n].CalcForB(calcB, ProbaBackward[n+1], self.__FS, Tab_GaussXY[n+1])
            except:
                print('ProbaBackward[n+1]=', ProbaBackward[n+1])
                input('pb')

            # normalisation (devijver)
            loinorm.ProductFB(ProbaForwardNorm[n], ProbaBackward[n])
            #loinorm.nextAfterZeros() # on evite des proba de zero
            if loinorm.Integ() == 0.:
                print('loinorm.Integ()=', loinorm.Integ())
                print('ProbaForwardNorm[n]=', ProbaForwardNorm[n])
                print('ProbaBackward[n]=', ProbaBackward[n])
                input('pb loinorm.Integ() == 0.')
            ProbaBackward[n].normalisation(loinorm.Integ())
            #ProbaBackward[n].print()
            # print('sum back=', ProbaBackward[n].Integ())
            #ProbaBackward[n].nextAfterZeros() # on evite des proba de zero
            #print('\n  -->sum backw=', ProbaBackward[n].Integ())
            #print('  -->tab_normalis[n+1]=', tab_normalis[n+1])
            #ProbaBackward[n].normalisation(tab_normalis[n+1])
            #print('  -->sum backw=', ProbaBackward[n].Integ())
            #ProbaBackward[n].plot('$p(r_{n+1} | y_1^{n+1})$')
            # ProbaBackward[n].print()
            # input('backward')

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
            # tab_gamma[n].print()
            # input('pause')

            # normalisation : uniquement due pour compenser des pb liés aux approximations numeriques de forward et de backward
            # Si F= 20, on voit que la normalisation n'est pas necessaire (deja la somme == 1.)
            if abs(1.-tab_gamma[n].Integ()) > 1E-3:
                print('tab_gamma[n].Integ()=', tab_gamma[n].Integ())
                input('PB PB PB Gamma')
                tab_gamma[n].normalisation(tab_gamma[n].Integ())

            # calcul de psi (loi jointe a posteriori) ****************************************************************
            tab_psi.append(Loi2DDiscreteFuzzy_TMC(self.__EPS, self.__STEPS, self.__Rcentres))
            tab_psi[n].CalcPsi(ProbaForward[n], ProbaBackward[n+1], self.__FS, Tab_GaussXY[n+1])
            # normalisation
            integ = tab_psi[n].Integ()
            if integ == 0.:
                print("Tab Spi normalisation")
                tab_psi[n].print()
                input('pause')
            tab_psi[n].normalisation(tab_psi[n].Integ())
            # print('tab_psi[n].Integ()=', tab_psi[n].Integ())
            # input('STOP')

            # calcul de p(rnp1 | rn, z_1^M)  ****************************************************************
            # on créé une liste, pour chaque valeur de rn, de lois 1D
            Liste = []
            for indrn in range(self.__STEPS+2):
                
                if   indrn == 0:              rn=0.
                elif indrn == self.__STEPS+1: rn=1.
                else:                         rn=self.__Rcentres[indrn-1]
                
                Liste.append(Loi1DDiscreteFuzzy_TMC(self.__EPS, self.__STEPS, self.__Rcentres))
                if tab_gamma[n].get(rn) != 0.:
                    Liste[indrn].CalcCond(rn, tab_gamma[n].get(rn), tab_psi[n])
                else:
                    if self.__verbose>1:
                        print('PBPB tab_gamma[n].get(rn) == 0.')
                        input('ATTENTE ERROR')
                    else:
                        # loi uniforme
                        Liste[indrn].setValCste(1.)
                        Liste[indrn].normalisation(Liste[indrn].Integ())
                
            tab_cond.append(Liste)

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

        N   = np.shape(Rlabels)[0]
        # print('N=', N)

        # Remise à 0.
        self.__Mean_Zf.fill(0.)
        self.__Cov_Zf.fill(0.)
        self.__cpt.fill(0)
    
        # The means
        for n in range(N):
            self.__Mean_Zf[Rlabels[n], :] += self.__Ztrain[:, n]
            self.__cpt[Rlabels[n]] += 1
        for q in range(self.__STEPS+2):
            self.__Mean_Zf[q, :] /= self.__cpt[q]
        # print('cpt=', self.__cpt, ', sum=', np.sum(self.__cpt))
        # print(self.__Mean_Zf)
        # input('hard mean')
        
        # The variances
        VectZ = np.zeros(shape=(self.__n_z, 1))
        for n in range(N):
            VectZ = (np.transpose(self.__Ztrain[:, n]) - self.__Mean_Zf[Rlabels[n], :]).reshape(self.__n_z, 1)
            self.__Cov_Zf[Rlabels[n], :, :] += np.dot(VectZ, np.transpose(VectZ))
        for q in range(self.__STEPS+2):
            self.__Cov_Zf[q, :, :] /= self.__cpt[q]
            # check if cov matrix
            if Check_CovMatrix( self.__Cov_Zf[q, :, :]) == False:
                print(self.__Cov_Zf[q, :, :])
                input('This is not a cov matrix!!')

        # print(self.__Cov_Zf)
        # input('hardcov')

    def getMean(self, q):
        return self.__Mean_Zf[q, :]

    def getCov(self, q):
        return self.__Cov_Zf[q, :]

    def getMeanAll(self):
        return self.__Mean_Zf

    def getCovAll(self):
        return self.__Cov_Zf

    def setMeanAll(self, Mean_Zf):
        self.__Mean_Zf = Mean_Zf

    def setCovAll(self, Cov_Zf):
        self.__Cov_Zf = Cov_Zf

