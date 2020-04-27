#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 11:06:52 2017

@author: MacBook_Derrode
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib.lines import Line2D


from Fuzzy.LoisDiscreteFuzzy import Loi2DDiscreteFuzzy, Loi1DDiscreteFuzzy
#from LoisDiscreteFuzzy import Loi2DDiscreteFuzzy, Loi1DDiscreteFuzzy


fontS = 13 # fontSize
mpl.rc('xtick', labelsize=fontS)
mpl.rc('ytick', labelsize=fontS)

def echelle(val, a, b):
    """
    To stretch values between a and b
    """
    #assert a <= b, print('PB dans echelle car a>b!')
    # value = val * (b - a) + a
    # if (value<a or value>=b):
    #     print("a=", a ,", value=", value, ", b=", b)
    #     input('fonction echelle')
    return val * (b - a) + a

def plotSample(rv, nbsample, filename, dpi=150):
    """
    Plot the pdf and cdf of rv, together with the hist of a sample of \
    nb sample drawed from rv, to check the algorihtm implementation.
    """

    sample = []
    for n in range(nbsample):
        sample.append(rv.rvs(1))
    hist, bin_edges = np.histogram(sample, bins=30, density=True)
    R = np.linspace(start=1E-10, stop=1.0 - 1E-10, num=100, endpoint=True)
    #print(hist, bin_edges)
    # print(np.sum(hist*np.diff(bin_edges)))
    fig = plt.figure()
    ax = fig.gca()
    pdf, = ax.plot(R, rv.pdf(R), alpha=0.6, color='g', label='pdf')
    cdf, = ax.plot(R, rv.cdf(R), alpha=0.6, color='b', label='cdf')
    hist, = ax.plot(bin_edges[0:-1], hist, alpha=0.6, color='r', label='hist' + str(nbsample) + ' samples')
    plt.legend(handles=[pdf, cdf, hist])
    ax.set_xlabel('$r$')
    ax.set_xlim(0, 1)
    # plt.show()
    plt.savefig(filename, bbox_inches='tight', dpi=dpi)
    plt.close()

    return np.mean(sample), np.var(sample)


class LoiAPriori:
    """
    Super class to factorize code for APrioriFuzzyLaw_Seriesx.py series of classes.
    """

    def __init__(self, EPS, discretization):

        self._EPS            = EPS
        self.__discretization = discretization

        if self.__discretization != 0:
            self.__Rcentres = np.linspace(start=1./(2.*self.__discretization), stop=1.0-1./(2.*self.__discretization), num=self.__discretization, endpoint=True)
        else:
            self.__Rcentres = np.empty(shape=(0,))

        self.__uneLoi1D = Loi1DDiscreteFuzzy(self._EPS, self.__discretization, self.__Rcentres)
        self.__uneLoi2D = Loi2DDiscreteFuzzy(self._EPS, self.__discretization, self.__Rcentres)

    def maxiFuzzyJump(self):
        """
        return the percent of fuzzy jump for the joint a priori law
        """
        return 1.0 - self.maxiHardJump()

    def maxiHardJump(self):
        """
        return the percent of hard jump for the joint a priori law
        """
        return self.probaR(0.) + self.probaR(1.)

    def getMTProbaFormJProba(self, JProba, n_r):
        MProba = np.sum(JProba, axis= 1).T
        TProba = np.zeros(shape=(n_r, n_r))
        for r in range(n_r):
            TProba[r, :] = JProba[r, :] / MProba[r]
        return MProba, TProba


    def __filInLoi2D(self):

        # Pour les masses
        self.__uneLoi2D.setr(0., 0., self.probaR1R2(0., 0.))
        self.__uneLoi2D.setr(0., 1., self.probaR1R2(0., 1.))
        self.__uneLoi2D.setr(1., 0., self.probaR1R2(1., 0.))
        self.__uneLoi2D.setr(1., 1., self.probaR1R2(1., 1.))

         # Pour les arrètes et le coeur
        for r1 in self.__Rcentres:
            self.__uneLoi2D.setr(0., r1, self.probaR1R2(0., r1))
            self.__uneLoi2D.setr(r1, 0., self.probaR1R2(r1, 1.))
            self.__uneLoi2D.setr(1., r1, self.probaR1R2(1., r1))
            self.__uneLoi2D.setr(r1, 1., self.probaR1R2(r1, 1.))

            for r2 in self.__Rcentres:
                self.__uneLoi2D.setr(r1, r2, self.probaR1R2(r1, r2))

    def sumR1R2(self):
        """
        Integration of density p(r1,R2), should sum one
        """
        self.__filInLoi2D()
        return self.__uneLoi2D.Integ()

    def getNumericalHardTransition(self, n_r):
        """
        Integration of density p(r1,R2), should sum one
        """

        if n_r != 2:
            input('ProbaHard : n_r != 2 - IMP0SSIBLE')

        # on rempli la loi 2D
        self.__filInLoi2D()

        JProba = np.zeros(shape=(n_r, n_r), dtype=float)
        JProba[0,0] = self.__uneLoi2D.partialInteg(0.0, 0.5, 0.0, 0.5)
        JProba[1,0] = self.__uneLoi2D.partialInteg(0.5, 1.0, 0.0, 0.5)
        JProba[0,1] = self.__uneLoi2D.partialInteg(0.0, 0.5, 0.5, 1. )
        JProba[1,1] = self.__uneLoi2D.partialInteg(0.5, 1.0, 0.5, 1. )
        
        somme = np.sum(sum(JProba))
        if abs(1.0 - somme)> 1E-2:
            print('JProba=', JProba)
            print('sum=', somme)
            input('PROBLEM getNumericalHardTransition')
        JProba /= somme

        MProba, TProba = self.getMTProbaFormJProba(JProba, n_r)

        return MProba, TProba, JProba

    def sumR1(self):
        """
        Integration of density p(r1), should sum to one
        """

        self.__uneLoi1D.setindr(0,                       self.probaR(0.))
        self.__uneLoi1D.setindr(self.__discretization+1, self.probaR(1.))
        for i, r in enumerate(self.__Rcentres):
            self.__uneLoi1D.setindr(i+1, self.probaR(r))

        integ = self.__uneLoi1D.Integ()
        return integ


    def sumR2CondR1(self, r1):
        """
        Integration of density p(r2 | r1), should sum to one
        """

        self.__uneLoi1D.setindr(0,                       self.probaR2CondR1 (r1, 0.))
        self.__uneLoi1D.setindr(self.__discretization+1, self.probaR2CondR1 (r1, 1.))
        for i, r in enumerate(self.__Rcentres):
            self.__uneLoi1D.setindr(i+1, self.probaR2CondR1 (r1, r))

        integ = self.__uneLoi1D.Integ()
        return integ


    def testModel(self, verbose=False, epsilon=1E-2):

        print('THE MODEL')
        print('  -->', self)
        # ALPHA, BETA, ETA, DELTA = self.getParam()
        # print('2:'+str(ALPHA)+':'+str(ETA)+':'+str(DELTA)+' #pH='+str(self.maxiHardJump()))
        print('  --> model string', self.stringName())
        
        # Test de sommes à 1
        OKSumtoOne = self.testSum_to_one(verbose=verbose, epsilon=epsilon)
        if OKSumtoOne==True:
            print('SUM to 1: PASSED')
        else:
            print('SUM to 1: FAIL')

        # Calcul théorique et empirique de la proportion de sauts durs
        #print('THE (THEORETICAL AND) NUMERICAL MODEL')
        MethodExists = getattr(self, "getTheoriticalHardTransition", False)
        if MethodExists != False:
            MProbaTh, TProbaTh, JProbaTh = self.getTheoriticalHardTransition(2)
            if verbose == True:
                sumth = sum(sum(JProbaTh))
                print('Joint proba Hard (theoretical), sum=', sumth, ', J=\n', JProbaTh)
                if abs(1.-sumth)>epsilon :
                    print('JProbaTh does not sum to 1 : ', sumth)

        MProbaNum, TProbaNum, JProbaNum = self.getNumericalHardTransition(2)
        if verbose == True:
            sumnum = sum(sum(JProbaNum))
            print('Joint proba Hard (numerical), sum=', sumnum, ', J=\n', JProbaNum)
            if abs(1.-sumnum)>epsilon :
                print('JProbaNum does not sum to 1 : ', sumnum)

        OKJointMat = True
        if MethodExists != False:
            OKJointMat = np.allclose(JProbaNum, JProbaTh, rtol=epsilon, atol=epsilon)
            if OKJointMat == False:
                print('Joint matrices are all close: PASSED')
            else:
                print('Joint matrices are all close: FAILED')

        return OKSumtoOne and OKJointMat


    def testSimulMC(self, N=10000, verbose=False, epsilon=1E-2):

        chain = np.zeros(shape=(N))
        chain[0] = self.tirageR1()
        for i in range(1, N):
            chain[i] = self.tirageRnp1CondRn(chain[i-1])

        # Comptage des quarts
        JProbaEch = np.zeros(shape=(2,2))
        for i in range(N-1):
            if chain[i]<0.5:
                if chain[i+1]<0.5:
                    JProbaEch[0,0] += 1.
                else:
                    JProbaEch[0,1] += 1.
            else:
                if chain[i+1]<0.5:
                    JProbaEch[1,0] += 1.
                else:
                    JProbaEch[1,1] += 1.
        JProbaEch /= (N-1.)
        if verbose == True:
            sumech = sum(sum(JProbaEch))
            print('Joint proba Hard (MC simul), sum=', sumech, ', J=\n', JProbaEch)
        
        cpt0 = 0
        cpt1 = 0
        for i in range(N):
            if chain[i] == 0.:
                cpt0 += 1
            elif chain[i] == 1.0:
                cpt1 += 1

        if verbose == True:
            print('  Nbre saut 0 :', cpt0/N, ', Theorique :', self.probaR(0.))
            print('  Nbre saut 1 :', cpt1/N, ', Theorique :', self.probaR(1.))
            print('  Nbre saut durs (0+1) :', (cpt0+cpt1)/N, ', Theorique :', self.maxiHardJump())

        if abs((cpt0+cpt1)/N - self.maxiHardJump())<=epsilon:
            print('MC simul: PASSED')
        else:
            print('MC simul: FAILED')

        return chain


    def testSum_to_one(self, verbose=False, epsilon=1E-2):

        OKSumtoOne = True
        tabsum = np.zeros(shape=(7))
        tabsum[0] = self.sumR1R2()
        tabsum[1] = self.sumR1()
        tabsum[2] = self.sumR2CondR1(0.)
        tabsum[3] = self.sumR2CondR1(0.10)
        tabsum[4] = self.sumR2CondR1(0.50)
        tabsum[5] = self.sumR2CondR1(0.90)
        tabsum[6] = self.sumR2CondR1(1.)
        
        for somme in tabsum:
            if abs(1.-somme)>epsilon: OKSumtoOne = False
    
        if verbose == True or OKSumtoOne == False:
            print("sum_R1R2         = ", tabsum[0])
            print("sum_R1           = ", tabsum[1])
            print("sum_R2CondR1_0   = ", tabsum[2])
            print("sum_R2CondR1_20  = ", tabsum[3])
            print("sum_R2CondR1_50  = ", tabsum[4])
            print("sum_R2CondR1_90  = ", tabsum[5])
            print("sum_R2CondR1_100 = ", tabsum[6])

        return OKSumtoOne


    def plotR1R2(self, filename, dpi=150):
        """
        Plot of the joint density p(r1, r2)
        """

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')

        #mpl.style.use('seaborn') # les couleurs
        pR1R2 = np.ndarray(shape=(self.__discretization, self.__discretization))
        R1Grid1, R2Grid1 = np.meshgrid(self.__Rcentres, self.__Rcentres)

        ############# Dessin de la surface centrale
        for i, r1 in enumerate(self.__Rcentres):
            for j, r2 in enumerate(self.__Rcentres):
                pR1R2[i, j] = self.probaR1R2(r1, r2)
        ax.plot_surface(R1Grid1, R2Grid1, pR1R2, alpha=0.6, color='xkcd:green')

        maxim = np.amax(pR1R2)

        ############# Dessin des bords
        # for i, r1 in enumerate(self.__Rcentres):
        #     pR1R2[i, 0] = self.probaR1R2(r1, 0.)
        # for i, r1 in enumerate(self.__Rcentres):
        #     pR1R2[i, self.__discretization-1] = self.probaR1R2(r1, 1.)
        # for j, r2 in enumerate(self.__Rcentres):
        #     pR1R2[0, j] = self.probaR1R2(0., r2)
        # for j, r2 in enumerate(self.__Rcentres):
        #     pR1R2[self.__discretization-1, j] = self.probaR1R2(1., r2)
        #ax.plot_surface(R1Grid1, R2Grid1, pR1R2, alpha=0.9, color='xkcd:lavender'

        maxim = max(maxim, np.amax(pR1R2))

        ############# Dessin des masses
        x, y, z=[0., 0.], [0., 0.], [0., self.probaR1R2(0., 0.)]
        ax.plot(x,y,z,'k', alpha=0.8, linewidth=3.0)
        x, y, z=[0., 0.], [1., 1.], [0., self.probaR1R2(0., 1.)]
        ax.plot(x,y,z,'k', alpha=0.8, linewidth=3.0)
        x, y, z=[1., 1.], [0., 0.], [0., self.probaR1R2(1., 0.)]
        ax.plot(x,y,z,'k', alpha=0.8, linewidth=3.0)
        x, y, z=[1., 1.], [1., 1.], [0., self.probaR1R2(1., 1.)]
        ax.plot(x,y,z,'k', alpha=0.8, linewidth=3.0)
        maxim = max(maxim, self.probaR1R2(0., 0.))
        maxim = max(maxim, self.probaR1R2(0., 1.))
        maxim = max(maxim, self.probaR1R2(1., 0.))
        maxim = max(maxim, self.probaR1R2(1., 1.))
        #print(maxim)

        ############# Dessin des axes
        ax.set_xlabel('$r_1$', fontsize=16)
        #ax.set_xlim(-0.02, 1.02)
        ax.set_ylabel('$r_2$', fontsize=16)
        #ax.set_ylim(-0.02, 1.02)
        #ax.set_zlabel('$p(r_1,r_2)$', fontsize=fontS)
        #ax.set_zlim(0., maxim*1.02)
        ax.view_init(25, 238)

        # plt.show()
        if filename != None:
            plt.savefig(filename, bbox_inches='tight', dpi=dpi)
        plt.close()


    def plotR1(self, filename, dpi):
        """
        Plot of the marginal density p(r)
        """

        proba = np.zeros(shape=(self.__discretization))
        for i, r in enumerate(self.__Rcentres):
            proba[i] = self.probaR(r)

        fig, ax = plt.subplots()
        ax.plot(self.__Rcentres, proba, alpha=0.6, color='g', linewidth=2)
        ax.set_xlabel('$r$', fontsize=fontS)
        #ax.set_xlim(0, 1)
        #ax.set_ylabel('$p(r)$', fontsize=fontS)
        #ax.set_ylim(0., max(pR)*1.05)
        # plt.show()

        line1 = Line2D([0., 0.], [0., self.probaR(0.) ], linewidth=4)
        line2 = Line2D([1., 1.], [0., self.probaR(1.) ], linewidth=4)
        ax.add_line(line1)
        ax.add_line(line2)
        plt.savefig(filename, bbox_inches='tight', dpi=dpi)
        plt.close()


    def PlotMCchain(self, filename, chain, mini, maxi, dpi):
        
        fig, ax = plt.subplots()
        abscisse= np.linspace(start=mini, stop=maxi, num=maxi-mini)
        ax.plot(abscisse, chain[mini:maxi], 'g')
        #plt.title('Trajectory (Fuzzy jumps)')
        ax.set_xlabel('$n$', fontsize=fontS)
        ax.set_ylim(0., 1.05)
        plt.savefig(filename, bbox_inches='tight', dpi=dpi)
        plt.close()
