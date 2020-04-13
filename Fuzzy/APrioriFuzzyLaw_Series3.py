#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 11:06:52 2017

@author: MacBook_Derrode
"""

import sys
import random
import scipy.stats as stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

fontS = 13 # fontSize
mpl.rc('xtick', labelsize=fontS)
mpl.rc('ytick', labelsize=fontS)
dpi = 150

if __name__ == '__main__':
    from APrioriFuzzyLaw import LoiAPriori, plotSample
else:
    from Fuzzy.APrioriFuzzyLaw import LoiAPriori, plotSample

def main():

    discretization = 1000
    EPS            = 1E-8
    epsilon        = 1E-2
    verbose        = False
    graphics       = True

    # seed = random.randrange(sys.maxsize)
    # seed = 5039309497922655937
    # rng = random.Random(seed)
    # print("Seed was:", seed)

    # SERIES 3
    print('*********************SERIES 3')
    series = 'Serie3'
    #P, case = LoiAPrioriSeries3(alpha=0.05, delta=0.05, EPS=EPS, discretization=discretization), 1
    P,case = LoiAPrioriSeries3(alpha=0.5, delta=0.1, EPS=EPS, discretization=discretization), 2
    #P,case = LoiAPrioriSeries3(alpha=0.1, delta=0.5, EPS=EPS, discretization=discretization), 3
    #P,case = LoiAPrioriSeries3(alpha=0.05, delta=0.0, EPS=EPS, discretization=discretization), 10

    print(P)
    print('model string', P.stringName())

    # Test le modele
    OKtestModel = P.testModel(verbose=verbose, epsilon=epsilon)

    # Simulation d'un chaine de markov flou suivant ce modèle
    N = 10000
    chain = P.testSimulMC(N, verbose=verbose, epsilon=epsilon)

    if graphics == True:
        P.plotR1R2   ('./figures/LoiCouple_' + series + '_' + str(case) + '.png', dpi=dpi)
        P.plotR1     ('./figures/LoiMarg_'   + series + '_' + str(case) + '.png', dpi=dpi)
        # Dessins
        mini, maxi = 100, 150
        P.PlotMCchain('./figures/Traj_'      + series + '_' + str(case) + '.png', chain, mini=mini, maxi=maxi, dpi=dpi)



########### SERIE 3 #############
#################################
class LoiAPrioriSeries3(LoiAPriori):
    """
        Implementation of the third law described in the report Calcul_Simu_CGOFMSM.pdf
    """

    def __init__(self, alpha, delta, EPS=1E-8, discretization=100):
        """Constructeur to set the parameters of the density."""

        LoiAPriori.__init__(self, EPS=EPS, discretization=discretization)

        self.__alpha = alpha

        assert delta > 0. or delta <= 0.5, print('PB : delta=', delta)
        self.__delta = delta

        if self.__delta != 0.:
            self.__gamma = (1. - 2. * self.__alpha) / (self.__delta * (6. - self.__delta))
        else:
            self.__gamma = 0. # normalement non utilisé
            self.__alpha = 0.5

        self.update()

    def update(self):
        self.__Coeff = self.__alpha + self.__delta * self.__gamma

    def setParametersFromSimul(self, Rsimul, nbcl):
        
        input('setParametersFromSimul : to be done')
        Nsimul = len(Rsimul)

        self.update()
        
    def getParam(self):
        """ Return the params of the law model."""
        return self.__alpha, self.__delta, self.__gamma

    def __str__(self):
        return "alpha=" + str(self.__alpha) + ", delta=" + str(self.__delta) + ", gamma=" + str(self.__gamma)

    def stringName(self):
        return '3:'+str('%.4f'%self.__alpha)+':'+str('%.4f'%self.__delta)

    def probaR1R2(self, r1, r2):
        """ Return the joint proba at r1, r2."""

        if (r1 == 0. and r2 == 0.) or (r1 == 1. and r2 == 1.):
            return self.__alpha

        if (r1 == 1. and r2 == 0.) or (r1 == 0. and r2 == 1.):
            return 0.

        if abs(r1 - r2) <= self.__delta:
            return self.__gamma

        return 0.


    def probaR(self, r):
        """ Return the marginal proba at r."""

        if r == 0. or r == 1.:
            return self.__alpha + self.__delta * self.__gamma

        if (r > 0.) and (r <= self.__delta):
            return self.__gamma * (self.__delta + r + 1)

        if (r >= 1 - self.__delta) and (r < 1.):
            return self.__gamma * (self.__delta + 2. - r)

        return 2. * self.__delta * self.__gamma  # self.A(r)

    def probaR2CondR1(self, r1, r2, verbose=False):
        """ Return the conditional proba at r2 knowing r1."""

        if r1 == 0.0:
            if r2 == 0.:
                return self.__alpha / self.__Coeff
            if r2 > 0 and r2 <= self.__delta:
                return self.__gamma / self.__Coeff
            return 0.

        elif r1 == 1.0:
            if r2 == 1.:
                return self.__alpha / self.__Coeff
            if (r2 >= 1. - self.__delta) and (r2 < 1.):
                return self.__gamma / self.__Coeff
            return 0.

        elif (r1 > 0.) and (r1 <= self.__delta):
            if (r2 >= 0) and (r2 <= r1 + self.__delta):
                return 1. / (self.__delta + r1 + 1.)
            return 0.

        elif (r1 >= 1. - self.__delta) and (r1 < 1.):
            if (r2 >= r1 - self.__delta) and (r2 <= 1):
                return 1. / (2. + self.__delta - r1)
            return 0.

        else:
            if (r2 >= r1 - self.__delta) and (r2 <= r1 + self.__delta):
                return 1. / (2. * self.__delta)
            return 0.

    def tirageR1(self):
        """ Return a draw according to the marginal density p(r1) """

        proba = [self.maxiFuzzyJump(), self.maxiHardJump()]
        typeSample = np.random.choice(a=['flou', 'dur'], size=1, p=proba)[0]
        # print(typeSample)

        if typeSample == 'dur':
            norm = self.probaR(0.0) + self.probaR(1.0)
            probaR1 = [self.probaR(0.0) / norm, self.probaR(1.0) / norm]
            r1 = np.random.choice(a=[0.0, 1.0], size=1, p=probaR1)[0]
        else:
            # Normalement tirage selon la loi par morceaux
            r1 = 0.
            while r1 == 0.:
                #r1 = self.__rv_trapeze.rvs(self.__alpha, self.__beta, self.__gamma, self.__delta_d, self.__delta_u)
                r1 = np.random.random_sample()
        return r1

    def tirageRnp1CondRn(self, rn):
        """ Return a draw according to the conditional density p(r2 | r1) """

        if rn == 0.0:
            proba = [self.__alpha/self.__Coeff, 1.-self.__alpha/self.__Coeff]
            typeSample = np.random.choice(a=['dur', 'flou'], size=1, p=proba)[0]
            if typeSample == 'dur':
                return 0.
            else:
                return echelle(np.random.random_sample(), 0, self.__delta)

        elif rn == 1.0:
            proba = [self.__alpha/self.__Coeff, 1.-self.__alpha/self.__Coeff]
            typeSample = np.random.choice(a=['dur', 'flou'], size=1, p=proba)[0]
            if typeSample == 'dur':
                return 1.
            else:
                return echelle(np.random.random_sample(), 1 - self.__delta, 1.)

        elif rn > 0. and rn <= self.__delta:
            proba = [1./(self.__delta+rn+1), 1.-1./(self.__delta+rn+1)]
            typeSample = np.random.choice(a=['dur', 'flou'], size=1, p=proba)[0]
            if typeSample == 'dur':
                return 0.
            else:
                return echelle(np.random.random_sample(), 0, rn + self.__delta)

        elif rn >= 1. - self.__delta and rn < 1:
            proba = [1./(2.+self.__delta-rn), 1.-1./(2.+self.__delta-rn)]
            typeSample = np.random.choice(a=['dur', 'flou'], size=1, p=proba)[0]
            if typeSample == 'dur':
                return 1.
            else:
                return echelle(np.random.random_sample(), rn - self.__delta, 1.)

        else:
            return echelle(np.random.random_sample(), rn - self.__delta, rn + self.__delta)


if __name__ == '__main__':
    main()
