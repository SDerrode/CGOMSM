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

    # SERIES 1
    print('*********************SERIES 1')
    series = 'Serie1'
    P, case, = LoiAPrioriSeries1(alpha=0.2, gamma=0.1, EPS=EPS, discretization=discretization), 1
    # P,case = LoiAPrioriSeries1(alpha=0.3, gamma=0.0, EPS=EPS, discretization=discretization), 2
    # P,case = LoiAPrioriSeries1(alpha=0.0, gamma=0.2, EPS=EPS, discretization=discretization), 3
    
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


########### SERIE 1 ##################
######################################
class LoiAPrioriSeries1(LoiAPriori):
    """
    Implementation of the first law described in the report Calcul_Simu_CGOFMSM.pdf.
    """

    def __init__(self, alpha, gamma, EPS=1E-8, discretization=100):
        """Constructeur to set the parameters of the density."""

        LoiAPriori.__init__(self, EPS=EPS, discretization=discretization)

        assert alpha <= (1. - 3. * gamma) / 2., \
                    print('PB, you should set alpha to a maximum value of ', \
                    (1. - 5. * gamma) / 2., ' which corresponds to beta = 0')

        self.__alpha = alpha
        self.__gamma = gamma
        self.__beta = (1. - 5 * self.__gamma) / 2. - self.__alpha

        temp = 2. * self.__alpha + 2. * self.__beta + 5. * self.__gamma
        assert temp == 1.0, print('Pb : this value should be 1 : ', temp)
        #print(' 2 alpha + 2 beta = ', 2.*self.__alpha+2.*self.__beta)
        #print('beta = ', self.__beta)

        assert self.__alpha >= 0. and self.__alpha <= 1., \
            print('PB : alpha=', self.__alpha)
        assert self.__beta >= 0. and self.__beta <= 1., \
            print('PB : beta =', self.__beta)
        assert self.__gamma >= 0. and self.__gamma <= 1., \
            print('PB : gamma=', self.__gamma)

        self.update()


    def update(self):
        # rien à faire ici
        temp=0.


    def setParametersFromSimul(self, Rsimul, nbcl):
        
        input('setParametersFromSimul : to be done')
        Nsimul = len(Rsimul)

        alpha, gamma = 0., 0.
        # for n in range(1, Nsimul):
        #     if Rsimul[n-1] == 0      and Rsimul[n] == 0:      alpha0 += 1.
        #     if Rsimul[n-1] == nbcl-1 and Rsimul[n] == nbcl-1: alpha1 += 1.
        #     if Rsimul[n-1] == 0      and Rsimul[n] == nbcl-1: beta   += 1.
        #     if Rsimul[n-1] == nbcl-1 and Rsimul[n] == 0:      beta   += 1.
        
        self.__alpha = alpha / (Nsimul-1.)
        self.__gamma = gamma / (Nsimul-1.)

        self.__beta = (1. - 5 * self.__gamma) / 2. - self.__alpha

        self.update()


    def getParam(self):
        """ Return the params of the law model."""
        return self.__alpha, self.__beta, self.__gamma

    def __str__(self):
        return "alpha=" + str(self.__alpha) + ", beta=" + str(self.__beta) + \
                    ", gamma=" + str(self.__gamma)

    def stringName(self):
        return '1:'+str('%.4f'%self.__alpha)+':'+str('%.4f'%self.__gamma)

    def probaR1R2(self, r1, r2):
        """ Return the joint proba at r1, r2."""

        if (r1 == 0. and r2 == 0.) or (r1 == 1. and r2 == 1.):
            return self.__alpha

        if (r1 == 1. and r2 == 0.) or (r1 == 0. and r2 == 1.):
            return self.__beta

        return self.__gamma

    def probaR(self, r):
        """ Return the marginal proba at r."""
        if r == 0. or r == 1.:
            return self.__alpha + self.__beta + self.__gamma

        return 3 * self.__gamma

    def probaR2CondR1(self, r1, r2, verbose=False):
        """ Return the conditional proba at r2 knowing r1."""

        temp = self.__alpha + self.__beta + self.__gamma

        if r1 == 0.0:
            if r2 == 0.0:
                return self.__alpha / temp
            elif r2 == 1.0:
                return self.__beta / temp

            return self.__gamma / temp

        elif r1 == 1.0:
            if r2 == 0.0:
                return self.__beta / temp
            elif r2 == 1.0:
                return self.__alpha / temp

            return self.__gamma / temp

        return 1.0 / 3.0

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
            r1 = 0.
            while r1 == 0.:
                r1 = np.random.random_sample()
        return r1

    def tirageRnp1CondRn(self, rn):
        """ Return a draw according to the conditional density p(r2 | r1) """

        if rn == 0. or rn == 1.:
            norm = self.__alpha + self.__beta + self.__gamma
            proba = [(self.__alpha + self.__beta)/norm, self.__gamma/norm]
        else:
            proba = [2./3., 1./3.]
        typeSample = np.random.choice(a=['dur', 'flou'], size=1, p=proba)[0]
        # print('rn=', rn, ', proba=', proba)
        # print(typeSample)
        # input('attente')

        if typeSample == 'dur':
            if self.probaR2CondR1(rn, 0.0) == 0. and self.probaR2CondR1(rn, 1.0) == 0.:
                probaR2CondR1 = [0.5, 0.5]
            else:
                norm = self.probaR2CondR1(rn, 0.0) + self.probaR2CondR1(rn, 1.0)
                probaR2CondR1 = [self.probaR2CondR1(rn, 0.0) / norm, self.probaR2CondR1(rn, 1.0) / norm]
            rnp1 = np.random.choice(a=[0.0, 1.0], size=1, p=probaR2CondR1)[0]
        else:
            rnp1 = 0.
            while rnp1 == 0.:
                rnp1 = np.random.random_sample()

        return rnp1


if __name__ == '__main__':
    main()
