#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 11:06:52 2017

@author: MacBook_Derrode
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

fontS = 16 # fontSize
matplotlib.rc('xtick', labelsize=fontS)
matplotlib.rc('ytick', labelsize=fontS)
dpi=150

from Fuzzy.APrioriFuzzyLaw import LoiAPriori, echelle
#from APrioriFuzzyLaw import LoiAPriori, echelle


########### SERIE 1 ##################
######################################
class LoiAPrioriSeries1(LoiAPriori):
    """
    Implementation of the first law described in the report Calcul_Simu_CGOFMSM.pdf.
    """

    def __init__(self, alpha, gamma):
        """Constructeur to set the parameters of the density."""

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

        self.__probaFH = [self.maxiFuzzyJump(), self.maxiHardJump()]

    def getParam(self):
        """ Return the params of the law model."""
        return self.__alpha, self.__beta, self.__gamma

    def __str__(self):
        return "alpha=" + str(self.__alpha) + ", beta=" + str(self.__beta) + \
                    ", gamma=" + str(self.__gamma)

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


    discretization = 40
    # np.random.seed(0)
    np.random.seed(None)

    # SERIES 1
    print('*********************SERIES 1')
    series = 'Serie1'
    #P, case, = LoiAPrioriSeries1(alpha=0.2, gamma=0.1), 1
    #P,case = LoiAPrioriSeries1(alpha=0.3, gamma=0.0), 2
    P,case = LoiAPrioriSeries1(alpha=0.0, gamma=0.2), 3

    print(P)
    sum_R1R2 = P.sumR1R2(discretization)
    print("sum_R1R2 = ", sum_R1R2)
    sum_R1 = P.sumR1(discretization)
    print("sum_R1 = ", sum_R1)
    print('maxiHardJump = ', P.maxiHardJump())

    N = 10000
    chain = np.zeros((1, N))
    chain[0] = P.tirageR1()
    for i in range(1, N):
        chain[0, i] = P.tirageRnp1CondRn(chain[0, i - 1])

    cpt0 = 0
    cpt1 = 0
    for i in range(N):
        if chain[0, i] == 0.:
            cpt0 += 1
        elif chain[0, i] == 1.0:
            cpt1 += 1
    print('Nbre saut 0 :', cpt0/N, ', Theorique :', P.probaR(0.))
    print('Nbre saut 1 :', cpt1/N, ', Theorique :', P.probaR(1.))
    print('Nbre saut durs (0+1) :', (cpt0+cpt1)/N, ', Theorique :', P.maxiHardJump())

    mini = 100
    maxi = 250
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    P.plotR1R2(discretization, 'LoiCouple_' + series + '_' + str(case) + '.png', ax, dpi=dpi)
    P.plotR1(discretization, 'LoiMarg_' + series + '_' + str(case) + '.png')
    FIG = plt.figure()
    AX = FIG.gca()
    abscisse= np.linspace(start=mini, stop=maxi, num=maxi-mini)
    AX.plot(abscisse, chain[0, mini:maxi], 'g')
    #plt.title('Trajectory (Fuzzy jumps)')
    AX.set_xlabel('$n$', fontsize=fontS)
    AX.set_ylim(0., 1.05)
    plt.savefig('Traj_' + series + '_' + str(case) + '.png',bbox_inches='tight', dpi=dpi)
