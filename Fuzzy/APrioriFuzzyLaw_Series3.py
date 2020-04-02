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

fontS = 16 # fontSize
mpl.rc('xtick', labelsize=fontS)
mpl.rc('ytick', labelsize=fontS)
dpi = 300

if __name__ == '__main__':
    from APrioriFuzzyLaw import LoiAPriori, plotSample
else:
    from Fuzzy.APrioriFuzzyLaw import LoiAPriori, plotSample

def main():

    discretization = 200
    EPS            = 1E-10

    seed = random.randrange(sys.maxsize)
    seed = 5039309497922655937
    rng = random.Random(seed)
    print("Seed was:", seed)

    # SERIES 3
    print('*********************SERIES 3')
    series = 'Serie3'
    #P, case = LoiAPrioriSeries3(EPS, discretization, alpha=0.05, delta=0.05), 1
    P,case = LoiAPrioriSeries3(EPS, discretization, alpha=0.5, delta=0.1), 2
    #P,case = LoiAPrioriSeries3(EPS, discretization, alpha=0.1, delta=0.5), 3
    #P,case = LoiAPrioriSeries3(EPS, discretization, alpha=0.05, delta=0.0), 10

    print(P)
    ALPHA, DELTA, GAMMA = P.getParam()
    print('3:'+str(ALPHA)+':'+str(DELTA)+' #pH='+str(P.maxiHardJump()))

# Test de sommes à 1
    sum_R1R2 = P.sumR1R2()
    sum_R1   = P.sumR1()
    sum_R2CondR1_0   = P.sumR2CondR1(0.)
    sum_R2CondR1_20  = P.sumR2CondR1(0.10)
    sum_R2CondR1_50  = P.sumR2CondR1(0.50)
    sum_R2CondR1_90  = P.sumR2CondR1(0.90)
    sum_R2CondR1_100 = P.sumR2CondR1(1.)
    print("sum_R1R2 = ", sum_R1R2)
    print("sum_R1 = ", sum_R1)
    print("sum_R2CondR1_0   = ", sum_R2CondR1_0)
    print("sum_R2CondR1_20  = ", sum_R2CondR1_20)
    print("sum_R2CondR1_50  = ", sum_R2CondR1_50)
    print("sum_R2CondR1_90  = ", sum_R2CondR1_90)
    print("sum_R2CondR1_100 = ", sum_R2CondR1_100)
    print('maxiHardJump = ', P.maxiHardJump())
    
    # Calcul théorique et empirique de la proportion de suats durs
    # MProbaTh, TProbaTh, JProbaTh = P.getTheoriticalHardTransition(2)
    # print('JProba Hard Theorique=\n', JProbaTh)
    # print('sum=', sum(sum(JProbaTh)))

    MProbaNum, TProbaNum, JProbaNum = P.getNumericalHardTransition(2)
    print('Jproba Hard Numerique, J=\n', JProbaNum)
    print('sum=', sum(sum(JProbaNum)))

    # Simulation d'un chaine de markov flou suivant ce modèle
    N = 10000
    chain = np.zeros(shape=(N))
    # Le premier
    chain[0] = P.tirageR1()
    # les suivantes...
    for i in range(1, N):
        chain[i] = P.tirageRnp1CondRn(chain[i-1])

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
    print('Jproba Hard Echantillon, J=\n', JProbaEch)
    print('sum=', sum(sum(JProbaEch)))

    cpt0 = 0
    cpt1 = 0
    for i in range(N):
        if chain[i] == 0.:
            cpt0 += 1
        elif chain[i] == 1.0:
            cpt1 += 1
    print('Nbre saut 0 :', cpt0/N, ', Theorique :', P.probaR(0.))
    print('Nbre saut 1 :', cpt1/N, ', Theorique :', P.probaR(1.))
    print('Nbre saut durs (0+1) :', (cpt0+cpt1)/N, ', Theorique :', P.maxiHardJump())

    mini = 100
    maxi = 150
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    P.plotR1R2('./figures/LoiCouple_' + series + '_' + str(case) + '.png', ax, dpi=dpi)
    P.plotR1('./figures/LoiMarg_' + series + '_' + str(case) + '.png', dpi=dpi)
    FIG = plt.figure()
    AX = FIG.gca()
    abscisse= np.linspace(start=mini, stop=maxi, num=maxi-mini)
    AX.plot(abscisse, chain[mini:maxi], 'g')
    #plt.title('Trajectory (Fuzzy jumps)')
    AX.set_xlabel('$n$', fontsize=fontS)
    AX.set_ylim(0., 1.05)
    plt.savefig('./figures/Traj_' + series + '_' + str(case) + '.png', bbox_inches='tight', dpi=dpi)



########### SERIE 3 #############
#################################
class LoiAPrioriSeries3(LoiAPriori):
    """
        Implementation of the third law described in the report Calcul_Simu_CGOFMSM.pdf
    """

    def __init__(self, EPS, discretization, alpha, delta):
        """Constructeur to set the parameters of the density."""

        LoiAPriori.__init__(self, EPS, discretization)

        self.__alpha = alpha

        assert delta > 0. or delta <= 0.5, print('PB : delta=', delta)
        self.__delta = delta

        if self.__delta != 0.:
            self.__gamma = (1. - 2. * self.__alpha) / (self.__delta * (6. - self.__delta))
        else:
            self.__gamma = 0. # normalement non utilisé
            self.__alpha = 0.5
        self.__Coeff = self.__alpha + self.__delta * self.__gamma

    def getParam(self):
        """ Return the params of the law model."""
        return self.__alpha, self.__delta, self.__gamma

    def __str__(self):
        return "alpha=" + str(self.__alpha) + ", delta=" + str(self.__delta) + ", gamma=" + str(self.__gamma)

    def stringName(self):
        return '3:'+str(self.__alpha)+':'+str(self.__delta)

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
