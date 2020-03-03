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

from Fuzzy.APrioriFuzzyLaw import LoiAPriori, plotSample
#from APrioriFuzzyLaw import LoiAPriori, plotSample


def main():

    discretization = 200
    EPS            = 1E-10

    seed = random.randrange(sys.maxsize)
    seed = 5039309497922655937
    rng = random.Random(seed)
    print("Seed was:", seed)

    # SERIES 2 bis
    print('*********************SERIES 2 bis')
    series = 'Serie2bis'
    #P, case = LoiAPrioriSeries2bis(EPS, discretization, alpha=0.07, eta=0.21, delta=0.05, lamb=0.), 3
    #P, case = LoiAPrioriSeries2bis(EPS, discretization, alpha=0.07, eta=0.21, delta=0.05, lamb=1.), 2
    P, case = LoiAPrioriSeries2bis(EPS, discretization, alpha=0.07, eta=0.21, delta=0.10, lamb=0.3), 1
    #P, case = LoiAPrioriSeries2bis(EPS, discretization, alpha=0.12, eta=0., delta=0., lamb=0.3), 5
    #P, case = LoiAPrioriSeries2bis(EPS, discretization, alpha=0.1, eta=0.3, delta=0.3, lamb=0.5), 6

    # Le cas suivant assure que alpha=beta=0
    # eta = 0.21
    # lamb= 0.6
    # delta = 6.*(0.5 - eta*(lamb+1./3.))
    # P, case = LoiAPrioriSeries2bis(EPS, discretization, alpha=0, eta=eta, delta=delta, lamb=lamb), 4

    print(P)
    ALPHA, BETA, ETA, DELTA, LAMBDA = P.getParam()
    print('2bis:'+str(ALPHA)+':'+str(ETA)+':'+str(DELTA)+':'+str(LAMBDA)+' - Beta='+ str(BETA) +', #pH='+str(P.maxiHardJump()))

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

    # if not (ETA ==0. and DELTA==0.) :
    #     # Dessin de la pente a partir de la quelle on doit faire des tirages
    #     pente = pente_Serie2bis_gen(momtype=0, name='pente_Serie2bis', a=0., b=1.)
    #     rv = pente()
    #     #print(pente.pdf(0.54))
    #     mean, var = plotSample(rv, 10000, 'pente_' + series + '_'+str(case)+'.png')
    #     print('mean echantillon = ', mean)
    #     print('var echantillon = ', var)
    #     print(rv.stats('mvsk'))

    #     # Dessin de la pente a partir de la quelle on doit faire des tirages
    #     pente2 = pente2_Serie2bis_gen(momtype=0, name='pente2_Serie2bis', a=0., b=1.)
    #     rv = pente2()
    #     #print(pente2.pdf(0.54))
    #     mean, var = plotSample(rv, 10000, 'pente2_' + series + '_'+str(case)+'.png')
    #     print('mean echantillon = ', mean)
    #     print('var echantillon = ', var)
    #     print(rv.stats('mvsk'))

    #     # Dessin de la pente a partir de laquelle on doit faire des tirages
    #     parab = parabole_Serie2bis_gen(momtype=0, name='parabole_Serie2bis', a=0., b=1., shapes="ETA, DELTA, LAMB")
    #     rv = parab(ETA, DELTA, LAMBDA)
    #     #print(parab.pdf(0.54, ETA, DELTA))
    #     mean, var = plotSample(rv, 10000, 'parab_' + series + '_'+str(case)+'.png')
    #     print('mean echantillon = ', mean)
    #     print('var echantillon = ', var)
    #     print(rv.stats('mvsk'))

    #     triangle = triangle_Serie2bis_gen(momtype=0, name='triangle_Serie2bis', a=0., b=1., shapes="ETA, DELTA, r1")
    #     rv = triangle(ETA, DELTA, 0.3)
    #     #print(triangle.pdf(0.54, ETA, DELTA, 0.3))
    #     mean, var = plotSample(rv, 10000, 'triangle_' + series + '_'+str(case)+'.png')
    #     print('mean echantillon = ', mean)
    #     print('var echantillon = ', var)
    #     print(rv.stats('mvsk'))


########### SERIE 2 bis ##############
######################################
class LoiAPrioriSeries2bis(LoiAPriori):
    """
    Implementation of the second law described in the report Calcul_Simu_CGOFMSM.pdf
    """

    def __init__(self, EPS, discretization, alpha, eta, delta, lamb):
        """Constructeur to set the parameters of the density."""

        LoiAPriori.__init__(self, EPS, discretization)

        self.__alpha = alpha
        self.__eta = eta
        self.__delta = delta
        self.__lamb = lamb
        self.__beta = 0.5 - delta/6. - eta*(lamb+1./3.) - self.__alpha
        #print('test ph=', 2.*(self.__alpha + self.__beta) + self.__lamb * self.__eta)
        #print('test pf=', 1./3.*(self.__delta-self.__eta) + self.__eta *(self.__lamb +1.))

        self.__D1 = self.__alpha + self.__beta + self.__lamb * self.__eta / 2.

        # pour les tirages aléatoires
        pente_01 = pente_Serie2bis_gen(momtype=0, name='pente1_Serie2bis', a=0., b=1.)
        self.__rv_pente0 = pente_01()
        pente2_01 = pente2_Serie2bis_gen(momtype=0, name='pente2_Serie2bis', a=0., b=1.)
        self.__rv_pente1 = pente2_01()

        parab = parabole_Serie2bis_gen(momtype=0, name='parabole_Serie2bis', a=0., b=1., shapes="eta, delta, lamb")
        self.__rv_parab = parab(self.__eta, self.__delta, self.__lamb)

        self.__rv_triangle = triangle_Serie2bis_gen(momtype=0, name='triangle_Serie2bis', a=0., b=1., shapes="eta, delta, r1")

    def getParam(self):
        """ Return the params of the law model."""
        return self.__alpha, self.__beta, self.__eta, self.__delta, self.__lamb

    def __str__(self):
        return "alpha=" + str(self.__alpha) + ", beta=" + str(self.__beta) + ", eta=" + str(self.__eta) + ", delta=" + str(self.__delta) + ", lambda=" + str(self.__lamb)

    def stringName(self):
        return '2bis:'+str(self.__alpha)+':'+str(self.__beta)+':'+str(self.__eta)+':'+str(self.__lamb)

    def probaR1R2(self, r1, r2):
        """ Return the joint proba at r1, r2."""

        if (r1 == 0. and r2 == 0.) or (r1 == 1. and r2 == 1.):
            return self.__alpha

        if (r1 == 1. and r2 == 0.) or (r1 == 0. and r2 == 1.):
            return self.__beta

        if r2 == 0.:
            return self.__lamb * self.__eta*(1.-r1)
        if r2 == 1.:
            return self.__lamb * self.__eta*r1
        if r1 == 0.:
            return self.__lamb * self.__eta*(1.-r2)
        if r1 == 1.:
            return self.__lamb * self.__eta*r2

        return self.__eta + (self.__delta - self.__eta) * np.abs(r1 - r2)

    def probaR(self, r):
        """ Return the marginal proba at r."""

        if r == 0. or r == 1.:
            return self.__alpha + self.__beta + self.__eta * self.__lamb / 2.

        return self.__eta * (self.__lamb +1.) + (self.__delta - self.__eta) * (0.5 + r * r - r)

    def probaR2CondR1(self, r1, r2, verbose=False):
        """ Return the conditional proba at r2 knowing r1."""

        if r1 == 0.0:
            if r2 == 0.0:
                return self.__alpha / self.__D1
            elif r2 == 1.0:
                return self.__beta / self.__D1
            else:
                return (self.__lamb * self.__eta * (1. - r2)) / self.__D1

        elif r1 == 1.0:
            if r2 == 0.0:
                return self.__beta / self.__D1
            elif r2 == 1.0:
                return self.__alpha / self.__D1
            else:
                return (self.__lamb * self.__eta * r2) / self.__D1

        else:
            D2 = self.__eta * (self.__lamb + 1) + (self.__delta - self.__eta) * (0.5 + r1 * r1 - r1)
            if r2 == 0.0:
                return (self.__lamb * self.__eta * (1. - r1)) / D2
            elif r2 == 1.0:
                return (self.__lamb * self.__eta * r1) / D2
            else:
                return (self.__eta + (self.__delta - self.__eta) * np.abs(r1 - r2)) / D2

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
            r1 = self.__rv_parab.rvs()
        return r1

    def tirageRnp1CondRn(self, rn):
        """ Return a draw according to the conditional density p(r2 | r1) """

        if self.__eta == 0. and self.__delta == 0.:
            # cas particulier dune chaîne de markov dure à 2 états
            norm = self.__alpha + self.__beta
            if rn == 0.:
                probaR2CondR1 = [self.__alpha / norm, self.__beta / norm]
            else:
                probaR2CondR1 = [self.__beta / norm, self.__alpha / norm]
            return np.random.choice(a=[0.0, 1.0], size=1, p=probaR2CondR1)[0]
        else:

            D2 = self.__eta * (self.__lamb + 1) + (self.__delta - self.__eta) * (0.5 + rn * rn - rn)
            if rn == 0. or rn == 1.:
                proba = [(self.__alpha + self.__beta)/self.__D1, 1.-(self.__alpha + self.__beta)/self.__D1]
            else:
                proba = [self.__lamb * self.__eta / D2, 1.-self.__lamb * self.__eta / D2]

            typeSample = np.random.choice(a=['dur', 'flou'], size=1, p=proba)[0]

            if typeSample == 'dur':
                norm = self.probaR2CondR1(rn, 0.0) + self.probaR2CondR1(rn, 1.0)
                probaR2CondR1 = [self.probaR2CondR1(rn, 0.0) / norm, self.probaR2CondR1(rn, 1.0) / norm]
                return np.random.choice(a=[0.0, 1.0], size=1, p=probaR2CondR1)[0]
            else:
                if rn == 0.:
                    return self.__rv_pente0.rvs()
                if rn == 1.:
                    return self.__rv_pente1.rvs()
                return self.__rv_triangle.rvs(self.__eta, self.__delta, rn)

# depuis :
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html
class parabole_Serie2bis_gen(stats.rv_continuous):
    "Parabole de la série 2 bis, lorsque r1 in ]0,1["

    def _pdf(self, x, eta, delta, lamb):
        K = lamb+1.
        L = delta - eta
        # print('K=', K, ', L=', L, ', value=', (1./3.*L+eta*K), flush=True)
        return (eta * K + L*(0.5 + x * (x-1.))) / (1./3.*L+eta*K)

    def _cdf(self, y, eta, delta, lamb):
        K = lamb+1.
        L = delta - eta
        # print('K=', K, ', L=', L, ', value=', (1./3.*L+eta*K), flush=True)
        return (eta*K*y + L*(0.5*y + 1./3.*y*y*y - 1./2.*y*y)) / (1./3.*L+eta*K)

    def _stats(self, eta, delta, lamb):
        K = lamb+1.
        L = delta - eta
        N = (1./3.*L+eta*K)
        # print('N=', N, flush=True)
        moment1 = 1./N*(1./2 * eta * K + L/6.)
        moment2 = 1./N*(1./3 * eta * K + 7.*L/60.)
        return moment1, moment2 - moment1**2, 0., None

    def _argcheck(self, *args):
        # Aucune condition sur les paramètres (on renvoie que des 1)
        cond = 1
        for arg in args:
            cond = np.logical_and(cond, 1)
        return cond


class pente_Serie2bis_gen(stats.rv_continuous):
    "Pente de la série 2, lorsque r1==0 (eq 13)"

    def _pdf(self, x):
        return 2.*(1.-x)

    def _cdf(self, y):
        return 2.*y*(1. - 0.5*y)

    def _stats(self):
        moment1 = 1./3.
        moment2 = 1./6.
        return moment1, moment2 - moment1**2, None, None

class pente2_Serie2bis_gen(stats.rv_continuous):
    "Pente de la série 2, lorsque r1==0 (eq 13)"

    def _pdf(self, x):
        return 2.*x

    def _cdf(self, y):
        return y*y

    def _stats(self):
        moment1 = 2./3.
        moment2 = 1./2.
        return moment1, moment2 - moment1**2, None, None


class triangle_Serie2bis_gen(stats.rv_continuous):
    "triangle de la série 2, lorsque r2 in ]0,1[ (eq 15)"

    def _pdf(self, x, eta, delta, r1):
        L = delta - eta
        # print('L=', L, ', value=', (eta + L * (0.5 + r1*r1 - r1)), flush=True)
        return (eta + L * np.abs(x - r1)) / (eta + L * (0.5 + r1*r1 - r1))

    def _argcheck(self, *args):
        # Aucune condition sur les paramètres (on renvoie que des 1)
        cond = 1
        for arg in args:
            cond = np.logical_and(cond, 1)
        return cond

#    def _cdf(self, y, eta, delta, r1):
#        #K = 1.5*(delta+eta)
#        L = delta-eta
#        M = eta + L*(r1*r1-r1+0.5)
#        #if (y.size>1):
#        #    print(y<=r1)
#        #a = y<=r1
#
#        #return 1
#        y = np.array(y)
#        if y<=r1:
#            return y/M*(eta + L*(r1-0.5*y))
#        else:
#            return 1./M*(eta*y + L*(0.5*y*y-r1*y+r1*r1))
#        a = np.zeros((y.size))
#        for i, yval in enumerate(y):
#            print(yval, yval.size)
#            if yval<=r1:
#                a[i] = yval/M*(eta + L*(r1-0.5*yval))
#            else:
#                a[i] = 1./M*(eta*yval + L*(0.5*yval*yval-r1*yval+r1*r1))
#        return a
#    def _stats(self, eta, delta, r1):
#        K = 1.5*(delta+eta)
#        L = delta-eta
#        moment1 = 0.5
#        moment2 = 1./(K-L/6.)*(K/3.-L/20.)
#        return moment1, moment2-moment1**2, 0., None


if __name__ == '__main__':
    main()