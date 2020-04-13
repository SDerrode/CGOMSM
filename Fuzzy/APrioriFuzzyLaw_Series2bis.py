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
    verbose        = True
    graphics       = True

    # seed = random.randrange(sys.maxsize)
    # seed = 5039309497922655937
    # rng = random.Random(seed)
    # print("Seed was:", seed)

    # SERIES 2 bis
    print('*********************SERIES 2 bis')
    series = 'Serie2bis'
    #P, case = LoiAPrioriSeries2bis(alpha=0.07, eta=0.21, delta=0.05, lamb=0., EPS=EPS, discretization=discretization), 3
    #P, case = LoiAPrioriSeries2bis(alpha=0.07, eta=0.21, delta=0.05, lamb=1., EPS=EPS, discretization=discretization), 2
    P, case = LoiAPrioriSeries2bis(alpha=0.07, eta=0.21, delta=0.10, lamb=0.3, EPS=EPS, discretization=discretization), 1
    #P, case = LoiAPrioriSeries2bis(alpha=0.12, eta=0., delta=0., lamb=0.3, EPS=EPS, discretization=discretization), 5
    #P, case = LoiAPrioriSeries2bis(alpha=0.1, eta=0.3, delta=0.3, lamb=0.5, EPS=EPS, discretization=discretization), 6

    # Le cas suivant assure que alpha=beta=0
    # eta = 0.21
    # lamb= 0.6
    # delta = 6.*(0.5 - eta*(lamb+1./3.))
    # P, case = LoiAPrioriSeries2bis(alpha=0, eta=eta, delta=delta, lamb=lamb, EPS=EPS, discretization=discretization), 4

    print(P)
    print('model string', P.stringName())

    # Test le modele
    OKtestModel = P.testModel(verbose=verbose, epsilon=epsilon)

    # Simulation d'un chaine de markov flou suivant ce modèle
    N = 10000
    chain = P.testSimulMC(N, verbose=verbose, epsilon=epsilon*2)

    if graphics == True:
        P.plotR1R2   ('./figures/LoiCouple_' + series + '_' + str(case) + '.png', dpi=dpi)
        P.plotR1     ('./figures/LoiMarg_'   + series + '_' + str(case) + '.png', dpi=dpi)
        # Dessins
        mini, maxi = 100, 150
        P.PlotMCchain('./figures/Traj_'      + series + '_' + str(case) + '.png', chain, mini=mini, maxi=maxi, dpi=dpi)


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

    def __init__(self, alpha, eta, delta, lamb, EPS=1E-8, discretization=100):
        """Constructeur to set the parameters of the density."""

        LoiAPriori.__init__(self, EPS=EPS, discretization=discretization)

        self.__alpha = alpha
        self.__eta = eta
        self.__delta = delta
        self.__lamb = lamb
        self.__beta = 0.5 - delta/6. - eta*(lamb+1./3.) - self.__alpha
        #print('test ph=', 2.*(self.__alpha + self.__beta) + self.__lamb * self.__eta)
        #print('test pf=', 1./3.*(self.__delta-self.__eta) + self.__eta *(self.__lamb +1.))

        self.update()

    def update(self):
        self.__D1 = self.__alpha + self.__beta + self.__lamb * self.__eta / 2.

        # pour les tirages aléatoires
        pente_01 = pente_Serie2bis_gen(momtype=0, name='pente1_Serie2bis', a=0., b=1.)
        self.__rv_pente0 = pente_01()
        pente2_01 = pente2_Serie2bis_gen(momtype=0, name='pente2_Serie2bis', a=0., b=1.)
        self.__rv_pente1 = pente2_01()

        parab = parabole_Serie2bis_gen(momtype=0, name='parabole_Serie2bis', a=0., b=1., shapes="eta, delta, lamb")
        self.__rv_parab = parab(self.__eta, self.__delta, self.__lamb)

        self.__rv_triangle = triangle_Serie2bis_gen(momtype=0, name='triangle_Serie2bis', a=0., b=1., shapes="eta, delta, r1")

    def setParametersFromSimul(self, Rsimul, nbcl):
        
        input('setParametersFromSimul : to be done')
        Nsimul = len(Rsimul)

        self.update()
        
    def getParam(self):
        """ Return the params of the law model."""
        return self.__alpha, self.__beta, self.__eta, self.__delta, self.__lamb

    def __str__(self):
        return "alpha=" + str(self.__alpha) + ", beta=" + str(self.__beta) + ", eta=" + str(self.__eta) + ", delta=" + str(self.__delta) + ", lambda=" + str(self.__lamb)

    def stringName(self):
        return '2bis:'+str('%.4f'%self.__alpha)+':'+str('%.4f'%self.__beta)+':'+str('%.4f'%self.__eta)+':'+str('%.4f'%self.__lamb)

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