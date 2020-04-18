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
    from LoisDiscreteFuzzy import Loi2DDiscreteFuzzy, Loi1DDiscreteFuzzy
else:
    from Fuzzy.APrioriFuzzyLaw import LoiAPriori, plotSample

def main():

    discretization = 200
    EPS            = 1E-8
    epsilon        = 1E-2
    verbose        = True
    graphics       = True

    # seed = random.randrange(sys.maxsize)
    # seed = 5039309497922655937
    # rng = random.Random(seed)
    # print("Seed was:", seed)

    # SERIES 2 ter (papier sur la prédicton de trafic)
    print('*********************SERIES 2 ter')
    series = 'Serie2ter'

    #P, case = LoiAPrioriSeries2ter(alpha0 = 0.10, alpha1 = 0.20, beta=0.06, EPS=EPS, discretization=discretization), 1
    #P, case = LoiAPrioriSeries2ter(alpha0 = 0.05, alpha1 = 0.02, beta=0.01, EPS=EPS, discretization=discretization), 2
    # P, case = LoiAPrioriSeries2ter(alpha0 = 0.144, alpha1 = 0.050, beta=0.000, EPS=EPS, discretization=discretization), 3
    P, case = LoiAPrioriSeries2ter(alpha0 = 0.07, alpha1 = 0.24, beta=0.09, EPS=EPS, discretization=discretization), 4

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
        mini, maxi = 100, 150
        P.PlotMCchain('./figures/Traj_'      + series + '_' + str(case) + '.png', chain, mini=mini, maxi=maxi, dpi=dpi)


########### SERIE 2 ter ##############
######################################
class LoiAPrioriSeries2ter(LoiAPriori):
    """
    Implementation of the second law described in the paper about traffic flows (Zied)
    """

    def __init__(self, alpha0, alpha1, beta, EPS=1E-8, discretization=100):
        """Constructeur to set the parameters of the density."""

        LoiAPriori.__init__(self, EPS=EPS, discretization=discretization)

        self.__alpha0 = alpha0
        self.__alpha1 = alpha1
        self.__beta   = beta
        self.__eta = 3./8.*(1. - self.__alpha0 - self.__alpha1 - 2.*self.__beta)
        if self.__eta<0.: self.__eta=0. # erreur d'arrondi
        #print('self.__eta=', self.__eta)

        self.update()

    def update(self):

        self.__D0 = self.__alpha0 + self.__beta + self.__eta / 2.
        self.__D1 = self.__alpha1 + self.__beta + self.__eta / 2.

        # pour les tirages aléatoires
        pente_0 = pente0_Serie2ter_gen(momtype=0, name='pente0_Serie2', a=0., b=1., shapes="alpha0, alpha1, beta")
        self.__rv_pente0 = pente_0(self.__alpha0, self.__alpha1, self.__beta)
        pente_1 = pente1_Serie2ter_gen(momtype=0, name='pente1_Serie2', a=0., b=1., shapes="alpha0, alpha1, beta")
        self.__rv_pente1 = pente_1(self.__alpha0, self.__alpha1, self.__beta)

        self.__rv_triangle = triangle_Serie2ter_gen(momtype=0, name='triangle_Serie2ter', a=0., b=1., shapes="alpha0, alpha1, beta, r1")

        parab = parabole_Serie2ter_gen(momtype=0, name='parabole_Serie2ter', a=0., b=1., shapes="alpha0, alpha1, beta")
        self.__rv_parab = parab(self.__alpha0, self.__alpha1, self.__beta)

    def getParam(self):
        """ Return the params of the law model."""
        return self.__alpha0, self.__alpha1, self.__beta, self.__eta

    def getEta(self):
        """ Return the eta param of the law model."""
        self.__eta = 3./8.*(1. - self.__alpha0 - self.__alpha1 - 2.*self.__beta)
        if self.__eta<0.: self.__eta=0. # erreur d'arrondi
        return self.__eta
        
    def __str__(self):
        return "alpha0=" + str('%.3f'%(self.__alpha0)) +  ", alpha1=" + str('%.3f'%(self.__alpha1)) + ", beta=" + str('%.3f'%(self.__beta)) + ", eta=" + str('%.3f'%(self.__eta))

    def stringName(self):
        return '2ter:'+str('%.4f'%self.__alpha0)+':'+str('%.4f'%self.__alpha1)+':'+str('%.4f'%self.__beta)

    def getTheoriticalHardTransition(self, n_r):

        JProba = np.zeros(shape=(2,2))
        JProba[0,0] = self.__alpha0 + 23./24. * self.__eta
        JProba[0,1] = self.__beta   +  3./8.  * self.__eta
        JProba[1,0] = JProba[0,1]
        JProba[1,1] = self.__alpha1 + 23./24. * self.__eta
        # construite ainsi a matrice est necessairement stationnaire (les summ des colonnes sont identiques)

        if len(JProba[JProba<0.0]) > 0 or len(JProba[JProba>1.0]) > 0:
            print('Array JProba = ', JProba)
            exit('    --> PROBLEM getTheoriticalHardTransition because all values must be between 0 and 1')

        somme = np.sum(sum(JProba))
        if abs(1.0 - somme)> 1E-10:
            print('JProba=', JProba)
            print('sum=', somme)
            input('PROBLEM getTheoriticalHardTransition because the sum must be 1.0')

        MProba, TProba = self.getMTProbaFormJProba(JProba, n_r)

        return MProba, TProba, JProba


    def setParametersFromSimul(self, Rsimul, nbcl):
        
        Nsimul = len(Rsimul)

        alpha0, alpha1, beta = 0., 0., 0.
        for n in range(1, Nsimul):
            if Rsimul[n-1] == 0      and Rsimul[n] == 0:      alpha0 += 1.
            if Rsimul[n-1] == nbcl-1 and Rsimul[n] == nbcl-1: alpha1 += 1.
            if Rsimul[n-1] == 0      and Rsimul[n] == nbcl-1: beta   += 1.
            if Rsimul[n-1] == nbcl-1 and Rsimul[n] == 0:      beta   += 1.
        beta /= 2. # ca compte les transitions 0-1, 1-0, donc on divise par deux
        
        self.__alpha0 = alpha0 / (Nsimul-1.)
        self.__alpha1 = alpha1 / (Nsimul-1.)
        self.__beta = beta   / (Nsimul-1.)
        self.__eta = 3./8.*(1. - self.__alpha0 - self.__alpha1 - 2.*self.__beta)
        if self.__eta<0.: self.__eta=0. # erreur d'arrondi

        self.update()


    def probaR1R2(self, r1, r2):
        """ Return the joint proba at r1, r2."""
        if r1 == 0.: 
            if r2 == 0.:
                return self.__alpha0
            elif r2 == 1.:
                return self.__beta
            else:
                return self.__eta * (1. - r2)

        if r1 == 1.: 
            if r2 == 0.:
                return self.__beta
            elif r2 == 1.:
                return self.__alpha1
            else:
                return self.__eta * (1. - np.abs(1. - r2))

        return self.__eta * (1. - np.abs(r1 - r2))

    def probaR(self, r):
        """ Return the marginal proba at r."""

        if r == 0.:
            return self.__alpha0 + self.__beta + self.__eta  / 2.

        if r == 1.:
            return self.__alpha1 + self.__beta + self.__eta  / 2.

        return self.__eta * (3./2. + r - r*r)

    def probaR2CondR1(self, r1, r2):
        """ Return the conditional proba at r2 knowing r1."""

        if r1 == 0.0:
            if r2 == 0.0:
                return self.__alpha0          / self.__D0
            elif r2 == 1.0:
                return self.__beta            / self.__D0
            else:
                return self.__eta * (1. - r2) / self.__D0

        elif r1 == 1.0:
            if r2 == 0.0:
                return self.__beta      / self.__D1
            elif r2 == 1.0:
                return self.__alpha1    / self.__D1
            else:
                return self.__eta  * r2 / self.__D1

        else:
            D = 1.5 + r1 - r1 * r1 
            if r2 == 0.0:
                return  (1. - r1)              / D
            elif r2 == 1.0:
                return  r1                     / D
            else:
                return  (1. - np.abs(r1 - r2)) / D

        

    def tirageR1(self):
        """ Return a draw according to the marginal density p(r1) """

        proba = np.zeros(shape=(3))
        proba[0] = self.__alpha0 + self.__beta + self.__eta/2.
        proba[1] = self.__alpha1 + self.__beta + self.__eta/2.
        proba[2] = 1. - (proba[0]+proba[1])
        # typeSample = np.random.choice(a=['0.', '1.', 'F'], size=1, p=proba)[0]
        typeSample = random.choices(population=['0.', '1.', 'F'], weights=proba)[0]
        if typeSample != 'F':
            r1 = float(typeSample)
        else:
            r1 = self.__rv_parab.rvs()

        return r1


    def tirageRnp1CondRn(self, rn):
        """ Return a draw according to the conditional density p(r2 | r1) """

        proba = np.zeros(shape=(3))
        if rn == 0.:
            proba[0] = self.__alpha0/self.__D0
            proba[1] = self.__beta  /self.__D0
            
        elif rn == 1.:
            proba[0] = self.__beta  /self.__D1
            proba[1] = self.__alpha1/self.__D1
        else:
            Dr1      = 1.5 + rn - rn*rn
            proba[0] = (1.-rn) / Dr1
            proba[1] = rn      / Dr1
        proba[2] = 1. - (proba[0]+proba[1])
        
        typeSample = random.choices(population=['0.', '1.', 'F'], weights=proba)[0]
        
        if typeSample != 'F':
            rnp1 = float(typeSample)
        else:
            if rn == 0.:
                rnp1 =  self.__rv_pente0.rvs()
            elif rn == 1.:
                rnp1 = self.__rv_pente1.rvs()
            else:
                rnp1 = self.__rv_triangle.rvs(self.__alpha0, self.__alpha1, self.__beta, rn)

        return rnp1

# depuis :
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html
class parabole_Serie2ter_gen(stats.rv_continuous):
    "Parabole de la série 2 ter, lorsque r1 in ]0,1["

    def _pdf(self, x, alpha0, alpha1, beta):
        return 3./5.*(3./2.+x-x*x)

    def _argcheck(self, *args):
        # Aucune condition sur les paramètres (on renvoie que des 1)
        cond = 1
        for arg in args:
            cond = np.logical_and(cond, 1)
        return cond

class pente0_Serie2ter_gen(stats.rv_continuous):
    "Pente de la série 2 ter, lorsque r1==0 "

    def _pdf(self, x, alpha0, alpha1, beta):
        return 2.*(1.-x)

    def _argcheck(self, *args):
        # Aucune condition sur les paramètres (on renvoie que des 1)
        cond = 1
        for arg in args:
            cond = np.logical_and(cond, 1)
        return cond

class pente1_Serie2ter_gen(stats.rv_continuous):
    "Pente de la série 2 ter, lorsque r1==1 "

    def _pdf(self, x, alpha0, alpha1, beta):
        return 2.*x

    def _argcheck(self, *args):
        # Aucune condition sur les paramètres (on renvoie que des 1)
        cond = 1
        for arg in args:
            cond = np.logical_and(cond, 1)
        return cond


class triangle_Serie2ter_gen(stats.rv_continuous):
    "triangle de la série 2 ter, lorsque r2 in ]0,1["

    def _pdf(self, x, alpha0, alpha1, beta, r1):
        return  (1.-np.abs(r1-x))/(0.5+r1-r1*r1)

    def _argcheck(self, *args):
        # Aucune condition sur les paramètres (on renvoie que des 1)
        cond = 1
        for arg in args:
            cond = np.logical_and(cond, 1)
        return cond


if __name__ == '__main__':
    main()
