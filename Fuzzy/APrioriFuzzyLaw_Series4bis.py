##!/usr/bin/env python3
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
    from APrioriFuzzyLaw import LoiAPriori, plotSample, echelle
else:
    from Fuzzy.APrioriFuzzyLaw import LoiAPriori, plotSample, echelle

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

    print('*********************SERIES 4 Extended')
    series = 'Serie4bis'
    #P, case = LoiAPrioriSeries4bis(alpha=0.10, gamma = 0.65, delta_d=0.15, delta_u=0.05, lamb=0.5, EPS=EPS, discretization=discretization), 1
    P, case = LoiAPrioriSeries4bis(alpha=0.10, gamma = 0.65, delta_d=0.15, delta_u=0.15, lamb=0.8, EPS=EPS, discretization=discretization), 1

    print(P)
    # ALPHA, BETA, DELTA_D, DELTA_U, GAMMA, LAMB = P.getParam()
    # print('4bis:'+str(ALPHA)+':'+str(GAMMA)+':'+str(DELTA_D)+':'+str(DELTA_U)+':'+str(LAMB)+', beta='+str(BETA)+', #pH='+str(P.maxiHardJump()))
    print('model string', P.stringName())
    
    print(P)
    print('model string', P.stringName())

    # Test le modele
    OKtestModel = P.testModel(verbose=verbose, epsilon=epsilon)

    # Simulation d'un chaine de markov flou suivant ce modèle
    N = 30000
    chain = P.testSimulMC(N, verbose=verbose, epsilon=epsilon)

    if graphics == True:
        P.plotR1R2   ('./figures/LoiCouple_' + series + '_' + str(case) + '.png', dpi=dpi)
        P.plotR1     ('./figures/LoiMarg_'   + series + '_' + str(case) + '.png', dpi=dpi)
        # Dessins
        mini, maxi = 100, 150
        P.PlotMCchain('./figures/Traj_'      + series + '_' + str(case) + '.png', chain, mini=mini, maxi=maxi, dpi=dpi)


    # # Dessin de la pente a partir de la quelle on doit faire des tirages
    # pente = pente_Serie4bis_gen(momtype=0, name='pente_Serie4bis', a=0., b=1., shapes="DELTA_U")
    # rv = pente(DELTA_U)
    # #print(pente.pdf(0.54, DELTA_U))
    # mean, var = plotSample(rv, 1000, 'pente_' + series + '_'+str(case)+'.png')
    # print('mean echantillon = ', mean)
    # print('var echantillon = ', var)
    # print(rv.stats('mvsk'))

    # # Dessin de la pente a partir de la quelle on doit faire des tirages
    # pente2 = pente2_Serie4bis_gen(momtype=0, name='pente2_Serie4bis', a=0., b=1., shapes="DELTA_D")
    # rv = pente2(DELTA_D)
    # #print(pente2.pdf(0.54, DELTA_D))
    # mean, var = plotSample(rv, 1000, 'pente2_' + series + '_'+str(case)+'.png')
    # print('mean echantillon = ', mean)
    # print('var echantillon = ', var)
    # print(rv.stats('mvsk'))

    # trapeze = trapeze_Serie4bis_gen(momtype=0, name='trapeze_Serie4bis', a=0., b=1., shapes="ALPHA, BETA, GAMMA, DELTA_D, DELTA_U, LAMB")
    # rv = trapeze(ALPHA, BETA, GAMMA, DELTA_D, DELTA_U, LAMB)
    # #print(trapeze.pdf(0.54, ALPHA, BETA, GAMMA, DELTA_D, DELTA_U, LAMB))
    # mean, var = plotSample(rv, 1000, 'trapeze_' + series + '_'+str(case)+'.png')
    # print('mean echantillon = ', mean)
    # print('var echantillon = ', var)
    # print(rv.stats('mvsk'))

########### SERIE 4 Extended ##################
######################################
class LoiAPrioriSeries4bis(LoiAPriori):
    """
    Implementation of the fourth law described in the report Calcul_Simu_CGOFMSM.pdf
    """

    def __init__(self, alpha, gamma, delta_d, delta_u, lamb, EPS=1E-8, discretization=100):
        """Constructeur to set the parameters of the density."""

        LoiAPriori.__init__(self, EPS=EPS, discretization=discretization)

        self.__alpha = alpha

        assert delta_d >= 0. and delta_d <= 0.5, print('PB : delta_d=', delta_d)
        assert delta_u >= 0. and delta_u <= 0.5, print('PB : delta_u=', delta_u)
        self.__delta_d = delta_d
        self.__delta_u = delta_u
        self.__lamb = lamb

        # M = 3.*(self.__delta_u + self.__delta_d) - 0.5*(self.__delta_d*self.__delta_d+self.__delta_u*self.__delta_u)
        # if M != 0.:
        #     if gamma >= (1.-2.*self.__alpha)/M:
        #         self.__gamma = (1.-2.*self.__alpha)/M-10E-10
        #     else:
        #         self.__gamma = gamma
        # else:
        #     self.__gamma = 0.

        self.__gamma = gamma

        self.update()

    def update(self):

        M = (self.__delta_u + self.__delta_d) *(self.__lamb+1.) - 0.5*(self.__delta_u*self.__delta_u + self.__delta_d*self.__delta_d)
        self.__beta = (1. - self.__gamma * M)/2. - self.__alpha

        # pour les tirages aléatoires
        pente_01 = pente_Serie4bis_gen(momtype=0, name='pente1_Serie4bis', a=0., b=1., shapes="delta_u")
        self.__rv_pente0 = pente_01(self.__delta_u)
        pente2_01 = pente2_Serie4bis_gen(momtype=0, name='pente2_Serie4bis', a=0., b=1., shapes="delta_d")
        self.__rv_pente1 = pente2_01(self.__delta_d)
        
        # print(self.maxiHardJump())
        # print(self.maxiFuzzyJump())
        # print(2.*(self.__alpha+self.__beta) + self.__lamb * self.__gamma / 2.*(self.__delta_d+self.__delta_u))
        # print(self.__gamma*( (self.__lamb/2.+1.) * (self.__delta_d+self.__delta_u) - 0.5 * (self.__delta_d*self.__delta_d + self.__delta_u*self.__delta_u)))
        # input('Perfecto!')

    def setParametersFromSimul(self, Rsimul, nbcl):
        
        input('setParametersFromSimul : to be done')
        Nsimul = len(Rsimul)

        self.update()
        
    def getParam(self):
        """ Return the params of the law model."""
        return self.__alpha, self.__beta, self.__delta_d, self.__delta_u, self.__gamma, self.__lamb

    def __str__(self):
        str1 = "alpha=" + str(self.__alpha) + ", beta=" + str(self.__beta)
        str1 += ", delta_d=" + str(self.__delta_d) + ", delta_u=" + str(self.__delta_u) + ", gamma=" + str(self.__gamma)
        str1 += ", lambda=" + str(self.__lamb)
        return str1

    def stringName(self):
        return '4bis:'+str('%.4f'%self.__alpha)+':'+str('%.4f'%self.__gamma)+':'+str('%.4f'%self.__delta_d)+':'+str('%.4f'%self.__delta_u)+':'+str('%.4f'%self.__lamb)

    def probaR1R2(self, r1, r2):
        """ Return the joint proba at r1, r2."""

        # Les masses aux coins
        if (r1 == 0. and r2 == 0.) or (r1 == 1. and r2 == 1.):
            return self.__alpha
        if (r1 == 1. and r2 == 0.) or (r1 == 0. and r2 == 1.):
            return self.__beta

        # les bords
        if r2 == 0. and r1 <= self.__delta_d:
            return self.__lamb * self.__gamma*(1.-r1/self.__delta_d)
        if r2 == 1. and r1 >= 1.-self.__delta_u:
            return self.__lamb * self.__gamma*(r1/self.__delta_u + (1.-1./self.__delta_u))
        if r1 == 0. and r2 <= self.__delta_u:
            return self.__lamb * self.__gamma*(1.-r2/self.__delta_u)
        if r1 == 1. and r2 >= 1.-self.__delta_d:
            return self.__lamb * self.__gamma*(r2/self.__delta_d + (1.-1./self.__delta_d))

        # Le coeurs
        if (-self.__delta_d <r2-r1) and (r2-r1< self.__delta_u):
            return self.__gamma
        return 0.

    def probaR(self, r):
        """ Return the marginal proba at r."""

        if r == 0.:
            return self.__alpha + self.__beta + self.__delta_u * self.__gamma * self.__lamb / 2.

        if r == 1.:
            return self.__alpha + self.__beta + self.__delta_d * self.__gamma * self.__lamb / 2.

        if (r > 0.) and (r <= self.__delta_d):
            return self.__gamma * ( (1.-self.__lamb / self.__delta_d) * r + self.__lamb + self.__delta_u)

        if (r >= 1. - self.__delta_u) and (r < 1.):
            return self.__gamma * ( (self.__lamb / self.__delta_u - 1.) * r + self.__lamb * (1. - 1./self.__delta_u) + 1. + self.__delta_d)

        return self.__gamma * (self.__delta_d + self.__delta_u)

    def probaR2CondR1(self, r1, r2):
        """ Return the conditional proba at r2 knowing r1."""

        if (r1 > 0.) and (r1 <= self.__delta_d):
            temp = (1.-self.__lamb / self.__delta_d)*r1 + self.__lamb + self.__delta_u
            if r2 == 0.:
                return self.__lamb * (1. - r1/self.__delta_d)/ temp
            if r2 <= r1 + self.__delta_u:
                return 1. / temp
            else:
                return 0.

        elif (r1 <1.) and (r1 >= 1-self.__delta_u):
            temp = (self.__lamb / self.__delta_u - 1.)*r1 + self.__lamb * (1. - 1./self.__delta_u) + 1. + self.__delta_d
            if r2 == 1.:
                return self.__lamb * (r1/self.__delta_u + 1. - 1./self.__delta_u)/ temp
            if r2 >= r1 - self.__delta_d:
                return 1. / temp
            else:
                return 0.

        elif r1 == 0.0:
            temp_u = self.__alpha + self.__beta + self.__gamma * self.__lamb / 2. * self.__delta_u
            if r2 > 0 and r2 <= self.__delta_u:
                return self.__lamb * self.__gamma * (1. - r2/self.__delta_u) / temp_u
            elif r2 == 0.:
                return self.__alpha / temp_u
            elif r2 == 1.:
                return self.__beta / temp_u
            else:
                return 0.

        elif r1 == 1.0:
            temp_d = self.__alpha + self.__beta + self.__gamma * self.__lamb / 2. * self.__delta_d
            if r2 >= 1. - self.__delta_d and r2 < 1.:
                return self.__lamb * self.__gamma * ( (r2/self.__delta_d) + (1. - 1./self.__delta_d)) / temp_d
            elif r2 == 0.:
                return self.__beta / temp_d
            elif r2 == 1.:
                return self.__alpha / temp_d
            else:
                return 0.

        else:
            if r2 >= r1 - self.__delta_d and r2 <= r1 + self.__delta_u:
                return 1. / (self.__delta_d + self.__delta_u)
            else:
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
            temp_u = self.__alpha+self.__beta+self.__gamma*self.__lamb/2.*self.__delta_u
            proba = [(self.__alpha+self.__beta)/temp_u, 1.-(self.__alpha+self.__beta)/temp_u]
            typeSample = np.random.choice(a=['dur', 'flou'], size=1, p=proba)[0]
            if typeSample == 'dur':
                sumi = self.__alpha/temp_u + self.__beta/temp_u
                proba = [self.__alpha/(temp_u*sumi), self.__beta/(temp_u*sumi)]
                rnp1 = np.random.choice(a=[0.0, 1.0], size=1, p=proba)[0]
            else:
                rnp1 = self.__rv_pente0.rvs()
                # print('rnp1=', rnp1)
                # input('rn==0')

        elif rn == 1.0:
            temp_d = self.__alpha+self.__beta+self.__gamma*self.__lamb/2.*self.__delta_d
            proba = [(self.__alpha+self.__beta)/temp_d, 1.-(self.__alpha+self.__beta)/temp_d]
            typeSample = np.random.choice(a=['dur', 'flou'], size=1, p=proba)[0]
            if typeSample == 'dur':
                sumi = self.__beta + self.__alpha
                proba = [self.__beta/sumi, self.__alpha/sumi]
                rnp1 = np.random.choice(a=[0.0, 1.0], size=1, p=proba)[0]
            else:
                rnp1 = self.__rv_pente1.rvs()

        elif (rn > 0.) and (rn <= self.__delta_d):
            temp = (1.-self.__lamb / self.__delta_d)*rn + self.__lamb + self.__delta_u
            proba = [self.__lamb*(1.-rn/self.__delta_d)/temp, 1.-self.__lamb*(1.-rn/self.__delta_d)/temp]
            typeSample = np.random.choice(a=['dur', 'flou'], size=1, p=proba)[0]
            if typeSample == 'dur':
                rnp1 = 0.
            else:
                rnp1 = echelle(np.random.random_sample(), 0, rn+self.__delta_u)

        elif (rn >= 1. - self.__delta_u) and (rn < 1.):
            temp = (self.__lamb / self.__delta_u - 1.)*rn + self.__lamb * (1. - 1./self.__delta_u) + 1. + self.__delta_d
            proba = [self.__lamb*(rn/self.__delta_u +1. - 1./self.__delta_u)/temp, 1.-self.__lamb*(rn/self.__delta_u +1. - 1./self.__delta_u)/temp]
            typeSample = np.random.choice(a=['dur', 'flou'], size=1, p=proba)[0]
            if typeSample == 'dur':
                rnp1 = 1.
            else:
                rnp1 = echelle(np.random.random_sample(), rn - self.__delta_d, 1.)

        else:
            rnp1 = echelle(np.random.random_sample(), rn - self.__delta_d, rn + self.__delta_u)
            
        return rnp1

class pente_Serie4bis_gen(stats.rv_continuous):
    "Pente de la serie 4 bis, lorsque r1==0 "

    def _pdf(self, x, delta_u):
        if x.size == 1:
            if x>0. and x<=delta_u:
                K = 2. / delta_u
                return K*(1.-x/delta_u)
            else:
                return 0.

        solution = np.zeros(x.shape)
        for i,val in enumerate(x):
            delta_u_i = delta_u[i]
            if val>0. and val<=delta_u_i:
                K = 2. / delta_u_i
                solution[i] = K*(1.-val/delta_u_i)
            else:
                solution[i] = 0.
        return solution

    def _cdf(self, x, delta_u):
        if x.size == 1:
            if x>0. and x<=delta_u:
                return x/delta_u*(2.-x/delta_u)
            else:
                return 1.

        solution = np.zeros(x.shape)
        for i,val in enumerate(x):
            delta_u_i = delta_u[i]
            if val>0. and val<=delta_u_i:
                solution[i] = val/delta_u_i*(2.-val/delta_u_i)
            else:
                solution[i] = 1.
        return solution

    def _stats(self, delta_u):
        moment1 = 1./3. * delta_u
        moment2 = 1./6. *  delta_u * delta_u
        return moment1, moment2 - moment1**2, None, None

    def _argcheck(self, *args):
        # Aucune condition sur les paramètres (on renvoie que des 1)
        cond = 1
        for arg in args:
            cond = np.logical_and(cond, 1)
        return cond

class pente2_Serie4bis_gen(stats.rv_continuous):
    "Pente de la série 4 bis, lorsque r1==1"

    def _pdf(self, x, delta_d):
        if x.size == 1:
            if x>=1-delta_d and x<1.:
                K = 2. / delta_d
                Z = 1. - 1./delta_d
                return K*(x/delta_d + Z)
            else:
                return 0.

        solution = np.zeros(x.shape)
        for i,val in enumerate(x):
            delta_d_i = delta_d[i]
            if val>=1-delta_d_i and val<1.:
                K = 2. / delta_d_i
                Z = 1. - 1./delta_d_i
                solution[i] = K*(val/delta_d_i + Z)
            else:
                solution[i] = 0.
        return solution

    def _cdf(self, x, delta_d):
        if x.size == 1:
            if x>=1-delta_d and x<1.:
                K = 2. / delta_d
                Z = 1. - 1./delta_d
                return K*(Z*(x-1.+delta_d) + (x*x-1.-delta_d*delta_d+2. *delta_d)/(2. * delta_d))
            else:
                return 0.

        solution = np.zeros(x.shape)
        for i,val in enumerate(x):
            delta_d_i = delta_d[i]
            if val>=1-delta_d_i and val<1.:
                K = 2. / delta_d_i
                Z = 1. - 1./delta_d_i
                solution[i] = K*(Z*(val-1.+delta_d_i) + (val*val-1.-delta_d_i*delta_d_i+2. *delta_d_i)/(2. * delta_d_i))
            else:
                solution[i] = 0.
        return solution

    def _stats(self, delta_d):
        moment1 = 1.-delta_d/3
        Z = 1. - 1./delta_d
        K = 2. / delta_d
        moment2 = K*(Z/3.*(1-(1. - delta_d)**3) + 1./(4. * delta_d)*(1. - (1. - delta_d)**4) )
        return moment1, moment2 - moment1**2, None, None
        
    def _argcheck(self, *args):
        # Aucune condition sur les paramètres (on renvoie que des 1)
        cond = 1
        for arg in args:
            cond = np.logical_and(cond, 1)
        return cond

# class trapeze_Serie4bis_gen(stats.rv_continuous):
#     "trapeze de la serie 4 extended, pour p(r_1)"

#     def _pdf(self, x, alpha, beta, gamma, delta_d, delta_u):
#         norm = 2. * (alpha + beta) + gamma*(delta_d+delta_u)
#         if not isfloat(x):
#             print(x)
#         if x > 0 and x <= delta_d:
#             s = gamma*(delta_u + x + 1.)
#         elif x>delta_d and x<1.-delta_u:
#             s = gamma*(delta_u + delta_d)
#         else:
#             s = gamma*(2. + delta_d - x)
#         return s*norm



if __name__ == '__main__':
    main()

    