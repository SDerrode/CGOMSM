#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 11:06:52 2017

@author: MacBook_Derrode
"""

import numpy as np
import random
import scipy.stats as stats
import matplotlib as mpl
import matplotlib.pyplot as plt

fontS = 16 # fontSize
mpl.rc('xtick', labelsize=fontS)
mpl.rc('ytick', labelsize=fontS)
dpi = 300

from Fuzzy.APrioriFuzzyLaw import LoiAPriori, plotSample
#from APrioriFuzzyLaw import LoiAPriori, plotSample

def main():

    discretization = 200

    seed = random.randrange(sys.maxsize)
    seed = 5039309497922655937
    rng = random.Random(seed)
    print("Seed was:", seed)

    # SERIES 2
    print('*********************SERIES 2')
    series = 'Serie2'

    #P,case = LoiAPrioriSeries2(alpha = 0.10, eta = 0.12, delta=0.08), 1
    #P, case = LoiAPrioriSeries2(alpha=0.02, eta=0.12, delta=0.001), 2
    # si delta = (3-8 eta)/7, alors alpha = beta = 0.
    #P,case = LoiAPrioriSeries2(alpha = 0, eta = 0.1, delta=0.3142), 3

    #P, case = LoiAPrioriSeries2(alpha=0.07, eta=0.275, delta=0.05), 10 #--> 48%
    #P, case = LoiAPrioriSeries2(alpha=0.07, eta=0.21, delta=0.05), 11 #--> 58%
    #P, case = LoiAPrioriSeries2(alpha=0.07, eta=0.16, delta=0.05), 12 #--> 67%
    #P, case = LoiAPrioriSeries2(alpha=0.07, eta=0.108, delta=0.05), 13 #--> 75%
    #P, case = LoiAPrioriSeries2(alpha=0.07, eta=0.005, delta=0.05), 14 #--> 93%

    # alpha1 = 0.
    # delta1 = 0.2
    # eta1 = -1./8.*(6.*alpha1-3.+7*delta1)
    # print('eta1=', eta1)
    # P, case = LoiAPrioriSeries2(alpha=alpha1, eta=eta1, delta=delta1), 100

    # alpha1 = 0.15
    # delta1 = 0.
    # eta1 = -1./8.*(6.*alpha1-3.+7*delta1)
    # print('eta1=', eta1)
    # P, case = LoiAPrioriSeries2(alpha=alpha1, eta=eta1, delta=delta1), 101


    #P, case = LoiAPrioriSeries2(alpha=0.14, eta=0.25, delta=0.02), 20
    #P, case = LoiAPrioriSeries2(alpha=0.10, eta=0.10, delta=0.10), 20
    P, case = LoiAPrioriSeries2(alpha=0.10, eta=0.21, delta=0.076), 200
    # P, case = LoiAPrioriSeries2(alpha=0.1, eta=0.23723214285714284, delta=0.0), 1200

    sum_R1R2 = P.sumR1R2(discretization)
    sum_R1   = P.sumR1(discretization)
    sum_R2CondR1_0   = P.sumR2CondR1(discretization, 0.)
    sum_R2CondR1_20  = P.sumR2CondR1(discretization, 0.10)
    sum_R2CondR1_50  = P.sumR2CondR1(discretization, 0.50)
    sum_R2CondR1_90  = P.sumR2CondR1(discretization, 0.90)
    sum_R2CondR1_100 = P.sumR2CondR1(discretization, 1.)
    ALPHA, BETA, ETA, DELTA = P.getParam()

    print(P)
    print("sum_R1R2 = ", sum_R1R2)
    print("sum_R1 = ", sum_R1)
    print("sum_R2CondR1_0   = ", sum_R2CondR1_0)
    print("sum_R2CondR1_20  = ", sum_R2CondR1_20)
    print("sum_R2CondR1_50  = ", sum_R2CondR1_50)
    print("sum_R2CondR1_90  = ", sum_R2CondR1_90)
    print("sum_R2CondR1_100 = ", sum_R2CondR1_100)
    print('maxiHardJump = ', P.maxiHardJump())
    print('2:'+str(ALPHA)+':'+str(ETA)+':'+str(DELTA)+' #pH='+str(P.maxiHardJump()))

    MProbaTh, TProbaTh, JProbaTh = P.getTheoriticalHardTransition(2)
    print('JProba Hard Theorique=\n', JProbaTh)
    print('sum=', sum(sum(JProbaTh)))

    MProbaNum, TProbaNum, JProbaNum = P.getNumericalHardTransition(2, discretization)
    print('Jproba Hard Numerique, J=\n', JProbaNum)
    print('sum=', sum(sum(JProbaNum)))

    N = 10000
    chain = np.zeros((1, N))
    chain[0] = P.tirageR1()
    # les suivantes...
    for i in range(1, N):
        chain[0, i] = P.tirageRnp1CondRn(chain[0, i-1])

    # Comptage des quarts
    JProbaEch = np.zeros(shape=(2,2))
    for i in range(N-1):
        if chain[0, i]<0.5:
            if chain[0, i+1]<0.5:
                JProbaEch[0,0] += 1.
            else:
                JProbaEch[0,1] += 1.
        else:
            if chain[0, i+1]<0.5:
                JProbaEch[1,0] += 1.
            else:
                JProbaEch[1,1] += 1.
    JProbaEch /= (N-1.)
    print('Jproba Hard Echantillon, J=\n', JProbaEch)
    print('sum=', sum(sum(JProbaEch)))

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
    maxi = 150
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    P.plotR1R2(discretization, 'LoiCouple_' + series + '_' + str(case) + '.png', ax, dpi=dpi)
    P.plotR1(discretization, 'LoiMarg_' + series + '_' + str(case) + '.png', dpi=dpi)
    FIG = plt.figure()
    AX = FIG.gca()
    abscisse= np.linspace(start=mini, stop=maxi, num=maxi-mini)
    AX.plot(abscisse, chain[0, mini:maxi], 'g')
    #plt.title('Trajectory (Fuzzy jumps)')
    AX.set_xlabel('$n$', fontsize=fontS)
    AX.set_ylim(0., 1.05)
    plt.savefig('Traj_' + series + '_' + str(case) + '.png',bbox_inches='tight', dpi=dpi)

    # if not (ETA ==0. and DELTA==0.) :
        
    #     # Dessin de la pente a partir de la quelle on doit faire des tirages
    #     pente0 = pente_Serie2_gen(momtype=0, name='pente_Serie2', a=0., b=1., shapes="ETA, DELTA")
    #     rv = pente0(ETA, DELTA)
    #     #print(pente.pdf(0.54, ETA, DELTA))
    #     mean, var = plotSample(rv, 10000, 'pente0_' + series + '_'+str(case)+'.png', dpi=dpi)
    #     print('mean echantillon = ', mean)
    #     print('var echantillon = ', var)
    #     print(rv.stats('mvsk'))


    #     pente1 = pente_Serie2_gen(momtype=0, name='pente_Serie2', a=0., b=1., shapes="DELTA, ETA")
    #     rv = pente1(DELTA, ETA)
    #     #print(pente1.pdf(0.54, DELTA, ETA))
    #     mean, var = plotSample(rv, 10000, 'pente1_' + series + '_'+str(case)+'.png', dpi=dpi)
    #     print('mean echantillon = ', mean)
    #     print('var echantillon = ', var)
    #     print(rv.stats('mvsk'))

    #     # Dessin de la pente a partir de laquelle on doit faire des tirages
    #     parab = parabole_Serie2_gen(momtype=0, name='parabole_Serie2', a=0., b=1., shapes="ETA, DELTA")
    #     rv = parab(ETA, DELTA)
    #     #print(parab.pdf(0.54, ETA, DELTA))
    #     mean, var = plotSample(rv, 10000, 'parab_' + series + '_'+str(case)+'.png', dpi=dpi)
    #     print('mean echantillon = ', mean)
    #     print('var echantillon = ', var)
    #     print(rv.stats('mvsk'))

    #     triangle = triangle_Serie2_gen(momtype=0, name='triangle_Serie2', a=0., b=1., shapes="ETA, DELTA, r1")
    #     rv = triangle(ETA, DELTA, 0.3)
    #     #print(triangle.pdf(0.54, ETA, DELTA, 0.3))
    #     mean, var = plotSample(rv, 10000, 'triangle_' + series + '_'+str(case)+'.png', dpi=dpi)
    #     print('mean echantillon = ', mean)
    #     print('var echantillon = ', var)
    #     print(rv.stats('mvsk'))


########### SERIE 2 ##################
######################################
class LoiAPrioriSeries2(LoiAPriori):
    """
    Implementation of the second law described in the report Calcul_Simu_CGOFMSM.pdf
    """

    def __init__(self, alpha, eta, delta):
        """Constructeur to set the parameters of the density."""

        #print('(3.-7.*eta-8.*delta)/6.0=', (3.-7.*eta-8.*delta)/6.0)
        #assert  alpha <= (3.-7.*eta-8.*delta)/6.0, print('PB, you should set alpha to a maximum value of ', (3.-7.*eta-8.*delta)/6.0, ' which corresponds to beta = 0')

        self.__alpha = alpha
        self.__eta = eta
        self.__delta = delta
        self.__beta = (1. - 2.5 * (delta + eta) + 1. / 6. * (delta - eta)) / 2.0 - self.__alpha
        #print('self.__beta=', self.__beta)

        temp = 2. * (self.__alpha + self.__beta) + 2.5 * (self.__eta + self.__delta) - 1. / 6. * (self.__delta - self.__eta)
        assert np.abs(temp - 1.0)<1.E-6, print('Pb : this value should be 1 : ', temp)
        #print(' 2 alpha + 2 beta = ', 2.*self.__alpha+2.*self.__beta)
        #print('beta = ', self.__beta)

#        assert self.__alpha>=0. and self.__alpha<=1., print('PB : alpha=', self.__alpha)
#        assert self.__beta>=0.  and self.__beta<=1.,  print('PB : beta =', self.__beta)
#        assert self.__eta>=0.   and self.__eta<=1.,   print('PB : eta  =', self.__eta)
#        assert self.__delta>=0. and self.__delta<=1., print('PB : delta=', self.__delta)

        self.__D1 = self.__alpha + self.__beta + (self.__delta + self.__eta) / 2.

        # pour les tirages aléatoires
        pente_01 = pente_Serie2_gen(momtype=0, name='pente1_Serie2', a=0., b=1., shapes="eta, delta")
        self.__rv_pente0 = pente_01(self.__eta, self.__delta)
        # meme pente mais avec coeff échangés
        self.__rv_pente1 = pente_01(self.__delta, self.__eta)

        parab = parabole_Serie2_gen(momtype=0, name='parabole_Serie2', a=0., b=1., shapes="eta, delta")
        self.__rv_parab = parab(self.__eta, self.__delta)

        self.__rv_triangle = triangle_Serie2_gen(momtype=0, name='triangle_Serie2', a=0., b=1., shapes="eta, delta, r1")


    def getParam(self):
        """ Return the params of the law model."""
        return self.__alpha, self.__beta, self.__eta, self.__delta

    def __str__(self):
        return "alpha=" + str(self.__alpha) + ", beta=" + str(self.__beta) \
                + ", eta=" + str(self.__eta) + ", delta=" + str(self.__delta)

    def getTheoriticalHardTransition(self, n_r):

        Q10Th = self.__beta+3./8.*self.__eta+7./8.*self.__delta
        Q00Th = 1./2. - Q10Th

        JProba = np.zeros(shape=(2,2))
        JProba[0,0] = Q00Th
        JProba[0,1] = Q10Th
        JProba[1,0] = JProba[0,1]
        JProba[1,1] = JProba[0,0]
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

    def probaR1R2(self, r1, r2):
        """ Return the joint proba at r1, r2."""

        if (r1 == 0. and r2 == 0.) or (r1 == 1. and r2 == 1.):
            return self.__alpha

        if (r1 == 1. and r2 == 0.) or (r1 == 0. and r2 == 1.):
            return self.__beta

        return self.__eta + (self.__delta - self.__eta) * np.abs(r1 - r2)

    def probaR(self, r):
        """ Return the marginal proba at r."""

        if r == 0. or r == 1.:
            # Same value : (3.0 - 4.0 * self.__eta - 5. * self.__delta)/6.0
            return self.__alpha + self.__beta + (self.__eta + self.__delta) / 2.

        return (self.__delta - self.__eta) * (r * r - r) + 1.5 * (self.__eta + self.__delta)

    def probaR2CondR1(self, r1, r2, verbose=False):
        """ Return the conditional proba at r2 knowing r1."""

        if r1 == 0.0:
            if r2 == 0.0:
                return self.__alpha / self.__D1
            elif r2 == 1.0:
                return self.__beta / self.__D1
            else:
                return (self.__eta + (self.__delta - self.__eta) * r2) / self.__D1

        elif r1 == 1.0:
            if r2 == 0.0:
                return self.__beta / self.__D1
            elif r2 == 1.0:
                return self.__alpha / self.__D1
            else:
                return (self.__delta + (self.__eta - self.__delta) * r2) / self.__D1

        else:
            D2 = 1.5 * (self.__delta + self.__eta) + (self.__delta - self.__eta) * (r1 * r1 - r1)
            if r2 == 0.0:
                return (self.__eta + (self.__delta - self.__eta) * r1) / D2
            elif r2 == 1.0:
                return (self.__delta + (self.__eta - self.__delta) * r1) / D2
            else:
                return (self.__eta + (self.__delta - self.__eta) * np.abs(r1 - r2)) / D2


    def tirageR1(self):
        """ Return a draw according to the marginal density p(r1) """

        proba = np.zeros(shape=(3))
        proba[0] = self.__alpha + self.__beta + (self.__delta * self.__eta)/2.
        proba[1] = proba[0]
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

        if self.__eta == 0. and self.__delta == 0.:
            input('TO BE CHECKED')

            # cas particulier dune chaîne de markov dure à 2 états
            norm = self.__alpha + self.__beta
            if rn == 0.:
                probaR2CondR1 = [self.__alpha / norm, self.__beta / norm]
            else:
                probaR2CondR1 = [self.__beta / norm, self.__alpha / norm]
            # return np.random.choice(a=[0.0, 1.0], size=1, p=probaR2CondR1)[0]
            return random.choices(population=[0.0, 1.0], weights=probaR2CondR1)[0]

        else:

            proba = np.zeros(shape=(3))
            if rn == 0.:
                proba[0] = self.__alpha/self.__D1
                proba[1] = self.__beta /self.__D1
                proba[2] = 1. - (proba[0]+proba[1])
            elif rn == 1.:
                proba[0] = self.__beta /self.__D1
                proba[1] = self.__alpha/self.__D1
                proba[2] = 1. - (proba[0]+proba[1])
            else:
                D2       = 1.5*(self.__delta+self.__eta)+(self.__delta-self.__eta)*(rn*rn-rn)
                proba[0] = (self.__eta   + (self.__delta-self.__eta)*rn) / D2
                proba[1] = (self.__delta + (self.__eta-self.__delta)*rn) / D2
                proba[2] = 1. - (proba[0]+proba[1])
            
            # typeSample = np.random.choice(a=['0.', '1.', 'F'], size=1, p=proba)[0]
            typeSample = random.choices(population=['0.', '1.', 'F'], weights=proba)[0]
            
            if typeSample != 'F':
                rnp1 = float(typeSample)
            else:
                if rn == 0.:
                    rnp1 =  self.__rv_pente0.rvs()
                elif rn == 1.:
                    rnp1 = self.__rv_pente1.rvs()
                else:
                    rnp1 = self.__rv_triangle.rvs(self.__eta, self.__delta, rn)

            return rnp1

# depuis :
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html
class parabole_Serie2_gen(stats.rv_continuous):
    "Parabole de la série 2, lorsque r1 in ]0,1[ (eq 10)"

    def _pdf(self, x, eta, delta):
        K = 1.5 * (delta + eta)
        L = delta - eta
        return (K + L * x * (x - 1.)) / (K - L / 6.)

    def _cdf(self, y, eta, delta):
        K = 1.5 * (delta + eta)
        L = delta - eta
        return y / (K - L / 6.) * (L * y * y / 3. - L * y / 2. + K)

    def _stats(self, eta, delta):
        K = 1.5 * (delta + eta)
        L = delta - eta
        moment1 = 0.5
        moment2 = 1. / (K - L / 6.) * (K / 3. - L / 20.)
        return moment1, moment2 - moment1**2, 0., None

    def _argcheck(self, *args):
        # Aucune condition sur les paramètres (on renvoie que des 1)
        cond = 1
        for arg in args:
            cond = np.logical_and(cond, 1)
        return cond

class pente_Serie2_gen(stats.rv_continuous):
    "Pente de la série 2, lorsque r1==0 (eq 13)"

    def _pdf(self, x, eta, delta):
        # norm = 0.5 * (3. * eta - delta)
        # return (eta + (eta - delta) * x) / norm
        return (eta + (delta - eta) * x) * 2. / (eta + delta)

    # def _cdf(self, y, eta, delta):
    #     Ces calculs sont faux - mais non nécessaires
    #     ynorm = y/(eta + delta)
    #     return eta * ynorm + 0.5 * (delta - eta) * ynorm*ynorm

    def _argcheck(self, *args):
        # Aucune condition sur les paramètres (on renvoie que des 1)
        cond = 1
        for arg in args:
            cond = np.logical_and(cond, 1)
        return cond


class triangle_Serie2_gen(stats.rv_continuous):
    "triangle de la série 2, lorsque r2 in ]0,1[ (eq 15)"

    def _pdf(self, x, eta, delta, r1):
        #K = 1.5*(delta+eta)
        L = delta - eta
        return (eta + L * abs(x - r1)) / (eta + L * (r1 * r1 - r1 + 0.5))

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
