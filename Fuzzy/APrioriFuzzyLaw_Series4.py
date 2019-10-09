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

from Fuzzy.APrioriFuzzyLaw import LoiAPriori, echelle, plotSample
#from APrioriFuzzyLaw import LoiAPriori, echelle, plotSample

def main():
    discretization = 200

    seed = random.randrange(sys.maxsize)
    seed = 5039309497922655937
    rng = random.Random(seed)
    print("Seed was:", seed)
   

    print('*********************SERIES 4')
    series = 'Serie4'
    #P, case = LoiAPrioriSeries4(alpha=0.05, gamma = 0.0, delta_d=0.05, delta_u=0.05), 1
    #P, case = LoiAPrioriSeries4(alpha=0.05, gamma = 0.15, delta_d=0.3, delta_u=0.), 2
    #P, case = LoiAPrioriSeries4(alpha=0.15, gamma = 0.15, delta_d=0.0, delta_u=0.1), 3
    
    # CAS PROBELMATIQUE
    #M = 3.*(0.2+0.1) - 0.5*(0.2*0.2+0.1*0.1)
    #P, case = LoiAPrioriSeries4(alpha=0.2, gamma = (1.-4.*0.25)/M, delta_d=0.1, delta_u=0.2), 100
    #P, case = LoiAPrioriSeries4(alpha=0.2, gamma = 0.3, delta_d=0.05, delta_u=0.20), 11
    
    #P, case = LoiAPrioriSeries4(alpha=0.10, gamma = 0.55, delta_d=0.30, delta_u=0.30), 12  #--> 48%
    #P, case = LoiAPrioriSeries4(alpha=0.15, gamma = 0.65, delta_d=0.15, delta_u=0.15), 12 #--> 62%
    #P, case = LoiAPrioriSeries4(alpha=0.15, gamma = 0.65, delta_d=0.10, delta_u=0.10), 12 #--> 75%
    #P, case = LoiAPrioriSeries4(alpha=0.15, gamma = 0.55, delta_d=0.06, delta_u=0.06), 12 #--> 87%
    #P, case = LoiAPrioriSeries4(alpha=0.15, gamma = 0.65, delta_d=0.00, delta_u=0.00), 12 #--> 100%

    #P, case = LoiAPrioriSeries4(alpha=0.10, gamma = 0.55, delta_d=0.0, delta_u=0.1), 200
    #P, case = LoiAPrioriSeries4(alpha=0.15, gamma = 0.34, delta_d=0.2, delta_u=0.2), 200

    #P, case = LoiAPrioriSeries4(alpha=0.10, gamma = 0.65, delta_d=0.15, delta_u=0.15), 1

    # alpha = 0.15
    # delta_d=0.15
    # delta_u=0.15
    # M = 3.*(delta_u + delta_d) - 0.5*(delta_d*delta_d+delta_u*delta_u)
    # gamma = (1. - 2. * alpha)/M
    # P, case = LoiAPrioriSeries4(alpha=alpha, gamma = gamma, delta_d=delta_d, delta_u=delta_u), 57

    # P, case = LoiAPrioriSeries4(alpha=0.15, gamma = 0.65, delta_d=0., delta_u=0.), 58
    P, case = LoiAPrioriSeries4(alpha=0.15, gamma = 0.60, delta_d=0.20, delta_u=0.20), 57
    
    sum_R1R2 = P.sumR1R2(discretization)
    sum_R1   = P.sumR1(discretization)
    sum_R2CondR1_0   = P.sumR2CondR1(discretization, 0.)
    sum_R2CondR1_20  = P.sumR2CondR1(discretization, 0.10)
    sum_R2CondR1_50  = P.sumR2CondR1(discretization, 0.50)
    sum_R2CondR1_90  = P.sumR2CondR1(discretization, 0.90)
    sum_R2CondR1_100 = P.sumR2CondR1(discretization, 1.)
    ALPHA, BETA, DELTA_D, DELTA_U, GAMMA = P.getParam()

    print(P)
    print("sum_R1R2 = ", sum_R1R2)
    print("sum_R1 = ", sum_R1)
    print("sum_R2CondR1_0   = ", sum_R2CondR1_0)
    print("sum_R2CondR1_20  = ", sum_R2CondR1_20)
    print("sum_R2CondR1_50  = ", sum_R2CondR1_50)
    print("sum_R2CondR1_90  = ", sum_R2CondR1_90)
    print("sum_R2CondR1_100 = ", sum_R2CondR1_100)
    print('maxiHardJump = ', P.maxiHardJump())
    print('4:'+str(ALPHA)+':'+str(GAMMA)+':'+str(DELTA_D)+':'+str(DELTA_U)+', beta='+str(BETA)+', #pH='+str(P.maxiHardJump()))

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

    mini = 0
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

    # NORMTRAP = 1. - (2.*(ALPHA+BETA) + GAMMA*(DELTA_U+DELTA_D))
    # trapeze = trapeze_Serie4_gen(momtype=0, name='trapeze_serie4', a=0., b=1., shapes="NORMTRAP, GAMMA, DELTA_D, DELTA_U")
    # rv = trapeze(NORMTRAP, GAMMA, DELTA_D, DELTA_U)
    # print(trapeze.pdf(0.54, ALPHA, BETA, GAMMA, DELTA_D, DELTA_U))
    # mean, var = plotSample(rv, 10000, 'trapeze_' + series + '_'+str(case)+'.png')
    # print('mean echantillon = ', mean)
    # print('var echantillon = ', var)
    # print(rv.stats('mvsk'))


########### SERIE 4 ##################
######################################
class LoiAPrioriSeries4(LoiAPriori):
    """
    Implementation of the fourth law described in the report Calcul_Simu_CGOFMSM.pdf
    """

    def __init__(self, alpha, gamma, delta_d, delta_u):
        """Constructeur to set the parameters of the density."""

        if delta_d != delta_u:
            input('ATTENTION: LES CALCULS NE SONT PAS BONS LORSQUE DELTA_D >< DELTA_U !!!!!')

        self.__alpha = alpha

        assert delta_d >= 0. and delta_d <= 0.5, print('PB : delta_d=', delta_d)
        assert delta_u >= 0. and delta_u <= 0.5, print('PB : delta_u=', delta_u)
        self.__delta_d = delta_d
        self.__delta_u = delta_u

        M = 3.*(self.__delta_u+self.__delta_d) - 0.5*(self.__delta_d*self.__delta_d+self.__delta_u*self.__delta_u)

        if M != 0.:
            if gamma >= (1.-2.*self.__alpha)/M:
                self.__gamma = (1.-2.*self.__alpha)/M - 1E-10
            else:
                self.__gamma = gamma
        else:
            self.__gamma = 0.
            print('   ---> Attention : gamma == 0.')
            #input('pause')

        self.__beta = (1. - self.__gamma * M)/2. - self.__alpha

        NormTrap = 1. - (2.*(self.__alpha+self.__beta) + self.__gamma*(self.__delta_u+self.__delta_d))
        # NormTrap = self.__gamma*(2*(self.__delta_u+self.__delta_d) - 0.5*(self.__delta_u*self.__delta_u + self.__delta_d*self.__delta_d)  ) # equivalently
        gamma          = self.__gamma
        beta           = self.__beta
        trap_01        = trapeze_Serie4_gen(momtype=0, name='trapeze_serie4', a=0., b=1., shapes="NormTrap, gamma, delta_d, delta_u")
        self.__rv_trap = trap_01(NormTrap, self.__gamma, self.__delta_d, self.__delta_u)


    def getParam(self):
        """ Return the params of the law model."""
        return self.__alpha, self.__beta, self.__delta_d, self.__delta_u, self.__gamma

    def __str__(self):
        str1 = "alpha=" + str(self.__alpha) + ", beta=" + str(self.__beta)
        str1 += ", delta_d=" + str(self.__delta_d) + ", delta_u=" + str(self.__delta_u) + ", gamma=" + str(self.__gamma)
        return str1

    def getTheoriticalHardTransition(self, n_r):

        Q00Th = self.__alpha + (self.__delta_u + self.__delta_d + 1./4. - (1./2.-self.__delta_d)**2/2. - (1./2.-self.__delta_u)**2/2.) * self.__gamma
        Q10Th = self.__beta  + self.__delta_d*self.__delta_d * self.__gamma/2.
        Q01Th = self.__beta  + self.__delta_u*self.__delta_u * self.__gamma/2.
        
        JProba = np.zeros(shape=(2,2))
        JProba[0,0] = Q00Th
        JProba[0,1] = Q01Th
        JProba[1,0] = Q10Th
        JProba[1,1] = JProba[0,0]

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

        elif (r1 == 1. and r2 == 0.) or (r1 == 0. and r2 == 1.):
            return self.__beta

        elif (-self.__delta_d <=r2-r1) and (r2-r1<= self.__delta_u):
            return self.__gamma

        else:
            return 0.

    def probaR(self, r):
        """ Return the marginal proba at r."""

        if r == 0.:
            return self.__alpha + self.__beta + self.__delta_u * self.__gamma

        elif r == 1.:
            return self.__alpha + self.__beta + self.__delta_d * self.__gamma

        elif (r > 0.) and (r <= self.__delta_d):
            return self.__gamma * (self.__delta_u + r + 1.)

        elif (r >= 1. - self.__delta_u) and (r < 1.):
            return self.__gamma * (self.__delta_d + 2. - r)

        else:
            return self.__gamma * (self.__delta_d + self.__delta_u)


    def probaR2CondR1(self, r1, r2):
        """ Return the conditional proba at r2 knowing r1."""

        if (r1 > 0.) and (r1 <= self.__delta_d):
            if r2 >= 0. and r2 <= r1 + self.__delta_u:
                return 1. / (self.__delta_u + r1 + 1.)
            else:
                return 0.

        elif r1 >= (1. - self.__delta_u) and (r1 < 1.):
            if r2 >= r1 - self.__delta_d and r2 <= 1.:
                return 1. / (2. + self.__delta_d - r1)
            else:
                return 0.

        elif r1 == 0.0:
            temp_u = self.__alpha + self.__beta + self.__gamma * self.__delta_u
            if r2 > 0. and r2 <= self.__delta_u:
                return self.__gamma / temp_u
            elif r2 == 0.:
                return self.__alpha / temp_u
            elif r2 == 1.:
                return self.__beta / temp_u
            else:
                return 0.

        elif r1 == 1.0:
            temp_d = self.__alpha + self.__beta + self.__gamma * self.__delta_d
            if r2 >= 1. - self.__delta_d and r2 < 1.:
                return self.__gamma / temp_d
            elif r2 == 0.:
                return self.__beta / temp_d
            elif r2 == 1.:
                return self.__alpha / temp_d
            else:
                return 0.

        else:
            if (r2 >= r1 - self.__delta_d) and (r2 <= r1 + self.__delta_u):
                if self.__delta_d + self.__delta_u == 0. :
                    return 1E-100
                return 1. / (self.__delta_d + self.__delta_u)
            else:
                return 0.

    def tirageR1(self):
        """ Return a draw according to the marginal density p(r1) """

        proba    = np.zeros(shape=(3))
        proba[0] = self.__alpha + self.__beta +self.__gamma * self.__delta_u
        proba[1] = self.__alpha + self.__beta +self.__gamma * self.__delta_d
        proba[2] = 1. - (proba[0]+proba[1])
        #typeSample = np.random.choice(a=['0.', '1.', 'F'], size=1, p=proba)[0]
        typeSample = random.choices(population=['0.', '1.', 'F'], weights=proba)[0]

        if typeSample != 'F':
            r1 = float(typeSample)
        else:
            r1 = self.__rv_trap.rvs()

        return r1


    def tirageRnp1CondRn(self, rn):
        """ Return a draw according to the conditional density p(r2 | r1) """

        if rn == 0.:
            temp_u     = self.__alpha+self.__beta+self.__gamma*self.__delta_u
            proba      = [self.__alpha/temp_u, self.__beta/temp_u, 1.-(self.__alpha+self.__beta)/temp_u]
            #typeSample = np.random.choice(a=['0.', '1.', 'F'], size=1, p=proba)[0]
            typeSample = random.choices(population=['0.', '1.', 'F'], weights=proba)[0]
            if typeSample != 'F':
                rnp1 = float(typeSample)
            else:
                # rnp1 = echelle(np.random.random_sample(), 0., self.__delta_u)
                rnp1 = echelle(random.random(), 0., self.__delta_u)

        elif rn == 1.:
            temp_d = self.__alpha+self.__beta+self.__gamma*self.__delta_d
            proba      = [self.__beta/temp_d, self.__alpha/temp_d, 1.-(self.__alpha+self.__beta)/temp_d]
            #typeSample = np.random.choice(a=['0.', '1.', 'F'], size=1, p=proba)[0]
            typeSample = random.choices(population=['0.', '1.', 'F'], weights=proba)[0]
            if typeSample != 'F':
                rnp1 = float(typeSample)
            else:
                #rnp1 = echelle(np.random.random_sample(), 1. - self.__delta_d, 1.)
                rnp1 = echelle(random.random(), 1. - self.__delta_d, 1.)

        elif (rn > 0.) and (rn <= self.__delta_d):
            proba = [1./(self.__delta_u + rn + 1.), 1.-1./(self.__delta_u + rn + 1.)]
            #typeSample = np.random.choice(a=['dur', 'flou'], size=1, p=proba)[0]
            typeSample = random.choices(population=['dur', 'flou'], weights=proba)[0]
            if typeSample == 'dur':
                rnp1 = 0.
            else:
                # rnp1 = echelle(np.random.random_sample(), 0., rn+self.__delta_u)
                rnp1 = echelle(random.random(), 0., rn+self.__delta_u)

        elif (rn >= 1. - self.__delta_u) and (rn < 1.):
            proba = [1./(2. + self.__delta_d - rn), 1.-1./(2. + self.__delta_d - rn)]
            # typeSample = np.random.choice(a=['dur', 'flou'], size=1, p=proba)[0]
            typeSample = random.choices(population=['dur', 'flou'], weights=proba)[0]
            if typeSample == 'dur':
                rnp1 = 1.
            else:
                # rnp1 = echelle(np.random.random_sample(), rn - self.__delta_d, 1.)
                rnp1 = echelle(random.random(), rn - self.__delta_d, 1.)

        elif (rn > self.__delta_d) and (rn< 1.-self.__delta_u):
            #rnp1 = echelle(np.random.random_sample(), rn - self.__delta_d, rn + self.__delta_u)
            rnp1 = echelle(random.random(), rn - self.__delta_d, rn + self.__delta_u)

        else:
            input('impossible')
        #print('rn=', rn, ', rnp1=', rnp1)
        #input('Attente')

        return rnp1

class trapeze_Serie4_gen(stats.rv_continuous):
    "Pente de la série 4, pour la simulation selon la loi marginale p(r1)"

    def _pdf(self, x, Norm, gamma, delta_d, delta_u):
        if type(x)!=np.ndarray:
            if x > 0. and x <= delta_d:
                return gamma*(delta_u + x + 1.) / Norm
            if x>delta_d and x<1.-delta_u:
                return  gamma*(delta_u + delta_d) / Norm
            return gamma*(2. + delta_d - x) / Norm
        else:
            y = np.ndarray(shape=np.shape(x))
            for i in range(np.shape(x)[0]):
                if x[i] > 0. and x[i] <= delta_d[i]:
                    y[i] = gamma[i]*(delta_u[i] + x[i] + 1.) / Norm[i]
                elif x[i]>delta_d[i] and x[i]<1.-delta_u[i]:
                    y[i] = gamma[i]*(delta_u[i] + delta_d[i]) / Norm[i]
                else:
                    y[i] = gamma[i]*(2. + delta_d[i] - x[i]) / Norm[i]
            return y

    # def _cdf(self, y, alpha, beta, gamma, delta_d, delta_u):
    #     return 1.

    # def _argcheck(self, *args):
    #     # Aucune condition sur les paramètres (on renvoie que des 1)
    #     cond = 1
    #     for arg in args:
    #         cond = np.logical_and(cond, 1)
    #     return cond


if __name__ == '__main__':
    main()

 