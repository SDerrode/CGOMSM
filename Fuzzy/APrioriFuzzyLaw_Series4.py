#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 11:06:52 2017

@author: MacBook_Derrode
"""

import sys
import random
import numpy             as np
import scipy.stats       as stats
import matplotlib        as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection



fontS = 13 # fontSize
mpl.rc('xtick', labelsize=fontS)
mpl.rc('ytick', labelsize=fontS)
dpi = 150

if __name__ == '__main__':
    from APrioriFuzzyLaw import LoiAPriori, plotSample, echelle
else:
    from Fuzzy.APrioriFuzzyLaw import LoiAPriori, plotSample, echelle

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

    print('*********************SERIES 4')
    series = 'Serie4'
    #P, case = LoiAPrioriSeries4(alpha=0.05, gamma = 0.0, delta=0.05, EPS=EPS, discretization=discretization), 1
    #P, case = LoiAPrioriSeries4(alpha=0.05, gamma = 0.15, delta=0.3, EPS=EPS, discretization=discretization), 2
    #P, case = LoiAPrioriSeries4(alpha=0.15, gamma = 0.15, delta=0.0, EPS=EPS, discretization=discretization), 3
    
    # CAS PROBELMATIQUE
    #M = 3.*(0.2+0.1) - 0.5*(0.2*0.2+0.1*0.1)
    #P, case = LoiAPrioriSeries4(alpha=0.2, gamma = (1.-4.*0.25)/M, delta=0.1, EPS=EPS, discretization=discretization), 100
    #P, case = LoiAPrioriSeries4(alpha=0.2, gamma = 0.3, delta=0.05, EPS=EPS, discretization=discretization), 11
    
    #P, case = LoiAPrioriSeries4(alpha=0.10, gamma = 0.55, delta=0.30, EPS=EPS, discretization=discretization), 12  #--> 48%
    #P, case = LoiAPrioriSeries4(alpha=0.15, gamma = 0.65, delta=0.15, EPS=EPS, discretization=discretization), 12 #--> 62%
    #P, case = LoiAPrioriSeries4(alpha=0.15, gamma = 0.65, delta=0.10, EPS=EPS, discretization=discretization), 12 #--> 75%
    #P, case = LoiAPrioriSeries4(alpha=0.15, gamma = 0.55, delta=0.06, EPS=EPS, discretization=discretization), 12 #--> 87%
    #P, case = LoiAPrioriSeries4(alpha=0.15, gamma = 0.65, delta=0.00, EPS=EPS, discretization=discretization), 12 #--> 100%

    #P, case = LoiAPrioriSeries4(alpha=0.10, gamma = 0.55, delta=0.0, EPS=EPS, discretization=discretization), 200
    #P, case = LoiAPrioriSeries4(alpha=0.15, gamma = 0.34, delta=0.2, EPS=EPS, discretization=discretization), 200

    #P, case = LoiAPrioriSeries4(alpha=0.10, gamma = 0.65, delta=0.15, EPS=EPS, discretization=discretization), 1

    # alpha = 0.15
    # delta=0.15
    # M = 6.*(delta) - (delta*delta)
    # gamma = (1. - 2. * alpha)/M
    # P, case = LoiAPrioriSeries4(alpha=alpha, gamma = gamma, delta=delta, EPS=EPS, discretization=discretization), 57

    # P, case = LoiAPrioriSeries4(alpha=0.15, gamma = 0.65, delta=0., EPS=EPS, discretization=discretization), 58
    P, case = LoiAPrioriSeries4(alpha=0.15, gamma = 0.30, delta=0.30, EPS=EPS, discretization=discretization), 57

    print(P)
    print('model string', P.stringName())

    # Test le modele
    OKtestModel = P.testModel(verbose=verbose, epsilon=epsilon)

    # Simulation d'un chaine de markov flou suivant ce modèle
    N = 10000
    chain = P.testSimulMC(N, verbose=verbose, epsilon=epsilon)

    if graphics == True:
        P.plotR1R2   ('./figures/LoiCouple_'   + series + '_' + str(case) + '.png', dpi=dpi)
        P.plotR1R2th ('./figures/LoiCoupleTh_' + series + '_' + str(case) + '.png', dpi=dpi)
        P.plotR1     ('./figures/LoiMarg_'     + series + '_' + str(case) + '.png', dpi=dpi)
        # Dessins
        mini, maxi = 100, 150
        P.PlotMCchain('./figures/Traj_'      + series + '_' + str(case) + '.png', chain, mini=mini, maxi=maxi, dpi=dpi)


    # NORMTRAP = 1. - (2.*(ALPHA+BETA) + 2.*GAMMA*DELTA)
    # trapeze = trapeze_Serie4_gen(momtype=0, name='trapeze_serie4', a=0., b=1., shapes="NORMTRAP, GAMMA, DELTA")
    # rv = trapeze(NORMTRAP, GAMMA, DELTA)
    # print(trapeze.pdf(0.54, ALPHA, BETA, GAMMA, DELTA))
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

    def __init__(self, alpha, gamma, delta, EPS=1E-8, discretization=100):
        """Constructeur to set the parameters of the density."""

        LoiAPriori.__init__(self, EPS=EPS, discretization=discretization)

        self.__alpha = alpha

        assert delta >= 0. and delta <= 0.5, print('PB : delta=', delta)
        self.__delta = delta
        
        M = self.__delta * (6. -self.__delta)
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

        self.update()

    def update(self):
        NormTrap       = 1. - (2.*(self.__alpha+self.__beta) + 2.*self.__gamma*self.__delta)
        gamma          = self.__gamma
        beta           = self.__beta
        trap_01        = trapeze_Serie4_gen(momtype=0, name='trapeze_serie4', a=0., b=1., shapes="NormTrap, gamma, delta")
        self.__rv_trap = trap_01(NormTrap, self.__gamma, self.__delta)

    def setParametersFromSimul(self, Rsimul, nbcl):
        
        input('setParametersFromSimul : to be done')
        Nsimul = len(Rsimul)
        self.update()

    def getParam(self):
        """ Return the params of the law model."""
        return self.__alpha, self.__beta, self.__delta, self.__gamma

    def __str__(self):
        str1 = "alpha=" + str(self.__alpha) + ", beta=" + str(self.__beta)
        str1 += ", delta=" + str(self.__delta) + ", gamma=" + str(self.__gamma)
        return str1

    def stringName(self):
        return '4:'+str('%.4f'%self.__alpha)+':'+str('%.4f'%self.__gamma)+':'+str('%.4f'%self.__delta)

    def getTheoriticalHardTransition(self, n_r):

        Q00Th = self.__alpha + (2.*self.__delta + 1./4. - (1./2.-self.__delta)**2) * self.__gamma
        Q10Th = self.__beta  + self.__delta*self.__delta * self.__gamma/2.
        Q01Th = self.__beta  + self.__delta*self.__delta * self.__gamma/2.
        
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

        if (r1 == 1. and r2 == 0.) or (r1 == 0. and r2 == 1.):
            return self.__beta

        if (-self.__delta <=r2-r1) and (r2-r1<= self.__delta):
            return self.__gamma

        return 0.

    def probaR(self, r):
        """ Return the marginal proba at r."""

        if r == 0. or r == 1.:
            return self.__alpha + self.__beta + self.__delta * self.__gamma

        elif (r > 0.) and (r <= self.__delta):
            return self.__gamma * (self.__delta + r + 1.)

        elif (r >= 1. - self.__delta) and (r < 1.):
            return self.__gamma * (self.__delta + 2. - r)

        else:
            return 2.*self.__gamma * self.__delta


    def probaR2CondR1(self, r1, r2):
        """ Return the conditional proba at r2 knowing r1."""

        if (r1 > 0.) and (r1 <= self.__delta):
            if r2 >= 0. and r2 <= r1 + self.__delta:
                return 1. / (self.__delta + r1 + 1.)
            else:
                return 0.

        elif r1 >= (1. - self.__delta) and (r1 < 1.):
            if r2 >= r1 - self.__delta and r2 <= 1.:
                return 1. / (2. + self.__delta - r1)
            else:
                return 0.

        elif r1 == 0.0:
            temp_d = self.__alpha + self.__beta + self.__gamma * self.__delta
            if r2 > 0. and r2 <= self.__delta:
                return self.__gamma / temp_d
            elif r2 == 0.:
                return self.__alpha / temp_d
            elif r2 == 1.:
                return self.__beta / temp_d
            else:
                return 0.

        elif r1 == 1.0:
            temp_d = self.__alpha + self.__beta + self.__gamma * self.__delta
            if r2 >= 1. - self.__delta and r2 < 1.:
                return self.__gamma / temp_d
            elif r2 == 0.:
                return self.__beta / temp_d
            elif r2 == 1.:
                return self.__alpha / temp_d
            else:
                return 0.

        else:
            if (r2 >= r1 - self.__delta) and (r2 <= r1 + self.__delta):
                if self.__delta + self.__delta == 0. :
                    return 1E-100
                return 1. / (2.*self.__delta)
            else:
                return 0.

    def tirageR1(self):
        """ Return a draw according to the marginal density p(r1) """

        proba    = np.zeros(shape=(3))
        proba[0] = self.__alpha + self.__beta +self.__gamma * self.__delta
        proba[1] = proba[0]
        proba[2] = 1. - (proba[0]+proba[1])
        typeSample = random.choices(population=['0.', '1.', 'F'], weights=proba)[0]

        if typeSample != 'F':
            r1 = float(typeSample)
        else:
            r1 = self.__rv_trap.rvs()

        return r1


    def tirageRnp1CondRn(self, rn):
        """ Return a draw according to the conditional density p(r2 | r1) """

        if rn == 0.:
            temp_d = self.__alpha+self.__beta+self.__gamma*self.__delta
            proba  = [self.__alpha/temp_d, self.__beta/temp_d, 1.-(self.__alpha+self.__beta)/temp_d]
            typeSample = random.choices(population=['0.', '1.', 'F'], weights=proba)[0]
            if typeSample != 'F':
                rnp1 = float(typeSample)
            else:
                rnp1 = echelle(random.random(), 0., self.__delta)

        elif rn == 1.:
            temp_d = self.__alpha+self.__beta+self.__gamma*self.__delta
            proba  = [self.__beta/temp_d, self.__alpha/temp_d, 1.-(self.__alpha+self.__beta)/temp_d]
            typeSample = random.choices(population=['0.', '1.', 'F'], weights=proba)[0]
            if typeSample != 'F':
                rnp1 = float(typeSample)
            else:
                #rnp1 = echelle(np.random.random_sample(), 1. - self.__delta, 1.)
                rnp1 = echelle(random.random(), 1. - self.__delta, 1.)

        elif (rn > 0.) and (rn <= self.__delta):
            proba = [1./(self.__delta + rn + 1.), 1.-1./(self.__delta + rn + 1.)]
            typeSample = random.choices(population=['dur', 'flou'], weights=proba)[0]
            if typeSample == 'dur':
                rnp1 = 0.
            else:
                rnp1 = echelle(random.random(), 0., rn+self.__delta)

        elif (rn >= 1. - self.__delta) and (rn < 1.):
            proba = [1./(2. + self.__delta - rn), 1.-1./(2. + self.__delta - rn)]
            typeSample = random.choices(population=['dur', 'flou'], weights=proba)[0]
            if typeSample == 'dur':
                rnp1 = 1.
            else:
                # rnp1 = echelle(np.random.random_sample(), rn - self.__delta, 1.)
                rnp1 = echelle(random.random(), rn - self.__delta, 1.)

        elif (rn > self.__delta) and (rn< 1.-self.__delta):
            rnp1 = echelle(random.random(), rn - self.__delta, rn + self.__delta)

        else:
            input('impossible')

        return rnp1


    def plotR1R2th(self, filename, dpi=150):
        """
        Plot of the joint density p(r1, r2)
        """

        d = self.__delta
        g = self.__gamma
        a = self.__alpha
        b = self.__beta
        p = self.probaR1R2
        E = self._EPS

        r1 = [0, 0,      1, 1,      0, 0,      1, 1]
        r2 = [0, 0,      1, 1,      1, 1,      0, 0]
        z  = [0, p(0,0), 0, p(1,1), 0, p(0,1), 0, p(1,0)]
        tupleListMasses = list(zip(r1, r2, z))

        r1 = [E, E,       d,       d,      1,      1,         1,        1,       1-E, 1-E,         1-d,      1-d,     0,  0,     0,      0]
        r2 = [0, 0,       0,       0,      1-E,    1-E,       1-d,      1-d,       1,   1,         1,          1,     E,  E,     d,      d]
        z  = [0, p(E, 0), p(d, 0), 0,      0, p(1, 1-E), p(1,1-d), 0,         0,   p(1-E, 1), p(1-d, 1),  0,     0, p(0,E), p(0,d), 0]
        tupleListBord = list(zip(r1, r2, z))

        r1 = [E,      d+E,      d+E, 1-E, 1-E,            1,    1, 1.-d, 0, 0, 0, 1.-d]
        r2 = [E,      E,        E,   E,   1-d-2*E,        1.-d, 1, 1,    d, d, 1, 1,  ]
        z  = [p(E,E), p(d+E,E), 0,   0,   p(1-E,1-d-2*E), g,    g, g,    g, 0, 0, 0,  ]
        
        tupleList = list(zip(r1, r2, z))

        edge_color  = 'xkcd:dark blue'
        colorMasses = 'xkcd:orange'
        colorBords  = 'xkcd:turquoise' 
        colorInside = 'xkcd:sky blue'

        fig = plt.figure()
        ax  = fig.add_subplot(1, 1, 1, projection='3d')

        # on ajoute la masse derriere ############
        #####################################
        verticesBarreDerriere = [[2, 3]]
        barres3D = []
        for ix in range(len(verticesBarreDerriere)):
            Liste=[]
            for iy in range(len(verticesBarreDerriere[ix])):
                Liste.append(tupleListMasses[verticesBarreDerriere[ix][iy]])
            barres3D.append(Liste)
        # print('barres3D=', barres3D)

        collection=Poly3DCollection(barres3D, linewidths=4, alpha=1., zsort='max', zorder=1)
        collection.set_facecolor(colorMasses)
        collection.set_edgecolor(colorMasses)
        ax.add_collection3d(collection)

        # on ajoute les 2 bords derriere ############
        #####################################
        verticesBordDerriere = [[4, 5, 6, 7, 4], [8, 9, 10, 11, 8]]
        bords3D = []
        for ix in range(len(verticesBordDerriere)):
            Liste=[]
            for iy in range(len(verticesBordDerriere[ix])):
                Liste.append(tupleListBord[verticesBordDerriere[ix][iy]])
            bords3D.append(Liste)
        # print('bords3D=', bords3D)

        collection=Poly3DCollection(bords3D, linewidths=2, alpha=0.9, zsort='max', zorder=2)
        collection.set_facecolor(colorBords)
        #collection.set_edgecolor(edge_color)
        ax.add_collection3d(collection)
        

        # on ajoute l'intérieur #################
        #########################################
        # vertices = [[2, 3, 4, 2], [1, 2, 4, 5, 1], [0, 1, 5, 6, 7, 8, 0], [7, 8, 9, 11, 7], [9, 10, 11, 9]]
        vertices = [[9, 10, 11, 9], [7, 8, 9, 11, 7], [2, 3, 4, 2], [1, 2, 4, 5, 1], [0, 1, 5, 6, 7, 8, 0]]
        poly3d = []
        for ix in range(len(vertices)):
            Liste=[]
            for iy in range(len(vertices[ix])):
                Liste.append(tupleList[vertices[ix][iy]])
            poly3d.append(Liste)
        # print('poly3d=', poly3d)

        collection=Poly3DCollection(poly3d, linewidths=2, alpha=1., zsort='max', zorder=3)
        collection.set_facecolor(colorInside)
        collection.set_edgecolor(edge_color)
        ax.add_collection3d(collection)

        # on ajoute les 2 bords devant ############
        #####################################
        verticesBordDevant = [[0, 1, 2, 3, 0], [12, 13, 14, 15, 12]]
        bords3D = []
        for ix in range(len(verticesBordDevant)):
            Liste=[]
            for iy in range(len(verticesBordDevant[ix])):
                Liste.append(tupleListBord[verticesBordDevant[ix][iy]])
            bords3D.append(Liste)
        # print('bords3D=', bords3D)

        collection=Poly3DCollection(bords3D, linewidths=2, alpha=0.9, zsort='max', zorder=4)
        collection.set_facecolor(colorBords)
        #collection.set_edgecolor(edge_color)
        ax.add_collection3d(collection)

        # on ajoute les 3 masses devant ############
        #####################################
        verticesBarreDevant = [[0, 1], [4, 5], [6, 7]]
        barres3D = []
        for ix in range(len(verticesBarreDevant)):
            Liste=[]
            for iy in range(len(verticesBarreDevant[ix])):
                Liste.append(tupleListMasses[verticesBarreDevant[ix][iy]])
            barres3D.append(Liste)
        # print('barres3D=', barres3D)

        collection=Poly3DCollection(barres3D, linewidths=4, alpha=1, zsort='max', zorder=5)
        collection.set_facecolor(colorMasses)
        collection.set_edgecolor(colorMasses)
        ax.add_collection3d(collection)
        

        ############# Dessin des axes
        ax.set_xlabel('$r_1$', fontsize=16)
        # #ax.set_xlim(-0.02, 1.02)
        ax.set_ylabel('$r_2$', fontsize=16)
        # #ax.set_ylim(-0.02, 1.02)
        # #ax.set_zlabel('$p(r_1,r_2)$', fontsize=fontS)
        ax.set_zlim(0., g*1.02)
        ax.view_init(20, 238)

        # plt.show()
        if filename != None:
            plt.savefig(filename, bbox_inches='tight', dpi=dpi)
        plt.close()

   






class trapeze_Serie4_gen(stats.rv_continuous):
    "Pente de la série 4, pour la simulation selon la loi marginale p(r1)"

    def _pdf(self, x, Norm, gamma, delta):
        if type(x)!=np.ndarray:
            if x > 0. and x <= delta:
                return gamma*(delta + x + 1.) / Norm
            if x>delta and x<1.-delta:
                return  2.*gamma*delta / Norm
            return gamma*(2. + delta - x) / Norm
        else:
            y = np.ndarray(shape=np.shape(x))
            for i in range(np.shape(x)[0]):
                if x[i] > 0. and x[i] <= delta[i]:
                    y[i] = gamma[i]*(delta[i] + x[i] + 1.) / Norm[i]
                elif x[i]>delta[i] and x[i]<1.-delta[i]:
                    y[i] = 2.*gamma[i]*delta[i] / Norm[i]
                else:
                    y[i] = gamma[i]*(2. + delta[i] - x[i]) / Norm[i]
            return y

    # def _cdf(self, y, alpha, beta, gamma, delta):
    #     return 1.

    # def _argcheck(self, *args):
    #     # Aucune condition sur les paramètres (on renvoie que des 1)
    #     cond = 1
    #     for arg in args:
    #         cond = np.logical_and(cond, 1)
    #     return cond



if __name__ == '__main__':
    main()

 