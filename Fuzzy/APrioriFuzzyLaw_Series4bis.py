##!/usr/bin/env python3
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

    print('*********************SERIES 4 Extended')
    series = 'Serie4bis'
    #P, case = LoiAPrioriSeries4bis(alpha=0.10, beta = 0.10, delta=0.15, lamb=0.5, EPS=EPS, discretization=discretization), 1
    #P, case = LoiAPrioriSeries4bis(alpha=0.10, beta = 0.10, delta=0.15, lamb=0.10, EPS=EPS, discretization=discretization), 2
    P, case = LoiAPrioriSeries4bis(alpha=0.05, beta = 0.05, delta=0.10, lamb=0.001, EPS=EPS, discretization=discretization), 3
    
    print(P)
    print('model string', P.stringName())

    # Test le modele
    OKtestModel = P.testModel(verbose=verbose, epsilon=epsilon)

    # Simulation d'un chaine de markov flou suivant ce modèle
    N = 100000
    chain = P.testSimulMC(N, verbose=verbose, epsilon=epsilon)

    if graphics == True:
        P.plotR1R2   ('./figures/LoiCouple_' + series + '_' + str(case) + '.png', dpi=dpi)
        P.plotR1R2th ('./figures/LoiCoupleTh_' + series + '_' + str(case) + '.png', dpi=dpi)
        P.plotR1     ('./figures/LoiMarg_'   + series + '_' + str(case) + '.png', dpi=dpi)
        # Dessins
        mini, maxi = 100, 4000
        P.PlotMCchain('./figures/Traj_'      + series + '_' + str(case) + '.png', chain, mini=mini, maxi=maxi, dpi=dpi)


    # # Dessin de la pente a partir de la quelle on doit faire des tirages
    # pente = pente_Serie4bis_gen(momtype=0, name='pente_Serie4bis', a=0., b=1., shapes="DELTA")
    # rv = pente(DELTA)
    # #print(pente.pdf(0.54, DELTA))
    # mean, var = plotSample(rv, 1000, 'pente_' + series + '_'+str(case)+'.png')
    # print('mean echantillon = ', mean)
    # print('var echantillon = ', var)
    # print(rv.stats('mvsk'))

    # # Dessin de la pente a partir de la quelle on doit faire des tirages
    # pente2 = pente2_Serie4bis_gen(momtype=0, name='pente2_Serie4bis', a=0., b=1., shapes="DELTA")
    # rv = pente2(DELTA)
    # #print(pente2.pdf(0.54, DELTA))
    # mean, var = plotSample(rv, 1000, 'pente2_' + series + '_'+str(case)+'.png')
    # print('mean echantillon = ', mean)
    # print('var echantillon = ', var)
    # print(rv.stats('mvsk'))

    # trapeze = trapeze_Serie4bis_gen(momtype=0, name='trapeze_Serie4bis', a=0., b=1., shapes="ALPHA, BETA, GAMMA, DELTA, LAMB")
    # rv = trapeze(ALPHA, BETA, GAMMA, DELTA, LAMB)
    # #print(trapeze.pdf(0.54, ALPHA, BETA, GAMMA, DELTA, LAMB))
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

    def __init__(self, alpha, beta, delta, lamb, EPS=1E-8, discretization=100):
        """Constructeur to set the parameters of the density."""

        LoiAPriori.__init__(self, EPS=EPS, discretization=discretization)

        self.__alpha = alpha
        self.__beta = beta

        assert delta >= 0. and delta <= 0.5, print('PB : delta=', delta)
        self.__delta = delta
        assert lamb >= 0.
        self.__lamb = lamb

        self.update()

    def update(self):

        M = 2.*self.__delta *(self.__lamb+1.) - self.__delta*self.__delta
        self.__gamma = (1 - 2. *(self.__beta+self.__alpha)) / M

        # pour les tirages aléatoires
        pente_01 = pente_Serie4bis_gen(momtype=0, name='pente1_Serie4bis', a=0., b=1., shapes="delta")
        self.__rv_pente0 = pente_01(self.__delta)
        pente2_01 = pente2_Serie4bis_gen(momtype=0, name='pente2_Serie4bis', a=0., b=1., shapes="delta")
        self.__rv_pente1 = pente2_01(self.__delta)
        
        # print(self.maxiHardJump())
        # print(self.maxiFuzzyJump())
        # print(2.*(self.__alpha+self.__beta) + self.__lamb * self.__gamma / 2.*2.*self.__delta)
        # print(self.__gamma*( (self.__lamb/2.+1.) * (2.*self.__delta) - self.__delta*self.__delta))
        # input('Perfecto!')

    def setParametersFromSimul(self, Rsimul, nbcl):
        
        input('setParametersFromSimul : to be done')
        Nsimul = len(Rsimul)

        self.update()
        
    def getParam(self):
        """ Return the params of the law model."""
        return self.__alpha, self.__beta, self.__delta, self.__gamma, self.__lamb

    def __str__(self):
        str1 = "alpha=" + str(self.__alpha) + ", beta=" + str(self.__beta)
        str1 += ", delta=" + str(self.__delta) + ", gamma=" + str(self.__gamma)
        str1 += ", lambda=" + str(self.__lamb)
        return str1

    def stringName(self):
        return '4bis:'+str('%.4f'%self.__alpha)+':'+str('%.4f'%self.__beta)+':'+str('%.4f'%self.__delta)+':'+str('%.4f'%self.__lamb)

    def probaR1R2(self, r1, r2):
        """ Return the joint proba at r1, r2."""

        # Les masses aux coins
        if (r1 == 0. and r2 == 0.) or (r1 == 1. and r2 == 1.):
            return self.__alpha
        if (r1 == 1. and r2 == 0.) or (r1 == 0. and r2 == 1.):
            return self.__beta

        # les bords
        if r2 == 0. and r1 <= self.__delta:
            return self.__lamb * self.__gamma*(1.-r1/self.__delta)
        if r2 == 1. and r1 >= 1.-self.__delta:
            return self.__lamb * self.__gamma*(r1/self.__delta + (1.-1./self.__delta))
        if r1 == 0. and r2 <= self.__delta:
            return self.__lamb * self.__gamma*(1.-r2/self.__delta)
        if r1 == 1. and r2 >= 1.-self.__delta:
            return self.__lamb * self.__gamma*(r2/self.__delta + (1.-1./self.__delta))

        # Le coeurs
        if (-self.__delta<=r2-r1) and (r2-r1<=self.__delta):
            return self.__gamma

        return 0.


    def probaR(self, r):
        """ Return the marginal proba at r."""

        if r == 0. or r == 1.:
            return self.__alpha + self.__beta + self.__delta * self.__gamma * self.__lamb / 2.

        if (r > 0.) and (r <= self.__delta):
            return self.__gamma * ( (1.-self.__lamb / self.__delta) * r + self.__lamb + self.__delta)

        if (r >= 1. - self.__delta) and (r < 1.):
            return self.__gamma * ( (self.__lamb / self.__delta - 1.) * r + self.__lamb * (1. - 1./self.__delta) + 1. + self.__delta)

        return 2.*self.__gamma * self.__delta 

    def probaR2CondR1(self, r1, r2):
        """ Return the conditional proba at r2 knowing r1."""

        if (r1 > 0.) and (r1 <= self.__delta):
            temp = (1.-self.__lamb / self.__delta)*r1 + self.__lamb + self.__delta
            if r2 == 0.:
                return self.__lamb * (1. - r1/self.__delta)/ temp
            if r2 <= r1 + self.__delta:
                return 1. / temp
            else:
                return 0.

        elif (r1 <1.) and (r1 >= 1-self.__delta):
            temp = (self.__lamb / self.__delta - 1.)*r1 + self.__lamb * (1. - 1./self.__delta) + 1. + self.__delta
            if r2 == 1.:
                return self.__lamb * (r1/self.__delta + 1. - 1./self.__delta)/ temp
            if r2 >= r1 - self.__delta:
                return 1. / temp
            else:
                return 0.

        elif r1 == 0.0:
            temp_u = self.__alpha + self.__beta + self.__gamma * self.__lamb / 2. * self.__delta
            if r2 > 0 and r2 <= self.__delta:
                return self.__lamb * self.__gamma * (1. - r2/self.__delta) / temp_u
            elif r2 == 0.:
                return self.__alpha / temp_u
            elif r2 == 1.:
                return self.__beta / temp_u
            else:
                return 0.

        elif r1 == 1.0:
            temp_d = self.__alpha + self.__beta + self.__gamma * self.__lamb / 2. * self.__delta
            if r2 >= 1. - self.__delta and r2 < 1.:
                return self.__lamb * self.__gamma * ( (r2/self.__delta) + (1. - 1./self.__delta)) / temp_d
            elif r2 == 0.:
                return self.__beta / temp_d
            elif r2 == 1.:
                return self.__alpha / temp_d
            else:
                return 0.

        else:
            if r2 >= r1 - self.__delta and r2 <= r1 + self.__delta:
                return 1. / (2.*self.__delta )
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
                #r1 = self.__rv_trapeze.rvs(self.__alpha, self.__beta, self.__gamma, self.__delta)
                r1 = np.random.random_sample()
        return r1

    def tirageRnp1CondRn(self, rn):
        """ Return a draw according to the conditional density p(r2 | r1) """

        if rn == 0.0:
            temp_d = self.__alpha+self.__beta+self.__gamma*self.__lamb/2.*self.__delta
            proba = [(self.__alpha+self.__beta)/temp_d, 1.-(self.__alpha+self.__beta)/temp_d]
            typeSample = np.random.choice(a=['dur', 'flou'], size=1, p=proba)[0]
            if typeSample == 'dur':
                sumi = self.__alpha/temp_d + self.__beta/temp_d
                proba = [self.__alpha/(temp_d*sumi), self.__beta/(temp_d*sumi)]
                rnp1 = np.random.choice(a=[0.0, 1.0], size=1, p=proba)[0]
            else:
                rnp1 = self.__rv_pente0.rvs()

        elif rn == 1.0:
            temp_d = self.__alpha+self.__beta+self.__gamma*self.__lamb/2.*self.__delta
            proba = [(self.__alpha+self.__beta)/temp_d, 1.-(self.__alpha+self.__beta)/temp_d]
            typeSample = np.random.choice(a=['dur', 'flou'], size=1, p=proba)[0]
            if typeSample == 'dur':
                sumi = self.__beta + self.__alpha
                proba = [self.__beta/sumi, self.__alpha/sumi]
                rnp1 = np.random.choice(a=[0.0, 1.0], size=1, p=proba)[0]
            else:
                rnp1 = self.__rv_pente1.rvs()

        elif (rn > 0.) and (rn <= self.__delta):
            temp = (1.-self.__lamb / self.__delta)*rn + self.__lamb + self.__delta
            proba = [self.__lamb*(1.-rn/self.__delta)/temp, 1.-self.__lamb*(1.-rn/self.__delta)/temp]
            typeSample = np.random.choice(a=['dur', 'flou'], size=1, p=proba)[0]
            if typeSample == 'dur':
                rnp1 = 0.
            else:
                rnp1 = echelle(np.random.random_sample(), 0, rn+self.__delta)

        elif (rn >= 1. - self.__delta) and (rn < 1.):
            temp = (self.__lamb / self.__delta - 1.)*rn + self.__lamb * (1. - 1./self.__delta) + 1. + self.__delta
            proba = [self.__lamb*(rn/self.__delta +1. - 1./self.__delta)/temp, 1.-self.__lamb*(rn/self.__delta +1. - 1./self.__delta)/temp]
            typeSample = np.random.choice(a=['dur', 'flou'], size=1, p=proba)[0]
            if typeSample == 'dur':
                rnp1 = 1.
            else:
                rnp1 = echelle(np.random.random_sample(), rn - self.__delta, 1.)

        else:
            rnp1 = echelle(np.random.random_sample(), rn - self.__delta, rn + self.__delta)
            
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



class pente_Serie4bis_gen(stats.rv_continuous):
    "Pente de la serie 4 bis, lorsque r1==0 "

    def _pdf(self, x, delta):
        if x.size == 1:
            if x>0. and x<=delta:
                K = 2. / delta
                return K*(1.-x/delta)
            else:
                return 0.

        solution = np.zeros(x.shape)
        for i,val in enumerate(x):
            delta_i = delta[i]
            if val>0. and val<=delta_i:
                K = 2. / delta_i
                solution[i] = K*(1.-val/delta_i)
            else:
                solution[i] = 0.
        return solution

    def _cdf(self, x, delta):
        if x.size == 1:
            if x>0. and x<=delta:
                return x/delta*(2.-x/delta)
            else:
                return 1.

        solution = np.zeros(x.shape)
        for i,val in enumerate(x):
            delta_i = delta[i]
            if val>0. and val<=delta_i:
                solution[i] = val/delta_i*(2.-val/delta_i)
            else:
                solution[i] = 1.
        return solution

    def _stats(self, delta):
        moment1 = 1./3. * delta
        moment2 = 1./6. *  delta * delta
        return moment1, moment2 - moment1**2, None, None

    def _argcheck(self, *args):
        # Aucune condition sur les paramètres (on renvoie que des 1)
        cond = 1
        for arg in args:
            cond = np.logical_and(cond, 1)
        return cond

class pente2_Serie4bis_gen(stats.rv_continuous):
    "Pente de la série 4 bis, lorsque r1==1"

    def _pdf(self, x, delta):
        if x.size == 1:
            if x>=1-delta and x<1.:
                K = 2. / delta
                Z = 1. - 1./delta
                return K*(x/delta + Z)
            else:
                return 0.

        solution = np.zeros(x.shape)
        for i,val in enumerate(x):
            delta_i = delta[i]
            if val>=1-delta_i and val<1.:
                K = 2. / delta_i
                Z = 1. - 1./delta_i
                solution[i] = K*(val/delta_i + Z)
            else:
                solution[i] = 0.
        return solution

    def _cdf(self, x, delta):
        if x.size == 1:
            if x>=1-delta and x<1.:
                K = 2. / delta
                Z = 1. - 1./delta
                return K*(Z*(x-1.+delta) + (x*x-1.-delta*delta+2. *delta)/(2. * delta))
            else:
                return 0.

        solution = np.zeros(x.shape)
        for i,val in enumerate(x):
            delta_i = delta[i]
            if val>=1-delta_i and val<1.:
                K = 2. / delta_i
                Z = 1. - 1./delta_i
                solution[i] = K*(Z*(val-1.+delta_i) + (val*val-1.-delta_i*delta_i+2. *delta_i)/(2. * delta_i))
            else:
                solution[i] = 0.
        return solution

    def _stats(self, delta):
        moment1 = 1.-delta/3
        Z = 1. - 1./delta
        K = 2. / delta
        moment2 = K*(Z/3.*(1-(1. - delta)**3) + 1./(4. * delta)*(1. - (1. - delta)**4) )
        return moment1, moment2 - moment1**2, None, None
        
    def _argcheck(self, *args):
        # Aucune condition sur les paramètres (on renvoie que des 1)
        cond = 1
        for arg in args:
            cond = np.logical_and(cond, 1)
        return cond

# class trapeze_Serie4bis_gen(stats.rv_continuous):
#     "trapeze de la serie 4 extended, pour p(r_1)"

#     def _pdf(self, x, alpha, beta, gamma, delta):
#         norm = 2. * (alpha + beta) + 2.*gamma*delta
#         if not isfloat(x):
#             print(x)
#         if x > 0 and x <= delta:
#             s = gamma*(delta + x + 1.)
#         elif x>delta and x<1.-delta:
#             s = 2.*gamma*delta
#         else:
#             s = gamma*(2. + delta - x)
#         return s*norm



if __name__ == '__main__':
    main()

    