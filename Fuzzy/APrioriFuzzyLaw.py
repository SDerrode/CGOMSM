#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 11:06:52 2017

@author: MacBook_Derrode
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import axes3d, Axes3D
# from matplotlib.patches import Circle, PathPatch
from mpl_toolkits.mplot3d import Axes3D
# import mpl_toolkits.mplot3d.art3d as art3d

fontS = 13 # fontSize
mpl.rc('xtick', labelsize=fontS)
mpl.rc('ytick', labelsize=fontS)

def echelle(val, a, b):
    """
    To stretch values between a and b
    """
    #assert a <= b, print('PB dans echelle car a>b!')
    # value = val * (b - a) + a
    # if (value<a or value>=b):
    #     print("a=", a ,", value=", value, ", b=", b)
    #     input('fonction echelle')
    return val * (b - a) + a

def plotSample(rv, nbsample, filename, dpi=150):
    """
    Plot the pdf and cdf of rv, together with the hist of a sample of \
    nb sample drawed from rv, to check the algorihtm implementation.
    """

    sample = []
    for n in range(nbsample):
        sample.append(rv.rvs(1))
    hist, bin_edges = np.histogram(sample, bins=30, density=True)
    R = np.linspace(start=1E-10, stop=1.0 - 1E-10, num=100, endpoint=True)
    #print(hist, bin_edges)
    # print(np.sum(hist*np.diff(bin_edges)))
    fig = plt.figure()
    ax = fig.gca()
    pdf, = ax.plot(R, rv.pdf(R), alpha=0.6, color='g', label='pdf')
    cdf, = ax.plot(R, rv.cdf(R), alpha=0.6, color='b', label='cdf')
    hist, = ax.plot(bin_edges[0:-1], hist, alpha=0.6, color='r', label='hist' + str(nbsample) + ' samples')
    plt.legend(handles=[pdf, cdf, hist])
    ax.set_xlabel('$r$')
    ax.set_xlim(0, 1)
    # plt.show()
    plt.savefig(filename, bbox_inches='tight', dpi=dpi)

    return np.mean(sample), np.var(sample)

class LoiAPriori:
    """
    Super class to factorize code for APrioriFuzzyLaw_Seriesx.py classes.
    """

    def maxiFuzzyJump(self):
        """
        return the percent of fuzzy jump for the joint a priori law
        """
        return 1.0 - self.maxiHardJump()

    def maxiHardJump(self):
        """
        return the percent of hard jump for the joint a priori law
        """
        return self.probaR(0.) + self.probaR(1.)

    def getMTProbaFormJProba(self, JProba, n_r):
        MProba = np.sum(JProba, axis= 1).T
        TProba = np.zeros(shape=(n_r, n_r))
        for r in range(n_r):
            TProba[r, :] = JProba[r, :] / MProba[r]
        return MProba, TProba

    def sumR1R2(self, discretization):
        """
        Integration of density p(r1,R2), should sum one
        """

        EPS = 1E-10

        R = np.linspace(start=EPS, stop=1.0 - EPS, num=discretization, endpoint=True)

        integ = 0
        # D'abord les 4 droites
        pR = np.ndarray((R.shape[0]))
        for j, r in enumerate(R): pR[j] = self.probaR1R2(0, r)
        integ += sum(pR) / discretization
        for j, r in enumerate(R): pR[j] = self.probaR1R2(1, r)
        integ += sum(pR) / discretization
        for j, r in enumerate(R): pR[j] = self.probaR1R2(r, 0)
        integ += sum(pR) / discretization
        for j, r in enumerate(R): pR[j] = self.probaR1R2(r, 1)
        integ += sum(pR) / discretization

        # Ensuite les 4 angles
        integ += self.probaR1R2(0, 0) + self.probaR1R2(1, 0) + self.probaR1R2(0, 1) + self.probaR1R2(1, 1)

        # La surface à l'intérieur
        R1 = np.linspace(start=EPS, stop=1.0 - EPS, num=discretization, endpoint=True)
        R2 = np.linspace(start=EPS, stop=1.0 - EPS, num=discretization, endpoint=True)
        pR1R2 = np.ndarray(shape=(R1.shape[0], R2.shape[0]))
        for i, r1 in enumerate(R1):
            for j, r2 in enumerate(R2):
                pR1R2[i, j] = self.probaR1R2(r1, r2)
        integ += sum(sum(pR1R2)) / (discretization * discretization)

        return integ

    def probaQuart(self, r1, r2, minR1, maxR1, minR2, maxR2, discretization):

        EPS = 1E-10

        R1 = np.linspace(start=minR1+EPS, stop=maxR1-EPS, num=discretization, endpoint=True)
        R2 = np.linspace(start=minR2+EPS, stop=maxR2-EPS, num=discretization, endpoint=True)

        # Ensuite les 4 angles
        integ = self.probaR1R2(r1, r2)

        pR = np.ndarray((R1.shape[0]))
        if r1==0.:
            for j, r in enumerate(R2): pR[j] = self.probaR1R2(0., r)
            integ += sum(pR) / discretization / 2.
        if r1==1.:
            for j, r in enumerate(R2): pR[j] = self.probaR1R2(1., r)
            integ += sum(pR) / discretization / 2.
        if r2==0.:
            for j, r in enumerate(R1): pR[j] = self.probaR1R2(r, 0.)
            integ += sum(pR) / discretization / 2.
        if r2==1.:
            for j, r in enumerate(R1): pR[j] = self.probaR1R2(r, 1.)
            integ += sum(pR) / discretization /2.

        # La surface à l'intérieur
        pR1R2 = np.ndarray(shape=(R1.shape[0], R2.shape[0]))
        for i, p in enumerate(R1):
            for j, q in enumerate(R2):
                pR1R2[i, j] = self.probaR1R2(p, q)
        integ += sum(sum(pR1R2)) / (discretization * discretization) / 4.

        return integ

    def getNumericalHardTransition(self, n_r, discretization):
        """
        Integration of density p(r1,R2), should sum one
        """

        if n_r != 2:
            input('ProbaHard : n_r != 2 - IMP0SSIBLE')

        JProba = np.zeros(shape=(n_r, n_r), dtype=float)
        JProba[0,0] = self.probaQuart(0., 0., 0.0, 0.5,    0.0, 0.5   , discretization)
        JProba[1,0] = self.probaQuart(1., 0., 0.5, 1.-0.0, 0.0, 0.5   , discretization)
        JProba[0,1] = self.probaQuart(0., 1., 0.0, 0.5,    0.5, 1.-0.0, discretization)
        JProba[1,1] = self.probaQuart(1., 1., 0.5, 1.-0.0, 0.5, 1.-0.0, discretization)
        
        somme = np.sum(sum(JProba))
        if abs(1.0 - somme)> 1E-2:
            print('JProba=', JProba)
            print('sum=', somme)
            input('PROBLEM getNumericalHardTransition')
        JProba /= somme

        MProba, TProba = self.getMTProbaFormJProba(JProba, n_r)

        return MProba, TProba, JProba

    def plotR1R2(self, discretization, filename, ax, dpi=150):
        """
        Plot of the joint density p(r1, r2)
        """


        EPS = 1E-10
        #mpl.style.use('seaborn') # les couleurs

        R1 = np.linspace(start=0+EPS, stop=1.0-EPS, num=discretization, endpoint=True)
        R2 = np.linspace(start=0+EPS, stop=1.0-EPS, num=discretization, endpoint=True)
        pR1R2 = np.ndarray(shape=(R1.shape[0], R2.shape[0]))
        R1Grid1, R2Grid1 = np.meshgrid(R1, R2)

        ############# Dessin de la surface centrale
        for i, r1 in enumerate(R1):
            for j, r2 in enumerate(R2):
                pR1R2[i, j] = self.probaR1R2(r1, r2)
        ax.plot_surface(R1Grid1, R2Grid1, pR1R2, alpha=1, color='xkcd:green')

        maxim = np.amax(pR1R2)

        ############# Dessin des bords
        # pR1R2.fill(0.)
        # for i, r1 in enumerate(R1):
        #     pR1R2[i, 0] = self.probaR1R2(r1, 0.)
        # for i, r1 in enumerate(R1):
        #     pR1R2[i, R2.shape[0]-1] = self.probaR1R2(r1, 1.)
        # for j, r2 in enumerate(R2):
        #     pR1R2[0, j] = self.probaR1R2(0., r2)
        # for j, r2 in enumerate(R2):
        #     pR1R2[R1.shape[0]-1, j] = self.probaR1R2(1., r2)
        # ax.plot_surface(R1Grid1, R2Grid1, pR1R2, alpha=0.6, color='xkcd:lavender')
        # maxim = max(maxim, np.amax(pR1R2))

        ############# Dessin des masses
        x, y, z=[0., 0.], [0., 0.], [0., self.probaR1R2(0., 0.)]
        ax.plot(x,y,z,'k',alpha=0.8, linewidth=3.0)
        x, y, z=[0., 0.], [1., 1.], [0., self.probaR1R2(0., 1.)]
        ax.plot(x,y,z,'k',alpha=0.8, linewidth=3.0)
        x, y, z=[1., 1.], [0., 0.], [0., self.probaR1R2(1., 0.)]
        ax.plot(x,y,z,'k',alpha=0.8, linewidth=3.0)
        x, y, z=[1., 1.], [1., 1.], [0., self.probaR1R2(1., 1.)]
        ax.plot(x,y,z,'k',alpha=0.8, linewidth=3.0)
        maxim = max(maxim, self.probaR1R2(0., 0.))
        maxim = max(maxim, self.probaR1R2(0., 1.))
        maxim = max(maxim, self.probaR1R2(1., 0.))
        maxim = max(maxim, self.probaR1R2(1., 1.))
        #print(maxim)

        ############# Dessin des axes
        ax.set_xlabel('$r_1$', fontsize=16)
        #ax.set_xlim(-0.02, 1.02)
        ax.set_ylabel('$r_2$', fontsize=16)
        #ax.set_ylim(-0.02, 1.02)
        #ax.set_zlabel('$p(r_1,r_2)$', fontsize=fontS)
        #ax.set_zlim(0., maxim*1.02)
        ax.view_init(25, 238)

        # plt.show()
        if filename != None:
            plt.savefig(filename, bbox_inches='tight', dpi=dpi)


    def sumR1(self, discretization):
        """
        Integration of density p(r), should sum one
        """

        EPS = 1E-10
        R = np.linspace(start=EPS, stop=1.0-EPS, num=discretization, endpoint=True)

        pR = np.zeros((R.shape[0]))
        for i, r in enumerate(R):
            pR[i] = self.probaR(r)
        integ = sum(pR) / discretization

        # Les deux bords
        integ += self.probaR(0.) + self.probaR(1.)

        return integ

    def sumR2CondR1(self, discretization, r1):
        """
        Integration of density p(r), should sum one
        """

        EPS = 1E-10
        R = np.linspace(start=EPS, stop=1.0-EPS, num=discretization, endpoint=True)

        pR = np.zeros((R.shape[0]))
        for i, r in enumerate(R):
            pR[i] = self.probaR2CondR1 (r1, r)
        integ = sum(pR) / discretization

        # Les deux bords
        integ += self.probaR2CondR1(r1, 0.) + self.probaR2CondR1(r1, 1.)

        return integ

    def plotR1(self, discretization, filename, dpi):
        """
        Plot of the marginal density p(r)
        """
        ESP = 1E-10
        R = np.linspace(start=0.+ESP, stop=1.0-ESP, num=discretization, endpoint=True)

        pR = np.zeros((R.shape[0]))
        for i, r in enumerate(R):
            pR[i] = self.probaR(r)

        fig = plt.figure()
        ax = fig.gca()
        ax.plot(R, pR, alpha=0.6, color='g')

        ax.set_xlabel('$r$', fontsize=fontS)
        #ax.set_xlim(0, 1)
        #ax.set_ylabel('$p(r)$', fontsize=fontS)
        #ax.set_ylim(0., max(pR)*1.05)

        # plt.show()
        plt.savefig(filename, bbox_inches='tight', dpi=dpi)

