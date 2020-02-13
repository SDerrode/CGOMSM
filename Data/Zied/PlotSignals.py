#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

import matplotlib.dates as md
years    = md.YearLocator()   # every year
months   = md.MonthLocator()  # every month
days     = md.DayLocator()    # every day
yearsFmt = md.DateFormatter('%Y      ')
monthFmt = md.DateFormatter('%B  ')
dayFmt   = md.DateFormatter('%d')

Prefix     = './inputs/'
pathToSave = './results/'

colors     = ['red', 'blue', 'green']
dpi        = 100
fontS = 16     # font size
matplotlib.rc('xtick', labelsize=fontS)
matplotlib.rc('ytick', labelsize=fontS)

def draw (X, i, pathToSave, filename, title, N1, N2, save=True):
    N = X.shape[0]
    
    # draw of signal
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(range(N), X, lw=1, alpha=0.9, color=colors[i])
    ax.set_title(title)
    if save == False:
        plt.show()
    else:
        plt.savefig(pathToSave + filename, bbox_inches='tight', dpi=dpi)
        plt.close()

        # zoom signal
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(range(N1, N2), X[N1:N2], lw=1, alpha=0.9, color=colors[i])
        ax.set_title(title + ' zoom')
        if save == True:
            plt.savefig(pathToSave + filename+'_'+str(N2-N1), bbox_inches='tight', dpi=dpi)
        else:
            plt.show()
        plt.close()


def draw1 (X, Y, Xlabel, Xcolor, Ylabel, Ycolor, pathToSave, filename, N1, N2, save=True):
    N = X.shape[0]

    # draw of signal
    fig, ax1 = plt.subplots(nrows=1, ncols=1)
    ax1.plot(range(N), X, lw=1, alpha=0.9, color=colors[Xcolor])
    ax1.set_ylabel(Xlabel, color=colors[Xcolor])
    ax1.tick_params(axis='y', labelcolor=colors[Xcolor])
    ax2 = ax1.twinx()
    ax2.plot(range(N), Y, lw=1, alpha=0.9, color=colors[Ycolor])
    ax2.set_ylabel(Ylabel, color=colors[Ycolor])
    ax2.tick_params(axis='y', labelcolor=colors[Ycolor])
    if save == False:
        plt.show()
    else:
        plt.savefig(pathToSave + filename, bbox_inches='tight', dpi=dpi)
        plt.close()

        # draw of signal
        fig, ax1 = plt.subplots(nrows=1, ncols=1)
        ax1.plot(range(N1, N2), X[N1:N2], lw=1, alpha=0.9, color=colors[Xcolor])
        ax1.set_ylabel(Xlabel, color=colors[Xcolor])
        ax1.tick_params(axis='y', labelcolor=colors[Xcolor])
        ax2 = ax1.twinx()
        ax2.plot(range(N1, N2), Y[N1:N2], lw=1, alpha=0.9, color=colors[Ycolor])
        ax2.set_ylabel(Ylabel, color=colors[Ycolor])
        ax2.tick_params(axis='y', labelcolor=colors[Ycolor])
        plt.savefig(pathToSave + filename +'_'+str(N2-N1), bbox_inches='tight', dpi=dpi)
        plt.close()

def plot_GT2(ch, s, absci, GT, X, legend1, legend2, sf=True):

    matplotlib.rcParams.update({'font.size': 16})

    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(absci, GT, label=legend1, linewidth=2.0)
    ax.plot(absci, X,  label=legend2, linewidth=2.0)
    # format the ticks
    # ax.xaxis.set_major_locator(years)
    # ax.xaxis.set_major_formatter(yearsFmt)
    # ax.xaxis.set_minor_locator(months)
    # ax.xaxis.set_minor_formatter(monthFmt)
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(monthFmt)
    ax.xaxis.set_minor_locator(days)
    ax.xaxis.set_minor_formatter(dayFmt)

    #  datemin = np.datetime64(absci[0], 'Y')
    #  datemax = np.datetime64(absci[-1], 'Y') + np.timedelta64(1, 'Y')
    #ax.set_xlim(datemin, datemax)
    ax.format_xdata = md.DateFormatter('%Y-%m-%d')
    #ax.format_ydata = Y
    ax.grid(True, which='minor', axis='both')
    #ax.set_title(s)
    plt.title(s, fontsize=18)
    # rotates and right aligns the x labels, and moves the bottom of the
    # axes up to make room for them
    fig.autofmt_xdate()
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.15)
    leg = plt.legend(bbox_to_anchor=(0.1, 0.1, 0.8, -0.25), loc="upper center", ncol=4, shadow=False, fancybox=False, fontsize=11)
    plt.xticks(rotation=35)
    fig.tight_layout()
    if sf == True:
        plt.savefig(ch, bbox_inches='tight', dpi=dpi)
    else:
        pyplot.show()
    plt.close()