#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as md
from matplotlib import pyplot

import matplotlib.dates as md
years    = md.YearLocator()   # every year
months   = md.MonthLocator()  # every month
days     = md.DayLocator()    # every day
yearsFmt = md.DateFormatter('%Y      ')
monthFmt = md.DateFormatter('%B  ')
dayFmt   = md.DateFormatter('%d')


dpi=300
fontS = 16     # font size
matplotlib.rc('xtick', labelsize=fontS)
matplotlib.rc('ytick', labelsize=fontS)

def plot2(ch, absci, Y1, Y2, legend1, legend2, sf=True):

    fig, ax = plt.subplots(figsize=(10,4))
    color = 'b'
    lns1=ax.plot(absci, Y1, label=legend1, color='b')#, dashes=[3, 3, 3, 3])
    ax.tick_params(axis='y', labelcolor=color)
    ax.set_ylabel(legend1, color=color)

    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'r'
    lns2=ax2.plot(absci, Y2, label=legend2, color='r')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylabel(legend2, color=color)

    # format the ticks
    # ax.xaxis.set_major_locator(years)
    # ax.xaxis.set_major_formatter(yearsFmt)
    # ax.xaxis.set_minor_locator(months)
    # ax.xaxis.set_minor_formatter(monthFmt)

    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(monthFmt)
    ax.xaxis.set_minor_locator(days)
    ax.xaxis.set_minor_formatter(dayFmt)

    # datemin = np.datetime64(absci[0], 'Y')
    # datemax = np.datetime64(absci[-1], 'Y') + np.timedelta64(1, 'Y')
    #ax.set_xlim(datemin, datemax)
    ax.format_xdata = md.DateFormatter('%Y-%m-%d')
    #ax.format_ydata = Y
    ax.grid(True, which='minor', axis='both')
    ax.set_title('Data from US Dpt of Energy')

    # rotates and right aligns the x labels, and moves the bottom of the
    # axes up to make room for them
    fig.autofmt_xdate()
    plt.legend()
    plt.xticks(rotation=35)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.15)
    fig.tight_layout()          # otherwise the right y-label is slightly clipped
    if sf == True:
        plt.savefig(ch, bbox_inches='tight', dpi=dpi)
    else:
        pyplot.show()
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

def plot3(ch, absci, Y1, Y2, Y3, legend1, legend2, legend3, sf=True):

    fig, ax = plt.subplots(figsize=(10,4))
    color = 'b'
    lns1= ax.plot(absci, Y1, linewidth=1.0, label=legend1, color=color, dashes=[3, 3, 3, 3])
    ax.tick_params(axis='y', labelcolor=color)
    ax.set_ylabel(legend1, color=color)
    

    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'r'
    lns2= ax2.plot(absci, Y2, linewidth=1.0, label=legend2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylabel(legend2, color=color)
    color = 'g'
    lns3= ax2.plot(absci, Y3, linewidth=2.0, label=legend3, color=color, dashes=[6, 3, 6, 3])

    # format the ticks
    # ax.xaxis.set_major_locator(years)
    # ax.xaxis.set_major_formatter(yearsFmt)
    # ax.xaxis.set_minor_locator(months)
    # ax.xaxis.set_minor_formatter(monthFmt)

    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(monthFmt)
    ax.xaxis.set_minor_locator(days)
    ax.xaxis.set_minor_formatter(dayFmt)

    # datemin = np.datetime64(absci[0], 'Y')
    # datemax = np.datetime64(absci[-1], 'Y') + np.timedelta64(1, 'Y')
    #ax.set_xlim(datemin, datemax)
    ax.format_xdata = md.DateFormatter('%Y-%m-%d')
    #ax.format_ydata = Y
    ax.grid(True, which='minor', axis='both')
    ax.set_title('Data from US Dpt of Energy')

    # rotates and right aligns the x labels, and moves the bottom of the
    # axes up to make room for them
    fig.autofmt_xdate()
    
    lns = lns1+lns2+lns3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs)


    plt.xticks(rotation=35)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.15)
    fig.tight_layout()          # otherwise the right y-label is slightly clipped
    if sf == True:
        plt.savefig(ch, bbox_inches='tight', dpi=dpi)
    else:
        pyplot.show()
    plt.close()

def plot_GT3(ch, s, absci, Y, X_GT, X_est, legend1, legend2, legend3, sf=True):

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(absci, Y,                    label=legend1)
    ax.plot(absci, X_GT,  linewidth=2.0, label=legend2)
    ax.plot(absci, X_est, linewidth=2.0, label=legend3)
    # format the ticks
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(yearsFmt)
    ax.xaxis.set_minor_locator(months)
    ax.xaxis.set_minor_formatter(monthFmt)
    # ax.xaxis.set_major_locator(days)
    # ax.xaxis.set_major_formatter(dayFmt)

#  datemin = np.datetime64(absci[0], 'Y')
#  datemax = np.datetime64(absci[-1], 'Y') + np.timedelta64(1, 'Y')
    #ax.set_xlim(datemin, datemax)
    ax.format_xdata = md.DateFormatter('%Y-%m-%d')
    #ax.format_ydata = Y
    ax.grid(True, which='minor', axis='both')
    ax.set_title(s)

    # rotates and right aligns the x labels, and moves the bottom of the
    # axes up to make room for them
    fig.autofmt_xdate()
    plt.legend()
    plt.xticks(rotation=35)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.15)
    if sf == True:
        plt.savefig(ch, bbox_inches='tight', dpi=dpi)
    else:
        pyplot.show()
    plt.close()
