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
days     = md.DayLocator()  # every month
yearsFmt = md.DateFormatter('%Y      ')
monthFmt = md.DateFormatter('%m')
dayFmt   = md.DateFormatter('%d')


def plot3(ch, absci, Y1, Y2, Y3, legend1, legend2, legend3, sf=True):

  fig, ax = plt.subplots(figsize=(10,4))
  ax.plot(absci, Y1, linewidth=1.0, label=legend1)
  ax.plot(absci, Y2, linewidth=1.0, label=legend2)
  ax.plot(absci, Y3, linewidth=1.0, label=legend3)
  # format the ticks
  ax.xaxis.set_major_locator(years)
  ax.xaxis.set_major_formatter(yearsFmt)
  ax.xaxis.set_minor_locator(months)
  ax.xaxis.set_minor_formatter(monthFmt)
  # ax.xaxis.set_major_locator(days)
  # ax.xaxis.set_major_formatter(dayFmt)

  # datemin = np.datetime64(absci[0], 'Y')
  # datemax = np.datetime64(absci[-1], 'Y') + np.timedelta64(1, 'Y')
  #ax.set_xlim(datemin, datemax)
  ax.format_xdata = md.DateFormatter('%Y-%m-%d')
  #ax.format_ydata = Y
  ax.grid(True, which='minor', axis='both')
  ax.set_title('Pollution in Paris (insee code 75)')

  # rotates and right aligns the x labels, and moves the bottom of the
  # axes up to make room for them
  fig.autofmt_xdate()
  plt.legend()
  plt.xticks(rotation=35)
  plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.15)
  if sf == True:
    plt.savefig(ch, bbox_inches='tight')
  else:
    pyplot.show()
  plt.close()


def plot1(ch, absci, Y1, legend1, c, sf=True):

  fig, ax = plt.subplots(figsize=(10,4))
  ax.plot(absci, Y1, label=legend1, color=c)
  # format the ticks
  ax.xaxis.set_major_locator(years)
  ax.xaxis.set_major_formatter(yearsFmt)
  ax.xaxis.set_minor_locator(months)
  ax.xaxis.set_minor_formatter(monthFmt)
  # ax.xaxis.set_major_locator(days)
  # ax.xaxis.set_major_formatter(dayFmt)

  # datemin = np.datetime64(absci[0], 'Y')
  # datemax = np.datetime64(absci[-1], 'Y') + np.timedelta64(1, 'Y')
  #ax.set_xlim(datemin, datemax)
  ax.format_xdata = md.DateFormatter('%Y-%m-%d')
  #ax.format_ydata = Y
  ax.grid(True, which='minor', axis='both')
  ax.set_title('Pollution in Paris (insee code 75)')

  # rotates and right aligns the x labels, and moves the bottom of the
  # axes up to make room for them
  fig.autofmt_xdate()
  plt.legend()
  plt.xticks(rotation=35)
  plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.15)
  if sf == True:
    plt.savefig(ch, bbox_inches='tight')
  else:
    pyplot.show()
  plt.close()

