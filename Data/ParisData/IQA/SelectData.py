#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

if __name__ == '__main__':

    INSEE=92 # 75 77
    name = 'indices_QA_commune_IDF_2016_2018'
    #name = 'indices_QA_commune_IDF_01_2018'

    # Check out the data
    IQA_df = pd.read_csv('input/' + name + '.csv', parse_dates=[0], dayfirst=True)
    print(IQA_df.head(5))
    print(IQA_df.columns)

    pd.to_datetime(IQA_df['date'])
    IQA_df.sort_values(by=['date'])
    print('count=', IQA_df.count())
    print('-->Date départ série temporelle = ', IQA_df['date'].iloc[0])
    print('-->Date fin    série temporelle = ', IQA_df['date'].iloc[-1])

    IQA_df.set_index('date', inplace=True)

    IQA_exerpt_df = IQA_df.loc[IQA_df['ninsee'] == INSEE]
    print('count=', IQA_exerpt_df.count())

    name1 = name + '_' + str(INSEE)
    IQA_exerpt_df.to_csv('input/' + name1 +'.csv', header=True, index=True)
