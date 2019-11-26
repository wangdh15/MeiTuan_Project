# -*- coding:utf-8 -*-

import os
import pandas as pd
path = '../Result2/'
for parent, dirnames, filenames in os.walk(path, followlinks=True):
    r_list = []
    for filename in filenames:
        r=pd.read_csv(path+filename)
        r_list.append(r)
    r_concat = pd.concat(r_list, axis=1).drop('ID', axis=1)
    r_mean = r_concat.mean(axis=1)
    r_out = pd.concat([r['ID'], r_mean], axis=1)
    r_out.columns = ['ID', 'action']
    r_out.to_csv(path+'ensemble.csv',index=False,header=True)
    print('Done!')
