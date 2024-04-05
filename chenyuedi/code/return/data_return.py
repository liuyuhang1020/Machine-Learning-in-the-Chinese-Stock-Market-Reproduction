import pandas as pd
import numpy as np
import os

path_040_lst = ["TRD_Dalyr/TRD_Dalyr.xlsx"]
#for files in os.listdir('TRD_Dalyr'):
#    for f in os.listdir('TRD_Dalyr/' + files):
#        if f.split('.')[-1] == 'xlsx':
#            path_040_lst.append('/'.join(['TRD_Dalyr',files,f]))
#print(len(path_040_lst))

d = pd.DataFrame()
for path_040 in path_040_lst: 
    print('-----reading-----')
    d = pd.concat([d,pd.read_excel(path_040)[['Stkcd', 'Trddt', 'Dretnd']].iloc[2:]])
    print('-----------------------finish {:s}'.format(path_040))
    print('-----success-----')
d = d.rename(columns = {'Dretnd': 'ret'})

d['Yearmon'] = d['Trddt'].astype('str').replace('\-', '', regex=True)
d['Yearmon'] = d['Yearmon'].astype('int') // 100
d['Yearmon'] = d['Yearmon'].astype('str')
mon_lst = []
for y in range(2022,2024):
    for m in range(1,13):
        mon_lst.append('{:d}{:02d}'.format(y,m))
#print(mon_lst[8:12])
stk_lst = d.drop_duplicates(subset ='Stkcd')['Stkcd']
R1 = pd.DataFrame(stk_lst,columns=['Stkcd'])
R2 = pd.DataFrame(mon_lst[8:12],columns=['Yearmon'])
d = pd.merge(R2,d,how='left',on='Yearmon')

maxret = d.groupby(['Stkcd','Yearmon']).max().reset_index()
maxret = maxret.rename(columns = {'ret': 'maxret'})
sumret = d.groupby(['Stkcd','Yearmon']).sum().reset_index()
sumret = sumret.rename(columns = {'ret': 'sumret'})
std = d.groupby(['Stkcd','Yearmon']).std().reset_index()
std = std.rename(columns = {'ret': 'std'})


d = pd.merge(d,maxret,how='left',on=['Stkcd','Yearmon'])
d = pd.merge(d,sumret,how='left',on=['Stkcd','Yearmon'])
d = pd.merge(d,std,how='left',on=['Stkcd','Yearmon'])

d1 = d.drop_duplicates(subset=['Stkcd','Yearmon'])

R2['tmp'] = 0
R1['tmp'] = 0
R = pd.merge(R1,R2,on=['tmp'],how='left')
#print(R)
d1 = pd.merge(R[['Stkcd','Yearmon']],d1,how='left',on=['Stkcd','Yearmon'])
d1 = d1.sort_values(by=['Stkcd','Yearmon']).reset_index()
d1 = d1[['Stkcd','Yearmon','maxret','sumret','std']]

d1['Yearmon'] = d1['Yearmon'].astype('str')
print('-----------------------------finish d1')
#print(d1.loc[d1['Yearmon'] == '199012'])

# maxret
d1['maxret'] = d1['maxret'].shift()

# ground truth
d1['ground_truth'] = d1['sumret']

# mom1m
d1['sumret'] = d1['sumret'].shift()
d1 = d1.rename(columns={'sumret':'mom1m'})

# volatility
d1['std'] = d1['std'].shift()
d1 = d1.rename(columns={'std':'volatility'})

# 清空错位数据
d1.loc[d1['Yearmon'] == '202209',['maxret','mom1m','volatility']] = np.nan

# mom6m
m6 = d1.groupby(['Stkcd'])['mom1m'].rolling(window=5).sum().shift()
d1['mom6m'] = m6.tolist()
print('-----------------------------finish mom6m')

# mom12m
m12 = d1.groupby(['Stkcd'])['mom1m'].rolling(window=11).sum().shift()
d1['mom12m'] = m12.tolist()
print('-----------------------------finish mom12m')

# mom36m
t36 = d1.groupby(['Stkcd'])['mom1m'].rolling(window=36).sum().shift()
t12 = d1.groupby(['Stkcd'])['mom1m'].rolling(window=12).sum().shift()
m36 = t36 - t12
d1['mom36m'] = m36.tolist()
print('-----------------------------finish mom36m')

d1.loc[d1['Yearmon'] == '202209',['mom6m','mom12m','mom36m']] = np.nan
d1['Yearmon']=d1['Yearmon'].astype(int)
d1['Stkcd']=d1['Stkcd'].astype(int)
print(d1.dtypes)
import sys
sys.path.append('../../utils')
from format_transfer import mon_freq_data

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path',default='.')
args = parser.parse_args()

mon_freq_data(d1,d1.columns[2:],args.path)
