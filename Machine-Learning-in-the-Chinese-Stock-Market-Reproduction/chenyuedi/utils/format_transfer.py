import numpy as np
import pandas as pd

def mon_freq_data(d,factors,path):   
    stk_lst = pd.read_csv("../../utils/all_stock.csv")['Stkcd']
    mon_lst = []
    for y in range(1990,2023):
        for m in range(1,13):
            mon_lst.append(int('{:d}{:02d}'.format(y,m)))
    #for y in range(2022,2024):
    #    for m in range(1,13):
    #        mon_lst.append(int('{:d}{:02d}'.format(y,m)))
    R1 = pd.DataFrame(stk_lst,columns=['Stkcd'])
    R2 = pd.DataFrame(mon_lst[11:-3],columns=['Yearmon'])
   # R2 = pd.DataFrame(mon_lst[8:12],columns=['Yearmon'])
    R1['tmp'] = 0
    R2['tmp'] = 0
    R6 = pd.merge(R1,R2,on=['tmp'],how='left')
    #R6['Stkcd']=R6['Stkcd'].astype(str)
    #print(R6.dtypes)
    d = pd.merge(R6[['Stkcd','Yearmon']],d,on=['Stkcd','Yearmon'],how='left')
    for f in factors:
        d1 = pd.DataFrame(stk_lst,columns=['Stkcd'])
        for ym in mon_lst[11:-3]:
            d1 = pd.merge(d1,d[d['Yearmon'] == ym][['Stkcd',f]],how='left',on=['Stkcd'])
            d1 = d1.rename(columns={f:ym})
    
        d1.to_csv(r"{:s}\{:s}.csv".format(path,f),encoding='utf_8_sig',index = False)
        print(r"-----finish {:s}\{:s}.csv".format(path,f))
def season_freq_data(d,factors,path):
    stk_lst = pd.read_csv("../../utils/all_stock.csv")['Stkcd']
    mon_lst = []
    for y in range(1990,2023):
        for m in range(1,13):
            mon_lst.append(int('{:d}{:02d}'.format(y,m)))
    season_lst = []
    for y in range(1990,2023):
        for m in range(3,13,3):
            season_lst.append(int('{:d}{:02d}'.format(y,m)))
    R1 = pd.DataFrame(stk_lst,columns=['Stkcd'])
    R2 = pd.DataFrame(mon_lst[11:-3],columns=['Yearmon'])
    R3 = pd.DataFrame(season_lst[3:-1],columns=['Season'])
    
    R2['num'] = R2['Yearmon'] // 100 * 10 + ((R2['Yearmon'] % 100) - 1) // 3
    R3['num'] = R3['Season'] // 100 * 10 + (R3['Season'] % 100) // 3 - 1
    R4 = pd.merge(R2,R3,on=['num'],how='left')
    R1['tmp'] = 0
    R4['tmp'] = 0
    #print(R4)
    R5 = pd.merge(R1,R4,on=['tmp'],how='left')
    R5 = R5[['Stkcd','Yearmon','Season']]
    #print(R5.dtypes)
    #print(d.dtypes)
    d = pd.merge(R5,d,on=['Stkcd','Season'],how='left')
    d = d.drop_duplicates(subset=['Stkcd','Yearmon'])
    #print(d)
    for f in factors:
        d1 = pd.DataFrame(stk_lst,columns=['Stkcd'])
        for ym in mon_lst[11:-3]:
            d1 = pd.merge(d1,d[d['Yearmon'] == ym][['Stkcd',f]],how='left',on=['Stkcd'])
            #print(d1.shape)
            d1 = d1.rename(columns={f:ym})
    
        d1.to_csv(r"{:s}\{:s}.csv".format(path,f),encoding='utf_8_sig',index = False)
        print(r"-----finish {:s}\{:s}.csv".format(path,f))

def year_freq_data(d,factors,path):   
    stk_lst = pd.read_csv("../../utils/all_stock.csv")['Stkcd']
    mon_lst = []
    for y in range(1990,2023):
        for m in range(1,13):
            mon_lst.append(int('{:d}{:02d}'.format(y,m)))
    year_lst = range(1990,2023)
    R1 = pd.DataFrame(stk_lst,columns=['Stkcd'])
    R2 = pd.DataFrame(mon_lst[11:-3],columns=['Yearmon'])
    R1['tmp'] = 0
    R2['tmp'] = 0
    R2['Year'] = R2['Yearmon'] // 100
    R6 = pd.merge(R1,R2,on=['tmp'],how='left')
    d = pd.merge(R6[['Stkcd','Yearmon','Year']],d,on=['Stkcd','Year'],how='left')
    for f in factors:
        d1 = pd.DataFrame(stk_lst,columns=['Stkcd'])
        for ym in mon_lst[11:-3]:
            d1 = pd.merge(d1,d[d['Yearmon'] == ym][['Stkcd',f]],how='left',on=['Stkcd'])
            d1 = d1.rename(columns={f:ym})
    
        d1.to_csv(r"{:s}\{:s}.csv".format(path,f),encoding='utf_8_sig',index = False)
        print(r"-----finish {:s}\{:s}.csv".format(path,f))