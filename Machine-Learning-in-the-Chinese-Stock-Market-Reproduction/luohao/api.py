"""
罗皓 
复现 Machine learning in the Chinese stock market Markus Leippold
因子整理部分
主要数据来源：CSMAR
将因子计算放在函数中，按照序号排列，比如cal_1_2_3_60，是计算第1,2,3,60个因子的函数。按照因子频率，分成月，季度，半年，年放在data文件夹里

2023.1.8 update: 将1_2_3_60因子的计算方式改为和别的因子一样，直接处理下好的文件，而不是用csmar的接口下载
"""
# from csmarapi.CsmarService import CsmarService
# from csmarapi.ReportUtil import ReportUtil
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm
import datetime
# csmar = CsmarService()
# csmar.login('1800011778@pku.edu.cn', 'yygwzzjsC1')

import warnings
warnings.filterwarnings("ignore")

def get_zcfz(start='2018-01-04',end='2019-12-31'):
    """资产负债表，半年"""
    data1 = csmar.query(['Stkcd','Accper','Typrep','A0b1103000','A001124000',
    'A001100000','A001000000','A002113000','A002100000'], "Accper LIKE '%____-06-__%'", 'FS_Combas',start,end)
    df1 = pd.DataFrame(data1)
    df1.rename({'Stkcd':"证券代码",'Accper':"会计期间",'Typrep':"报表类型",'A0b1103000':"现金及存放中央银行款项",'A001124000':"一年内到期的非流动资产",
    'A001100000':"流动资产合计",'A001000000':"资产总计",'A002113000':"应交税费",'A002100000':"流动负债合计"},axis='columns',inplace=True)
    df1 = df1[df1.报表类型=='A']
    # df.to_csv('data/资产负债表1.csv',index=False,encoding='utf-8_sig')

    data2 = csmar.query(['Stkcd','Accper','Typrep','A0b1103000','A001124000',
    'A001100000','A001000000','A002113000','A002100000'], "Accper LIKE '%____-12-__%'", 'FS_Combas',start,end)
    df2 = pd.DataFrame(data2)
    df2.rename({'Stkcd':"证券代码",'Accper':"会计期间",'Typrep':"报表类型",'A0b1103000':"现金及存放中央银行款项",'A001124000':"一年内到期的非流动资产",
    'A001100000':"流动资产合计",'A001000000':"资产总计",'A002113000':"应交税费",'A002100000':"流动负债合计"},axis='columns',inplace=True)
    df2 = df2[df2.报表类型=='A']
    # df.to_csv('data/资产负债表2.csv',index=False,encoding='utf-8_sig')

    df3 = pd.concat([df1,df2])
    df3 = df3.sort_values(by=["证券代码","会计期间"])
    df3.to_csv('data/资产负债表.csv',index=False,encoding='utf-8_sig')
    return df3

def get_lr(start='2018-01-01',end='2019-12-31'):
    """利润表，半年"""
    data1 = csmar.query(['Stkcd','Accper','Typrep','B002000000'], 
    "Accper LIKE '%____-06-__%'", 'FS_Comins',start,end)
    df1 = pd.DataFrame(data1)
    df1.rename({'Stkcd':"证券代码",'Accper':"会计期间",'Typrep':"报表类型",'B002000000':'净利润'},axis='columns',inplace=True)
    df1 = df1[df1.报表类型=='A']

    data2 = csmar.query(['Stkcd','Accper','Typrep','B002000000'], 
    "Accper LIKE '%____-12-__%'", 'FS_Comins',start,end)
    df2 = pd.DataFrame(data2)
    df2.rename({'Stkcd':"证券代码",'Accper':"会计期间",'Typrep':"报表类型",'B002000000':'净利润'},axis='columns',inplace=True)
    df2 = df2[df2.报表类型=='A']

    df3 = pd.concat([df1,df2])
    df3 = df3.sort_values(by=["证券代码","会计期间"])
    df3.to_csv('data/利润表.csv',index=False,encoding='utf-8_sig')
    return df3

def get_zjtx(start='2019-01-04',end='2019-12-31'):
    """折旧摊销表，半年
    折旧摊销 =（固定资产折旧、油气资产折耗、生产性生物资产折旧+无形资产摊销+长期待摊费用摊销）
    折旧 = D000103000固定资产折旧、油气资产折耗、生产性生物资产折旧
    摊销 = D000104000无形资产摊销 + D000105000长期待摊费用摊销
"""
    data1 = csmar.query(['Stkcd','Accper','Typrep','D000103000'], 
    "Accper LIKE '%____-06-__%'", 'FS_Comscfi',start,end)
    df1 = pd.DataFrame(data1)
    df1.rename({'Stkcd':"证券代码",'Accper':"会计期间",'Typrep':"报表类型",'D000103000':'折旧'},axis='columns',inplace=True)
    df1 = df1[df1.报表类型=='A']

    data2 = csmar.query(['Stkcd','Accper','Typrep','D000103000'], 
    "Accper LIKE '%____-12-__%'", 'FS_Comscfi',start,end)
    df2 = pd.DataFrame(data2)
    df2.rename({'Stkcd':"证券代码",'Accper':"会计期间",'Typrep':"报表类型",'D000103000':'折旧'},axis='columns',inplace=True)
    df2 = df2[df2.报表类型=='A']

    data3 = csmar.query(['Stkcd','Accper','Typrep','D000104000','D000105000'], 
    "Accper LIKE '%____-06-__%'", 'FS_Comscfi',start,end)
    df3 = pd.DataFrame(data3)
    df3.rename({'Stkcd':"证券代码",'Accper':"会计期间",'Typrep':"报表类型",'D000104000':"无形资产摊销",'D000105000':"长期待摊费用摊销"},axis='columns',inplace=True)
    df3 = df3[df3.报表类型=='A']

    data4 = csmar.query(['Stkcd','Accper','Typrep','D000104000','D000105000'], 
    "Accper LIKE '%____-12-__%'", 'FS_Comscfi',start,end)
    df4 = pd.DataFrame(data4)
    df4.rename({'Stkcd':"证券代码",'Accper':"会计期间",'Typrep':"报表类型",'D000104000':"无形资产摊销",'D000105000':"长期待摊费用摊销"},axis='columns',inplace=True)
    df4 = df4[df4.报表类型=='A']

    df5 = pd.concat([df1,df2])
    df5 = df5.sort_values(by=["证券代码","会计期间"])

    df6 = pd.concat([df3,df4])
    df6 = df6.sort_values(by=["证券代码","会计期间"])

    df5 = df5.fillna(0)
    df6 = df6.fillna(0)
    df7 = pd.merge(df5,df6)
    df7["折旧摊销"] = df7["折旧"]+df7["无形资产摊销"]+df7["长期待摊费用摊销"]
    df7 = df7[["证券代码","会计期间","报表类型","折旧摊销"]]
    df7.to_csv('data/折旧摊销表.csv',index=False,encoding='utf-8_sig')
    return df7

def get_yjlr():
    """将三张表合在一起
    保存到data/应记利润.csv
    """
    df1 = pd.read_csv('data/资产负债表.csv')
    df2 = pd.read_csv('data/折旧摊销表.csv')
    df3 = pd.read_csv('data/利润表.csv')
    df4 = pd.merge(pd.merge(df1,df2,how='outer'),df3,how='outer').fillna(0)
    df4.sort_values(by=["证券代码","会计期间"],inplace=True)
    df4.to_csv('data/应记利润.csv',index=False,encoding='utf-8_sig')
# get_yjlr()

def cal_1_2_3_60_old():
    """应记利润 因子acc,absacc,pctacc 半年; agr 年
    用到 应记利润 
    """
    df1 = pd.read_csv('data/应记利润.csv')
    df1 = df1.fillna(0)
    df1['delta_流动资产'] = df1['流动资产合计'] - df1['流动资产合计'].shift(1)
    df1['delta_现金及存放中央银行款项'] = df1['现金及存放中央银行款项'] - df1['现金及存放中央银行款项'].shift(1)
    df1['delta_流动负债合计'] = df1['流动负债合计'] - df1['流动负债合计'].shift(1)
    df1['delta_一年内到期的非流动资产'] = df1['一年内到期的非流动资产'] - df1['一年内到期的非流动资产'].shift(1)
    df1['delta_应交税费'] = df1['应交税费'] - df1['应交税费'].shift(1)
    df1["acc"] = ((df1['delta_流动资产']-df1['delta_现金及存放中央银行款项']) -\
                (df1['delta_流动负债合计']-df1['delta_一年内到期的非流动资产']-df1['delta_应交税费']) - df1['折旧摊销']) / df1['资产总计']
    
    df1["absacc"] = abs(df1["acc"])
    df1["pctacc"] = df1["acc"]*df1['资产总计']/df1["净利润"]
    for i in range(1,len(df1)):
        if df1.loc[i,"证券代码"] != df1.loc[i-1,"证券代码"]:
            df1.loc[i,"acc"] = df1.loc[i,"absacc"] = df1.loc[i,"pctacc"] = None
    df1[["证券代码","会计期间","acc","absacc","pctacc"]].to_csv('data/半年_1_2_60_old.csv',index=False,encoding='utf-8_sig')

    df3 = df1[["证券代码","会计期间","资产总计"]]
    lines = []
    for i in range(len(df3)):
        if '-12-' in df3.loc[i,"会计期间"]:
            lines.append(list(df3.loc[i].values))
    lines = np.array(lines)
    df4 = pd.DataFrame({"证券代码":lines[:,0],"会计期间":lines[:,1],"资产总计":map(float,lines[:,2])})
    df4.to_csv('data/应记利润.csv',index=False,encoding='utf-8_sig')
    df4["agr"] = (df4['资产总计']-df4['资产总计'].shift(1))/df4['资产总计'].shift(1)
    for i in range(1,len(df4)):
        if df4.loc[i,"证券代码"] != df4.loc[i-1,"证券代码"]:
            df4.loc[i,"agr"] = None
    df4[["证券代码","会计期间","agr"]].to_csv('data/年_3.csv',index=False,encoding='utf-8_sig')
# cal_1_2_3_60_old()

def cal_1_2_3_60():
    """资本增长与获利 acc,absacc,agr,pctacc 半年
    用到 资产负债FS_Combas 现金流量表(间接法)FS_Comscfi 利润FS_Comins
    """
    df1 = pd.read_csv('data/FS_Combas.csv')
    df2 = pd.read_csv('data/FS_Comscfi.csv')
    df3 = pd.read_csv('data/FS_Comins.csv')
    df1.rename({'Stkcd':"证券代码",'Accper':"会计期间",'Typrep':"报表类型",'A0b1103000':"现金及存放中央银行款项",'A001124000':"一年内到期的非流动资产",
    'A001100000':"流动资产合计",'A001000000':"资产总计",'A002113000':"应交税费",'A002100000':"流动负债合计"},axis='columns',inplace=True)
    # df1 = df1[df1.报表类型=='A']
    df1 = df1.sort_values(by=["证券代码","会计期间"])
    df1 = df1.fillna(0)
    
    df2.rename({'Stkcd':"证券代码",'Accper':"会计期间",'Typrep':"报表类型",'D000103000':'折旧','D000104000':"无形资产摊销",'D000105000':"长期待摊费用摊销"},axis='columns',inplace=True)
    df2["折旧摊销"] = df2["折旧"]+df2["无形资产摊销"]+df2["长期待摊费用摊销"]

    df3.rename({'Stkcd':"证券代码",'Accper':"会计期间",'Typrep':"报表类型",'B002000000':'净利润'},axis='columns',inplace=True)

    df1 = pd.merge(pd.merge(df1,df2),df3).fillna(0)
    #筛选6月和12月的出来
    df1["month"] = pd.DatetimeIndex(df1["会计期间"]).month.astype(str)
    df1_1 = df1[df1.month=='06']
    df1_2 = df1[df1.month=='12']
    df1 = pd.concat([df1_1,df1_2])
    df1.sort_values(by=["证券代码","会计期间"],inplace=True)
    df1.reset_index(inplace=True)

    df1['delta_流动资产'] = df1['流动资产合计'] - df1['流动资产合计'].shift(1)
    df1['delta_现金及存放中央银行款项'] = df1['现金及存放中央银行款项'] - df1['现金及存放中央银行款项'].shift(1)
    df1['delta_流动负债合计'] = df1['流动负债合计'] - df1['流动负债合计'].shift(1)
    df1['delta_一年内到期的非流动资产'] = df1['一年内到期的非流动资产'] - df1['一年内到期的非流动资产'].shift(1)
    df1['delta_应交税费'] = df1['应交税费'] - df1['应交税费'].shift(1)
    df1["acc"] = ((df1['delta_流动资产']-df1['delta_现金及存放中央银行款项']) -\
                (df1['delta_流动负债合计']-df1['delta_一年内到期的非流动资产']-df1['delta_应交税费']) - df1['折旧摊销']) / df1['资产总计']
    
    df1["absacc"] = abs(df1["acc"])
    df1["pctacc"] = df1["acc"]*df1['资产总计']/df1["净利润"]
    for i in range(1,len(df1)):
        if df1.loc[i,"证券代码"] != df1.loc[i-1,"证券代码"]:
            df1.loc[i,"acc"] = df1.loc[i,"absacc"] = df1.loc[i,"pctacc"] = None
    df1[["证券代码","会计期间","acc","absacc","pctacc"]].to_csv('data/半年_1_2_60.csv',index=False,encoding='utf-8_sig')
    
    #年
    df2 = df1[df1.month=='12']
    df2["agr"] = (df2['资产总计']-df2['资产总计'].shift(1))/df2['资产总计'].shift(1)
    df2.reset_index(inplace=True)
    for i in range(1,len(df2)):
        if df2.loc[i,"证券代码"] != df2.loc[i-1,"证券代码"]:
            df2.loc[i,"agr"] = None
    df2[["证券代码","会计期间","agr"]].to_csv('data/年_3.csv',index=False,encoding='utf-8_sig')
# cal_1_2_3_60()

def cal_4_5():
    """beta betasq
    用到 wind下载的beta_origin
    """
    #先将beta_origin转换为通常格式
    df = pd.read_csv("data/beta_origin1.csv")
    columns = [s for s in df.columns if not s.endswith('BJ')][1:]
    df = df[['Stkcd']+columns]
    df.rename({column:column[:-3]for column in columns},axis='columns',inplace=True)
    df.to_csv('output_data/beta.csv',index=False,encoding='utf-8_sig')

    #square
    for column in df.columns[1:]:
        df[column] = df[column]**2
    df.to_csv('output_data/betasq.csv',index=False,encoding='utf-8_sig')
# cal_4_5()

def cal_6_7():
    """账面市值比 经行业调整的账面市值比 bm bm_ia 季度
    用到 FI_T10
    """
    df1 = pd.read_excel('data/FI_T10.csv')
    df1 = df1[["Stkcd","Accper","Indcd","F101001A"]]
    df1 = df1[2:].reset_index()
    df1.rename({'Stkcd':"证券代码",'Accper':"会计期间","Indcd":"行业代码",'F101001A':"bm"},axis='columns',inplace=True)
    df2 = df1[["行业代码","会计期间",'bm']].groupby(["行业代码","会计期间"]).mean().reset_index()
    df2.rename({'bm':'bm_i'},axis='columns',inplace=True)
    df3 = pd.merge(df1,df2,on=["行业代码","会计期间"],how='left')
    df3['bm_ia'] = df3['bm']-df3['bm_i']
    df3 = df3[["证券代码","会计期间","bm","bm_ia"]]
    df3.to_csv('data/季_6_7.csv',index=False,encoding='utf-8_sig')
# cal_6_7()

def cal_8_83():
    """有形资产指数变量 cash tang 季度
    用到 FS_Combas
    """
    df1 = pd.read_csv('data/FS_Combas.csv')
    df1 = df1[["Stkcd","Accper",'A0b1103000','A001111000','A001123000','A001212000','A001000000']]
    df1.rename({'Stkcd':"证券代码",'Accper':"会计期间",'A0b1103000':"现金及存放中央银行款项",'A001111000':"应收账款净额",'A001123000':"存货净额",
    'A001212000':"固定资产净额",'A001000000':"资产总计"},axis='columns',inplace=True)
    df1.fillna(0,inplace=True)
    df1['tang'] = df1['现金及存放中央银行款项']+0.715*df1['应收账款净额']+0.547*df1['存货净额']+0.535*df1['固定资产净额']/df1['资产总计']
    df1["cash"] = df1["现金及存放中央银行款项"]/df1['资产总计']
    df2 = df1[["证券代码","会计期间","cash","tang"]]
    df2.to_csv('data/季_8_83.csv',index=False,encoding='utf-8_sig')
# cal_8_83()

def get_xjll(start,end):
    """现金流量表，季度"""
    def f(month):
        data1 = csmar.query(['Stkcd','Accper','Typrep','C001000000'], f"Accper LIKE '%____-{month}-__%'", 'FS_Comscfd',start,end)
        df1 = pd.DataFrame(data1)
        df1.rename({'Stkcd':"证券代码",'Accper':"会计期间",'Typrep':"报表类型",'C001000000':"经营活动产生的现金流量净额"},axis='columns',inplace=True)
        return df1
    df1,df2,df3,df4 = f("03"),f("06"),f("09"),f("12")
    df5 = pd.concat([df1,df2,df3,df4])
    df5.sort_values(by=["证券代码","会计期间"],inplace=True)
    df5.to_csv('data/现金流量表.csv',index=False,encoding='utf-8_sig')
start,end="1995-12-31", "2022-06-30"
# get_xjll(start,end)

def cal_9_old():
    """现金流量与当期债务比 cashdebt 季度
    用到现金流量表 FS_Combas
    """
    df1 = pd.read_csv('data/现金流量表.csv')
    df2 = pd.read_csv('data/FS_Combas.csv')
    df2.rename({'Stkcd':"证券代码",'Accper':"会计期间",'Typrep':"报表类型",'A002100000':"流动负债合计"},axis='columns',inplace=True)

    df3 = df2[["证券代码","会计期间","流动负债合计"]]
    lines = []
    for i in range(len(df3)):
        if '-03-' in df3.loc[i,"会计期间"] or '-06-' in df3.loc[i,"会计期间"] or '-09-' in df3.loc[i,"会计期间"] or '-12-' in df3.loc[i,"会计期间"]:
            lines.append(list(df3.loc[i].values))
    lines = np.array(lines)
    df4 = pd.DataFrame({"证券代码":map(float,lines[:,0]),"会计期间":lines[:,1],"流动负债合计":map(float,lines[:,2])})
    df4 = df4.fillna(0)
    df5 = pd.merge(df1,df4,on=['证券代码',"会计期间"])
    df5.drop_duplicates(['证券代码',"会计期间"], 'first', inplace=True)
    df5['cashbebt'] = df5["经营活动产生的现金流量净额"]/df5["流动负债合计"]
    df5[["证券代码","会计期间",'cashbebt']].to_csv('data/季_9_old.csv',index=False,encoding='utf-8_sig')
# cal_9_old()

def cal_9():
    """现金流量与当期债务比 cashdebt 季度
    用到FS_Comscfd FS_Combas
    """
    df1 = pd.read_csv('data/FS_Comscfd.csv')
    df2 = pd.read_csv('data/FS_Combas.csv')
    df1.rename({'Stkcd':"证券代码",'Accper':"会计期间",'Typrep':"报表类型",'C001000000':"经营活动产生的现金流量净额"},axis='columns',inplace=True)
    df2.rename({'Stkcd':"证券代码",'Accper':"会计期间",'Typrep':"报表类型",'A002100000':"流动负债合计"},axis='columns',inplace=True)

    df3 = df2[["证券代码","会计期间","流动负债合计"]]
    lines = []
    for i in range(len(df3)):
        if '-03-' in df3.loc[i,"会计期间"] or '-06-' in df3.loc[i,"会计期间"] or '-09-' in df3.loc[i,"会计期间"] or '-12-' in df3.loc[i,"会计期间"]:
            lines.append(list(df3.loc[i].values))
    lines = np.array(lines)
    df4 = pd.DataFrame({"证券代码":map(float,lines[:,0]),"会计期间":lines[:,1],"流动负债合计":map(float,lines[:,2])})
    df4 = df4.fillna(0)
    df5 = pd.merge(df1,df4,on=['证券代码',"会计期间"])
    df5.drop_duplicates(['证券代码',"会计期间"], 'first', inplace=True)
    df5['cashbebt'] = df5["经营活动产生的现金流量净额"]/df5["流动负债合计"]
    df5[["证券代码","会计期间",'cashbebt']].to_csv('data/季_9.csv',index=False,encoding='utf-8_sig')
# cal_9()

def cal_10():
    """现金生产率 cashspr 季度
    quarter-end market capitalization plus longterm
debt minus total assets divided by cash and equivalents.
    (市值+长期负债-资产总计)/期末现金及现金等价物余额
    用到FS_Comscfd FS_Combas daily_return
    """
    df1 = pd.read_csv('data/日个股回报率/daily_return.csv')[["Stkcd","Trddt","Dsmvtll"]]
    df1.rename({'Stkcd':"证券代码",'Trddt':"会计期间",'Dsmvtll':"市值"},axis='columns',inplace=True)
    df1.to_csv('data/tmp1.csv',index=False,encoding='utf-8_sig')
    df2 = pd.read_csv('data/FS_Combas.csv')
    df2.rename({'Stkcd':"证券代码",'Accper':"会计期间",'A002206000':"长期负债合计","A001000000":"资产总计"},axis='columns',inplace=True)
    df2 = df2[["证券代码","会计期间","长期负债合计","资产总计"]]
    df2.to_csv('data/tmp2.csv',index=False,encoding='utf-8_sig')
    df3 = pd.read_csv('data/FS_Comscfd.csv')
    df3.rename({'Stkcd':"证券代码",'Accper':"会计期间",'Typrep':"报表类型",'C006000000':"期末现金及现金等价物余额"},axis='columns',inplace=True)
    df3 = df3[["证券代码","会计期间","期末现金及现金等价物余额"]]
    df4 = pd.merge(df1,df2,on=["证券代码","会计期间"])
    df4["会计期间"]=pd.to_datetime(df4["会计期间"])
    df3["会计期间"]=pd.to_datetime(df3["会计期间"])

    df5 = pd.merge(df4,df3,on=["证券代码","会计期间"])
    df5.fillna(0,inplace=True)
    df5["cashspr"] = (df5["市值"] + df5["长期负债合计"] - df5["资产总计"])/df5["期末现金及现金等价物余额"]
    df5.sort_values(by=["证券代码","会计期间"],inplace=True)
    df5[["证券代码","会计期间","cashspr"]].to_csv('data/季_10.csv',index=False,encoding='utf-8_sig')
# cal_10()

def cal_11_12():
    """营业产生的现金流/市值，以及经行业调整过的比值 cfp cfp_ia 季度
    用到 FS_Comscfd daily_return FI_T10
        市值：Dsmvtll
    """
    df1 = pd.read_csv('data/FS_Comscfd.csv')
    df1 = df1[['Stkcd','Accper','C001000000']]
    df1.rename({'Stkcd':"证券代码",'Accper':"会计期间",'C001000000':"现金流"},axis='columns',inplace=True)
   
    df2 = pd.read_csv('data/日个股回报率/daily_return.csv')
    df2 = df2[['Stkcd','Trddt','Dsmvtll']]
    df2.rename({'Stkcd':"证券代码",'Trddt':"会计期间",'Dsmvtll':"市值"},axis='columns',inplace=True)
    df3 = pd.merge(df1,df2, on = ["证券代码","会计期间"])
    df3["cfp"] = df3["现金流"]/df3["市值"]

    #求行业平均
    df4 = pd.read_excel('data/FI_T10.csv')[['Stkcd','Indcd']][2:]
    df4.rename({'Stkcd':'证券代码','Indcd':"行业代码"},axis='columns',inplace=True)
    df4.drop_duplicates('证券代码', 'first', inplace=True)
    # df3["证券代码"] = df3["证券代码"].astype(int)
    df4["证券代码"] = df4["证券代码"].astype(int)
    df5 = pd.merge(df3,df4,on=["证券代码"])
    df6 = df5[["行业代码","会计期间",'cfp']].groupby(["行业代码","会计期间"]).mean().reset_index()
    df6.rename({'cfp':'cfp_i'},axis='columns',inplace=True)

    df7 = pd.merge(df5,df6,on=["行业代码","会计期间"])
    df7["cfp_ia"] = df7["cfp"] - df7["cfp_i"]
    df7.sort_values(by=["证券代码","会计期间"],inplace=True)
    df7[["证券代码","会计期间",'cfp','cfp_ia']].to_csv('data/季_11_12.csv',index=False,encoding='utf-8_sig')
# cal_11_12()

def cal_13_14():
    """资产周转率 chato chato_ia 季度
    chato: 销售变化量除以平均总资产
    chato_ia:经行业调整的销售变化除以平均总资产
    用到FS_Comins FS_Combas
    """
    df1 = pd.read_csv('data/FS_Comins.csv')
    df1 = df1[['Stkcd','Accper','Typrep','B001100000']]
    df1.rename({'Stkcd':"证券代码",'Accper':"会计期间",'B001100000':"营业总收入"},axis='columns',inplace=True)
    df1['DTSale'] = df1["营业总收入"]-df1["营业总收入"].shift(1)

    df2 = pd.read_csv('data/FS_Combas.csv')
    df2 = df2[['Stkcd','Accper','Typrep','A001000000']]
    df2.rename({'Stkcd':"证券代码",'Accper':"会计期间",'A001000000':"总资产"},axis='columns',inplace=True)
    df2['AveTA'] = (df2["总资产"]+df2["总资产"].shift(1))/2

    df1["会计期间"] = pd.to_datetime(df1["会计期间"])
    df2["会计期间"] = pd.to_datetime(df2["会计期间"])

    df3 = pd.merge(df1,df2,on=["证券代码","会计期间"])
    df3.sort_values(by=["证券代码","会计期间"],inplace=True)
    df3['chato'] = df3['DTSale']/df3['AveTA']
    for i in range(1,len(df3)):
        if df3.loc[i,"证券代码"] != df3.loc[i-1,"证券代码"]:
            df3.loc[i,'chato'] = None
    

    #加入行业代码
    df4 = pd.read_excel('data/FI_T10.csv')[['Stkcd','Indcd']][2:]
    df4.rename({'Stkcd':'证券代码','Indcd':"行业代码"},axis='columns',inplace=True)
    df4.drop_duplicates('证券代码', 'first', inplace=True)
    df4['证券代码'] = df4['证券代码'].astype(int)
    df3 = pd.merge(df3,df4,on='证券代码',how='left')
    
    #计算行业平均chato_i
    df5 = df3[["行业代码","会计期间",'chato']].groupby(["行业代码","会计期间"]).mean().reset_index()

    df5.rename({'chato':'chato_i'},axis='columns',inplace=True)
    df6 = pd.merge(df3,df5,on=["行业代码","会计期间"],how='left')
    #计算chato_ia
    df6.fillna(0,inplace=True)
    df6["chato_ia"] = df6["chato"] - df6["chato_i"]
    df6.sort_values(by=["证券代码","会计期间"],inplace=True)
    for i in range(1,len(df6)):
        if df6.loc[i,"证券代码"] != df6.loc[i-1,"证券代码"]:
            df6.loc[i,'chato'] = df6["chato_i"] = None
    df6[["证券代码","会计期间","chato","chato_ia"]].to_csv('data/季_13_14.csv',index=False,encoding='utf-8_sig')
# cal_13_14()

def cal_15():
    """股本变动数 chcsho 月
    用到 TRD_Capchg
    """
    print("cal 15")
    df1 = pd.read_csv('data/TRD_Capchg.csv')
    df1 = df1[["Stkcd","Shrchgdt","Nshrttl"]]
    df1.rename({'Stkcd':"证券代码",'Shrchgdt':"会计期间",'Nshrttl':"总股数"},axis='columns',inplace=True)
    #修改时间表示
    df1["year"] = pd.DatetimeIndex(df1["会计期间"]).year.astype(int)
    df1["year"] = df1["year"].astype(str)
    df1["month"] = pd.DatetimeIndex(df1["会计期间"]).month.astype(str)
    for i in range(len(df1)):
        if len(df1.loc[i,"month"]) == 1:
             df1.loc[i,"month"] = '0'+df1.loc[i,"month"]
    df1["会计期间"] = df1["year"] + df1["month"]
    #每月保留最后一个
    df1.drop_duplicates(["证券代码","会计期间"],keep='last',inplace=True)
    df1.reset_index(drop=True,inplace=True)

    def next_ym(ym):
        #下一个月
        y,m = ym[:4],ym[4:]
        if m == '12':
            y = str(int(y) + 1)
            m = '01'
        elif m in['11','10','09']:
            m = str(int(m)+1)
        else:
            m = '0' + str(int(m)+1)
        return y+m

    def fill(df,id,start,end,value):
        #填充一行
        while end != next_ym(start):
            start = next_ym(start)
            df.loc[len(df)] = [id,start,value,None,None]
            
    #填充空月
    length = len(df1)
    print(length)
    for i in range(1,length):
        if i%10000 == 0:
            print(i)
        if df1.loc[i,"证券代码"] == df1.loc[i-1,"证券代码"]:
            value = df1.loc[i-1,"总股数"]
            fill(df1,df1.loc[i,"证券代码"],df1.loc[i-1,"会计期间"],df1.loc[i,"会计期间"],value)
    df1["会计期间"] = [x.date() for x in pd.to_datetime(df1["会计期间"],format='%Y%m')]
    df1.sort_values(["证券代码","会计期间"],inplace=True)
    df1["chcsho"] = (df1["总股数"] - df1["总股数"].shift(1))/df1["总股数"].shift(1)
    for i in range(1,len(df1)):
        if df1.loc[i,"证券代码"] != df1.loc[i-1,"证券代码"]:
            df1.loc[i,"chcsho"] = None
    df1.fillna(0,inplace=True)
    df1[["证券代码","会计期间","chcsho"]].to_csv('data/月_15.csv',index=False,encoding='utf-8_sig')
# cal_15()

def cal_16_34():
    """行业调整后的员工人数变化 chempia 年
    员工人数变化百分比 hire 年
    用到 CG_Ybasic FI_T10
    """
    df1 = pd.read_excel('data/CG_Ybasic.xlsx')
    df1 = df1[["Stkcd","Reptdt","Y0601b"]]
    df1.rename({'Stkcd':"证券代码",'Reptdt':"会计期间",'Y0601b':"员工人数"},axis='columns',inplace=True)
    df1 = df1[2:].reset_index()
    df1["delta_员工人数"] = df1["员工人数"]-df1["员工人数"].shift(1)
    for i in range(1,len(df1)):
        if df1.loc[i,"证券代码"]!=df1.loc[i-1,"证券代码"]:
            df1.loc[i,"delta_员工人数"] = None
    df1["hire"] = df1["delta_员工人数"]/(df1["员工人数"].shift(1)+1)#员工人数可能为0

    #加入行业代码
    df3 = pd.read_excel('data/FI_T10.csv')[['Stkcd','Indcd']][2:]
    df3.rename({'Stkcd':'证券代码','Indcd':"行业代码"},axis='columns',inplace=True)
    df3.drop_duplicates('证券代码', 'first', inplace=True)
    df4 = pd.merge(df1,df3,on='证券代码',how='left')

    #计算行业平均值
    df5 = df4[["行业代码","会计期间",'hire']].groupby(["行业代码","会计期间"]).mean().reset_index()
    df5.rename({'hire':'hire_i'},axis='columns',inplace=True)

    #计算chempia
    df6 = pd.merge(df4,df5,how='left')
    df6["chempia"] = df6["hire"] - df6["hire_i"]
    for i in range(1,len(df1)):
        if df6.loc[i,"证券代码"]!=df6.loc[i-1,"证券代码"]:
            df6.loc[i,"chempia"] = None
    df6[["证券代码","会计期间","chempia","hire"]].to_csv('data/年_16_34.csv',index=False,encoding='utf-8_sig')
# cal_16_34()

def cal_17():
    """存货变动资产比 chinv 季度
    用到FS_Combas
    """
    df1 = pd.read_csv('data/FS_Combas.csv')
    df1 = df1[["Stkcd","Accper","A001123000",'A001000000']]
    df1.rename({'Stkcd':"证券代码",'Accper':"会计期间",'A001123000':"存货",'A001000000':"总资产"},axis='columns',inplace=True)
    df1['存货比'] = df1["存货"]/df1["总资产"]
    df1['chinv'] = df1['存货比'] - df1['存货比'].shift(1)
    for i in range(1,len(df1)):
        if df1.loc[i,"证券代码"]!=df1.loc[i-1,"证券代码"]:
            df1.loc[i,"chinv"] = None
    df1 = df1[["证券代码","会计期间","chinv"]]
    df1.to_csv('data/季_17.csv',index=False,encoding='utf-8_sig')
# cal_17()

def cal_18_40_41_42_43_44():
    """ 月
    6个月的动量变化 chmom
    最大日return maxret ，使用考虑现金红利的日个股回报率 Dretwd
    12月动量 mom12m
    1月动量 mom1m 
    6月动量 mom6m
    36月动量 mom36m
    用到 daily_return
    update 23/2/15: 在t月结束
    """
    df1 = pd.read_csv('data/日个股回报率/daily_return.csv')[["Stkcd","Trddt","Dsmvosd","Dsmvtll","Dretwd","Clsprc"]]
    # df1 = df1[-1000:].reset_index()
    df1.rename({'Stkcd':"证券代码",'Trddt':"会计期间","Dsmvosd":"流通市值","Dsmvtll":"总市值","Dretwd":"回报率","Clsprc":"收盘价"},axis='columns',inplace=True)
    df1["year"] = pd.DatetimeIndex(df1["会计期间"]).year.astype(str)
    # df1["year"] = df1["year"].astype(int)
    df1["year"] = df1["year"].astype(str)
    df1.reset_index(drop=True,inplace=True)
    df1["month"] = pd.DatetimeIndex(df1["会计期间"]).month.astype(str)
    for i in range(len(df1)):
        if len(df1.loc[i,"month"]) == 1:
             df1.loc[i,"month"] = '0'+df1.loc[i,"month"]

    #40 最大日回报
    df2 = df1
    df2["会计期间"] = df2["year"]+df2["month"]
    df2 = df2[["证券代码","会计期间","回报率"]].groupby(["证券代码","会计期间"]).max().reset_index()
    # df2["回报率"] = df2["回报率"].shift(1)
    df2.loc[0,"回报率"] = None
    for i in range(1,len(df2)):
        if df2.loc[i,"证券代码"] != df2.loc[i-1,"证券代码"]:
            df2.loc[i,"回报率"] = None
    df2.rename({"回报率":"maxret"},axis='columns',inplace=True)
    df2["会计期间"] = [x.date() for x in pd.to_datetime(df2["会计期间"],format='%Y%m')]
    df2.to_csv('data/月_40.csv',index=False,encoding='utf-8_sig')

    #41-44 动量
    df3 = df1
    df3["会计期间"] = df3["year"]+df3["month"]
    #mom1m :1-month cumulative return
    df3 = df3[["证券代码","会计期间","回报率"]].groupby(["证券代码","会计期间"]).sum().reset_index()
    df3.rename({"回报率":"mom1m"},axis='columns',inplace=True)
    # df3["mom1m"] = df3["mom1m"].shift(1)
    
    #mom6m: 5-month cumulative returns ,from t-5 to t-1
    df3["mom6m"] = df3["mom1m"].rolling(5).sum().shift(1)
    df3["mom12m"] = df3["mom1m"].rolling(11).sum().shift(1)

    df3["mom36m"] = df3["mom1m"].rolling(35).sum()
    df3["mom11m"] = df3["mom1m"].rolling(11).sum()
    #t-35 -> t-12 
    df3["mom36m"] = (df3["mom36m"] - df3["mom11m"])
    for i in range(36,len(df3)):
        if df3.loc[i,"证券代码"] != df3.loc[i-5,"证券代码"]:
            df3.loc[i,"mom6m"] = None
        if df3.loc[i,"证券代码"] != df3.loc[i-11,"证券代码"]:
            df3.loc[i,"mom12m"] = None
        if df3.loc[i,"证券代码"] != df3.loc[i-36,"证券代码"]:
            df3.loc[i,"mom36m"] = None
    df3.drop(columns='mom11m',inplace=True)
        
    #18 动量变化 chmom Cumulative returns from months t - 6 to t - 1 minus months t - 12 to t - 7
    df3["chmom"] = df3["mom6m"] - (df3["mom12m"] - df3["mom6m"])
    df3["会计期间"] = [x.date() for x in pd.to_datetime(df3["会计期间"],format='%Y%m')]
    # df3[["证券代码","会计期间","mon36m"]].to_csv('data/月_18_41_42_43_44_.csv',index=False,encoding='utf-8_sig')
    df3.to_csv('data/月_18_41_42_43_44.csv',index=False,encoding='utf-8_sig')
# cal_18_40_41_42_43_44()
#here

def cal_25_26():
    """两个虚拟变量divi divo，表示今年分红去年没分；今年没分去年分了 年
    不管Disttyp的类型，只要这年存在记录，就认为这年分红了
    用到 TRD_Cptl"""
    df1 = pd.read_csv('data/TRD_Cptl.csv')[["Stkcd","Exdistdt"]]
    df1['year'] = pd.DatetimeIndex(df1["Exdistdt"]).year
    df1 = df1[(df1.year>=2020) | (df1.year==2022)]

    df1['dividend'] = True
    df1 = df1.drop_duplicates(subset=['Stkcd','year']).reset_index()
    df1.drop(labels=['Exdistdt','index'],axis='columns',inplace=True)

    #插值
    def next_year(year):
        return year+1
    for i in range(1,len(df1)):
        if df1.loc[i,"Stkcd"] == df1.loc[i-1,"Stkcd"]:
            year = next_year(df1.loc[i-1,"year"])
            while year != df1.loc[i,"year"]:
                df1.loc[len(df1)] = [df1.loc[i,"Stkcd"],year,False]
                year = next_year(year)
    df1 = df1.sort_values(by=["Stkcd","year"]).reset_index()#需要遍历df时，要reset_index

    #按行计算 divi 与 divo
    for i in range(1,len(df1)):
        if df1.loc[i,"Stkcd"] == df1.loc[i-1,"Stkcd"]:
            if df1.loc[i,"dividend"] == True and df1.loc[i-1,"dividend"] == False:
                df1.loc[i,"divi"] = True
            if df1.loc[i,"dividend"] == False and df1.loc[i-1,"dividend"] == True:
                df1.loc[i,"divo"] = True
    df1.fillna(False,inplace=True)
    df1["会计期间"] = pd.to_datetime(df1["year"],format='%Y')
    df1.rename({"Stkcd":"证券代码"},axis='columns',inplace=True)
    df1[["证券代码","会计期间","divi","divo"]].to_csv('data/年_25_26.csv',index=False,encoding='utf-8_sig')
# cal_25_26()

def cal_28():
    """ dy 年
    Total dividends divided by market capitalization at year end.
    现金分红，Disttyp为CA，每股配发Amount
    股数：daily_return->Dnshrtrd
    用到 TRD_Cptl 市值：daily_return->Dsmvtll
    """
    df1 = pd.read_csv('data/TRD_Cptl.csv')[["Stkcd","Disttyp","Exdistdt","Amount"]]
    df2 = pd.read_csv('data/日个股回报率/daily_return.csv')[["Stkcd","Trddt","Dnshrtrd","Dsmvtll"]]
    df1 = df1[df1["Disttyp"]=='CA']
    df1.rename({"Exdistdt":"Trddt"},axis='columns',inplace=True)
    df3 = pd.merge(df1,df2,on=["Stkcd","Trddt"])
    df3["dividends"] = df3["Amount"] * df3["Dnshrtrd"]

    #以年为单位对分红求和
    df3['year'] = pd.DatetimeIndex(df3["Trddt"]).year
    df4 = df3[["Stkcd","year","dividends"]].groupby(["Stkcd","year"]).sum().reset_index()
    
    #保留个股回报率每年最后一天，作为市值的计算标准
    df2['year'] = pd.DatetimeIndex(df2["Trddt"]).year
    df2.drop_duplicates(subset=["Stkcd","year"],keep='last',inplace=True)

    df5 = pd.merge(df2,df4,on=["Stkcd","year"],how='left')#有的年份没有分红
    df5["dy"] = df5["dividends"]/df5["Dsmvtll"]
    df5.rename({"Stkcd":"证券代码","Trddt":"会计期间"},axis='columns',inplace=True)
    df5[["证券代码","会计期间","dy"]].to_csv('data/年_28.csv',index=False,encoding='utf-8_sig')
# cal_28()

def cal_29():
    """ ear 
    盈利公告前后共3天的日回报率之和。这里的3天指的是交易日
    用到ER_RelForcDate daily_return
    23/2/22 UPDATE: 改成月度数据
    """
    print("cal 29")
    df1 = pd.read_csv('data/ER_RelForcDate.csv')[["Symbol","ActRelDate"]][2:].reset_index()
    df1.rename({'Symbol':"证券代码",'ActRelDate':"会计期间"},axis='columns',inplace=True)
    df1['year'] = pd.DatetimeIndex(df1["会计期间"]).year
    df1.drop(columns='index',inplace=True)
    
    df1["证券代码"] = df1["证券代码"].astype(int)
    # df1["会计期间"] = pd.to_datetime(df1["会计期间"])
    df1['announce_token'] = df1.index#第几个公告
    df1['announce'] = True

    df2 = pd.read_csv('data/日个股回报率/daily_return.csv')[["Stkcd","Trddt","Dretwd"]]
    df2.rename({'Stkcd':"证券代码",'Trddt':"会计期间","Dretwd":"回报率"},axis='columns',inplace=True)
    df3 = pd.merge(df1,df2,on=["证券代码","会计期间"],how='right')#保留daily_return的所有行
    df1["会计期间"] = pd.to_datetime(df1["会计期间"])
    df2["会计期间"] = pd.to_datetime(df2["会计期间"])
    
    # 添加公告日前后1天
    for i in range(1,len(df3)):
        if df3.loc[i,"announce"] == True:
            df3.loc[i-1,"announce_token"] = df3.loc[i+1,"announce_token"] = df3.loc[i,"announce_token"]
    df3.to_csv('data/tmp3.csv',index=False,encoding='utf-8_sig')
    df3.dropna(subset=["announce_token"],inplace=True)
    df3["announce_token"] = df3["announce_token"].astype(int)#自动变成float了
    df4 = df3[["announce_token","回报率"]].groupby(["announce_token"]).sum()#加和3天
    print(len(df3),len(df4))

    df5 = pd.merge(df1,df4,on="announce_token")
    df5.rename({'回报率':"ear"},axis='columns',inplace=True)
    df5[["证券代码","会计期间","ear"]].to_csv('data/月_29.csv',index=False,encoding='utf-8_sig')
# cal_29()


def cal_30():
    """股东权益变化比例 egr 季度
    用到 FS_Combas
    """
    df1 = pd.read_csv('data/FS_Combas.csv')
    df1 = df1[["Stkcd","Accper","A003000000"]]
    df1.rename({'Stkcd':"证券代码",'Accper':"会计期间","A003000000":"股东权益"},axis='columns',inplace=True)
    for i in range(1,len(df1)):
        if df1.loc[i,"证券代码"] == df1.loc[i-1,"证券代码"]:
            try:
                df1.loc[i,"egr"] = (df1.loc[i,"股东权益"]-df1.loc[i-1,"股东权益"])/df1.loc[i-1,"股东权益"]
            except:
                df1.loc[i,"egr"] = None
    df1.drop(columns='股东权益',inplace=True)
    df1.to_csv('data/季_30.csv',index=False,encoding='utf-8_sig')
# cal_30()

def cal_31():
    """
    gma: Revenue minus cost of goods sold divided by lagged total assets. 季度
    Revenue: 现金流量表FS_Comscfd->C001001000 销售商品、提供劳务收到的现金
    cost of goods:  FS_Comscfd->C001014000 购买商品、接受劳务支付的现金
    lagged total assets: 资产负债表FS_Combas->	A001000000
    """
    df1 = pd.read_csv('data/FS_Comscfd.csv')[["Stkcd","Accper","C001001000","C001014000"]]
    df2 = pd.read_csv('data/FS_Combas.csv')[["Stkcd","Accper","A001000000"]]
    df1.rename({'Stkcd':"证券代码",'Accper':"会计期间",'C001001000':"收入",'C001014000':"支出"},axis='columns',inplace=True)
    df2.rename({'Stkcd':"证券代码",'Accper':"会计期间",'A001000000':"资产"},axis='columns',inplace=True)
    df3 = pd.merge(df1,df2,on=["证券代码","会计期间"])
    df3["gma"] = (df3["收入"]-df3["支出"])/df3["资产"]
    df3[["证券代码","会计期间","gma"]].to_csv('data/季_31.csv',index=False,encoding='utf-8_sig')
# cal_31()

def cal_32():
    """grCAPX : Percentage change in capital expenditures from year t-2 to year t. 年度
    capital expenditures:现金流量表FS_Comscfd->C002006000 购建固定资产、无形资产和其他长期资产支付的现金
    """
    df1 = pd.read_csv('data/FS_Comscfd.csv')[["Stkcd","Accper","C002006000"]]
    df1.rename({'Stkcd':"证券代码",'Accper':"会计期间",'C002006000':"支出"},axis='columns',inplace=True)
    df1['year'] = pd.DatetimeIndex(df1["会计期间"]).year
    df2 = df1[["证券代码","year","支出"]].groupby(["证券代码","year"]).sum().reset_index()
    df2["会计期间"] = pd.to_datetime(df2["year"],format='%Y')
    for i in range(2,len(df2)):
        if df2.loc[i,"证券代码"] == df2.loc[i-2,"证券代码"]:
            df2.loc[i,"grCAPX"] = (df2.loc[i,"支出"]-df2.loc[i-2,"支出"])/df2.loc[i-2,"支出"]
    df2[["证券代码","会计期间","grCAPX"]].to_csv('data/季_32.csv',index=False,encoding='utf-8_sig')
# cal_32()

def cal_33():
    """
    herf 销售占比平方 Sum of squared percentage sales in industry for each company 季度
    销售:FS_Comins->B001100000
    行业:FI_T10->Indcd
    """
    df1 = pd.read_csv('data/FS_Comins.csv')[["Stkcd","Accper","B001100000"]]
    df1.rename({'Stkcd':"证券代码",'Accper':"会计期间",'B001100000':"销售额"},axis='columns',inplace=True)
    df2 = pd.read_excel('data/FI_T10.csv')[["Stkcd","Accper","Indcd"]][2:]
    df2.rename({'Stkcd':"证券代码",'Accper':"会计期间",'Indcd':"行业代码"},axis='columns',inplace=True)
    df2["证券代码"] =df2["证券代码"].astype(int)
    df1["会计期间"] = pd.to_datetime(df1["会计期间"])
    df2["会计期间"] = pd.to_datetime(df2["会计期间"])
    df3 = pd.merge(df1,df2,on=["证券代码","会计期间"])
    #计算行业当季度销售总量
    df4 = df3[["行业代码","会计期间","销售额"]].groupby(["行业代码","会计期间"]).sum()
    df4.rename({'销售额':"销售总额"},axis='columns',inplace=True)
    #和df3合并
    df5 = pd.merge(df3,df4,on=["行业代码","会计期间"],how='left')
    #计算销售占比，及其平方
    df5["herf"] = (df5["销售额"]/df5["销售总额"])**2
    df5[["证券代码","会计期间","herf"]].to_csv('data/季_33.csv',index=False,encoding='utf-8_sig')
# cal_33()

def cal_35():
    """
    idiovol : Standard deviation of residuals of weekly returns on weekly equally-weighted market
returns for three years prior to month end. 月
    用到 daily_return
    """
    RET = pd.read_csv('data/日个股回报率/daily_return.csv')
    RET = RET[['Stkcd', 'Trddt','Dretwd']]

    RET["year"] = pd.DatetimeIndex(RET["Trddt"]).year
    RET["month"] = pd.DatetimeIndex(RET["Trddt"]).month

    def My_YM(Set,Col):
        Set['DATE'] = Set[Col].astype(str).replace('\-', '', regex=True)
        Set['Yearmon'] = Set['DATE'].astype(int) // 100
    My_YM(RET,'Trddt')

    # RETS = RET[['Dretwd', 'Stkcd','Yearmon']
    #         ].groupby(['Stkcd','Yearmon']
    #         ).std().reset_index()
    # RETS['volatility'] = RETS['Dretwd'].shift(1)
    # RETS = RETS.dropna()
    # RETS[['Stkcd','Yearmon','volatility']].to_csv(
    #     "data/RETS_volatility_treated.csv",
    #     encoding='utf_8_sig',index = False)

    # 月回报
    RETM = RET.drop_duplicates(subset = ['Stkcd','Yearmon'], keep = 'last').reset_index(drop=True)

    # 等权市场回报率
    RETMM = RETM[['Dretwd', 'Yearmon']
            ].groupby(['Yearmon']
            ).mean().reset_index()
    RETMM = RETMM.rename(
        columns = {'Dretwd': 'MKT'}) 
    
    RETM = pd.merge(RETM, RETMM, on='Yearmon', how='left')
    RETM = RETM.rename(
        columns = {'Dretwd': 'RET'}) 

    Stkcd_list = np.unique(RET['Stkcd'])
    Mon_list = RETMM['Yearmon']
    ID_list = np.arange(len(Mon_list))

    MID = pd.DataFrame(columns=['ID_Mon','Yearmon'],
                        index=np.arange(len(ID_list)))
    MID['ID_Mon'] = ID_list
    MID['Yearmon'] = Mon_list

    RETM = pd.merge(RETM, MID, on='Yearmon', how='left')

    RETM_ = RETM[['Stkcd', 'ID_Mon', 'Yearmon','RET', 'MKT']]
    def My_IVOL(Set,Stkcd,ID):
        # ['Stkcd', 'RET', 'Yearmon', 'MKT', 'ID_Mon']
        try:
            Set_s = Set[(Set['Stkcd'] == Stkcd)]
            ID_Mon_0 = ID - 36
            ID_Mon_1 = ID - 1
            Set_m = Set_s[
                (Set_s['ID_Mon'] >= ID_Mon_0) & 
                (Set_s['ID_Mon'] <= ID_Mon_1)
                ]        
            formula_test = 'RET~MKT'
            result = sm.ols(formula=formula_test, 
                            data = Set_m).fit()
            idiovol = np.std(result.resid)
            return idiovol
        except ValueError:
            return 0
        

    #%% 循环结构
    from itertools import product
    X1 = []
    X2 = []
    X3 = []
    cnt = 0
    tot = len(list(product(Stkcd_list,ID_list[36:])))
    for i,j in product(Stkcd_list,ID_list[36:]):
        if cnt % 10000 == 0:
            print(cnt,tot)
        X1.append(i)
        X2.append(Mon_list[j])
        X3.append(My_IVOL(RETM_,i,j))
        cnt += 1
    R4 = pd.DataFrame(columns=['证券代码','会计期间','idiovol'],
                        index=np.arange(len(X1)))
    R4['证券代码'] = X1
    R4['会计期间'] = X2
    R4['idiovol'] = X3
    R4["会计期间"] = [x.date() for x in pd.to_datetime(R4["会计期间"],format='%Y%m')]
    R4.to_csv("data/月_35.csv",encoding='utf_8_sig',index = False)
# cal_35()

def cal_36():
    """
    ill : Average of daily (absolute return/RMB volume) in month t. 月
    绝对收益: 不考虑现金红利的日个股回报率 daily_return->Dretwd 的绝对值 
    交易量: daily_info->Tolstknva 
    """
    print("cal 36")
    df1 = pd.read_csv('data/日个股回报率/daily_return.csv')[['Stkcd', 'Trddt','Dretwd']]
    df1.rename({'Stkcd':"证券代码",'Trddt':"会计期间",'Dretwd':"收益"},axis='columns',inplace=True)
    df2 = pd.read_csv('data/日交易统计文件/daily_info.csv')[['Stkcd', 'Trddt','Tolstknva']]
    df2.rename({'Stkcd':"证券代码",'Trddt':"会计期间",'Tolstknva':"交易量"},axis='columns',inplace=True)
    #2023.1及以后的
    df1["year"] = pd.DatetimeIndex(df1["会计期间"]).year
    df1["month"] = pd.DatetimeIndex(df1["会计期间"]).month
    df2["year"] = pd.DatetimeIndex(df2["会计期间"]).year
    df2["month"] = pd.DatetimeIndex(df2["会计期间"]).month
    df1 = df1[(df1["year"]>=2023) & (df1["month"]>=1)].reset_index()
    df2 = df2[(df2["year"]>=2023) & (df2["month"]>=1)].reset_index()
    print(df1.head(),df2.head())

    df1 = pd.merge(df1,df2,on=["证券代码","会计期间"])
    df1["绝对收益"] = abs(df1["收益"])
    df1["value"] = df1["绝对收益"]/df1["交易量"]
    df1["year"] = pd.DatetimeIndex(df1["会计期间"]).year.astype(str)
    df1["month"] = pd.DatetimeIndex(df1["会计期间"]).month.astype(str)
    # df1.to_csv('data/tmp1.csv',index=False,encoding='utf-8_sig')

    for i in range(len(df1)):
        if len(df1.loc[i,"month"]) == 1:
            df1.loc[i,"month"] = '0'+df1.loc[i,"month"]
    df1["会计期间"] = df1["year"] + df1["month"]
    # df1.to_csv('data/tmp2.csv',index=False,encoding='utf-8_sig')

    df2 = df1[["证券代码","会计期间","value"]].groupby(["证券代码","会计期间"]).mean().reset_index()
    df2["会计期间"] = [x.date() for x in pd.to_datetime(df2["会计期间"],format='%Y%m')]
    df2.rename({'value':"ill"},axis='columns',inplace=True)
    df2.to_csv('data/月_36.csv',index=False,encoding='utf-8_sig')
cal_36()  

def cal_37():
    """
    invest: The sum of annual change in fixed assets and annual change in inventories divided by
lagged total assets. 年
    fixed assets: FS_Combas->A001212000
    inventories: FS_Combas->A001123000
    total assets: FS_Combas->A001000000
    用到 FS_Combas
"""
    df = pd.read_csv('data/FS_Combas.csv')[["Stkcd","Accper","A001212000","A001123000","A001000000"]]
    df.rename({'Stkcd':"证券代码",'Accper':"会计期间",'A001212000':"固定资产",'A001123000':"存货",'A001000000':"总资产"},axis='columns',inplace=True)
    #按年
    df['会计期间'] = pd.DatetimeIndex(df["会计期间"]).year
    df = df.groupby(["证券代码","会计期间"]).sum().reset_index()
    df["invest"] = (df["固定资产"] - df["固定资产"].shift(1) + df["存货"] - df["存货"].shift(1))/df["总资产"].shift(1)
    for i in range(1,len(df)):
        if df.loc[i,"证券代码"] != df.loc[i-1,"证券代码"]:
            df.loc[i,"invest"] = None
    df['会计期间'] = pd.to_datetime(df['会计期间'],format='%Y')
    df[["证券代码","会计期间","invest"]].to_csv('data/年_37.csv',index=False,encoding='utf-8_sig')
# cal_37()

def cal_38():
    """lev Total liabilities divided by quarter-end market capitalization. 季
    liabilities: FS_Combas->A002000000
    capitalization: daily_return->Dsmvtll
    用到 FS_Combas daily_return
    """
    df1 = pd.read_csv('data/FS_Combas.csv')[["Stkcd","Accper","A002000000"]]
    df1.rename({'Stkcd':"证券代码",'Accper':"会计期间",'A002000000':"负债总计"},axis='columns',inplace=True)
    df2 = pd.read_csv('data/日个股回报率/daily_return.csv')[["Stkcd","Trddt","Dsmvtll"]]
    df2.rename({'Stkcd':"证券代码",'Trddt':"会计期间","Dsmvtll":"市值"},axis='columns',inplace=True)
    df3 = pd.merge(df1,df2,on=["证券代码","会计期间"],how='left')
    df3["lev"] = df3["负债总计"]/df3["市值"]
    df3[["证券代码","会计期间","lev"]].to_csv('data/季_38.csv',index=False,encoding='utf-8_sig')
# cal_38()

def cal_39():
    """lgr : Quarterly percentage change in total liabilities 季度
    liabilities: FS_Combas->A002000000
    用到 FS_Combas
    """
    df1 = pd.read_csv('data/FS_Combas.csv')[["Stkcd","Accper","A002000000"]]
    df1.rename({'Stkcd':"证券代码",'Accper':"会计期间",'A002000000':"负债总计"},axis='columns',inplace=True)
    df1["lgr"] = (df1["负债总计"]-df1["负债总计"].shift(1))/df1["负债总计"].shift(1)
    for i in range(1,len(df1)):
        if df1.loc[i,"证券代码"] != df1.loc[i-1,"证券代码"]:
            df1.loc[i,"lgr"] = None
    df1[["证券代码","会计期间","lgr"]].to_csv('data/季_39.csv',index=False,encoding='utf-8_sig')
# cal_39()

def cal_45():
    """ms Financial statement score 年度
    8个bool之和，bool表示与行业中位数的大小
    https://zhuanlan.zhihu.com/p/112102549
    """
    pass

def cal_46_47():
    """市值 mve mve_ia 季度
    用到FI_T10.csv
    附录里是月度，但数据是季度
    23/2/15 update: t月的市值
    TODO 月度对应的表
    """
    df1 = pd.read_csv('data/FI_T10.csv')
    df1 = df1[["Stkcd","Accper","Indcd","F100801A"]]
    # df1 = df1[2:].reset_index()
    df1 = df1.reset_index()
    df1.rename({'Stkcd':"证券代码",'Accper':"会计期间",'Indcd':"行业代码",'F100801A':"市值"},axis='columns',inplace=True)
    df1['mve'] = np.log(df1["市值"].astype(float))

    # for i in range(1,len(df1)):
    #     if df1.loc[i,"证券代码"] == df1.loc[i-1,"证券代码"]:
    #         try:
    #             df1.loc[i,'mve'] = np.log(int(df1.loc[i-1,"市值"]))
    #         except:
    #             df1.loc[i,'mve'] = None
    
    df2 = df1[["行业代码","会计期间",'mve']].groupby(["行业代码","会计期间"]).mean().reset_index()
    df2.rename({'mve':'mve_i'},axis='columns',inplace=True)
    df3 = pd.merge(df1,df2,on=["行业代码","会计期间"],how='left')
    df3['mve_ia'] = df3['mve']-df3['mve_i']
    df3 = df3[["证券代码","会计期间","mve","mve_ia"]]
    df3.to_csv('data/季_46_47.csv',index=False,encoding='utf-8_sig')
# cal_46_47()

def cal_48():
    """ 盈余连续增加数 nincr 季度
    用到FS_Comins"""
    df1 = pd.read_csv('data/FS_Comins.csv')
    df1 = df1[["Stkcd","Accper","B002000000"]]
    df1.rename({'Stkcd':"证券代码",'Accper':"会计期间",'B002000000':"净利润"},axis='columns',inplace=True)
    df1 = df1[2:].reset_index()
    nincr = 0
    for i in range(1,len(df1)):
        if df1.loc[i,"证券代码"] == df1.loc[i-1,"证券代码"] and df1.loc[i,"净利润"] > df1.loc[i-1,"净利润"]:
            nincr += 1
            nincr = min(nincr,8)
            df1.loc[i,"nincr"] = nincr
        else:
            nincr = 0
            df1.loc[i,"nincr"] = 0
    df1 = df1[["证券代码","会计期间","nincr"]]
    df1.to_csv('data/季_48.csv',index=False,encoding='utf-8_sig')
# cal_48()

def cal_50():
    """ 组织资本 orgcap 季度
    用到 FS_Comins
    # check https://www.tandfonline.com/doi/full/10.1080/1540496X.2022.2057846
    """
    df1 = pd.read_csv('data/FS_Comins.csv')[["Stkcd","Accper","B001210000","B001209000"]]
    df1 = df1[2:]
    df1.rename({'Stkcd':"证券代码",'Accper':"会计期间","B001210000":"管理费用","B001209000":"销售费用"},axis='columns',inplace=True)
    df1["orgcap"] = df1["管理费用"] + df1["销售费用"]
    df1[["证券代码","会计期间","orgcap"]].to_csv('data/季_50.csv',index=False,encoding='utf-8_sig')
# cal_50()

def cal_51():
    """pchcapx_ia % Industry adjusted % change in capital expenditures 年度
    capital expenditures:现金流量表FS_Comscfd->C002006000 购建固定资产、无形资产和其他长期资产支付的现金
    行业 :FI_T10 Indcd
    """
    df1 = pd.read_csv('data/FS_Comscfd.csv')[["Stkcd","Accper","C002006000"]]
    df1.rename({'Stkcd':"证券代码",'Accper':"会计期间","C002006000":"支出"},axis='columns',inplace=True)
    df2 = pd.read_excel('data/FI_T10.csv')[["Stkcd","Accper","Indcd"]][2:]
    df2.rename({'Stkcd':"证券代码",'Accper':"会计期间","Indcd":"行业"},axis='columns',inplace=True)

    #整理年度数据
    df1["会计期间"] = pd.DatetimeIndex(df1["会计期间"]).year.astype(str)
    df1 = df1.groupby(["证券代码","会计期间"]).sum().reset_index()
    df1["pchcapx"] = (df1["支出"]-df1["支出"].shift(1))/df1["支出"].shift(1)
    for i in range(1,len(df1)):
        if df1.loc[i,"证券代码"] != df1.loc[i-1,"证券代码"]:
            df1.loc[i,"pchcapx"] = None
    df2["会计期间"] = pd.DatetimeIndex(df2["会计期间"]).year.astype(str)
    df2["证券代码"] = df2["证券代码"].astype(int)
    df2.drop_duplicates(["证券代码","会计期间"],keep='last',inplace=True)

    #合并行业数据
    df3 = pd.merge(df1,df2,on=["证券代码","会计期间"],how='left')
    df4 = df3[["行业","会计期间","pchcapx"]].groupby(["行业","会计期间"]).median()
    df4.rename({'pchcapx':"pchcapx_i"},axis='columns',inplace=True)
    df5 = pd.merge(df3,df4,on=["行业","会计期间"])
    df5["pchcapx_ia"] = df5["pchcapx"]-df5["pchcapx_i"]
    df5['会计期间'] = pd.to_datetime(df5['会计期间'],format='%Y')
    df5.sort_values(["证券代码","会计期间"],inplace=True)
    df5[["证券代码","会计期间","pchcapx_ia"]].to_csv('data/年_51.csv',index=False,encoding='utf-8_sig')
# cal_51()

def cal_61():
    """pricedelay The proportion of variation in weekly returns for 36 months ending in month
t explained by four lags of weekly market returns incremental to contemporaneous market
return. 月
    """
    pass

def cal_71():
    """roeq: Income before extraordinary items divided by lagged common shareholders' equity.
    收益: FS_Comins->B001100000
    common shareholders' equity: FS_Combas->A003000000
    用到 FS_Comins FS_Combas 
    """
    df1 = pd.read_csv('data/FS_Comins.csv')[['Stkcd','Accper','B001100000']]
    df1.rename({'Stkcd':"证券代码",'Accper':"会计期间",'B001100000':"收益"},axis='columns',inplace=True)
    df2 = pd.read_csv('data/FS_Combas.csv')[['Stkcd','Accper','A003000000']]
    df2.rename({'Stkcd':"证券代码",'Accper':"会计期间",'A003000000':"股东权益"},axis='columns',inplace=True)
    df1.fillna(0,inplace=True)
    
    df1["会计期间"] = pd.to_datetime(df1["会计期间"])
    df2["会计期间"] = pd.to_datetime(df2["会计期间"])
    df3 = pd.merge(df1,df2,on=["证券代码","会计期间"])
    df3["roeq"] = df3["收益"]/(df3["股东权益"].shift(1))
    for i in range(1,len(df3)):
        if df3.loc[i,"证券代码"] != df3.loc[i-1,"证券代码"]:
            df3.loc[i,"roeq"] = None
    df3[["证券代码","会计期间","roeq"]].to_csv('data/季_71.csv',index=False,encoding='utf-8_sig')
# cal_71()

def cal_72():
    """
    roic: Quarterly earnings before interest and taxes minus nonoperating income divided by noncash
enterprise value. 季
    收益: FS_Comins->B001100000 
    非营业收入:FS_Comins->B0f1105000
    noncash enterprise value：
        存货:FS_Combas -> A001123000
        固定资产:FS_Combas -> A001212000
        无形资产:FS_Combas -> A001218000
    用到 FS_Comins FS_Combas
    """
    df1 = pd.read_csv('data/FS_Comins.csv')[['Stkcd','Accper','B001100000','B0f1105000']]
    df1.rename({'Stkcd':"证券代码",'Accper':"会计期间",'B001100000':"收益",'B0f1105000':"非营业收入"},axis='columns',inplace=True)
    df2 = pd.read_csv('data/FS_Combas.csv')[['Stkcd','Accper','A001123000','A001212000','A001218000']]
    df2.rename({'Stkcd':"证券代码",'Accper':"会计期间",'A001123000':"存货",'A001212000':"固定资产",'A001218000':"无形资产"},axis='columns',inplace=True)
    df1.fillna(0,inplace=True)
    df2.fillna(0,inplace=True)
    df1["income"] = df1["收益"] - df1["非营业收入"]
    df2["noncash"] = df2["存货"]+df2["固定资产"]+df2["无形资产"]

    df1["会计期间"] = pd.to_datetime(df1["会计期间"])
    df2["会计期间"] = pd.to_datetime(df2["会计期间"])
    df3 = pd.merge(df1,df2,on=["证券代码","会计期间"])
    df3["roic"] = df3["income"]/df3["noncash"]
    df3[["证券代码","会计期间","roic"]].to_csv('data/季_72.csv',index=False,encoding='utf-8_sig')
# cal_72()

def cal_81():
    """
    stdacc: Standard deviation of 16 quarters of accruals from month t - 16 to t - 1.
    用半年_1_2_60中的acc，3个半年的标准差
    用到半年_1_2_60.csv 
    """
    df1 = pd.read_csv('data/半年_1_2_60.csv')[['证券代码','会计期间','acc']]
    df1["stdacc"] = df1["acc"].rolling(window=3).std()
    df1[['证券代码','会计期间','stdacc']].to_csv('data/半年_81.csv',index=False,encoding='utf-8_sig')
# cal_81()

def cal_87(st_ym,ed_ym,start = "1990-01-01", end = "2023-02-29"):
# def cal_87(start = "2020-01-01", end = "2021-01-01"):
    """异常周转率 atr 季度
    用到 announcement dailyinfo daily_return
    见https://zhuanlan.zhihu.com/p/523209413
    特殊事件：
        季度、半年度及年度收益公告
        公司并购公告及其他股权结构重大变化，包括股票回购、重大资本投资、大股东买卖股票等
        债券及股票发行公告
        中国股权改革的公告与实施
        宣布主要管理人员的变动
        定期/特别现金股息公告
        ['8','9','11','12','13','14','16','99','nan']
    周转率=交易量/市值 
        交易量：daily_info->Tolstknva
        市值：daily_return->Dsmvtll
    市场周转率=总交易量/总市值
"""
    # #以证券代码1-5 交易日期2010/1/1-2010/2/29为例
    # df3 = df3[(df3["Symbol"]<=5) & (df3.DeclareDate>=start) & (df3.DeclareDate<=end)]
    # # 截取数据
    # def data_process():
    #     df1 = pd.read_csv('data/日个股回报率/daily_return.csv')
    #     df1 = df1[ (df1["Trddt"]>=start) & (df1["Trddt"]<=end)]
    #     df1 = df1[["Stkcd","Trddt","Dsmvtll"]].rename({'Stkcd':"证券代码",'Trddt':"会计期间","Dsmvtll":"市值"},axis='columns')
    #     df1.to_csv('data/atr/daily_return.csv',index=False,encoding='utf-8_sig')

    #     df2 = pd.read_csv('data/日交易统计文件/daily_info.csv')
    #     df2 = df2[(df2.Trddt>=start) & (df2.Trddt<=end)]
    #     df2 = df2[["Stkcd","Trddt","Tolstknva"]].rename({'Stkcd':"证券代码",'Trddt':"会计期间","Tolstknva":"交易量"},axis='columns')
    #     df2.to_csv('data/atr/daily_info.csv',index=False,encoding='utf-8_sig')

    #     df3 = pd.read_csv('data/公司公告/announcement.csv')[2:]
    #     df3 = df3[(df3.DeclareDate>=start) & (df3.DeclareDate<=end)]
    #     df3 = df3[["Symbol","DeclareDate","ClassID"]].rename({'Symbol':"证券代码",'DeclareDate':"会计期间","ClassID":"事件类型"},axis='columns')
    #     df3.sort_values(by=["证券代码","会计期间"],inplace=True)
    #     df3.reset_index(inplace=True)
    #     df3.fillna(100,inplace=True)
    #     df3["事件类型"] = df3["事件类型"].astype(str)
    #     #处理同日期的情况
    #     #8,9->8+9
    #     for i in range(len(df3)-1):
    #         if df3.loc[i,"事件类型"] == None:
    #             continue
    #         for j in range(i+1,len(df3)):
    #             if df3.loc[i,"证券代码"] == df3.loc[j,"证券代码"] and df3.loc[i,"会计期间"] == df3.loc[j,"会计期间"]:
    #                 df3.loc[i,"事件类型"] += "+"+df3.loc[j,"事件类型"]
    #                 df3.loc[j,"事件类型"] = None
    #             else:
    #                 break
    #     df3.dropna(subset="事件类型",inplace=True)
    #     df3.drop(columns=["index"],inplace=True)
    #     df3.to_csv('data/atr/announcement.csv',index=False,encoding='utf-8_sig')   
    # # data_process()

    # df1 = pd.read_csv('data/atr/announcement.csv')
    # df2 = pd.read_csv('data/atr/daily_info.csv')
    # df3 = pd.read_csv('data/atr/daily_return.csv')
    # df1 = df1[(df1["会计期间"]>=st_ym) & (df1["会计期间"]<ed_ym)]
    # df2 = df2[(df2["会计期间"]>=st_ym) & (df2["会计期间"]<ed_ym)]
    # df3 = df3[(df3["会计期间"]>=st_ym) & (df3["会计期间"]<ed_ym)]
    # df4 = pd.DataFrame(columns=["证券代码","会计期间","事件_8","事件_9","事件_11","事件_12","事件_13","事件_14","事件_16","事件_99","事件_100"])
    # df4 = pd.merge(df1,df4,how='left')
    # df4 = pd.merge(pd.merge(df4,df2,how='outer'),df3,how='outer')
    # df4.fillna(0,inplace=True)
    # df4["会计期间"] = pd.to_datetime(df4["会计期间"])
    # df4.sort_values(by=["证券代码","会计期间"],inplace=True)
    # df4.reset_index(inplace=True)

    # #丢弃非交易日：交易量 and 市值 = 0
    # df4.drop(df4[df4["市值"]==0].index,inplace=True)
    # for i in range(len(df4)):#设置虚拟变量
    #     for j in range(i-3,i+4):
    #         try:
    #             if df4.loc[j,"证券代码"] != df4.loc[i,"证券代码"]:
    #                 continue
    #             for announce_type in [int(float((x))) for x in df4.loc[j,"事件类型"].split('+')]:
    #                 df4.loc[i,f"事件_{announce_type}"] = 1
    #         except:
    #             pass
    
    # #周转率
    # df4["周转率"] = df4["交易量"] / df4["市值"]
    # df5 = df4[["会计期间","交易量"]].groupby(by=["会计期间"]).sum().reset_index()
    # df6 = df4[["会计期间","市值"]].groupby(by=["会计期间"]).sum().reset_index()
    # df7 = pd.merge(df5,df6)
    # df7["市场周转率"] = df7["交易量"]/df7["市值"]
    # df4 = pd.merge(df4,df7[["会计期间","市场周转率"]],on='会计期间',how='left')
    # df4.drop(columns=["事件类型","index"],inplace=True)

    # #保存
    # df4.to_csv(f'data/atr/tmp_{str(ed_ym)[:4]}.csv',index=False,encoding='utf-8_sig')   

    # 回归
    # 每个月，用t-7 到 t-1月的每日数据进行回归，得到模型
    def excess(ym1,ym2):
        """ym1在ym2的前7-1个月内则判断为True,否则为False
        ym1比ym2小
        ym形如202210
        """
        if ym1[:4] == ym2[:4]:
            if int(ym1[4:])+6 >= int(ym2[4:]):#1 7
                return True
            else:
                return False
        else:
            if int(ym1[4:])-12+6 >= int(ym2[4:]):#12 6
                return True
            else:
                return False
    df4 = pd.read_csv(f'data/atr/tmp_{str(ed_ym)[:4]}.csv')   
    df4["year"] = pd.DatetimeIndex(df4["会计期间"]).year.astype(str)
    df4["month"] = pd.DatetimeIndex(df4["会计期间"]).month.astype(str)
    ids = [] #存储已经搜过的股票id
    yms = [] #存储已经搜过的year month
    for i in range(len(df4)):
        if i%1000 == 0:
            print(ed_ym,i,len(df4),datetime.datetime.now().strftime("%m-%d %X"))
        if df4.loc[i,"证券代码"] not in ids:
            ids.append(df4.loc[i,"证券代码"])
            yms = [df4.loc[i,"year"] + df4.loc[i,"month"]] #初始化yms为当前年月，并将当前年月的股票误差设为None
            for j in range(i,len(df4)):
                if df4.loc[i,"证券代码"] == df4.loc[j,"证券代码"] and df4.loc[j,"month"] == df4.loc[i,"month"]:
                    df4.loc[j,"预测周转率"] = None
        if df4.loc[i,"year"] + df4.loc[i,"month"] not in yms:
            yms.append(df4.loc[i,"year"] + df4.loc[i,"month"])
            #新的ym，需要做回归。从i-1行向上找6个月的股票数据
            top = i
            for j in range(i-1,0,-1):
                if df4.loc[i,"证券代码"] != df4.loc[j,"证券代码"] or excess(df4.loc[j,"year"]+df4.loc[j,"month"],df4.loc[i,"year"]+df4.loc[i,"month"]):
                    top = j+1
            #截取j-(i-1)行，做回归
            lr = LinearRegression()
            y = np.array(df4.loc[top:i,"周转率"])
            x = np.array(df4.loc[top:i,["市场周转率","事件_8","事件_9","事件_11","事件_12","事件_13","事件_14","事件_16","事件_99","事件_100"	]])
            lr.fit(x, y)
            #对该股票该月的数据做预测求误差
            #从i+1行向下找1个月的数据
            down = i+1
            for j in range(i,len(df4)):
                if df4.loc[i,"证券代码"] != df4.loc[j,"证券代码"] or df4.loc[j,"month"] != df4.loc[i,"month"]:
                    down = j
            # y = np.array(df4.loc[i:down,"周转率"])
            x = np.array(df4.loc[i:down,["市场周转率","事件_8","事件_9","事件_11","事件_12","事件_13","事件_14","事件_16","事件_99","事件_100"	]])
            df4.loc[i:down,["预测周转率"]] = lr.predict(x)

    df4['误差'] = df4['周转率'] - df4['预测周转率']

    df4.to_csv(f'data/atr/月_87_{str(ed_ym)[:4]}_tmp.csv',index=False,encoding='utf-8_sig')

    #没有周转率数据，不计算误差
    for i in range(len(df4)):
        if df4.loc[i,'周转率'] == 0:
            df4.loc[i,'误差'] = None

    #误差以月为单位求平均
    df5 = df4[["证券代码","year","month","误差"]].groupby(by=["证券代码","year","month"]).mean().reset_index()
    df5["month"] = df5["month"].astype(str)
    df5["year"] = df5["year"].astype(str)
    for i in range(len(df5)):
        if len(df5.loc[i,"month"]) == 1:
            df5.loc[i,"month"] = '0'+df5.loc[i,"month"]
    df5["会计期间"] = df5["year"]+df5["month"]+"01"
    df5["会计期间"] = pd.to_datetime(df5["会计期间"])
    df5.rename({"误差":"atr"},axis='columns',inplace=True)
    df5[["证券代码","会计期间","atr"]].to_csv(f'data/atr/月_87_{str(ed_ym)[:4]}.csv',index=False,encoding='utf-8_sig')
# yms = [f'200{x}/01/01'for x in range(4,7)]  2004 05 06 ok
# yms = [f'200{x}/01/01'for x in range(7,10)] 07 08 09 ok
# yms = [f'20{x}/01/01'for x in range(10,14)] 10 11 12 13 ok
# yms = [f'20{x}/01/01'for x in range(14,18)] 14 15 16 17 
# yms = [f'20{x}/01/01'for x in range(18,23)] 18 19 20 21 22 -> 18 20 21 22
# yms = ['2017/01/01','2018/01/01']
# yms = ['2019/01/01','2020/01/01']
# yms = ['2020/01/01','2021/01/01']
# cd ../../S1/luohao/stock
# yms = ['2021/01/01','2022/01/01']
# for i in range(1,len(yms)):
#     print(str(yms[i])[:4])
#     cal_87(yms[i-1],yms[i])

def cal_88(start = "2000-01-01",end = "2003-01-01"):
    """
    趋势项 er_trend 月
        收益指 不考虑现金红利的月个股回报率Mretwd
        价格指 收盘价Clsprc
        数量指 日个股交易股数Dnshrtrd
    用到daily_return TRD_Mnth
    """
    print("cal 88")
    
    #计算MAP 3-, 5-, 10-, 20-, 50-, 100-, 200-, 300-, and 400-days. moving average price
    #用map_1..map_9表示

    # df1 = pd.read_csv('data/日个股回报率/daily_return.csv')
    # df1 = df1[["Stkcd",	"Trddt", "Clsprc", "Dnshrtrd"]]
    # df1.rename({'Stkcd':"证券代码",'Trddt':"会计期间","Clsprc":"价格", "Dnshrtrd":"交易量"},axis='columns',inplace=True)
    # df1["会计期间"] = df1["会计期间"].astype('datetime64')
    # df1["year"] = pd.DatetimeIndex(df1["会计期间"]).year.astype(int)
    # df1 = df1[df1.year>=2021]
    # df1.sort_values(by=["证券代码","会计期间"],inplace=True)
    # df1.to_csv('data/trend/tmp1.csv',index=False,encoding='utf-8_sig')

    # #计算MAV moving average volume
    # def get_ma(df,i,L,price):
    #     """计算moving average
    #     args:
    #         df:dataframe
    #         i:第i行
    #         L:要平均的长度
    #         price:bool, True表示求价格，False表示求交易量
    #     return: float
    #     """
    #     tot = 0
    #     for j in range(i,i-L,-1):
    #         if df.loc[i,"证券代码"] != df.loc[j,"证券代码"]:
    #             raise ValueError('Unenough data')
    #         if(price):
    #             tot += df.loc[j,"价格"]
    #         else:
    #             tot += df.loc[j,"交易量"]
    #     return tot/L

    # # #3-, 5-, 10-, 20-, 50-, 100-, 200-, 300-, and 400-days
    # lays = [3, 5, 10, 20, 50, 100, 200, 300, 400]
    # df1 = pd.read_csv(f'data/trend/tmp1.csv')
    # df1["month"] = pd.DatetimeIndex(df1["会计期间"]).month.astype(str)
    # for i in range(len(df1)-1):
    #     if i%10000 == 0:
    #         print(i,len(df1),datetime.datetime.now().strftime("%m-%d %X"))
    #     try:
    #         if df1.loc[i,"month"] == df1.loc[i+1,"month"]:#TODO 表格最后一天不能是月的最后一天
    #             continue
    #         df1.loc[i,'save'] = True
    #         #正则化
    #         for lay in lays:
    #             df1.loc[i,f"map_{lay}"] = get_ma(df1,i,lay,price=True)/df1.loc[i,"价格"]
    #             df1.loc[i,f"mav_{lay}"] = get_ma(df1,i,lay,price=False)/df1.loc[i,"交易量"]
                
    #     except:
    #         pass
    # df1.to_csv('data/trend/tmp2.csv',index=False,encoding='utf-8_sig')

    #只用2021年以后的
    # 处理月数据
    # df2 = pd.read_csv('data/TRD_Mnth.csv')
    # df2 = df2[["Stkcd",	"Trdmnt", "Mretwd"]]
    # df2.rename({'Stkcd':"证券代码",'Trdmnt':"会计期间","Mretwd":"收益"},axis='columns',inplace=True)

    # df2["year"] = pd.DatetimeIndex(df2["会计期间"]).year.astype(int)
    # df2 = df2[df2.year>=2021]

    # df2.sort_values(by=["证券代码","会计期间"],inplace=True)
    # # df2.drop(columns=["index"],inplace=True)
    # df2.to_csv('data/trend/monthly_return.csv',index=False,encoding='utf-8_sig')

    # df1 = pd.read_csv('data/trend/tmp2.csv')
    # df1 = df1[df1.year>=2021] 
    # df2 = pd.read_csv('data/trend/monthly_return.csv')
    # #结合两个表格
    # df1 = df1[df1['save']==True]
    # df1.reset_index(inplace=True)#仅保留每月最后一天

    # #将df1和df2的会计期间改为年+月，比如200001
    # df1["year"] = pd.DatetimeIndex(df1["会计期间"]).year.astype(str)
    # df1["month"] = pd.DatetimeIndex(df1["会计期间"]).month.astype(str)
    # df2["year"] = pd.DatetimeIndex(df2["会计期间"]).year.astype(str)
    # df2["month"] = pd.DatetimeIndex(df2["会计期间"]).month.astype(str)
    # #补齐月的位数
    # for i in range(len(df1)):
    #     if len(df1.loc[i,"month"]) == 1:
    #         df1.loc[i,"month"] = '0'+df1.loc[i,"month"]
    # for i in range(len(df2)):
    #     if len(df2.loc[i,"month"]) == 1:
    #         df2.loc[i,"month"] = '0'+df2.loc[i,"month"]
    # df1["会计期间"] = df1["year"]+df1["month"]
    # df2["会计期间"] = df2["year"]+df2["month"]

    # df3 = pd.merge(df1,df2,on=["证券代码","会计期间"])

    
    # df1.to_csv('data/trend/tmp3.csv',index=False,encoding='utf-8_sig')
    # df2.to_csv('data/trend/tmp4.csv',index=False,encoding='utf-8_sig')

    # df3 = pd.merge(df1,df2,on=["证券代码","year","month"])
    # df3["会计期间"] = df3["year"] + "-" + df3["month"]
    # df3["会计期间"] = [x.date() for x in pd.to_datetime(df3["会计期间"],format='%Y-%m')]

    # df3.drop(columns=["index"],inplace=True)
    # df3.to_csv('data/trend/tmp5.csv',index=False,encoding='utf-8_sig')

    #用map 和 mav 对股票的return(Dretnd)做回归
    df3 = pd.read_csv('data/trend/tmp5.csv')
    lays = [3, 5, 10, 20, 50, 100, 200, 300, 400]

    df3.fillna(0,inplace=True)
    x_list = [f"map_{lay}" for lay in lays]+[f"mav_{lay}" for lay in lays]
    
    for i in range(len(df3)):
        for x in x_list:
            if df3.loc[i,x] == float("inf"):
                df3.loc[i,x] = 1000 #最大605984.23

    #先对所有数据求回归，得截距。用这个截距对每行做一次回归。
    x = np.array(df3[x_list])
    y = np.array(df3["收益"])
    lr = LinearRegression()
    lr.fit(x,y)
    all_intercept = lr.intercept_#全局的截距，原文的beta_0

    fore_coef = [] #上一次回归的系数
    for i in range(len(df3)-1):
        if i%5000 == 0:
            print(i,len(df3),datetime.datetime.now().strftime("%m-%d %X"))
        if i==0 or df3.loc[i,"证券代码"] != df3.loc[i-1,"证券代码"]:#每只股票第一个月，不用衰减
            fore_coef = []
        if df3.loc[i,"证券代码"] != df3.loc[i+1,"证券代码"]:#最后一个月，不用计算
            continue

        #每个月做回归，除了该月这个点外，还有截距所在的点
        x = np.array([list(df3.loc[i,x_list]),[0]*18])
        y = np.array([df3.loc[i,"收益"],all_intercept])
        lr = LinearRegression()
        lr.fit(x,y)

        #修改coef
        coef = lr.coef_
        lambda_ = 0.022 #论文设置
        if len(fore_coef) != 0:
            tmp_coef = []
            for j in range(len(fore_coef)):
                tmp_coef.append((1-lambda_)*fore_coef[j]+lambda_*coef[j])
            coef = tmp_coef

        #计算er_trend
        df3.loc[i+1,'er_trend'] = 0
        for j in range(len(coef)):
            df3.loc[i+1,'er_trend'] += coef[j]*df3.loc[i,x_list][j]

        #更新fore_coef
        fore_coef = coef

    df3.to_csv('data/trend/tmp6.csv',index=False,encoding='utf-8_sig')
    # df3["会计期间"] = [x.date() for x in pd.to_datetime(df3["会计期间"],format='%Y%m')]
    df3[["证券代码","会计期间","er_trend"]].to_csv('data/月_88.csv',index=False,encoding='utf-8_sig')
# cal_88()