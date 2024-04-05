import pandas as pd
import os
import numpy as np
import zipfile
import datetime
import time
#1.beta 从wind上下载的
# df = pd.read_csv('data/beta_origin.csv',encoding='gbk',low_memory=False)
# stocks = df.columns[4:]
# values = {}
# for stock in stocks:
#     value={}
#     for line in range(5,338):
#         value[df.loc[line,"month"]] = df.loc[line,stock]
#     values[stock] = value
# df1 = pd.DataFrame(columns=["Stkcd","Accper","beta"])
# for stock in stocks:
#     for k,v in values[stock].items():
#         df1.loc[len(df1)] = [stock[:-3],k,v]
# df1.to_csv('data/beta.csv',index=False)

#2.删除FS文件中1/1以及Typrep=B的文件
# files = ['FS_Combas.csv','FS_Comins.csv','FS_Comscfd.csv']
# for file in files:
#     df = pd.read_csv(f'data/{file}')
#     df.drop(df[(df.Accper.str.endswith('01-01')) | (df.Typrep=='B')].index,inplace=True)
#     # df = df[df['Typrep'] == 'A']
#     df.to_csv(f'data/{file}',index=False,encoding='utf-8_sig')

#3.解压zip
# base = 'data/公司公告'
# paths = os.listdir(base)
# c = 0 
# for path in paths:
#     zf = zipfile.ZipFile(os.path.join(base,path))
#     for info in zf.infolist():#zipfile的所有文件对象
#         if info.filename.endswith('.xlsx'):
#             info.filename = f'{c}.xlsx'
#             zf.extract(info,path=base)
#             c += 1

# data = []
# for path in paths:
#     if path.endswith(".xlsx"):
#         data.append(pd.read_excel(os.path.join(base,path)))
# df = pd.concat(data)
# df.to_csv('data/announcement.csv',index=False)

#4.仅保留announcement中['8','9','11','12','13','14','16','99','nan']
# df = pd.read_csv('data/公司公告/announcement.csv')[2:].reset_index()
# save =  ['8','9','11','12','13','14','16','99']
# for i in range(len(df)):
#     if df.loc[i,"ClassID"] in save or pd.isnull(df.loc[i,"ClassID"]):
#         df.loc[i,"save"] = True
#     else:
#         df.loc[i,"save"] = False
# df = df[df["save"]==True]
# df.drop(columns=["index","save"],inplace=True)
# df.to_csv('data/公司公告/tmp.csv',index=False,encoding='utf-8_sig')

#5.整理daily_return
# df = pd.read_csv('data/日个股回报率/daily_return.csv')
# df.sort_values(by=["Stkcd","Trddt"],inplace=True)
# df.drop(df[df['Stkcd']=='证券代码'].index,inplace=True)
# df.to_csv('data/日个股回报率/daily_return.csv',index=False,encoding='utf-8_sig')

#6.合成daily_info
# base = 'data/日交易统计文件'
# dfs = []
# for file in os.listdir(base):
#     dfs.append(pd.read_excel(os.path.join(base,file))[2:])
# df = pd.concat(dfs)
# df["Stkcd"] = df["Stkcd"].astype(int)
# df["Trddt"] = pd.to_datetime(df["Trddt"])
# df.sort_values(by=["Stkcd","Trddt"])
# df.to_csv(os.path.join(base,'daily_info.csv'),index=False,encoding='utf-8_sig')


def transform():
    #7.结果表格转换
    #202201: 202201-202212
    #202204: 202201-202204
    # import warnings
    # warnings.filterwarnings("ignore")
    # end_ym = "202304" #end_ym之前的所有yearmon都有值
    # files = os.listdir('output_data')
    # files.remove('output')
    # files.remove('prolonged_output')
    # files.remove('prolonged_output_new')
    # for file in files:
    #     print(file,str(datetime.datetime.now()))
    #     max_fill_times = 0
    #     if file.startswith('季'):
    #         max_fill_times = 2
    #     elif file.startswith('半年'):
    #         max_fill_times = 5
    #     if file.startswith('年'):
    #         max_fill_times = 11
    #     df = pd.read_csv(f'output_data/{file}')
    #     df["year"] = pd.DatetimeIndex(df["会计期间"]).year.astype(str)
    #     df["month"] = pd.DatetimeIndex(df["会计期间"]).month.astype(str)
    #     for i in range(len(df)):
    #         if len(df.loc[i,"month"]) == 1:
    #             df.loc[i,"month"] = '0'+df.loc[i,"month"]
    #     if file.startswith('年'):
    #         df["month"] = "12"
    #     df["会计期间"] = df["year"] + df["month"]
        
    #     df.drop_duplicates(["证券代码","会计期间"],keep='last',inplace=True)
    #     df.sort_values(by=["证券代码","会计期间"],inplace=True)
    #     df.reset_index(inplace=True)
    #     df.drop(columns=["index","year","month"],inplace=True)

    #     #建立新df
    #     ids = []
    #     before = 0
    #     for i in range(len(df)):
    #         if df.loc[i,"证券代码"] != before:
    #             ids.append(df.loc[i,"证券代码"])
    #             before = df.loc[i,"证券代码"]

    #     def get_next_ym(ym):
    #         """得到下一个月的表达。 ym形如"202010"
    #         return:next_ym
    #         """
    #         if ym[-2:] == '12':
    #             return str(int(ym[:4])+1) + '01'
    #         elif ym[-2:] in ['11','10'] or ym[-1] == '9':
    #             return ym[:4] + str(int(ym[-2:])+1)
    #         else:
    #             return ym[:4] + '0' + str(int(ym[-2:])+1)

    #     def ym_minus(ym1:str,ym2:str)->int:
    #         """ 
    #         返回ym1-ym2的月数
    #         "202010"-"202008" -> 2
    #         """
    #         return (int(ym1[:4])-int(ym2[:4]))*12 + (int(ym1[4:])-int(ym2[4:]))
        
    #     for i in range(2,len(df.columns)):#每一个因子
    #         df1 = pd.DataFrame({"Stkcd":ids})
    #         factor_name = df.columns[i]
    #         c = -1
    #         for id in ids:#每一个股票
    #             c += 1
    #             tmp_df = df[df["证券代码"]==id][["会计期间",factor_name]].reset_index()
    #             tmp_df.drop(labels=["index"],axis="columns",inplace=True)
    #             tmp_df.loc[len(tmp_df)] = [end_ym,None]
    #             try:
    #                 df1.loc[c,tmp_df.loc[0,"会计期间"]] = tmp_df.loc[0,factor_name]
    #                 next_ym = tmp_df.loc[1,"会计期间"] #形如202010
    #                 next_value = tmp_df.loc[1,factor_name]
    #             except:
    #                 continue
                    
    #             for j in range(len(tmp_df)-1):#每一个月
    #                 ym = tmp_df.loc[j+1,"会计期间"]

    #                 while ym != next_ym and next_ym[:4]!="2024":
    #                     #ym-next_ym <= max_fill_times才填充
    #                     if len(next_ym)>=7:
    #                         print("error",next_ym)
    #                     if ym_minus(ym,next_ym) <= max_fill_times:
    #                         #在两段之间填充之后的值
    #                         df1.loc[c,next_ym] = next_value
    #                     next_ym = get_next_ym(next_ym)
    #                 df1.loc[c,ym] = next_value
    #                 next_ym = get_next_ym(next_ym)
    #                 if j != len(tmp_df)-2:
    #                     next_value = tmp_df.loc[j+2,factor_name]
    #         cols = list(df1.columns)[1:]
    #         cols.sort(key = lambda x: int(x))
    #         df1 = df1[["Stkcd"]+cols]
    #         df1.to_csv(f'output_data/output/{factor_name}.csv',index=False,encoding='utf-8_sig')

    #9.将列名扩展到199012 - 202303
    ys = [str(x) for x in range(1990,2024)]
    ms = ["01","02","03","04","05","06","07","08","09"] + [str(x) for x in range(10,13)]
    yms = [y+m for y in ys for m in ms]
    yms = yms[11:-8]
    
    files = os.listdir('output_data/output')
    for file in files:
        df = pd.read_csv(f'output_data/output/{file}')
        print(file,len(df))
        for ym in yms:
            if ym not in df.columns:
                df[ym] = None
        
        df = df[["Stkcd"]+yms]
        df.to_csv(f'output_data/prolonged_output/{file}',index=False,encoding='utf-8_sig')

    # 10.统一股票代码，共5245个
    df = pd.read_csv(f'output_data/prolonged_output/chpm_标准.csv')
    indexs = list(df["Stkcd"])
    #看列表股票数
    files = os.listdir('output_data/prolonged_output')
    flag = False
    for file in files:
        # if file.startswith('mon1m.csv'):
        #     flag = True
        # if not flag:
        #     continue
        if file.startswith('chpm_标准'):
            continue
        df = pd.read_csv(f'output_data/prolonged_output/{file}')
        print(file,len(df))

        for index in indexs:
            if index not in list(df["Stkcd"]):
                df.loc[len(df)] = [index]+[None]*(len(df.columns)-1)

        for i in range(len(df)):
            if df.loc[i,"Stkcd"] not in indexs:
                df.loc[i,"Stkcd"] = None

        df.dropna(subset=["Stkcd"],inplace=True)

        df.sort_values(by="Stkcd",inplace=True)
        df.to_csv(f'output_data/prolonged_output_new/{file}',index=False,encoding='utf-8_sig')
transform()
#合并daily_return与TRD_Dalyr.csv
# df1 = pd.read_csv('data/日个股回报率/daily_return_old.csv')
# df2 = pd.read_csv('data/日个股回报率/TRD_Dalyr.csv')
# df3 = pd.concat([df1,df2])
# df3 = df3.sort_values(by=["Stkcd","Trddt"])
# df3.drop_duplicates(subset=["Stkcd","Trddt"], keep='first', inplace=True)
# df3.to_csv('data/日个股回报率/daily_return.csv',index=False,encoding='utf-8_sig')

#合并TRD_Mnth_update与TRD_Mnth_old.csv
# df1 = pd.read_csv('data/TRD_Mnth_update.csv')
# df2 = pd.read_csv('data/TRD_Mnth_old.csv')
# df3 = pd.concat([df1,df2])
# df3["Trdmnt"] = pd.to_datetime(df3["Trdmnt"], format="%Y-%m")
# df3 = df3.sort_values(by=["Stkcd","Trdmnt"])
# df3.drop_duplicates(subset=["Stkcd","Trdmnt"], keep='first', inplace=True)
# df3.to_csv('data/TRD_Mnth.csv',index=False,encoding='utf-8_sig')

#合并BETA_Mbeta_update与BETA_Mbeta_old
# df1 = pd.read_excel('data/BETA_Mbeta_update.xlsx')[2:]
# df2 = pd.read_excel('data/BETA_Mbeta_old.xlsx')[2:]
# df1["Stkcd"] = df1["Stkcd"].astype(int)
# df2["Stkcd"] = df2["Stkcd"].astype(int)
# df3 = pd.concat([df1,df2])
# df3["Trdmnt"] = pd.to_datetime(df3["Trdmnt"], format="%Y-%m")
# df3 = df3.sort_values(by=["Stkcd","Trdmnt"])
# df3.drop_duplicates(subset=["Stkcd","Trdmnt"], keep='first', inplace=True)
# df3.to_csv('data/BETA_Mbeta.csv',index=False,encoding='utf-8_sig')

#处理csmar的BETA_Mbeta为月_4 月_5
# df = pd.read_csv('data/BETA_Mbeta.csv')[['Stkcd','Trdmnt','Betavals']]
# df.rename({'Stkcd':"证券代码",'Trdmnt':"会计期间",'Betavals':"beta"},axis='columns',inplace=True)
# df["会计期间"] = pd.to_datetime(df["会计期间"],format='%Y-%m')
# df.to_csv('data/月_4.csv',index=False,encoding='utf-8_sig')
# df["beta"] = df["beta"] ** 2
# df.rename({"beta":"betasq"},axis='columns',inplace=True)
# df.to_csv('data/月_5.csv',index=False,encoding='utf-8_sig')

#合并月_88 old与update，有重合
# df1 = pd.read_csv('data/月_88_update.csv')
# df2 = pd.read_csv('data/月_88_old.csv')
# # df1 = df1[df1["证券代码"]==1]
# # df2 = df2[df2["证券代码"]==1]
# df1["year"] = pd.DatetimeIndex(df1["会计期间"]).year.astype(int)
# df1 = df1[df1["year"]>=2023]
# df1.drop(columns='year',inplace=True)
# df3 = pd.concat([df1,df2])
# df3.drop_duplicates(subset=["证券代码","会计期间"], keep='first', inplace=True)
# df3 = df3.sort_values(by=["证券代码","会计期间"])
# df3.to_csv('data/月_88.csv',index=False,encoding='utf-8_sig')

#合并dailyinfo
# df1 = pd.read_csv('data/日交易统计文件/daily_info_old.csv')
# df2 = pd.read_excel('data/日交易统计文件/LT_Dailyinfo.xlsx')[2:]
# df2["Stkcd"] = df2["Stkcd"].astype(int)
# # df1 = df1[df1["Stkcd"]==1]
# # df2 = df2[df2["Stkcd"]==1]
# df3 = pd.concat([df1,df2])
# df3 = df3.sort_values(by=["Stkcd","Trddt"])
# df3.drop_duplicates(subset=["Stkcd","Trddt"], keep='last', inplace=True)
# print(df3.tail(10))
# df3.to_csv('data/日交易统计文件/daily_info.csv',index=False,encoding='utf-8_sig')

