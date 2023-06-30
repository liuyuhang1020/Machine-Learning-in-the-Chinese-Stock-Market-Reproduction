import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def merge(df1, df2, df3, on, bias=0.05, fill=np.nan):
    col12 = list((set(df1.columns) & set(df2.columns)) - set(df3.columns) - set(on))
    col13 = list((set(df1.columns) & set(df3.columns)) - set(df2.columns) - set(on))
    col23 = list((set(df2.columns) & set(df3.columns)) - set(df1.columns) - set(on))
    col123 = list((set(df1.columns) & set(df2.columns) & set(df3.columns)) - set(on))
    df1.columns = [c + '1' if c in col12 + col13 + col123 else c for c in df1.columns]
    df2.columns = [c + '2' if c in col12 + col23 + col123 else c for c in df2.columns]
    df3.columns = [c + '3' if c in col13 + col23 + col123 else c for c in df3.columns]
    df = pd.merge(pd.merge(df1, df2, on=on, how='left'), df3, on=on, how='left')
    df1.columns = [c.replace('1', '') if c.replace('1', '') in col12 + col13 + col123 else c for c in df1.columns]
    df2.columns = [c.replace('2', '') if c.replace('2', '') in col12 + col23 + col123 else c for c in df2.columns]
    df3.columns = [c.replace('3', '') if c.replace('3', '') in col13 + col23 + col123 else c for c in df3.columns]
    for col, id in zip([col12, col13, col23], [['1', '2'], ['1', '3'], ['2', '3']]):
        for c in col:
            df[c] = df[c + id[0]].fillna(df[c + id[1]])
            if df[c].dtype == 'object':
                df[c][(df[c + id[0]] != df[c + id[1]]) & (~df[c + id[0]].isna()) & (~df[c + id[1]].isna())] = fill
            else:
                gap = 2*(df[c + id[0]] - df[c + id[1]]).abs()/(df[c + id[0]].abs() + df[c + id[1]].abs())
                df[c][gap > bias] = fill
            df.drop(columns=[c + id[0], c + id[1]], inplace=True)
    for c in col123:
        df[c] = df[c + '1'].fillna(df[c + '2']).fillna(df[c + '3'])
        if df[c].dtype == 'object':
            df[c][(df[c + '2'] != df[c + '3']) & df[c + '1'].isna() & (~df[c + '2'].isna()) & (~df[c + '3'].isna())] = fill
            df[c][(df[c + '1'] != df[c + '3']) & df[c + '2'].isna() & (~df[c + '1'].isna()) & (~df[c + '3'].isna())] = fill
            df[c][(df[c + '1'] != df[c + '2']) & df[c + '3'].isna() & (~df[c + '1'].isna()) & (~df[c + '2'].isna())] = fill
            df[c][(df[c + '1'] != df[c + '2']) & (df[c + '1'] != df[c + '3']) & (df[c + '2'] != df[c + '3']) & (~df[c + '1'].isna()) & (~df[c + '2'].isna()) & (~df[c + '3'].isna())] = fill
        else:
            gap12 = 2*(df[c + '1'] - df[c + '2']).abs()/(df[c + '1'].abs() + df[c + '2'].abs())
            gap13 = 2*(df[c + '1'] - df[c + '3']).abs()/(df[c + '1'].abs() + df[c + '3'].abs())
            gap23 = 2*(df[c + '2'] - df[c + '3']).abs()/(df[c + '2'].abs() + df[c + '3'].abs())
            df[c][df[c + '1'].isna() & (gap23 > bias)] = fill
            df[c][df[c + '2'].isna() & (gap13 > bias)] = fill
            df[c][df[c + '3'].isna() & (gap12 > bias)] = fill
            df[c][(gap12 > bias) & (gap13 > bias) & (gap23 > bias)] = fill
            df[c][(gap12 <= gap13) & (gap12 <= gap23)] = df[c + '1'][(gap12 <= gap13) & (gap12 <= gap23)]
            df[c][(gap13 <= gap12) & (gap13 <= gap23)] = df[c + '1'][(gap13 <= gap12) & (gap13 <= gap23)]
            df[c][(gap23 <= gap12) & (gap23 <= gap13)] = df[c + '2'][(gap23 <= gap12) & (gap23 <= gap13)]
        df.drop(columns=[c + '1', c + '2', c + '3'], inplace=True)
    return df
