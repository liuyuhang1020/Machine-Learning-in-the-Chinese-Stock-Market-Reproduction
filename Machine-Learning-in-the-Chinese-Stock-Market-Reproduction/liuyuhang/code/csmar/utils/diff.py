import pandas as pd


def diff(df, col=None, freq='季'):
    if not (('证券代码' in df.columns) and ('会计期间' in df.columns)):
        raise Exception('列名必须包含\'证券代码\'和\'会计期间\'')
    if not (df['证券代码'].dtypes == 'int'):
        raise Exception('证券代码必须为整数类型')
    if freq == '季' and not (set(df['会计期间'].apply(lambda x: x[5:])) == set(['03-31', '06-30', '09-30', '12-31'])):
        raise Exception('季度频率的会计期间格式错误')
    if freq == '半年' and not (set(df['会计期间'].apply(lambda x: x[5:])) == set(['06-30', '12-31'])):
        raise Exception('半年度频率的会计期间格式错误')
    if col is None:
        col = list(set(df.columns) - set(('证券代码', '会计期间')))
    df_shift = df.groupby(['证券代码', df['会计期间'].apply(lambda x: x[:4])]).shift(1)[col]
    if freq == '季':
        df_shift[df['会计期间'].apply(lambda x: x[5:]) == '03-31'] = 0
    elif freq == '半年':
        df_shift[df['会计期间'].apply(lambda x: x[5:]) == '06-30'] = 0
    else:
        raise Exception('差分频率只能为季度或半年度')
    df_diff = df.copy()
    df_diff[col] = df_diff[col].values - df_shift.values
    return df_diff
