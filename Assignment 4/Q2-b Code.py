import pandas as pd
import numpy as np

# ===== 1. 读取个股数据 =====
stock_df = pd.read_csv("TRD_Mnth.csv")

# 如果日期是字符串，转成时间
stock_df.columns = ['stock_id', 'date', 'Msmvosd','return','Markettype']
stock_df['date'] = pd.to_datetime(stock_df['date'])
stock_df = stock_df[
    (stock_df['date'] >= '2000-12-01') &
    (stock_df['date'] <= '2025-12-31')
]
# ===== 2. 读取因子数据 =====
factor_df = pd.read_csv("STK_MKT_THRFACMONTH.csv")
factor_df.columns = ['Markettype', 'date', 'MKT', 'SMB', 'HML']
factor_df = factor_df[factor_df['Markettype'] == 'P9706']
factor_df['date'] = pd.to_datetime(factor_df['date'])
factor_df = factor_df[
    (factor_df['date'] >= '2000-12-01') &
    (factor_df['date'] <= '2025-12-31')
]

stock_df = stock_df.dropna(subset=['return', 'Msmvosd'])
stock_df = stock_df.sort_values(['stock_id', 'date'])

# 上个月收益（momentum signal）
stock_df['ret_lag1'] = stock_df.groupby('stock_id')['return'].shift(1)

def assign_group(df):
    df = df.copy()
    df = df.dropna(subset=['ret_lag1'])
    
    # 如果数据太少，直接跳过
    if len(df) < 10:
        df['group'] = np.nan
        return df
    
    # 如果所有值一样，也跳过
    if df['ret_lag1'].nunique() < 5:
        df['group'] = np.nan
        return df
    
    df['group'] = pd.qcut(df['ret_lag1'], 5, labels=False) + 1
    return df

stock_df = stock_df.groupby('date', group_keys=False).apply(assign_group)

portfolio_returns = (
    stock_df
    .dropna(subset=['group'])
    .groupby(['date', 'group'])['return']
    .mean()
    .reset_index()
)
port_pivot = portfolio_returns.pivot(index='date', columns='group', values='return')

# 重命名列
port_pivot.columns = [f'P{int(c)}' for c in port_pivot.columns]

port_pivot['LS'] = port_pivot['P5'] - port_pivot['P1']

##
import scipy.stats as stats

summary = []

for col in port_pivot.columns:
    r = port_pivot[col].dropna()
    mean = r.mean()
    tval = stats.ttest_1samp(r, 0)[0]
    
    summary.append({
        'Portfolio': col,
        'Mean Return': mean,
        't-value': tval
    })

summary_df = pd.DataFrame(summary)

## plot
import matplotlib.pyplot as plt

plot_df = summary_df[summary_df['Portfolio'].str.startswith('P')]

plt.figure()
plt.plot(plot_df['Portfolio'], plot_df['Mean Return'], marker='o')
plt.xlabel('Portfolio')
plt.ylabel('Average Monthly Return')
plt.title('Momentum Portfolios (1-month)')
plt.savefig("momentum_plot.png", dpi=300, bbox_inches='tight')

reg_df = port_pivot.merge(factor_df, on='date', how='inner')

##Fama-French三因子回归分析
import statsmodels.api as sm

reg_results = []

for col in port_pivot.columns:
    df = reg_df[[col, 'MKT', 'SMB', 'HML']].dropna()
    
    Y = df[col]
    X = df[['MKT', 'SMB', 'HML']]
    X = sm.add_constant(X)
    
    model = sm.OLS(Y, X).fit()
    
    reg_results.append({
        'Portfolio': col,
        'Alpha': model.params['const'],
        'Alpha t': model.tvalues['const'],
        'Beta_MKT': model.params['MKT'],
        'MKT_t': model.tvalues['MKT'],
        'Beta_SMB': model.params['SMB'],
        'SMB_t': model.tvalues['SMB'],
        'Beta_HML': model.params['HML'],
        'HML_t': model.tvalues['HML']
    })

reg_results_df = pd.DataFrame(reg_results)


summary_df['t-value'] = summary_df['t-value'].apply(lambda x: f"{x:.6f}")
summary_df['Mean Return'] = summary_df['Mean Return'].apply(lambda x: f"{x*100:.2f}%")
reg_results_df['Alpha'] = reg_results_df['Alpha'].apply(lambda x: f"{x*100:.2f}%")
cols_to_round = [col for col in reg_results_df.columns if col != 'Alpha']

reg_results_df[cols_to_round] = (
    reg_results_df[cols_to_round].round(6)
)

summary_df.to_csv(
    "momentum_summary_results.csv",
    index=False
)
reg_results_df.to_csv(
    "momentum_ff3_regression_results.csv",
    index=False
)

