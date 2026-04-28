import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as stats
import matplotlib.pyplot as plt

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
factor_df = factor_df.dropna()

stock_df = stock_df.sort_values(['stock_id', 'date'])

stock_df['mkt_lag1'] = stock_df.groupby('stock_id')['Msmvosd'].shift(1)
df = stock_df.dropna(subset=['mkt_lag1', 'return'])

def assign_size_group(df):
    df = df.copy()
    df = df.dropna(subset=['mkt_lag1', 'return'])
    
    if df['mkt_lag1'].nunique() < 10:
        df['size_group'] = np.nan
        return df
    
    df['size_group'] = pd.qcut(
        df['mkt_lag1'],
        10,
        labels=False,
        duplicates='drop'
    ) + 1
    
    return df

stock_df = stock_df.groupby('date', group_keys=False).apply(assign_size_group)

portfolio_returns = (
    stock_df
    .dropna(subset=['size_group'])
    .groupby(['date', 'size_group'])['return']
    .mean()
    .reset_index()
)

port = portfolio_returns.pivot(index='date', columns='size_group', values='return')
port.columns = [f'G{int(c)}' for c in port.columns]

# long-short: big - small
port['LS'] = port['G10'] - port['G1']

summary = []

for col in port.columns:
    r = port[col].dropna()
    
    summary.append({
        'Portfolio': col,
        'Mean Return': r.mean(),
        't-value': stats.ttest_1samp(r, 0)[0]
    })

summary_df = pd.DataFrame(summary)

plot_df = summary_df[summary_df['Portfolio'].str.startswith('G')]

plt.figure()
plt.plot(plot_df['Portfolio'], plot_df['Mean Return'], marker='o')
plt.xlabel("Portfolio (Size Deciles)")
plt.ylabel("Average Monthly Return")
plt.title("Size Effect: Portfolio Returns")

plt.savefig("size_effect_plot.png", dpi=300, bbox_inches='tight')
plt.show()

reg_df = port.merge(factor_df[['date', 'MKT']], on='date', how='inner')

capm_results = []

for col in port.columns:
    df = reg_df[[col, 'MKT']].dropna()
    
    Y = df[col]
    X = sm.add_constant(df['MKT'])
    
    model = sm.OLS(Y, X).fit()
    
    capm_results.append({
        'Portfolio': col,
        'Alpha': model.params['const'],
        'Alpha t': model.tvalues['const'],
        'Beta_MKT': model.params['MKT'],
        'Beta t': model.tvalues['MKT']
    })

capm_df = pd.DataFrame(capm_results)

cols = [c for c in capm_df.columns if c != 'Portfolio']

capm_df[cols] = capm_df[cols].round(6)
summary_df['t-value'] = summary_df['t-value'].apply(lambda x: f"{x:.6f}")
summary_df['Mean Return'] = summary_df['Mean Return'].apply(lambda x: f"{x*100:.2f}%")
capm_df['Alpha'] = capm_df['Alpha'].apply(lambda x: f"{x*100:.2f}%")
capm_df[['Beta_MKT', 'Beta t']] = capm_df[['Beta_MKT', 'Beta t']].round(6)
summary_df.to_csv("size_summary.csv", index=False)
capm_df.to_csv("size_capm.csv", index=False)