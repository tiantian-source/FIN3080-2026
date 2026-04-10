import pandas as pd
import numpy as np

# read files
df_1a = pd.read_csv('Q1-1a_Data.csv')
df_q = pd.read_csv('FI_T9.csv')

# prepare quarterly data
df_q.columns = ['Stkcd', 'Name','Accper', 'Typrep','EPS', 'BPS', 'TotalAssets','TotalLiabilities', 'ROA', 'ROE','R&DExpenses']
df_q.columns = [c.split('.')[-1].strip() for c in df_q.columns]
df_q['Accper'] = pd.to_datetime(df_q['Accper'], errors='coerce')
df_q['Stkcd'] = df_q['Stkcd'].astype(int)
df_q = df_q.sort_values(['Stkcd', 'Accper']).dropna(subset=['Accper'])

# prepare monthly data
df_1a['Trdmnt'] = pd.to_datetime(df_1a['Trdmnt'], errors='coerce')
df_1a['Stkcd'] = df_1a['Stkcd'].astype(int)
df_1a = df_1a.sort_values(['Trdmnt']).dropna(subset=['Trdmnt'])

# convert Stkcd to int to ensure merge_asof works correctly
df_1a['Stkcd'] = df_1a['Stkcd'].astype(int)
df_q['Stkcd'] = df_q['Stkcd'].astype(int)

df_1a = df_1a.drop(columns=['ROA','ROE'], errors='ignore')
# drop rows with missing values
df_1a = df_1a.dropna(subset=['Trdmnt'])
df_q = df_q.dropna(subset=['Accper'])

# sort by Stkcd and date to ensure merge_asof works correctly
df_1a = df_1a.sort_values(['Stkcd','Trdmnt'])
df_q = df_q.sort_values(['Stkcd','Accper'])

# merge_asof to align quarterly data with monthly data
df_1a = df_1a.dropna(subset=['Trdmnt']).sort_values('Trdmnt')
df_q = df_q.dropna(subset=['Accper']).sort_values('Accper')

df_final = pd.merge_asof(
    df_1a,
    df_q[['Stkcd','Accper','ROA','ROE']],
    left_on='Trdmnt',
    right_on='Accper',
    by='Stkcd',
    direction='backward',
    tolerance=pd.Timedelta('92D')
)

# Market Grouping
def classify_market(m):
    if m in [1, 4, 64]: return 'Main Board'
    elif m in [16 , 32]: return 'GEM Board'
    return 'Other'

df_final['Market_Group'] = df_final['Markettype'].apply(classify_market)

# Replace inf with NaN for relevant columns before statistics
cols = ['Mretwd', 'PE_Ratio', 'PB_Ratio', 'ROA', 'ROE', 'RD_Ratio', 'Firm_Age']

def get_stats(group):
    for col in cols:
        group[col] = group[col].replace([np.inf, -np.inf], np.nan)
    s = group[cols].agg(['count', 'mean', 'median', 'std']).T
    s['p25'] = group[cols].quantile(0.25)
    s['p75'] = group[cols].quantile(0.75)
    s.columns = ['Count', 'Mean', 'Median', 'Std', 'P25', 'P75']
    return s

summary_table = df_final[df_final['Market_Group'] != 'Other'].groupby('Market_Group').apply(get_stats)
summary_table = summary_table.swaplevel(0, 1).sort_index()
pd.options.display.float_format = '{:.6f}'.format
summary_table.to_csv('Q1-1b_Data.csv',float_format='%.6f')