import pandas as pd
import numpy as np
#1-1a
# Read files
df_monthly = pd.read_csv('TRD_Mnth.csv') 
df_quarterly = pd.read_csv('FI_T9.csv')
df_basic = pd.read_csv('Company_Basic_Info.csv')
#rename
df_monthly.columns = ['Stkcd', 'Trdmnt', 'Mclsprc','Msmvosd', 'Mretwd','Markettype']
df_quarterly.columns = ['Stkcd', 'Name','Accper', 'Typrep','EPS', 'BPS', 'TotalAssets','TotalLiabilities', 'ROA', 'ROE','R&DExpenses']
df_basic.columns = ['Stkcd', 'Name', 'EndDate','EstablishDate','Markettype']

# convert "date" columns to datetime format
df_monthly['Trdmnt'] = pd.to_datetime(df_monthly['Trdmnt'], errors='coerce')
df_quarterly['Accper'] = pd.to_datetime(df_quarterly['Accper'], errors='coerce')
df_basic['EstablishDate'] = pd.to_datetime(df_basic['EstablishDate'], errors='coerce')

if 'Typrep' in df_quarterly.columns:
    df_quarterly = df_quarterly[df_quarterly['Typrep'] == 'A'].copy()
    
# R&D_ratio
df_quarterly['RD_Ratio'] = df_quarterly['R&DExpenses'] / df_quarterly['TotalAssets']

# Firm Age
df_q_merged = pd.merge(df_quarterly, df_basic[['Stkcd', 'EstablishDate']], on='Stkcd', how='left')
df_q_merged['Firm_Age'] = (df_q_merged['Accper'] - df_q_merged['EstablishDate']).dt.days / 365.25

# Sort
df_monthly = df_monthly.sort_values(['Stkcd', 'Trdmnt']).dropna(subset=['Trdmnt'])
df_monthly = df_monthly.sort_values('Trdmnt')
df_q_merged = df_q_merged.sort_values('Accper')
# Delete rows with missing values in key columns
df_q_merged = df_q_merged.dropna(subset=['Accper', 'EPS', 'BPS'])
df_monthly = df_monthly.dropna(subset=['Trdmnt'])

# Merge with asof to align quarterly data with monthly data
df_final = pd.merge_asof(
    df_monthly,
    df_q_merged,
    left_on='Trdmnt',
    right_on='Accper',
    by='Stkcd',
    direction='backward',
    tolerance=pd.Timedelta('92D')
)

# calculate PE and PB ratios
df_final.loc[df_final['EPS'] <= 0, 'PE_Ratio'] = np.nan
df_final.loc[df_final['BPS'] <= 0, 'PB_Ratio'] = np.nan
#EPS choice: remove outliers beyond 1% and 99% quantiles
eps_lower = df_final['EPS'].quantile(0.01)
df_final = df_final[df_final['EPS'] >= eps_lower]
eps_upper = df_final['EPS'].quantile(0.99) 
df_final = df_final[df_final['EPS'] <= eps_upper]

df_final['PE_Ratio'] = df_final['Mclsprc'] / df_final['EPS']
df_final['PB_Ratio'] = df_final['Mclsprc'] / df_final['BPS']
#PE_Ratio choice: remove outliers beyond 1% and 99% quantiles
pe_lower = df_final['PE_Ratio'].quantile(0.01)
pe_upper = df_final['PE_Ratio'].quantile(0.99)
df = df_final[(df_final['PE_Ratio'] >= pe_lower) & (df_final['PE_Ratio'] <= pe_upper)]

# finalize the dataset
df_final = df_final.sort_values(by='Stkcd')
cols = ['Stkcd','Trdmnt', 'Accper', 'Markettype', 'Mretwd', 'EPS', 'BPS', 'PE_Ratio','PB_Ratio','RD_Ratio', 'Firm_Age']
df_final = df_final[cols]

pd.options.display.float_format = '{:.6f}'.format
df_final.to_csv('Q1-1a_Data.csv', index=False, float_format='%.6f')

