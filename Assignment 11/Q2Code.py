import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

df = pd.read_csv('Q1-1a_Data.csv')
df['Trdmnt'] = pd.to_datetime(df['Trdmnt'])
df['PE_Ratio'] = pd.to_numeric(df['PE_Ratio'], errors='coerce')
df = df.dropna(subset=['PE_Ratio'])
df = df[np.isfinite(df['PE_Ratio'])]

if 'Market_Group' not in df.columns:
    def map_market(code):
        if code in [1, 4, 8, 64]:
            return 'Main Board'
        elif code in [16, 32]:
            return 'GEM Board'
        else:
            return 'Other'
    df['Market_Group'] = df['Markettype'].apply(map_market)
    df = df[df['Market_Group'].isin(['Main Board', 'GEM Board'])]

# Add YearMonth column for grouping
df['YearMonth'] = df['Trdmnt'].dt.to_period('M')
median_pe = df.groupby(['YearMonth', 'Market_Group'])['PE_Ratio'].median().unstack()
median_pe.index = median_pe.index.to_timestamp()
full_month_range = pd.date_range(start=median_pe.index.min(), 
                                 end=median_pe.index.max(), 
                                 freq='MS')

# Reindex to ensure all months are present, then forward-fill missing values
median_pe_full = median_pe.reindex(full_month_range)
median_pe_full['Main Board'] = median_pe_full['Main Board'].fillna(method='ffill')
median_pe_full['GEM Board'] = median_pe_full['GEM Board'].fillna(method='ffill')


plt.figure(figsize=(12, 6))
plt.plot(median_pe_full.index, median_pe_full['Main Board'], label='Main Board', color='#1f77b4', linewidth=1.5)
plt.plot(median_pe_full.index, median_pe_full['GEM Board'], label='GEM Board', color='#ff7f0e', linewidth=1.5)

plt.title('Median P/E Ratio by Market Type (Monthly)', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Median P/E Ratio', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

ax = plt.gca()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_major_locator(mdates.YearLocator())
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('Q2_PE_TimeSeries.pdf', format='pdf', dpi=300)
plt.show()
