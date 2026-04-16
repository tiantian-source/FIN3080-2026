import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
# 读取数据
df = pd.read_csv("IDX_Idxtrd.csv")
df.columns = ['Indexcd','Date', 'Close']  # 根据实际列名修改
# 确保日期格式正确
df['Date'] = pd.to_datetime(df['Date'])

# 排序（非常重要）
df = df.sort_values('Date')
df.set_index('Date', inplace=True)
# 使用每月最后一个交易日的收盘价计算月度收益率
monthly_close = df['Close'].resample('M').last()
# 计算月度收益率 (百分比变化)
monthly_returns = monthly_close.pct_change().dropna()

# 4. 描述性统计量
mean_ret = monthly_returns.mean()
std_ret = monthly_returns.std()
skew_ret = monthly_returns.skew()
kurt_ret = monthly_returns.kurtosis()
# 汇总统计量
stats = pd.DataFrame({
    'Mean': [mean_ret],
    'Std Dev': [std_ret],
    'Skewness': [skew_ret],
    'Kurtosis': [kurt_ret]
}, index=['CSI 300'])
# 保存为 CSV
stats = stats.round(6)
stats.to_csv("csi300_summary.csv", index=True)


import matplotlib.pyplot as plt

plt.figure()
plt.hist(monthly_returns, bins=30)
plt.title("Histogram of CSI 300 Monthly Returns")
plt.xlabel("Return")
plt.ylabel("Frequency")
plt.savefig("csi300_histogram.png")
