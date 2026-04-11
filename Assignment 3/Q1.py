import pandas as pd
import numpy as np

# 读取数据
df = pd.read_csv("IDX_Idxtrd.csv")
df.columns = ['Indexcd','Date', 'Close']  # 根据实际列名修改
# 确保日期格式正确
df['Date'] = pd.to_datetime(df['Date'])

# 排序（非常重要）
df = df.sort_values('Date')

# 计算月度收益率（简单收益率）
df['Return'] = df['Close'].pct_change()

# 删除第一个NaN
returns = df['Return'].dropna()

# 描述统计
mean = returns.mean()
std = returns.std()
skew = returns.skew()
kurt = returns.kurtosis()

# 计算统计量
stats = pd.DataFrame({
    'Mean': [returns.mean()],
    'Std': [returns.std()],
    'Skewness': [returns.skew()],
    'Kurtosis': [returns.kurtosis()]
})

# 保存为 CSV
stats = stats.round(6)
stats.to_csv("csi300_summary.csv", index=True)


import matplotlib.pyplot as plt

plt.figure()
plt.hist(returns, bins=30)
plt.title("Histogram of CSI 300 Monthly Returns")
plt.xlabel("Return")
plt.ylabel("Frequency")
plt.savefig("csi300_histogram.png")