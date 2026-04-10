import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===================== 1. 读取数据 =====================
# 假设数据文件名为 Q3data.dta，包含以下字段：
# Code (公司代码), year (年份), ROA (总资产收益率), Revenue (总收入)
df = pd.read_stata('Q3 data.dta')

# 确保年份为整数
df['year'] = df['year'].astype(int)

# 剔除缺失值
df = df.dropna(subset=['ROA', 'Revenue'])

# ===================== 2. 计算收入增长率 =====================
# 按公司排序
df = df.sort_values(['Code', 'year'])

# 计算每个公司滞后一年的收入，用于计算增长率
df['Revenue_lag'] = df.groupby('Code')['Revenue'].shift(1)
df['Rev_Growth'] = df['Revenue'] / df['Revenue_lag'] - 1

# 剔除增长率缺失值（如首年数据）
df = df.dropna(subset=['Rev_Growth'])

# ===================== 3. 计算各年份的中位数 =====================
# ROA 中位数（每年）
roa_median = df.groupby('year')['ROA'].median().rename('ROA_median')
# 收入增长率中位数（每年）
growth_median = df.groupby('year')['Rev_Growth'].median().rename('growth_median')

# 将中位数合并回原数据
df = df.merge(roa_median, on='year', how='left')
df = df.merge(growth_median, on='year', how='left')

# 标记当年是否高于中位数
df['roa_above'] = df['ROA'] > df['ROA_median']
df['growth_above'] = df['Rev_Growth'] > df['growth_median']

# ===================== 4. 持续性分析（基年2012）=====================
# 定义分析年份范围（2012-2024）
years = list(range(2012, 2025))

# 获取基年（2012）有数据的公司（对于ROA和增长率分别处理）
# 注意：增长率从2012年开始计算，所以基年公司需在2012年有增长率数据
base_year = 2012
base_companies_roa = set(df[df['year'] == base_year]['Code'])
base_companies_growth = set(df[df['year'] == base_year]['Code'])  # 增长率基年也是2012

# 初始化结果字典
persist_roa = {}
persist_growth = {}

# 逐年计算持续性比例
for t in years:
    # 当前年份及之前年份的数据
    df_up_to_t = df[df['year'] <= t]
    
    # ROA：在基年样本中，要求从2012到t每年都存在且高于中位数
    # 获取每年都高于中位数的公司：按公司分组，检查每年是否都为True
    roa_above_by_company = df_up_to_t.groupby('Code')['roa_above'].all()
    # 仅保留基年样本中的公司
    roa_above_in_base = [stk for stk in base_companies_roa if stk in roa_above_by_company.index and roa_above_by_company[stk]]
    persist_roa[t] = len(roa_above_in_base) / len(base_companies_roa) * 100

    # 收入增长率：同样处理
    growth_above_by_company = df_up_to_t.groupby('Code')['growth_above'].all()
    growth_above_in_base = [stk for stk in base_companies_growth if stk in growth_above_by_company.index and growth_above_by_company[stk]]
    persist_growth[t] = len(growth_above_in_base) / len(base_companies_growth) * 100

# 转换为DataFrame便于绘图
persist_df = pd.DataFrame({
    'year': list(persist_roa.keys()),
    'roa_persist': list(persist_roa.values()),
    'growth_persist': list(persist_growth.values())
})

# ===================== 5. 绘图 =====================
# 设置统一风格
plt.style.use('default')
plt.figure(figsize=(10, 6))

# 绘制ROA持续性线
plt.plot(persist_df['year'], persist_df['roa_persist'],
         marker='o', linestyle='-', color='#1f77b4', linewidth=2, markersize=6, label='ROA')

# 绘制收入增长率持续性线
plt.plot(persist_df['year'], persist_df['growth_persist'],
         marker='s', linestyle='-', color='#ff7f0e', linewidth=2, markersize=6, label='Revenue Growth')

plt.xlabel('Year', fontsize=12)
plt.ylabel('Percentage of Companies (%)', fontsize=12)
plt.title('Percentage of Companies Consistently Above Median: ROA vs Revenue Growth (2012-2024)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(years, rotation=45)
# 自动调整y轴范围，或根据需要设置固定范围
plt.ylim(0, max(persist_df['roa_persist'].max(), persist_df['growth_persist'].max()) * 1.1)

plt.tight_layout()
plt.savefig('Q3_Combined.pdf', format='pdf', dpi=300)
plt.show()
