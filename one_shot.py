import pandas as pd
import numpy as np
import time
from scipy.stats import laplace

# ========== 1. 参数设置 ==========
k = 1200
epsilon = 0.2
delta = 1e-5
num_trials = 20

# ========== 2. 加载数据 ==========
column_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'sex',
    'capital_income', 'capital_loss', 'hours_per_week', 'native_country', 'income'
]

df = pd.read_csv("census.csv", names=column_names, skipinitialspace=True)
df['income'] = df['income'].str.strip().str.replace('.', '')

# 年龄分箱
df['age'] = pd.to_numeric(df['age'], errors='coerce')
df['age_group'] = pd.cut(df['age'], bins=10, include_lowest=True)

# 六特征组合
df['feature'] = (df['education'].astype(str) + '|' +
                 df['occupation'].astype(str) + '|' +
                 df['race'].astype(str) + '|' +
                 df['sex'].astype(str) + '|' +
                 df['age_group'].astype(str) + '|' +
                 df['workclass'].astype(str))

print("=" * 60)
print("Oneshot Laplace - 效率测试")
print("=" * 60)

# ========== 3. 获取基础信息 ==========
filtered = df[df['income'] == '>50K']
grouped = filtered.groupby('feature').size()
m = len(grouped)
print(f"m = {m}, k = {k}, ε = {epsilon}\n")


# ========== 4. Oneshot Laplace ==========
def oneshot_laplace(df, k=k, epsilon=epsilon, delta=delta):
    filtered = df[df['income'] == '>50K']
    grouped = filtered.groupby('feature').size()
    values = grouped.values.astype(float)
    labels = grouped.index.values
    m = len(values)

    lam = 8 * np.sqrt(k * np.log(m / delta)) / epsilon

    gi = laplace.rvs(scale=lam, size=m)
    yi = values + gi
    top_idx = np.argsort(yi)[::-1][:k]

    return list(labels[top_idx])


# ========== 5. 运行测试 ==========
times = []
for i in range(num_trials):
    start = time.time()
    result = oneshot_laplace(df)
    elapsed = time.time() - start
    times.append(elapsed)
    print(f"第{i + 1:2d}次: {elapsed:.4f} 秒")

print(f"\n平均运行时间: {np.mean(times):.4f} 秒")
print(f"标准差: {np.std(times):.4f} 秒")