import pandas as pd
import numpy as np
import time
from scipy.stats import laplace

#参数设置
k = 800
epsilon = 0.2
num_trials = 20

#加载数据集
column_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'sex',
    'capital_income', 'capital_loss', 'hours_per_week', 'native_country', 'income'
]

df = pd.read_csv("census.csv", names=column_names, skipinitialspace=True)
df['income'] = df['income'].str.strip().str.replace('.', '')

df['age'] = pd.to_numeric(df['age'], errors='coerce')
df['age_group'] = pd.cut(df['age'], bins=10, include_lowest=True)

df['feature'] = (df['education'].astype(str) + '|' +
                 df['occupation'].astype(str) + '|' +
                 df['race'].astype(str) + '|' +
                 df['sex'].astype(str) + '|' +
                 df['age_group'].astype(str) + '|' +
                 df['workclass'].astype(str))

print("=" * 60)
print("Baseline 简单组合 (平方×100拉伸)")
print("=" * 60)

# 获取真实 Top-k
filtered = df[df['income'] == '>50K']
grouped = filtered.groupby('feature').size()
original_values = grouped.values.astype(float)
values = (original_values ** 2) * 100
labels = grouped.index.values
m = len(values)

sorted_idx = np.argsort(values)[::-1]
true_labels = list(labels[sorted_idx[:k]])

print(f"m = {m}, k = {k}, ε = {epsilon}")
print(f"每轮噪声尺度 = {k / epsilon:.2f}")
print(f"原始计数范围: {original_values.min():.0f} ~ {original_values.max():.0f}")
print(f"拉伸后范围: {values.min():.0f} ~ {values.max():.0f}\n")


#Baseline
def baseline_peeling(df):
    filtered = df[df['income'] == '>50K']
    grouped = filtered.groupby('feature').size()
    original_vals = grouped.values.astype(float)
    values = (original_vals ** 2) * 100
    labels = grouped.index.values

    epsilon_prime = epsilon / k
    noise_scale = 1 / epsilon_prime

    remaining_values = values.copy()
    remaining_labels = labels.copy()
    selected_labels = []

    for _ in range(k):
        gi = laplace.rvs(scale=noise_scale, size=len(remaining_values))
        noisy = remaining_values + gi
        best_idx = np.argmax(noisy)
        selected_labels.append(remaining_labels[best_idx])
        remaining_values = np.delete(remaining_values, best_idx)
        remaining_labels = np.delete(remaining_labels, best_idx)

    return selected_labels


def evaluate(true_labels, dp_labels):
    return len(set(true_labels) & set(dp_labels)) / len(true_labels)


#测试
scores = []
times = []
for i in range(num_trials):
    start = time.time()
    result = baseline_peeling(df)
    elapsed = time.time() - start
    times.append(elapsed)
    score = evaluate(true_labels, result)
    scores.append(score)
    print(f"第{i + 1:2d}次: Recall = {score * 100:.1f}%, 时间 = {elapsed:.3f}s")

print(f"\n平均 Recall = {np.mean(scores) * 100:.1f}%")
print(f"平均时间 = {np.mean(times):.3f}s")
print(f"随机基准 = {k / m * 100:.1f}%")