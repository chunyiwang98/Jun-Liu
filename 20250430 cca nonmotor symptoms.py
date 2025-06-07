import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# 加载 Excel 文件
file_path = r'E:/研三/MSA EEG/1225结果/CCA/总表.xlsx'
df = pd.read_excel(file_path)

# 提取临床量表数据（G 到 T 列，索引 6 到 19）和 EEG 特征数据（AA 到 AR 列，索引 26 到 44）
scales = df.iloc[:, 20:26].values  # G 到 T
eeg_features = df.iloc[:, 26:45].values  # AA 到 AR

# 处理缺失值（均值填充）
imputer = SimpleImputer(strategy='mean')
scales = imputer.fit_transform(scales)
eeg_features = imputer.fit_transform(eeg_features)

# 标准化数据
scaler = StandardScaler()
scales = scaler.fit_transform(scales)
eeg_features = scaler.fit_transform(eeg_features)

# 自动选择最佳 n_components（基于累积方差贡献率）
def select_best_n_components(X, threshold=0.9):
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    explained_variance_ratio = S**2 / np.sum(S**2)
    cumulative_variance = np.cumsum(explained_variance_ratio)
    best_n = np.searchsorted(cumulative_variance, threshold) + 1
    return min(best_n, X.shape[1])  # 限制最大值为变量数

n_components_scales = select_best_n_components(scales)
n_components_eeg = select_best_n_components(eeg_features)
n_components = min(n_components_scales, n_components_eeg)  # 取最小值
print(f"自动选择的最佳 n_components: {n_components}")

# 执行 CCA
cca = PLSRegression(n_components=n_components, scale=True, max_iter=1000, tol=1e-06)
cca.fit(scales, eeg_features)

# 计算典型相关分数
scales_c, eeg_features_c = cca.transform(scales, eeg_features)

# 计算观察到的典型相关系数
observed_corrs = np.array([np.corrcoef(scales_c[:, i], eeg_features_c[:, i])[0, 1] for i in range(n_components)])

# 置换检验
np.random.seed(800)
n_permutations = 10000  # 置换次数
permuted_corrs = np.zeros((n_permutations, n_components))
for i in range(n_permutations):
    if np.random.rand() > 0.5:
        permuted_scales = np.random.permutation(scales)  # 随机打乱临床量表
        permuted_eeg = eeg_features
    else:
        permuted_scales = scales
        permuted_eeg = np.random.permutation(eeg_features)  # 随机打乱 EEG 特征
    cca.fit(permuted_scales, permuted_eeg)
    permuted_scales_c, permuted_eeg_c = cca.transform(permuted_scales, permuted_eeg)
    permuted_corrs[i, :] = [np.corrcoef(permuted_scales_c[:, j], permuted_eeg_c[:, j])[0, 1] for j in range(n_components)]

# 计算 p 值
p_values = np.mean(np.abs(permuted_corrs) >= np.abs(observed_corrs), axis=0)

# 打印典型相关系数及 p 值
print("\n典型相关系数和 p 值:")
for i in range(n_components):
    print(f"成分 {i + 1}: 相关系数 = {observed_corrs[i]:.4f}, p 值 = {p_values[i]:.4f}")

# 计算典型成分的方差贡献率
explained_variance_scales = np.var(scales_c, axis=0) / np.sum(np.var(scales_c, axis=0))
explained_variance_eeg = np.var(eeg_features_c, axis=0) / np.sum(np.var(eeg_features_c, axis=0))

# 计算累积方差贡献率
cumulative_variance_scales = np.cumsum(explained_variance_scales)
cumulative_variance_eeg = np.cumsum(explained_variance_eeg)

# 打印方差贡献率
print("\n典型成分的方差贡献率：")
for i in range(n_components):
    print(f"成分 {i+1}: 临床量表方差贡献率 = {explained_variance_scales[i]:.4f}, EEG 特征方差贡献率 = {explained_variance_eeg[i]:.4f}")

print("\n典型成分的累积方差贡献率：")
for i in range(n_components):
    print(f"成分 {i+1}: 临床量表累积方差贡献率 = {cumulative_variance_scales[i]:.4f}, EEG 特征累积方差贡献率 = {cumulative_variance_eeg[i]:.4f}")

# 绘制累积方差贡献率图
plt.figure(figsize=(8, 4))
plt.plot(range(1, n_components+1), cumulative_variance_scales, marker='o', label='Clinical Scales')
plt.plot(range(1, n_components+1), cumulative_variance_eeg, marker='o', label='EEG Features')
plt.title('Cumulative Explained Variance by Canonical Components')
plt.xlabel('Canonical Component')
plt.ylabel('Cumulative Explained Variance')
plt.xticks(range(1, n_components+1))
plt.grid()
plt.legend()
plt.show()

# 计算载荷（Loadings）
def calculate_loadings(X, X_c):
    """
    计算原始变量与典型变量之间的载荷（相关性）
    :param X: 原始变量矩阵 (n_samples, n_features)
    :param X_c: 典型变量矩阵 (n_samples, n_components)
    :return: 载荷矩阵 (n_features, n_components)
    """
    loadings = np.zeros((X.shape[1], X_c.shape[1]))
    for i in range(X.shape[1]):  # 遍历每个原始变量
        for j in range(X_c.shape[1]):  # 遍历每个典型变量
            loadings[i, j] = np.corrcoef(X[:, i], X_c[:, j])[0, 1]
    return loadings

# 计算临床量表和EEG特征的载荷
loadings_scales = calculate_loadings(scales, scales_c)
loadings_eeg = calculate_loadings(eeg_features, eeg_features_c)

# 输出临床量表的载荷
clinical_scales = df.columns[20:26].tolist()  # 临床量表名称
print("\n每个临床量表对每个典型成分的贡献度（载荷）：")
for i in range(n_components):
    print(f"典型成分 {i+1}:")
    for j, scale in enumerate(clinical_scales):
        print(f"  {scale}: {loadings_scales[j, i]:.4f}")

# 输出EEG特征的载荷
eeg_features_list = df.columns[26:44].tolist()  # EEG 特征名称
print("\n每个 EEG 特征对每个典型成分的贡献度（载荷）：")
for i in range(n_components):
    print(f"典型成分 {i+1}:")
    for j, feature in enumerate(eeg_features_list):
        print(f"  {feature}: {loadings_eeg[j, i]:.4f}")

# 定义颜色方案
colors = {
    'PSD': '#CC7C71',  # 蓝色
    'AEC': '#EAA558',  # 橙色
    'wPLI': '#EAA558', # 橙色
    'PAC': '#CB9475'   # 绿色
}

# 根据 EEG 特征类型分配颜色
eeg_colors = []
for feature in eeg_features_list:
    if 'PSD' in feature:
        eeg_colors.append(colors['PSD'])
    elif 'AEC' in feature or 'wPLI' in feature:
        eeg_colors.append(colors['AEC'])
    elif 'PAC' in feature:
        eeg_colors.append(colors['PAC'])
    else:
        eeg_colors.append('gray')  # 默认颜色

# 绘制 CCA 载荷图（第三典型成分）
plt.figure(figsize=(6, 4))

# 临床量表的载荷图（第三典型成分）
plt.bar(range(loadings_scales.shape[0]), loadings_scales[:, 2], color='#1f77b4')  # 所有临床量表使用蓝色
plt.title('Non-motor Symptoms', fontsize=13)
plt.xlabel('Clinical Scales', fontsize=12)
plt.ylabel('CCA Loadings', fontsize=12)
plt.xticks(ticks=np.arange(len(clinical_scales)), labels=clinical_scales, rotation=45, ha='right', fontsize=10)
plt.tight_layout()
plt.show()

# EEG 特征的载荷图（第三典型成分）
plt.figure(figsize=(10, 4))
plt.bar(range(loadings_eeg.shape[0]), loadings_eeg[:, 2], color=eeg_colors)
plt.title('EEG Features', fontsize=13)
plt.xlabel('EEG Features', fontsize=12)
plt.ylabel('CCA Loadings', fontsize=12)
plt.xticks(ticks=np.arange(len(eeg_features_list)), labels=eeg_features_list, rotation=45, ha='right', fontsize=10)
plt.tight_layout()
plt.show()

# 选择成分 2 的典型相关分数
component_idx = 2  # 成分 3（索引从 0 开始）
scales_component2 = scales_c[:, component_idx]
eeg_component2 = eeg_features_c[:, component_idx]

# 绘制散点图并添加 95% 置信区间
plt.figure(figsize=(5, 4))
sns.regplot(x=scales_component2, y=eeg_component2, ci=95, scatter_kws={'alpha': 0.7}, line_kws={'color': 'red'})
plt.title("r = 0.348, p = 0.0226", fontsize=10)
plt.xlabel("Canonical Scores of Non-motor Symptoms", fontsize=10)
plt.ylabel("Canonical Scores of EEG Features", fontsize=10)
plt.show()