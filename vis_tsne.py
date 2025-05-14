# import numpy as np
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt

# # 生成样本数据
# num_samples_per_class = 15
# num_samples = num_samples_per_class * 2

# # 生成T类数据
# X_T = np.zeros((num_samples_per_class, 2))
# X_T[:, 0] = np.random.normal(loc=0, scale=1, size=num_samples_per_class)
# X_T[:, 1] = np.random.normal(loc=0, scale=1, size=num_samples_per_class)

# # 生成I类数据
# X_I = np.zeros((num_samples_per_class, 2))
# X_I[:, 0] = np.random.normal(loc=5, scale=1, size=num_samples_per_class)
# X_I[:, 1] = np.random.normal(loc=5, scale=1, size=num_samples_per_class)

# # 合并数据
# X = np.vstack([X_T, X_I])
# y = np.array(['T'] * num_samples_per_class + ['I'] * num_samples_per_class)

# # 使用T-SNE进行降维
# tsne = TSNE(n_components=2, random_state=42)
# X_tsne = tsne.fit_transform(X)

# # 可视化
# plt.figure(figsize=(8, 6))

# # 绘制T类样本
# plt.scatter(X_tsne[y=='T', 0], X_tsne[y=='T', 1], label='T', color='blue')

# # 绘制I类样本
# plt.scatter(X_tsne[y=='I', 0], X_tsne[y=='I', 1], label='I', color='red')

# plt.title('T-SNE Visualization of T and I Classes')
# plt.xlabel('T-SNE Component 1')
# plt.ylabel('T-SNE Component 2')
# plt.legend()
# plt.savefig('tsne_visualization.png')

# plt.show()


# import numpy as np
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt

# # 生成样本数据
# num_samples_per_class = 30
# num_samples = num_samples_per_class * 2

# # 生成T类数据
# mean_T = [2, 2]  # 调整T类的均值
# cov_T = [[1, 0], [0, 1]]  # 调整T类的方差
# X_T = np.random.multivariate_normal(mean_T, cov_T, num_samples_per_class)

# # 生成I类数据
# mean_I = [8, 8]  # 调整I类的均值
# cov_I = [[1, 0], [0, 1]]  # 调整I类的方差
# X_I = np.random.multivariate_normal(mean_I, cov_I, num_samples_per_class)

# # 合并数据
# X = np.vstack([X_T, X_I])
# y = np.array(['T'] * num_samples_per_class + ['I'] * num_samples_per_class)

# # 使用T-SNE进行降维
# tsne = TSNE(n_components=2, random_state=42)
# X_tsne = tsne.fit_transform(X)

# # 可视化
# plt.figure(figsize=(8, 6))

# # 绘制T类样本
# plt.scatter(X_tsne[y=='T', 0], X_tsne[y=='T', 1], label='T', color='blue')

# # 绘制I类样本
# plt.scatter(X_tsne[y=='I', 0], X_tsne[y=='I', 1], label='I', color='red')

# plt.title('T-SNE Visualization of T and I Classes')
# plt.xlabel('T-SNE Component 1')
# plt.ylabel('T-SNE Component 2')
# plt.legend()
# plt.savefig('tsne_visualization.png')
# plt.show()


import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 生成样本数据
num_samples_per_class = 30
num_samples = num_samples_per_class * 2

# 生成T类数据
X_T = np.random.rand(num_samples_per_class, 2) * 5  # 调整T类的数据分布范围
X_T += np.array([2, 2])  # 调整T类的中心位置

# 生成I类数据
X_I = np.random.rand(num_samples_per_class, 2) * 5  # 调整I类的数据分布范围
X_I += np.array([8, 8])  # 调整I类的中心位置

# 合并数据
X = np.vstack([X_T, X_I])
y = np.array(['T'] * num_samples_per_class + ['I'] * num_samples_per_class)

# 使用T-SNE进行降维
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# 可视化
plt.figure(figsize=(8, 6))

# 绘制T类样本
plt.scatter(X_tsne[y=='T', 0], X_tsne[y=='T', 1], label='T', color='blue')

# 绘制I类样本
plt.scatter(X_tsne[y=='I', 0], X_tsne[y=='I', 1], label='I', color='red')

# 不显示坐标轴
plt.xticks([])
plt.yticks([])

plt.title('T-SNE Visualization of T and I Classes')
plt.legend()
plt.savefig('tsne_visualization.png')
plt.show()
