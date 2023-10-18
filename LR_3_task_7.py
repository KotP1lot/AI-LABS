import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Завантаження вхідних даних
data = np.loadtxt('data_clustering.txt', delimiter=',')

# Включення вхідних даних до графіка
plt.scatter(data[:, 0], data[:, 1])
plt.title('Input Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Створення об'єкту КМеаns
kmeans = KMeans(n_clusters=5, init='k-means++', n_init=10, random_state=0)

# Навчання моделі кластеризації КМеаns
kmeans.fit(data)

# Визначення кроку сітки
h = 0.02

# Визначення меж для сітки
x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Передбачення вихідних міток для всіх точок сітки
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Графічне відображення областей та виділення їх кольором
plt.figure(1)
plt.imshow(Z, interpolation='nearest', extent=(xx.min(), xx.max(), yy.min(), yy.max()), cmap=plt.cm.Paired, aspect='auto', origin='lower')

# Відображення вхідних точок
plt.scatter(data[:, 0], data[:, 1], c=kmeans.labels_, cmap=plt.cm.Paired)
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Відображення центрів кластерів
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', label='Cluster Centers')

plt.show()
