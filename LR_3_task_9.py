import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Завантаження даних з файлу
data = np.loadtxt('data_clustering.txt', delimiter=',')  # Завантаження даних з файлу

# Оцінка оптимальної ширини вікна
bandwidth = estimate_bandwidth(data, quantile=0.2, n_samples=len(data))

# Створення об'єкта MeanShift для кластеризації з обраною шириною вікна
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)

# Навчання моделі кластеризації Mean Shift
ms.fit(data)

# Визначення центрів кластерів
cluster_centers = ms.cluster_centers_

# Оцінка кількості кластерів
num_clusters = len(cluster_centers)

# Візуалізація результатів
plt.scatter(data[:, 0], data[:, 1], c=ms.labels_, cmap='viridis', s=50)
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='black', s=200, alpha=0.7)

plt.title(f'Mean Shift Clustering (Кількість кластерів: {num_clusters})')
plt.xlabel('Ознака 1')
plt.ylabel('Ознака 2')
plt.show()

print(f'Оцінена кількість кластерів: {num_clusters}')
