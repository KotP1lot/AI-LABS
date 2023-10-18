import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

# Завантаження даних Iris
iris = load_iris()
X = iris.data  # Ознаки (довжина і ширина чашолистка і пелюстки)
y = iris.target  # Мітки класів

# Створення об'єкту KMeans для кластеризації на 3 кластери
kmeans = KMeans(n_clusters=3, random_state=0)

# Навчання моделі кластеризації KMeans
kmeans.fit(X)

# Отримання міток для кожного прикладу даних
y_kmeans = kmeans.predict(X)

# Візуалізація результатів
plt.figure(figsize=(10, 5))

# Відобразимо довжину чашолистка проти довжини пелюстки з кольорами на основі міток кластерів
plt.scatter(X[:, 0], X[:, 2], c=y_kmeans, cmap='viridis', s=50)
plt.xlabel('Довжина чашолистка')
plt.ylabel('Довжина пелюстки')
plt.title('Кластеризація K-Means (3 кластери) для Iris')

# Відобразимо центри кластерів
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 2], c='black', s=200, alpha=0.5, label='Центри кластерів')
plt.legend()

plt.show()
