import numpy as np
import json
import yfinance as yf
from sklearn.cluster import AffinityPropagation
from sklearn.preprocessing import StandardScaler
import pandas as pd

company_symbol_mapping = {
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corporation',
    # Додайте інші символічні позначення компаній тут
}
# Завантаження даних котирувань за допомогою yfinance
symbols = list(company_symbol_mapping.keys())
data = yf.download(symbols, start="2000-01-01", end="2010-01-01")
closing_prices = data['Adj Close']

# Обчислення різниць між котируваннями при відкритті та закритті біржі
price_diff = data['Open'] - data['Close']

# Нормалізація даних
scaler = StandardScaler()
price_diff_normalized = scaler.fit_transform(price_diff)

# Створення моделі графа
preference = -50  # Налаштований параметр вподобання (змініть за потребою)
affinity_propagation = AffinityPropagation(preference=preference, damping=0.9)
affinity_propagation.fit(price_diff_normalized)

# Визначення підгруп
cluster_centers_indices = affinity_propagation.cluster_centers_indices_
labels = affinity_propagation.labels_

# Виведення результатів
n_clusters_ = len(cluster_centers_indices)
print(f"Оцінена кількість підгруп: {n_clusters_}")

# Виведення символічних позначень компаній в кожній підгрупі
for cluster_idx in range(n_clusters_):
    cluster_mask = labels == cluster_idx
    companies_in_cluster = [company_symbol_mapping[symbols[i]] for i, is_in_cluster in enumerate(cluster_mask) if is_in_cluster]
    print(f"Підгрупа {cluster_idx + 1}: {', '.join(companies_in_cluster)}")
