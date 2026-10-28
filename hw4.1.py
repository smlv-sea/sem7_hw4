# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 10:16:33 2026

@author: Ольга
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler


# Генерация данных
def generate_data(n_samples=300, random_state=42):
    np.random.seed(random_state)
    
    # Группа 1: Равномерное распределение [-1, 1]
    a1 = np.random.uniform(-1, 1, n_samples)
    b1 = np.random.uniform(-1, 1, n_samples)
    eps1 = np.random.uniform(-1, 1, n_samples)
    z1 = 0.5 + a1 + b1 + eps1 
    group1 = np.column_stack((a1, b1, z1))
    
    # Группа 2: Нормальное распределение (mean=0, std=0.5)
    a2 = np.random.normal(0, 0.5, n_samples)
    b2 = np.random.normal(0, 0.5, n_samples)
    eps2 = np.random.normal(0, 0.5, n_samples)
    y2 = b2 + eps2
    z2 = 0.5 + a2 + eps2
    group2 = np.column_stack((a2, y2, z2))
    
    # Объединяем данные
    X = np.vstack((group1, group2))
    
    # Истинные метки (0 для первой группы, 1 для второй)
    true_labels = np.concatenate((np.zeros(n_samples), np.ones(n_samples)))
    
    return X, true_labels

# Генерируем данные
X_raw, true_labels = generate_data(n_samples=200, random_state=42)

# Масштабируем данные
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)


# Кластеризация методом Affinity Propagation
def evaluate_ap(X, true_labels, preference= None, random_state=42):
    # preference - это "насколько точка хочет быть центром кластера". 
    # Если preference=None, берется медиана сходств. Меньше медианы = меньше кластеров.
    ap = AffinityPropagation(preference=preference, random_state=random_state)
    ap.fit(X)
    
    
    # Получаем предсказанные метки
    pred_labels = ap.labels_
    n_clusters = len(ap.cluster_centers_indices_)
    
    # Считаем метрики (ARI требует истинных меток)
    ari = adjusted_rand_score(true_labels, pred_labels)
    # DBI не требует истинных меток, он оценивает разделимость кластеров
    if 1 < n_clusters < len(X):
     dbi = davies_bouldin_score(X, pred_labels)
    else:
     dbi = np.nan

    
    return n_clusters, ari, dbi, pred_labels


# Оценка устойчивости (разные seed)
print("Оценка устойчивости Affinity Propagation (10 запусков)")
results = []
for seed in range(10):
    # Генерируем новые данные и сразу кластеризуем
    X_iter, y_true_iter = generate_data(n_samples=200, random_state=seed)
    n_cl, ari, dbi, _ = evaluate_ap(X_iter, y_true_iter, random_state=seed)
    results.append((n_cl, ari, dbi))
    print(f"Запуск {seed+1}: Кластеров={n_cl}, ARI={ari:.3f}, DBI={dbi:.3f}")

mean_ari = np.mean([r[1] for r in results])
mean_dbi = np.mean([r[2] for r in results])
print(f"\nСредний ARI: {mean_ari:.3f} (1.0 - идеально)")
print(f"Средний DBI: {mean_dbi:.3f} (чем меньше, тем лучше)")


# Изменение параметров близости (Preference)
print("\nВлияние параметра Preference")
for pref_val in [-50, -25, -10, None, 10]:
    n_cl, ari, dbi, _ = evaluate_ap(X, true_labels, preference=pref_val, random_state=42)
    print(f"Preference={pref_val}: Кластеров={n_cl}, ARI={ari:.3f}, DBI={dbi:.3f}")


# Визуализация результатов
# Возьмем стандартный запуск для картинки
ap_final = AffinityPropagation(random_state=42)
ap_final.fit(X)
pred_labels = ap_final.labels_

fig = plt.figure(figsize=(15, 6))

# График 1: Истинные классы (как должно быть)
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(X[true_labels==0, 0], X[true_labels==0, 1], X[true_labels==0, 2], c='blue', label='Класс 1 (Uniform)', alpha=0.6)
ax1.scatter(X[true_labels==1, 0], X[true_labels==1, 1], X[true_labels==1, 2], c='orange', label='Класс 2 (Normal)', alpha=0.6)
ax1.set_title("Истинные классы")
ax1.legend()

# График 2: Результат Affinity Propagation
ax2 = fig.add_subplot(122, projection='3d')
scatter = ax2.scatter(X[:, 0], X[:, 1], X[:, 2], c=pred_labels, cmap='viridis', alpha=0.6)
# Отмечаем центры (экземпляры)
centers = X[ap_final.cluster_centers_indices_]
ax2.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='red', marker='X', s=100, label='Экземпляры (Centers)')
ax2.set_title(f"Affinity Propagation (Кластеров: {len(centers)})")
ax2.legend()

plt.tight_layout()
plt.show()