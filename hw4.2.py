# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 11:51:51 2026

@author: Ольга
"""

import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.cluster import AffinityPropagation, Birch
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import adjusted_rand_score
from pathlib import Path


# Загрузка файлов
 
# Первые 10 текстов — античные, остальные 10 — новые.
output_dir = Path(r'C:\1 Ольга работа\учёба\инфа\7 семестр Машинное обучение\дз\4.2\philosophy_texts')


# Формируем список файлов с суффиксом _new
input_files = [
    f'{output_dir}/ancient_{i}.txt' for i in range(1, 11)
] + [
    f'{output_dir}/modern_{i}.txt' for i in range(1, 11)
]

texts = []
for filename in input_files:
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            texts.append(f.read())
    except FileNotFoundError:
        print(f"ОШИБКА: Файл {filename} не найден!")
        exit()

true_labels = np.array([0]*10 + [1]*10)


# Извлечение признаков
tfidf_vect = TfidfVectorizer(stop_words='english')
X_tfidf = tfidf_vect.fit_transform(texts)

print(f"Исходная матрица признаков TF-IDF (документы x слова): {X_tfidf.shape}")

# Понижение размерности для работы алгоритмов (оставляем 15 компонент)
svd = TruncatedSVD(n_components=15, random_state=42)
X_dense = svd.fit_transform(X_tfidf)


# Кластеризация AFFINITY PROPAGATION
print("\n" + "="*60)
print("МЕТОД 1: AFFINITY PROPAGATION")
print("="*60)

# preference=-50 заставляет алгоритм искать меньше кластеров
ap = AffinityPropagation(preference=-50, random_state=42)
ap.fit(X_dense)
ap_labels = ap.labels_
n_clusters_ap = len(ap.cluster_centers_indices_)
ari_ap = adjusted_rand_score(true_labels, ap_labels)

print(f"Количество найденных кластеров: {n_clusters_ap}")
print(f"Adjusted Rand Index (ARI): {ari_ap:.3f}")


# Кластеризация BIRCH
print("\n" + "="*60)
print("МЕТОД 2: BIRCH")
print("="*60)

# BIRCH позволяет жестко задать число кластеров.
# threshold=0.1 — это радиус, в пределах которого точки объединяются в один подкластер.
# Мы точно знаем, что классов 2, поэтому ставим n_clusters=2.
birch = Birch(n_clusters=2, threshold=0.1)
birch.fit(X_dense)
birch_labels = birch.labels_
ari_birch = adjusted_rand_score(true_labels, birch_labels)

print(f"Количество найденных кластеров: 2 (задали вручную)")
print(f"Adjusted Rand Index (ARI): {ari_birch:.3f}")


# Сравнительная таблица результатой
print("\n" + "="*60)
print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ КЛАСТЕРИЗАЦИИ")
print("="*60)
print(f"{'Файл':<18} | {'Истина':<10} | {'Affinity Prop':<15} | {'BIRCH':<10}")
print("-" * 60)

for i in range(len(texts)):
    true_class = "Античный (0)" if true_labels[i] == 0 else "Новый (1)"
    short_name = os.path.basename(input_files[i])
    print(f"{short_name:<25} | {true_class:<10} | {ap_labels[i]:<15} | {birch_labels[i]:<10}")


# Вывод
print("\n" + "="*60)
print("ВЫВОД ОБ УСПЕШНОСТИ КЛАСТЕРИЗАЦИИ")
print("="*60)

def evaluate(ari, method_name):
    if ari > 0.8:
        print(f"[{method_name}] ОТЛИЧНО. Тексты разделены практически идеально (ARI={ari:.3f}).")
    elif ari > 0.4:
        print(f"[{method_name}] УДОВЛЕТВОРИТЕЛЬНО. Основная структура уловлена, но есть ошибки (ARI={ari:.3f}).")
    else:
        print(f"[{method_name}] НЕУДОВЛЕТВОРИТЕЛЬНО. Алгоритм не справился с задачей (ARI={ari:.3f}).")

evaluate(ari_ap, "Affinity Propagation")
evaluate(ari_birch, "BIRCH")

