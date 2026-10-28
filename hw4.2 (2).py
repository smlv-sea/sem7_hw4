# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 12:27:48 2026

@author: Ольга
"""

import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AffinityPropagation, Birch
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")


# Загрузка новых текстов из файла
#  В конец каждого текста добавляем одинаковые слова-маркеры soul, nature, god для античных текстов и reason, experience, consciousness для новых
output_dir = Path(r'C:\1 Ольга работа\учёба\инфа\7 семестр Машинное обучение\дз\4.2\philosophy_texts_new')

# Формируем список файлов с суффиксом _new
input_files = [
    f'{output_dir}/ancient_{i}_new.txt' for i in range(1, 11)
] + [
    f'{output_dir}/modern_{i}_new.txt' for i in range(1, 11)
]

texts_modified = []
for filename in input_files:
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            texts_modified.append(f.read())
    except FileNotFoundError:
        print(f"ОШИБКА: Файл {filename} не найден!")
        exit()

true_labels = np.array([0]*10 + [1]*10)


# 2. Извлечение признаков
tfidf_vect = TfidfVectorizer(stop_words='english')
X_tfidf = tfidf_vect.fit_transform(texts_modified)
print(f"Матрица TF-IDF (документы x слова): {X_tfidf.shape}")

# Сжатие и нормализация
svd = TruncatedSVD(n_components=10, random_state=42)
X_reduced = svd.fit_transform(X_tfidf)

scaler = StandardScaler()
X_dense = scaler.fit_transform(X_reduced)


# Кластеризация AFFINITY PROPAGATION
print("\n" + "="*60)
print("МЕТОД 1: AFFINITY PROPAGATION")
print("="*60)

best_ap = None
for pref in range(-20, -101, -5):
    ap = AffinityPropagation(preference=pref, random_state=42)
    ap.fit(X_dense)
    if len(ap.cluster_centers_indices_) == 2:
        best_ap = ap
        print(f"Найдено preference={pref}, создано 2 кластера.")
        break

if best_ap is None:
    best_ap = AffinityPropagation(preference=-50, random_state=42).fit(X_dense)

ap_labels = best_ap.labels_
ari_ap = adjusted_rand_score(true_labels, ap_labels)
print(f"Количество найденных кластеров: {len(set(ap_labels))}")
print(f"Adjusted Rand Index (ARI): {ari_ap:.3f}")


# Кластеризация BIRCH
print("\n" + "="*60)
print("МЕТОД 2: BIRCH")
print("="*60)

birch = Birch(n_clusters=2, threshold=0.1)
birch.fit(X_dense)
birch_labels = birch.labels_
ari_birch = adjusted_rand_score(true_labels, birch_labels)

print(f"Количество найденных кластеров: 2")
print(f"Adjusted Rand Index (ARI): {ari_birch:.3f}")


# Сравнительная таблица
print("\n" + "="*60)
print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
print("="*60)
print(f"{'Файл':<25} | {'Истина':<10} | {'Affinity Prop':<15} | {'BIRCH':<10}")
print("-" * 65)

for i in range(len(input_files)):
    true_class = "Античный" if true_labels[i] == 0 else "Новый"
    ap_match = "OK" if ap_labels[i] == true_labels[i] else "FAIL"
    br_match = "OK" if birch_labels[i] == true_labels[i] else "FAIL"
    short_name = os.path.basename(input_files[i])
    print(f"{short_name:<25} | {true_class:<10} | {ap_labels[i]} ({ap_match})   | {birch_labels[i]} ({br_match})")


# Вывод
print("\n" + "="*60)
print("ВЫВОД ОБ УСПЕШНОСТИ")
print("="*60)

def evaluate(ari, method_name):
    if ari > 0.8:
        print(f"[{method_name}] ОТЛИЧНО (ARI={ari:.3f}). Тексты разделены.")
    elif ari > 0.4:
        print(f"[{method_name}] УДОВЛЕТВОРИТЕЛЬНО (ARI={ari:.3f}). Есть ошибки.")
    else:
        print(f"[{method_name}] НЕУДОВЛЕТВОРИТЕЛЬНО (ARI={ari:.3f}). Не справился.")

evaluate(ari_ap, "Affinity Propagation")
evaluate(ari_birch, "BIRCH")

