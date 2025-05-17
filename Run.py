import os
from crop_patches import create_patches
from extract_features import extract_features
from classify_features import classify_features
import pandas as pd

# Ścieżki
images_dir = 'textures'
patches_dir = 'patches'
features_file = 'features.csv'

print("Etap 1: Tworzenie próbek tekstury (patches)")
create_patches(images_dir, patches_dir)
print("Próbki tekstury zapisane w folderze:", patches_dir)


print("\nStatystyki zbioru PATCHES:")
category_counts = {}
total_patches = 0
for category in os.listdir(patches_dir):
    cat_path = os.path.join(patches_dir, category)
    if os.path.isdir(cat_path):
        count = len([f for f in os.listdir(cat_path) if f.endswith('.png')])
        category_counts[category] = count
        total_patches += count
for category, count in category_counts.items():
    print(f" - {category}: {count} próbek")
print(f"Łączna liczba próbek: {total_patches}")
print(f"Liczba kategorii: {len(category_counts)}")

print("\nEtap 2: Ekstrakcja cech (GLCM)")
extract_features(patches_dir, features_file)
print("Zapisano cechy do:", features_file)

df = pd.read_csv(features_file)
print("\nStatystyki zbioru cech:")
print(f"Liczba przykładów: {len(df)}")
print(f"Liczba cech: {len(df.columns)-1}")
print("Kategorie:", df['label'].unique())

print("\nEtap 3: Klasyfikacja")
accuracy, report = classify_features(features_file)
print(f"\nOgólna dokładność klasyfikatora: {accuracy:.2%}")

print("\nDokładność dla poszczególnych kategorii:")
for label, metrics in report.items():
    if label not in ['accuracy', 'macro avg', 'weighted avg']:
        print(f" - {label}: {metrics['precision']:.2f} precision, {metrics['recall']:.2f} recall, {metrics['f1-score']:.2f} F1-score")


print("\nZakończono analizę tekstur.")
