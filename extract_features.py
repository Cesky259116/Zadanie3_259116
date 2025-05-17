import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops

def extract_features(patches_dir, output_csv):
    features = []

    for class_name in os.listdir(patches_dir):
        class_path = os.path.join(patches_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        for filename in os.listdir(class_path):
            if not filename.endswith('.png'):
                continue

            image_path = os.path.join(class_path, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            glcm = graycomatrix(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

            contrast = graycoprops(glcm, 'contrast')[0, 0]
            dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
            homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
            energy = graycoprops(glcm, 'energy')[0, 0]
            correlation = graycoprops(glcm, 'correlation')[0, 0]
            asm = graycoprops(glcm, 'ASM')[0, 0]

            features.append({
                'contrast': contrast,
                'dissimilarity': dissimilarity,
                'homogeneity': homogeneity,
                'energy': energy,
                'correlation': correlation,
                'asm': asm,
                'label': class_name
            })

    df = pd.DataFrame(features)
    df.to_csv(output_csv, index=False)
