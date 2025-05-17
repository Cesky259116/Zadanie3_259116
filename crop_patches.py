import os
import cv2
import numpy as np

def create_patches(input_folder, output_folder, patch_size=32):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for class_name in os.listdir(input_folder):
        class_path = os.path.join(input_folder, class_name)
        if not os.path.isdir(class_path):
            continue

        class_output_path = os.path.join(output_folder, class_name)
        os.makedirs(class_output_path, exist_ok=True)

        for filename in os.listdir(class_path):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            image_path = os.path.join(class_path, filename)
            image = cv2.imread(image_path)

            if image is None:
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape

            patch_id = 0
            for y in range(0, h - patch_size + 1, patch_size):
                for x in range(0, w - patch_size + 1, patch_size):
                    patch = gray[y:y+patch_size, x:x+patch_size]
                    patch_filename = f"{os.path.splitext(filename)[0]}_{patch_id}.png"
                    patch_path = os.path.join(class_output_path, patch_filename)
                    cv2.imwrite(patch_path, patch)
                    patch_id += 1
