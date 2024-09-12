# src/data_preprocessing.py
import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

def load_and_preprocess_images(folder_path):
    images = []
    labels = []
    for label in os.listdir(folder_path):
        label_folder = os.path.join(folder_path, label)
        for img_file in os.listdir(label_folder):
            img_path = os.path.join(label_folder, img_file)
            image = cv2.imread(img_path)
            image = cv2.resize(image, (256, 256))  # Redimensiona para 256x256

            images.append(image)
            labels.append(label)  # Adiciona o rótulo (healthy ou rust)
    return np.array(images), np.array(labels)

def preprocess_data(images, labels):
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)  # Converte os rótulos em formato one-hot
    return train_test_split(images, labels, test_size=0.2, random_state=42), lb