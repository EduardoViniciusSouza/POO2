import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader

class SoybeanDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]

        # Carregar imagem
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Redimensionar imagem para 256x256
        image = cv2.resize(image, (256, 256))

        if self.transform:
            image = self.transform(image)

        return image, label

def load_image_paths_and_labels(folder_path):
    image_paths = []
    labels = []

    # Mapear rótulos
    label_map = {
        'healthy': 0,
        'rust': 1
    }

    for label_folder in os.listdir(folder_path):
        if label_folder in label_map:  # Verifica se a pasta é uma das esperadas
            label_folder_path = os.path.join(folder_path, label_folder)
            if os.path.isdir(label_folder_path):
                for img_file in os.listdir(label_folder_path):
                    img_path = os.path.join(label_folder_path, img_file)
                    if os.path.isfile(img_path):
                        image_paths.append(img_path)
                        labels.append(label_map[label_folder])  # Usa o mapeamento

    return image_paths, labels

# Caminho para a pasta 'data'
data_folder = '../data'  # Ajuste o caminho conforme necessário

# Carregar os dados
train_image_paths, train_labels = load_image_paths_and_labels(data_folder)
test_image_paths, test_labels = load_image_paths_and_labels(data_folder)  # Use os mesmos dados para treino e teste como exemplo

# Criar datasets e dataloaders
train_dataset = SoybeanDataset(train_image_paths, train_labels)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = SoybeanDataset(test_image_paths, test_labels)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
#try to commit