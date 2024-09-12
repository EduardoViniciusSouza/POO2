# src/train.py
import numpy as np
from pre_processing import load_and_preprocess_images, preprocess_data
from model import create_model

# Carregando e pré-processando o conjunto de dados
images, labels = load_and_preprocess_images('dataset/')
(X_train, X_val, y_train, y_val), lb = preprocess_data(images, labels)

# Normaliza os dados
X_train = X_train / 255.0
X_val = X_val / 255.0

# Criando e treinando o modelo
model = create_model(num_classes=len(lb.classes_))
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Salvando o modelo
model.save('soybean_rust_model.h5')