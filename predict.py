# src/predict.py
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelBinarizer

def predict_image(model, img_path, lb):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (256, 256))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # Adiciona uma dimensão para o batch
    prediction = model.predict(img)
    
    
    print('[PREDICTION]: ', prediction);

    prediction_label = np.argmax(prediction, axis=1)

    return lb.inverse_transform(prediction_label)

# Carregando o modelo e o LabelBinarizer
model = load_model('soybean_rust_model.h5')
lb = LabelBinarizer()
# lb.classes_ = ['healthy', 'rust']  # Ajuste conforme suas classes

lb.fit(['healthy', 'rust']);
# Exemplo de uso

caminhoPastaDoentes = 'predict/doentes'
caminhoPastaSaudavel = 'predict/saudaveis'
rustPredictResults = []
healthyPredictResults = []

for item in os.listdir(caminhoPastaDoentes):
    item_path = os.path.join(caminhoPastaDoentes, item)
    resultRust = predict_image(model, item_path, lb)
    rustPredictResults.append(resultRust[0])

for item in os.listdir(caminhoPastaSaudavel):
    item_path = os.path.join(caminhoPastaSaudavel, item)
    resultHealthy = predict_image(model, item_path, lb)
    healthyPredictResults.append(resultHealthy[0])

qtdAcertos = 0
qtdErros = 0

for i in range(len(rustPredictResults)):
    if rustPredictResults[i] == 'rust':
        qtdAcertos += 1
    else:
        qtdErros += 1

for i in range(len(healthyPredictResults)):
    if healthyPredictResults[i] == 'healthy':
        qtdAcertos += 1
    else:
        qtdErros += 1

print(f'Quantidade de imagens corretas: {qtdAcertos}')
print(f'Quantidade de imagens erradas: {qtdErros}')

print('Rust predict results: ', rustPredictResults)
print('Healthy predict results: ', healthyPredictResults)