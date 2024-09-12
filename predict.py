# src/predict.py
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
result = predict_image(model, 'healthySoy.JPG', lb)
print(f'A imagem é: {result}')