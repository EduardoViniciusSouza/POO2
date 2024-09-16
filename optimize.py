import optuna
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from pre_processing import load_and_preprocess_images, preprocess_data
from model import create_model

def objective(trial):
    # Carregando e pré-processando o conjunto de dados
    images, labels = load_and_preprocess_images('dataset/')
    (X_train, X_val, y_train, y_val), lb = preprocess_data(images, labels)

    # Normaliza os dados
    X_train = X_train / 255.0
    X_val = X_val / 255.0

    model = create_model(trial)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

    val_acc = history.history['val_accuracy'][-1]  # Usa a última precisão de validação
    return val_acc


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

best_params = study.best_params

print('Melhores hiperparametros: ', best_params)