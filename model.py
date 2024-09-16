# src/model.py
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import regularizers

def create_model(trial):

    l2_reg = trial.suggest_loguniform('l2', 1e-5, 1e-2)
    input_shape = (256, 256, 3)
    num_classes = 2

    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape,  kernel_regularizer=regularizers.l2(l2_reg)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(l2_reg)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(l2_reg)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')  # Saída para o número de classes
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model