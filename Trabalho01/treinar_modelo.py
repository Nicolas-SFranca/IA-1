"""
treinar_mobilenet_balanceado.py

Script para treinamento de modelo usando MobileNetV2 e espectrogramas para classificação automática de animais no projeto EcoAuralia.

Autor: Seu Nome
Data: 2025-04-24
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Diretório das imagens
pasta_imgs = r"C:\Users\Administrador\Documents\meu_projeto_ecoauralia\spectrogramas"

# Data Augmentation
aug = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    validation_split=0.2
)

# Dados treino e validação
treino_gen = aug.flow_from_directory(
    pasta_imgs,
    target_size=(128, 128),
    batch_size=16,
    class_mode='categorical',
    subset='training'
)

valid_gen = aug.flow_from_directory(
    pasta_imgs,
    target_size=(128, 128),
    batch_size=16,
    class_mode='categorical',
    subset='validation'
)

# Pesos balanceados
pesos = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(treino_gen.classes),
    y=treino_gen.classes
)
pesos_classes = dict(zip(np.unique(treino_gen.classes), pesos))

print("Pesos ajustados:", pesos_classes)

# Carregar e congelar MobileNet
mobilenet = MobileNetV2(weights='imagenet', include_top=False, input_tensor=Input(shape=(128, 128, 3)))
for layer in mobilenet.layers:
    layer.trainable = False

# Cabeça personalizada
x = mobilenet.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
predicao = Dense(treino_gen.num_classes, activation='softmax')(x)

modelo = Model(inputs=mobilenet.input, outputs=predicao)

# Compilação
modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinamento
historico = modelo.fit(
    treino_gen,
    epochs=40,
    validation_data=valid_gen,
    class_weight=pesos_classes
)

# Salvar
modelo.save("modelo_mobilenet_balanceado.h5")
print("🎉 Pronto! Seu modelo MobileNetV2 está treinado e salvo com sucesso!")
