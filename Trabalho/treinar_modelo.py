# treinar_mobilenet_balanceado.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os

# Diretório com espectrogramas organizados por pasta
base_dir = r"C:\Users\Administrador\Documents\meu_projeto_ecoauralia\spectrogramas"

# Gerador de imagens com Data Augmentation
augmentador = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    validation_split=0.2
)

# Dados de treino
train_generator = augmentador.flow_from_directory(
    base_dir,
    target_size=(128, 128),
    batch_size=16,
    class_mode='categorical',
    subset='training'
)

# Dados de validação
val_generator = augmentador.flow_from_directory(
    base_dir,
    target_size=(128, 128),
    batch_size=16,
    class_mode='categorical',
    subset='validation'
)

# Calcular class weights automaticamente
labels_array = train_generator.classes
pesos = compute_class_weight(class_weight='balanced', classes=np.unique(labels_array), y=labels_array)
class_weights = dict(zip(np.unique(labels_array), pesos))

print("Pesos das classes:", class_weights)

# Carregar MobileNetV2 base
base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=Input(shape=(128, 128, 3)))
for layer in base_model.layers:
    layer.trainable = False

# Cabeça personalizada
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compilar
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinar
model.fit(
    train_generator,
    epochs=40,
    validation_data=val_generator,
    class_weight=class_weights
)

# Salvar modelo
model.save("modelo_mobilenet_balanceado.h5")
print("✅ Modelo salvo como modelo_mobilenet_balanceado.h5")
