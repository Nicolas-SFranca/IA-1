"""
treinar_modelo_mfcc.py

Escolhi usar MFCCs (Mel-Frequency Cepstral Coefficients) para treinar um modelo de classificação de sons de animais.
Esse método é amplamente utilizado na análise de áudio, especialmente em reconhecimento de fala e classificação de sons ambientais, como os de animais. 
Os MFCCs representam a frequência de áudio de forma semelhante à percepção auditiva humana, capturando aspectos importantes e ignorando ruídos irrelevantes. 
Eles são extraídos de arquivos de áudio, como os .ogg, e transformados em matrizes que representam as características sonoras.

Apos isso e feito o treinamento do modelo, conforme o codigo a seguir:
 
"""

import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Diretório com áudios separados por espécie
diretorio_audios = r"C:\Users\Administrador\Documents\meu_projeto_ecoauralia\train_audio"
especies = sorted(os.listdir(diretorio_audios))

mfcc_list = []
rotulos = []

print("🔍 Extraindo MFCCs dos arquivos de áudio...")

for idx, especie in enumerate(especies):
    caminho_especie = os.path.join(diretorio_audios, especie)

    if not os.path.isdir(caminho_especie):
        continue

    for audio in os.listdir(caminho_especie):
        if audio.endswith(".ogg"):
            caminho_audio = os.path.join(caminho_especie, audio)

            try:
                y_audio, sr = librosa.load(caminho_audio, sr=32000)
                mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=13)
                mfcc = mfcc[:, :216]  # Garantir mesmo tamanho

                if mfcc.shape[1] < 216:
                    mfcc = np.pad(mfcc, ((0, 0), (0, 216 - mfcc.shape[1])), mode='constant')

                mfcc_list.append(mfcc)
                rotulos.append(idx)

            except Exception as e:
                print(f"Erro ao processar '{audio}': {e}")

X = np.array(mfcc_list)
y = np.array(rotulos)
X = np.expand_dims(X, axis=-1)

print(f"✅ Extração concluída: Dados {X.shape}, Rótulos {y.shape}")

# Dividir dados
X_treino, X_valid, y_treino, y_valid = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Ajustar pesos para classes desbalanceadas
pesos = compute_class_weight(class_weight='balanced', classes=np.unique(y_treino), y=y_treino)
peso_classes = dict(zip(np.unique(y_treino), pesos))

# Construir modelo CNN
modelo = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(13, 216, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(especies), activation='softmax')
])

modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Treinamento
historico = modelo.fit(
    X_treino, y_treino,
    epochs=40,
    validation_data=(X_valid, y_valid),
    class_weight=peso_classes
)

# Salvar modelo treinado
modelo.save("modelo_mfcc.h5")
print("✅ Modelo salvo como 'modelo_mfcc.h5'!")
