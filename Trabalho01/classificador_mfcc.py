"""
classificador_mfcc.py

Script usado no projeto EcoAuralia para identificar espécies de animais do campo ( inicialmente)
Este classificado e do modelo MFCCS, utilizando os mesmos arquivos .ogg e sendo treinado pelo arquivo "treinar_modelo_mfcc.py"
"""

import librosa
import numpy as np
import tensorflow as tf
import os

# Carregar modelo já treinado (CNN usando MFCC)
arquivo_modelo = "modelo_mfcc.h5"
classificador = tf.keras.models.load_model(arquivo_modelo)

# Pasta contendo as classes usadas no treinamento
pasta_classes = r"C:\Users\Administrador\Documents\meu_projeto_ecoauralia\train_audio"
labels = sorted(os.listdir(pasta_classes))


def carregar_audio(caminho, taxa=32000):
    return librosa.load(caminho, sr=taxa)


def ajustar_mfcc(mfcc, tamanho_fixo=216):
    if mfcc.shape[1] < tamanho_fixo:
        return np.pad(mfcc, ((0, 0), (0, tamanho_fixo - mfcc.shape[1])), mode='constant')
    return mfcc[:, :tamanho_fixo]


def extrair_mfcc(caminho_audio):
    try:
        y, sr = carregar_audio(caminho_audio)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc = ajustar_mfcc(mfcc)

        mfcc = np.expand_dims(mfcc, axis=-1)  # (13, 216, 1)
        mfcc = np.expand_dims(mfcc, axis=0)   # (1, 13, 216, 1)
        return mfcc
    except Exception as e:
        print(f"Opa! Não consegui processar esse áudio. O motivo foi: {e}")
        return None


def classificar_audio_mfcc(caminho_audio):
    mfcc = extrair_mfcc(caminho_audio)
    if mfcc is None:
        return

    predicao = classificador.predict(mfcc)
    indice = np.argmax(predicao)
    confianca = np.max(predicao)
    especie = labels[indice]

    print(f"🐦 Parece que encontramos um '{especie}' com {confianca:.1%} de certeza!")
    return especie


if __name__ == "__main__":
    caminho = input("Arraste aqui o arquivo (.ogg) que você quer identificar: ").strip('"')

    if not os.path.isfile(caminho):
        print("Hmm... parece que esse arquivo não existe. Tente novamente!")
    else:
        classificar_audio_mfcc(caminho_audio)
