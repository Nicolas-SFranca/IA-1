# pipeline_completo_multimodal.py
import tensorflow as tf
import numpy as np
import cv2
import os
import google.generativeai as genai
import webbrowser
from huggingface_hub import login

# === CONFIGURAÇÕES ===
API_KEY_GEMINI ="_kXv3qG7EI"
HF_TOKEN = "hf_abfGhnOpQW76"  # substitua por seu token válido
CAMINHO_MODELO = "modelo_mobilenet_balanceado.h5"
CAMINHO_SPECTROS = r"C:\Users\Administrador\Documents\meu_projeto_ecoauralia\spectrogramas"
CAMINHO_IMAGEM = r"C:\Users\Administrador\Documents\meu_projeto_ecoauralia\spectrogramas\ovelhas\ovelhas_010.png"

# === AUTENTICAR GEMINI ===
genai.configure(api_key=API_KEY_GEMINI)

# === INICIALIZAR MODELOS ===
modelo_texto = genai.GenerativeModel("models/gemini-1.5-pro")
modelo_cnn = tf.keras.models.load_model(CAMINHO_MODELO)
labels = sorted(os.listdir(CAMINHO_SPECTROS))

# === FUNÇÃO: CLASSIFICAR ESPECTROGRAMA ===
def classificar_espectrograma(imagem_path):
    img = cv2.imread(imagem_path)
    if img is None:
        print(f"❌ Erro: Imagem não encontrada: {imagem_path}")
        return None
    img = cv2.resize(img, (128, 128))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    pred = modelo_cnn.predict(img)
    indice = np.argmax(pred)
    confianca = np.max(pred)
    especie = labels[indice]
    print(f"🔍 Previsão: {especie} (confiança: {confianca:.2%})")
    return especie

# === FUNÇÃO: GERAR TEXTO COM GEMINI ===
def gerar_descricao(animal):
    prompt = f"Descreva o animal '{animal}' em detalhes. Onde vive, como é o som que ele faz, e curiosidades sobre ele."
    resposta = modelo_texto.generate_content(prompt)
    print("\n📖 Descrição:")
    print(resposta.text)
    return resposta.text

# === FUNÇÃO: BUSCAR IMAGEM REAL ===
def buscar_imagem_real(animal):
    url = f"https://www.google.com/search?tbm=isch&q={animal}"
    print(f"\n🔎 Abrindo imagens reais de: {animal}")
    webbrowser.open(url)

# === EXECUÇÃO PRINCIPAL ===
if __name__ == "__main__":
    especie = classificar_espectrograma(CAMINHO_IMAGEM)
    if especie:
        gerar_descricao(especie)
        buscar_imagem_real(especie)
