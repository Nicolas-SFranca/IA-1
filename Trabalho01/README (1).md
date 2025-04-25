# 🌿 Ecoauralia: Identificação Sonora de Animais com Inteligência Artificial

## 📘 Introdução

**Ecoauralia** é um projeto de Inteligência Artificial desenvolvido para reconhecer o som de diferentes animais a partir de arquivos de áudio. Com foco na acessibilidade e aplicabilidade na área ambiental, o projeto simula um ecossistema de reconhecimento acústico, utilizando espectrogramas e MFCCs (Coeficientes Cepstrais de Frequência Mel) como entrada para modelos de deep learning.

O sistema é capaz de:
- Classificar sons de animais como gato, cachorro, vaca, porco e ovelha
- Gerar descrições detalhadas sobre as espécies com o modelo Gemini da Google
- Buscar imagens reais no navegador para visualização da espécie

## 🧰 Bibliotecas Utilizadas

| Biblioteca         | Finalidade |
|--------------------|-----------|
| `TensorFlow`       | Treinamento e uso de modelos de Deep Learning |
| `Keras`            | Interface de alto nível para redes neurais |
| `Librosa`          | Processamento de áudio e extração de MFCC/espectrogramas |
| `NumPy`            | Manipulação de arrays e dados científicos |
| `Matplotlib`       | Geração de espectrogramas em imagem |
| `Google Generative AI` | Geração de texto descritivo sobre os animais (Gemini) |
| `webbrowser`       | Abertura de pesquisas de imagem no navegador |
| `Hugging Face`     | (Opcional) Geração de imagens via Diffusion Pipeline - **Foi retirado do menu do projeto** |

## 🔧 Estrutura de Pastas

```bash
ecoauralia/
├── classificador.py               # Classificação com espectrogramas + Gemini + busca de imagem
├── classificador_mfcc.py          # Classificação com MFCCs
├── menu_principal.py              # Menu em terminal (interface de uso)
├── modelo_mobilenet_balanceado.h5 # Modelo treinado com espectrogramas
├── modelo_mfcc.h5                 # Modelo treinado com MFCCs
├── train_audio/                   # Áudios organizados por classe
└── spectrogramas/                 # Imagens geradas para treino/teste
```

## ⚙️ Funcionalidades principais

- 🎧 **Classificação de áudio com espectrograma**  
  Transforma áudio em imagem e classifica com modelo CNN (`MobileNetV2`)

- 📈 **Classificação com MFCC**  
  Extrai coeficientes de frequência e usa rede convolucional para classificar

- 📖 **Descrição com Gemini**  
  Usa LLM da Google para gerar informações detalhadas da espécie detectada

- 🌐 **Busca de imagens reais**  
  Redireciona o navegador para pesquisa de imagens no Google

## ▶️ Como executar

1. **Instale as dependências:**

```bash
pip install tensorflow librosa matplotlib numpy google-generativeai
```

2. **Execute o menu principal:**

```bash
python menu_principal.py
```

3. **Escolha uma das opções interativas do menu para usar as funcionalidades.**

## 🧪 Exemplo de uso

```bash
🌿 MENU - ECOAURALIA
1 - Classificar espectrograma (áudio .ogg)
2 - Classificar com MFCC (áudio .ogg)
3 - Gerar descrição com Gemini
4 - Buscar imagem real no navegador
5 - Executar tudo
6 - Sair
```

## 💡 Considerações

- A base de dados foi montada com sons organizados por classe animal
- O modelo foi treinado com data augmentation e class weights
- A performance pode variar conforme a qualidade dos áudios e o balanceamento entre as classes
- A geração de imagens com Diffusers foi descontinuada para manter o projeto leve e funcional localmente
- Bastante dificuldade para conseguir audios de animais pela internet.

