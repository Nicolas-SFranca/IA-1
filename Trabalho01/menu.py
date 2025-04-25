import os
from pathlib import Path
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from classificador import classificar_espectrograma, gerar_descricao, buscar_imagem_real
from classificador_mfcc import classificar_audio_mfcc

# Gera espectrograma automaticamente
def gerar_espectrograma(caminho_audio):
    nome_audio = Path(caminho_audio).stem
    pasta_saida = Path("spectrogramas/teste")
    pasta_saida.mkdir(parents=True, exist_ok=True)
    caminho_saida = pasta_saida / f"{nome_audio}.png"

    try:
        y, sr = librosa.load(caminho_audio, sr=32000)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_DB = librosa.power_to_db(S, ref=np.max)

        plt.figure(figsize=(3, 3))
        librosa.display.specshow(S_DB, sr=sr, x_axis=None, y_axis=None)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(caminho_saida, bbox_inches='tight', pad_inches=0)
        plt.close()

        return str(caminho_saida)
    except Exception as e:
        print(f"Oops, não consegui criar o espectrograma: {e}")
        return None

# Menu principal

def menu():
    while True:
        print("\n🌿 ECOAURALIA 🌿")
        print("1 - Classificar áudio por Espectrograma")
        print("2 - Classificar áudio por MFCC")
        print("3 - Obter descrição da espécie")
        print("4 - Buscar imagem real da espécie")
        print("5 - Executar fluxo completo (áudio -> espectrograma -> texto -> imagem)")
        print("6 - Sair")

        escolha = input("👉 Escolha uma opção: ")

        if escolha in ["1", "2"]:
            pasta_testes = Path(r"C:\Users\Administrador\Documents\meu_projeto_ecoauralia\train_audio\TesteAudios\convertidos")
            audios = list(pasta_testes.glob("*.ogg"))

            if not audios:
                print("⚠️  Nenhum áudio encontrado na pasta de testes.")
                continue

            print("\nÁudios disponíveis:")
            for idx, audio in enumerate(audios):
                print(f"{idx + 1} - {audio.name}")

            idx_audio = input("Número do áudio: ")
            if idx_audio.isdigit() and 1 <= int(idx_audio) <= len(audios):
                caminho_audio = str(audios[int(idx_audio) - 1])
                if escolha == "1":
                    caminho_img = gerar_espectrograma(caminho_audio)
                    if caminho_img:
                        classificar_espectrograma(caminho_img)
                else:
                    classificar_audio_mfcc(caminho_audio)
            else:
                print("⚠️ Opção inválida!")

        elif escolha == "3":
            especie = input("Digite a espécie desejada: ")
            gerar_descricao(especie)

        elif escolha == "4":
            especie = input("Digite a espécie desejada: ")
            buscar_imagem_real(especie)

        elif escolha == "5":
            pasta_testes = Path(r"C:\Users\Administrador\Documents\meu_projeto_ecoauralia\train_audio\TesteAudios\convertidos")
            audios = list(pasta_testes.glob("*.ogg"))

            if not audios:
                print("⚠️  Nenhum áudio encontrado na pasta de testes.")
                continue

            print("\nÁudios disponíveis:")
            for idx, audio in enumerate(audios):
                print(f"{idx + 1} - {audio.name}")

            idx_audio = input("Número do áudio para o fluxo completo: ")
            if idx_audio.isdigit() and 1 <= int(idx_audio) <= len(audios):
                caminho_audio = str(audios[int(idx_audio) - 1])
                caminho_img = gerar_espectrograma(caminho_audio)
                if caminho_img:
                    especie = classificar_espectrograma(caminho_img)
                    if especie:
                        gerar_descricao(especie)
                        buscar_imagem_real(especie)
            else:
                print("⚠️ Opção inválida!")

        elif escolha == "6":
            print("🌱 Obrigado por usar o EcoAuralia! Até mais!")
            break

        else:
            print("⚠️ Opção inválida! Tente novamente.")

if __name__ == "__main__":
    menu()
