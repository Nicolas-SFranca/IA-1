from diffusers import DiffusionPipeline
from huggingface_hub import login
import torch

# === token de identificação ===
HF_TOKEN = "Não vou colocar o token aqui, mas você deve colocar o seu token aqui."
# Você pode gerar um token em https://huggingface.co/settings/tokens

login(token=HF_TOKEN)

# Carregar modelo de geração de imagem
pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32)
pipe = pipe.to("cpu")

def gerar_imagem(prompt, caminho_saida="imagem_gerada.png"):
    print(f"\n🎨 Gerando imagem para o prompt: '{prompt}'...")
    imagem = pipe(prompt).images[0]
    imagem.save(caminho_saida)
    print(f"✅ Imagem salva como: {caminho_saida}")

if __name__ == "__main__":
    prompt_usuario = input("Digite o prompt para gerar a imagem: ")
    gerar_imagem(prompt_usuario)
