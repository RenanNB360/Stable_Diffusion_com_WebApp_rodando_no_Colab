# Imports
import diffusers
import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
from io import BytesIO

st.title('Geração de Imagens')

@st.cache_resource
def carrega_modelo():

    pipe = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', torch_dtype = torch.float16)

    if torch.cuda.is_available():
        pipe = pipe.to('cuda')
    
    return pipe

pipe = carrega_modelo()

prompt = st.text_input('Digite seu prompt para gerar a imagem:', value='Digite o prompt')
steps = st.slider('Escolha o número de passos de inferência', min_value=20, max_value=150, value=50)
seed = st.number_input('Escolha a seed (para reprodutibilidade)', value=1)

if st.button('Gerar Imagem com IA'):

    with st.spinner('Gerando Imagem...'):
        try:

            gerador = torch.Generator(device='cuda' if torch.cuda.is_available() else 'cpu').manual_seed(int(seed))
            image = pipe(prompt, num_inference_steps = steps, generator = gerador).images[0]
            st.image(image, caption='Imagem Gerada')

            img_buffer = BytesIO()
            image.save(img_buffer, format = 'PNG')
            img_buffer.seek(0)

            st.download_button(
                label='Baixar Imagem',
                data= img_buffer,
                file_name='imagem_gerada.png',
                mime = 'image/png'
            )

        except Exception as e:
            st.error(f'Erro ao gerar a imagem: {e}')
            