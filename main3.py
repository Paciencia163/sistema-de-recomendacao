import streamlit as st
import pandas as pd
import os
from PIL import Image
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import pickle
import tensorflow as tf

# Configuração do TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Função para extrair recursos
def extract_features(img_path, model):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = np.expand_dims(image.img_to_array(img), axis=0)
        preprocessed_img = preprocess_input(img_array)
        result = model.predict(preprocessed_img).flatten()
        return result / norm(result)
    except Exception as e:
        st.error(f"Erro ao extrair recursos: {e}")
        return None

# Função para recomendar produtos
def recommend(features, feature_list, n_neighbors=6):
    try:
        if len(feature_list) < n_neighbors:
            st.warning("Poucos dados disponíveis para recomendação.")
            n_neighbors = len(feature_list)
        neighbors = NearestNeighbors(n_neighbors=n_neighbors, algorithm='brute', metric='euclidean')
        neighbors.fit(feature_list)
        distances, indices = neighbors.kneighbors([features])
        return indices
    except Exception as e:
        st.error(f"Erro ao recomendar produtos: {e}")
        return None

# Carregar dados e modelo
@st.cache_data
def load_data(data_path):
    try:
        feature_list = np.array(pickle.load(open(os.path.join(data_path, 'embeddings.pkl'), 'rb')))
        filenames = pickle.load(open(os.path.join(data_path, 'filenames.pkl'), 'rb'))
        return feature_list, filenames
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return None, None

@st.cache_resource
def load_model():
    try:
        model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        model.trainable = False
        model = tf.keras.Sequential([model, GlobalMaxPooling2D()])
        return model
    except Exception as e:
        st.error(f"Erro ao carregar modelo: {e}")
        return None

@st.cache_data
def load_styles_df(data_path):
    try:
        styles_df = pd.read_csv(os.path.join(data_path, 'stylesv3.csv'))
        return styles_df
    except Exception as e:
        st.error(f"Erro ao carregar DataFrame de estilos: {e}")
        return None

# Função para exibir detalhes da imagem
def display_image_details(file_path, column, styles_df):
    try:
        image_id = os.path.basename(file_path).split('.')[0]
        if image_id.isdigit() and int(image_id) in styles_df['id'].values:
            image_details = styles_df[styles_df['id'] == int(image_id)].iloc[0]
            with column:
                st.image(file_path)
                st.caption(f"ID do Produto: {image_id}")
                st.write(f"**Nome:** {image_details['productDisplayName']}")
                st.write(f"**Categoria:** {image_details['subCategory']}")
            return {
                'ID do Produto': image_id,
                'Nome': image_details['productDisplayName'],
                'Categoria': image_details['subCategory']
            }
        else:
            with column:
                st.image(file_path)
                st.caption(f"ID do Produto: {image_id}")
                st.warning("Nenhuma informação encontrada para este produto.")
            return None
    except Exception as e:
        st.error(f"Erro ao exibir detalhes da imagem: {e}")
        return None

# Dados para sugestão de combinação de frutas
SUGESTOES = {
    "banana": [
        {"Alimento": "Aveia", "Sugestão": "Perfeito para um café da manhã saudável."},
        {"Alimento": "Mel", "Sugestão": "Adoça naturalmente a mistura."},
        {"Alimento": "Iogurte", "Sugestão": "Cria uma textura cremosa em smoothies."}
    ],
    "maçã": [
        {"Alimento": "Canela", "Sugestão": "Realça o sabor em sobremesas."},
        {"Alimento": "Pasta de Amendoim", "Sugestão": "Ótimo para um lanche energético."},
        {"Alimento": "Granola", "Sugestão": "Adiciona crocância e fibras."}
    ],
    "laranja": [
        {"Alimento": "Gengibre", "Sugestão": "Combinação refrescante em sucos."},
        {"Alimento": "Hortelã", "Sugestão": "Aumenta a frescura da bebida."},
        {"Alimento": "Cenoura", "Sugestão": "Boa base para um suco detox."}
    ]
}

def identificar_fruta(imagem):
    """Simula a identificação de uma fruta com base no nome do arquivo."""
    nome_arquivo = imagem.name.lower()
    for fruta in SUGESTOES.keys():
        if fruta in nome_arquivo:
            return fruta
    return None

# Carregar o DataFrame e o modelo
data_path = 'data'
upload_path = 'uploads'
feature_list, filenames = load_data(data_path)
model = load_model()
styles_df = load_styles_df(data_path)

# Interface principal do Streamlit com barra de navegação
st.title('Sistema de Recomendação de Produtos - Super Mercado Marivel')
st.image("marivel.jpg")

# Barra de navegação para selecionar a seção
page = st.sidebar.selectbox("Selecione uma página", ["Recomendação Geral", "Recomendação de Frutas"])

# Seção Recomendação Geral
if page == "Recomendação Geral":
    st.header("Recomendação Geral")
    st.write("Carregue ou capture uma imagem para receber recomendações de produtos similares.")

    # Selecionar método de upload
    upload_mode = st.radio("Escolha o método para carregar a imagem:", ("Upload", "Câmera"))
    uploaded_file = None
    
    if upload_mode == "Upload":
        uploaded_file = st.file_uploader("Escolha uma imagem")
    else:
        uploaded_file = st.camera_input("Tire uma foto")

    if uploaded_file is not None:
        # Salvar imagem capturada para processamento
        img_path = os.path.join(upload_path, "captured_image.jpg")
        with open(img_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        # Exibir a imagem capturada
        display_image = Image.open(uploaded_file)
        st.image(display_image, caption="Imagem capturada")

        # Extração de recursos
        with st.spinner('Extraindo recursos da imagem...'):
            features = extract_features(img_path, model)

        if features is not None:
            # Recomendação
            with st.spinner('Recomendando produtos similares...'):
                indices = recommend(features, feature_list)

            if indices is not None:
                # Exibir detalhes das imagens recomendadas
                product_info_list = []
                cols = st.columns(5)
                for i, col in enumerate(cols):
                    file_path = filenames[indices[0][i]].replace("\\", "/")
                    info = display_image_details(file_path, col, styles_df)
                    if info:
                        product_info_list.append(info)

                # Exibir tabela com informações dos produtos recomendados
                if product_info_list:
                    st.write("## Produtos Recomendados")
                    all_products_table = pd.DataFrame(product_info_list)
                    st.table(all_products_table)

# Seção Recomendação de Frutas
elif page == "Recomendação de Frutas":
    st.header("Sugestões de Combinações de Alimentos")
    imagem = st.file_uploader("Carregue uma imagem de uma fruta ou alimento", type=["jpg", "jpeg", "png"])

    if imagem:
        img = Image.open(imagem)
        st.image(img, caption="Imagem carregada", use_column_width=True)

        fruta = identificar_fruta(imagem)
        if fruta:
            st.success(f"Fruta identificada: {fruta.capitalize()}")
            st.write("Sugestões para combinar com a fruta:")
            st.table(SUGESTOES[fruta])
        else:
            st.error("Não foi possível identificar a fruta. Tente renomear a imagem com o nome da fruta.")
