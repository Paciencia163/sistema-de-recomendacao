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
import random

# Configuração
tf.get_logger().setLevel('ERROR')
IMAGE_DIR = 'images'

# Carregar dados
@st.cache_data
def load_data():
    feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
    filenames = pickle.load(open('filenames.pkl', 'rb'))
    filenames = [os.path.join(IMAGE_DIR, os.path.basename(f)) for f in filenames]
    filenames = [f for f in filenames if os.path.exists(f)]  # Ignorar arquivos ausentes
    return feature_list, filenames

@st.cache_resource
def load_model():
    model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model.trainable = False
    return tf.keras.Sequential([model, GlobalMaxPooling2D()])

@st.cache_data
def load_labels():
    df = pd.read_csv("styles.csv")
    return df

# Função para extrair features
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = np.expand_dims(image.img_to_array(img), axis=0)
    preprocessed_img = preprocess_input(img_array)
    result = model.predict(preprocessed_img).flatten()
    return result / norm(result)

# Função de recomendação
def recommend(features, feature_list, filenames, n_neighbors=6):
    if features is None:
        return []
    neighbors = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return [filenames[i] for i in indices[0] if os.path.exists(filenames[i])]

# Carregar modelo e dados
feature_list, filenames = load_data()
model = load_model()
labels_df = load_labels()

# Barra de navegação
st.sidebar.title("Menu")
page = st.sidebar.radio("Navegação", ["Recomendação de Produtos", "Sugestão de Combinação de Frutas"])

if page == "Recomendação de Produtos":
    st.title("Recomendação de Produtos")
    st.write("Clique em uma imagem para obter recomendações de produtos similares.")
    
    # Barra de pesquisa
    search_query = st.text_input("Buscar produto pelo nome")
    
    # Filtrar produtos pelo nome
    if search_query:
        filtered_filenames = []
        for file in filenames:
            produto_id = int(os.path.basename(file).split('.')[0])
            produto_info = labels_df[labels_df['id'] == produto_id]
            if not produto_info.empty and search_query.lower() in produto_info.iloc[0]['productDisplayName'].lower():
                filtered_filenames.append(file)
    else:
        filtered_filenames = filenames

    # Exibir imagens na interface
    cols = st.columns(5)
    selected_image = None

    for i, file in enumerate(filtered_filenames):
        produto_id = int(os.path.basename(file).split('.')[0])
        produto_info = labels_df[labels_df['id'] == produto_id]

        if not produto_info.empty:
            with cols[i % 5]:
                img = Image.open(file)
                if st.button("Selecionar Produto", key=file):
                    selected_image = file
                st.image(img, use_container_width=True)

    # Gerar recomendação ao selecionar uma imagem
    if selected_image:
        produto_id = int(os.path.basename(selected_image).split('.')[0])
        produto_info = labels_df[labels_df['id'] == produto_id]
        preco_ficticio = round(random.uniform(1000, 10000), 2)

        if not produto_info.empty:
            st.write("### Produto Selecionado")
            st.image(selected_image, use_container_width=True)
            st.write(f"**Nome:** {produto_info.iloc[0]['productDisplayName']}")
            st.write(f"**Categoria:** {produto_info.iloc[0]['subCategory']}")
            st.write(f"**Preço:** {preco_ficticio} Kzs")

            # Extração de features e recomendação
            features = extract_features(selected_image, model)
            recommended_files = recommend(features, feature_list, filenames)

            st.write("### Produtos Recomendados")
            rec_cols = st.columns(5)
            for i, file in enumerate(recommended_files):
                produto_id = int(os.path.basename(file).split('.')[0])
                produto_info = labels_df[labels_df['id'] == produto_id]
                preco_ficticio = round(random.uniform(1000, 10000), 2)

                if not produto_info.empty:
                    with rec_cols[i % 5]:
                        st.image(Image.open(file), use_container_width=True)
                        st.write(f"**Nome:** {produto_info.iloc[0]['productDisplayName']}")
                        st.write(f"**Preço:** {preco_ficticio} Kzs")

elif page == "Sugestão de Combinação de Frutas":
    st.title("Sugestão de Combinação de Frutas")
    SUGESTOES = {
        "banana": ["Aveia", "Mel", "Iogurte"],
        "maçã": ["Canela", "Pasta de Amendoim", "Granola"],
        "laranja": ["Gengibre", "Hortelã", "Cenoura"]
    }

    imagem_fruta = st.file_uploader("Carregue uma imagem de fruta", type=["jpg", "jpeg", "png"])

    if imagem_fruta:
        img = Image.open(imagem_fruta)
        st.image(img, use_container_width=True)
        fruta_identificada = os.path.basename(imagem_fruta.name).split('.')[0].lower()
        if fruta_identificada in SUGESTOES:
            st.success(f"Fruta identificada: {fruta_identificada.capitalize()}")
            st.write("Sugestões para combinar:")
            st.write(", ".join(SUGESTOES[fruta_identificada]))
        else:
            st.error("Não foi possível identificar a fruta.")
