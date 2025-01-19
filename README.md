# Sistema de Recomendação de Produtos - Super Mercado Marivel

Este projeto é uma aplicação de recomendação de produtos desenvolvida em Python utilizando a biblioteca **Streamlit**. Ele permite aos usuários carregar ou capturar imagens para obter recomendações de produtos similares com base em aprendizado profundo (Deep Learning).

## Funcionalidades

- **Recomendação Geral**: Carregue ou capture uma imagem de um produto para receber sugestões de produtos similares.
- **Recomendação de Frutas**: Identifique frutas em imagens carregadas e receba sugestões de alimentos que combinam com elas.

## Tecnologias Utilizadas

- **Python**: Linguagem principal utilizada para desenvolver a aplicação.
- **Streamlit**: Framework usado para criar a interface interativa.
- **TensorFlow/Keras**: Biblioteca usada para o modelo de aprendizado profundo (VGG16) e extração de recursos.
- **Scikit-learn**: Utilizado para cálculos de vizinhos mais próximos (Nearest Neighbors).
- **Pandas**: Para manipulação de dados tabulares.
- **Pillow (PIL)**: Para manipulação de imagens.

## Pré-requisitos

Certifique-se de que as seguintes dependências estão instaladas:

- Python 3.8 ou superior
- Streamlit
- TensorFlow
- Scikit-learn
- Pandas
- Pillow
- NumPy

Instale as dependências com o comando:

```bash
pip install -r requirements.txt
```

## Como Executar

1. Clone este repositório:
   ```bash
   git clone https://github.com/Paciencia163/sistema-de-recomendacao.git
   ```
2. Navegue para o diretório do projeto:
   ```bash
   cd sistema-recomendacao-produtos
   ```
3. Certifique-se de que os arquivos necessários estão presentes na pasta `data`:
   - `embeddings.pkl`: Recursos dos produtos pré-extraídos.
   - `filenames.pkl`: Caminhos das imagens dos produtos.
   - `stylesv3.csv`: Detalhes sobre os produtos.
4. Execute a aplicação Streamlit:
   ```bash
   streamlit run app.py
   ```
5. Acesse a aplicação no navegador, geralmente em [http://localhost:8501](http://localhost:8501).

## Estrutura do Projeto

```
.
├── app.py                # Código principal da aplicação
├── data/                 # Dados necessários para recomendações
│   ├── embeddings.pkl    # Recursos dos produtos
│   ├── filenames.pkl     # Caminhos das imagens
│   └── stylesv3.csv      # Informações sobre os produtos
├── uploads/              # Imagens carregadas pelos usuários
├── requirements.txt      # Dependências do projeto
└── README.md             # Documentação do projeto
```

## Funcionalidades Principais

### Recomendação Geral

1. O usuário carrega ou captura uma imagem de um produto.
2. A imagem é processada por um modelo pré-treinado (VGG16) para extração de recursos visuais.
3. O sistema utiliza um algoritmo de **Nearest Neighbors** para encontrar produtos similares no banco de dados.
4. Os detalhes dos produtos recomendados são exibidos junto com as imagens.

### Recomendação de Frutas

1. O sistema tenta identificar a fruta com base o arquivo da imagem carregada.
2. Caso a fruta seja reconhecida, sugestões de alimentos que combinam com ela são exibidas.

## Personalização

Você pode personalizar o conjunto de dados, adicionar novas categorias ou ajustar o número de vizinhos para recomendações no código-fonte.

## Licença

Este projeto está licenciado sob a [MIT License](LICENSE).

---

Feito com ❤️ por Paciência Aníbal Muienga.
