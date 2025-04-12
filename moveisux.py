# streamlit_app_regressao.py

import streamlit as st
import numpy as np
import joblib
import os

# Verifica existência dos arquivos
if not os.path.isfile('modelo_regressao1.pkl') or not os.path.isfile('scaler1.pkl'):
    st.error("Arquivos do modelo não encontrados. Execute 'treinar_modelo_regressao.py' primeiro.")
    st.stop()

# Carregar modelo e scaler
modelo = joblib.load('modelo_regressao1.pkl')
scaler = joblib.load('scaler1.pkl')

# Interface
st.title("Previsão de Preço de Imóvel")
st.markdown("Informe os dados do imóvel para estimar o **preço em reais (R$)** com base em um modelo de regressão.")

# Entradas do usuário
area_total = st.number_input("Área total (m²)", min_value=10, max_value=1000, value=100)
area_construida = st.number_input("Ano de construção", min_value=5, max_value=800, value=50)
numero_de_quartos = st.number_input("Número de quartos", min_value=1, max_value=10, value=2)
ano_de_construcao = st.number_input("Ano de construção", min_value=1900, max_value=2025, value=2000)
numero_de_banheiros = st.number_input("Número de Banheiros", min_value=1, max_value=10, value=2)
bairro_codificado = st.number_input("Classificação do Bairro", min_value=1, max_value=10, value=3)

if st.button("Estimar preço"):
    entrada = np.array([[area_total, numero_de_quartos, ano_de_construcao]])
    entrada_escalada = scaler.transform(entrada)
    preco_estimado = modelo.predict(entrada_escalada)[0]
    st.success(f"Preço estimado: **R$ {preco_estimado:,.2f}**")
