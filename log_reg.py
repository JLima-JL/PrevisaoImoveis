# treinar_modelo_regressao.py

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Verificação do arquivo
arquivo_excel = 'imoveisatualizado1.xlsx'
if not os.path.isfile(arquivo_excel):
    raise FileNotFoundError(f"Arquivo '{arquivo_excel}' não encontrado.")

# Carregar os dados
dados = pd.read_excel(arquivo_excel)

# Features e target
X = dados[['area_total', 'area_construida' , 'numero_de_quartos', 'ano_de_construcao', 'numero_de_banheiros',
           'bairro_codificado']]
y = dados['preco']

# Escalonamento
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Regressão Linear
modelo = LinearRegression()
modelo.fit(X_scaled, y)

# Salvar modelo e scaler
joblib.dump(modelo, 'modelo_regressao.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Modelo de regressão e scaler salvos com sucesso.")
