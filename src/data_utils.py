\
# src/data_utils.py
import requests
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import numpy as np

@st.cache_data(ttl=3600)
def get_brent_oil_data():
    """
    Busca os dados da série histórica do preço do petróleo Brent da API do IPEAData.

    Realiza o pré-processamento dos dados, incluindo:
    - Conversão da coluna de data para o tipo datetime.
    - Remoção de valores ausentes.
    - Ordenação dos dados por data.
    - Cálculo da média móvel de 7 dias.
    - Normalização das features 'price' e 'moving_average_7d' usando MinMaxScaler.

    Returns:
        tuple: Uma tupla contendo:
            - df (pd.DataFrame): DataFrame original com 'date', 'price', 'moving_average_7d'.
            - df_model_input (pd.DataFrame): DataFrame com as features normalizadas
                                             ('normalized_price', 'normalized_moving_average_7d')
                                             indexado por 'date', pronto para ser usado como entrada em modelos.
            - price_scaler (MinMaxScaler): O scaler ajustado para a coluna 'price'.
            - ma_scaler (MinMaxScaler): O scaler ajustado para a coluna 'moving_average_7d'.
        Retorna (None, None, None, None) em caso de erro na API ou processamento.
    """
    series_code = 'EIA366_PBRENT366'
    url = f"http://www.ipeadata.gov.br/api/odata4/ValoresSerie(SERCODIGO='{series_code}')"
    response = requests.get(url)
    if response.status_code == 200:
        raw_data = response.json()['value']
        df = pd.DataFrame(raw_data)
        df = df[['VALDATA', 'VALVALOR']]
        df.columns = ['date', 'price']
        try:
            df['date'] = pd.to_datetime(df['date'], utc=True)
            df['date'] = df['date'].dt.tz_localize(None)
        except Exception as e:
            st.warning(f"Aviso: Não foi possível converter datas para datetime: {e}")
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            if pd.api.types.is_datetime64_any_dtype(df['date']) and df['date'].dt.tz is not None:
                 df['date'] = df['date'].dt.tz_localize(None)
        df.dropna(subset=['date', 'price'], inplace=True)
        df = df.sort_values('date').reset_index(drop=True)
        df['moving_average_7d'] = df['price'].rolling(window=7).mean()
        df.dropna(subset=['moving_average_7d'], inplace=True) 
        df = df.reset_index(drop=True)
        price_scaler = MinMaxScaler()
        ma_scaler = MinMaxScaler()
        df_normalized_features = df.copy()
        df_normalized_features_indexed = df_normalized_features.set_index('date')
        df_normalized_features_indexed['normalized_price'] = price_scaler.fit_transform(df_normalized_features_indexed[['price']])
        df_normalized_features_indexed['normalized_moving_average_7d'] = ma_scaler.fit_transform(df_normalized_features_indexed[['moving_average_7d']])
        df_model_input = df_normalized_features_indexed[['normalized_price', 'normalized_moving_average_7d']].copy()
        return df, df_model_input, price_scaler, ma_scaler
    else:
        st.error(f"Erro ao acessar API do IPEA: {response.status_code}")
        return None, None, None, None

def create_sequences_multivariate(features_data, target_data, window_size):
    """
    Cria sequências de dados para modelos de séries temporais multivariadas.

    Args:
        features_data (np.array): Array NumPy contendo os dados das features (ex: preço normalizado, média móvel normalizada).
                                  A forma esperada é (n_observacoes, n_features).
        target_data (np.array): Array NumPy contendo os dados do alvo (ex: preço normalizado).
                                A forma esperada é (n_observacoes,).
        window_size (int): O tamanho da janela (número de passos temporais anteriores) para criar cada sequência.

    Returns:
        tuple: Uma tupla contendo:
            - X (np.array): Array NumPy de sequências de entrada. Forma: (n_sequencias, window_size, n_features).
            - y (np.array): Array NumPy de valores alvo correspondentes. Forma: (n_sequencias,).
    """
    X, y = [], []
    for i in range(len(features_data) - window_size):
        X.append(features_data[i:i + window_size, :])
        y.append(target_data[i + window_size])
    return np.array(X), np.array(y)
