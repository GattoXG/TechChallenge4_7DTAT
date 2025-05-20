\
# src/model_utils.py
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from .data_utils import create_sequences_multivariate

def evaluate_model(model, X_test, y_test, price_scaler):
    """
    Avalia um modelo de regressão calculando diversas métricas.

    Métricas calculadas:
    - MAE (Mean Absolute Error) - Erro Absoluto Médio
    - MSE (Mean Squared Error) - Erro Quadrático Médio
    - RMSE (Root Mean Squared Error) - Raiz do Erro Quadrático Médio
    - R² (R-squared) - Coeficiente de Determinação
    - RMSE (USD) - Raiz do Erro Quadrático Médio nos valores originais (desnormalizados) do preço.

    Args:
        model: O modelo de machine learning treinado (ex: XGBoost, RandomForest).
        X_test (np.array or pd.DataFrame): Dados de teste para as features.
        y_test (np.array or pd.Series): Dados de teste para o alvo (normalizado).
        price_scaler (sklearn.preprocessing.MinMaxScaler): Scaler usado para normalizar
                                                           a variável alvo ('price').
                                                           Usado para calcular o RMSE em USD.

    Returns:
        dict: Um dicionário contendo as métricas de avaliação.
    """
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    if price_scaler is not None:
        y_test_reshaped = y_test.reshape(-1, 1)
        y_pred_reshaped = y_pred.reshape(-1, 1)
        y_test_original = price_scaler.inverse_transform(y_test_reshaped).flatten()
        y_pred_original = price_scaler.inverse_transform(y_pred_reshaped).flatten()
        mse_original = mean_squared_error(y_test_original, y_pred_original)
        rmse_original = np.sqrt(mse_original)
        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R²': r2,
            'RMSE (USD)': rmse_original
        }
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R²': r2
    }

# --- XGBoost ---
def load_xgb_model(window, key_prefix="xgb"):
    """
    Carrega um modelo XGBoost treinado, juntamente com seus scalers de preço e média móvel.

    Args:
        window (int): A janela (em dias) para a qual o modelo foi treinado.
        key_prefix (str): Prefixo usado na nomeação dos arquivos de modelo e scalers.

    Returns:
        tuple: Contendo (model, price_scaler, ma_scaler) se os arquivos existirem.
               Caso contrário, retorna (None, None, None).
    """
    model_path = f"modelos/{key_prefix}_model_{window}d.pkl"
    price_scaler_path = f"modelos/{key_prefix}_price_scaler_{window}d.pkl"
    ma_scaler_path = f"modelos/{key_prefix}_ma_scaler_{window}d.pkl"
    if os.path.exists(model_path) and os.path.exists(price_scaler_path) and os.path.exists(ma_scaler_path):
        model = joblib.load(model_path)
        price_scaler = joblib.load(price_scaler_path)
        ma_scaler = joblib.load(ma_scaler_path)
        return model, price_scaler, ma_scaler
    return None, None, None

def train_xgb_model(window, df_features, price_scaler_base, ma_scaler_base, key_prefix="xgb"):
    """
    Treina um modelo XGBoost Regressor para prever o preço do petróleo.

    Os dados de entrada são transformados em sequências, e o modelo é treinado
    com essas sequências. O modelo treinado e os scalers são salvos em disco.

    Args:
        window (int): O tamanho da janela (número de passos temporais anteriores)
                      para criar as sequências de entrada.
        df_features (pd.DataFrame): DataFrame contendo as features normalizadas
                                    ('normalized_price', 'normalized_moving_average_7d').
        price_scaler_base (sklearn.preprocessing.MinMaxScaler): Scaler ajustado para 'price'.
        ma_scaler_base (sklearn.preprocessing.MinMaxScaler): Scaler ajustado para 'moving_average_7d'.
        key_prefix (str): Prefixo usado na nomeação dos arquivos de modelo e scalers.

    Returns:
        tuple: Contendo (model, price_scaler, ma_scaler) se o treinamento for bem-sucedido.
               Caso contrário, ou se não houver dados suficientes, retorna (None, None, None).
    """
    sequence_input_data = df_features[['normalized_price', 'normalized_moving_average_7d']].values
    sequence_target_data = df_features['normalized_price'].values
    X_seq, y_target = create_sequences_multivariate(sequence_input_data, sequence_target_data, window)

    if X_seq.shape[0] == 0:
        st.error(f"Não há dados suficientes para treinar o modelo XGBoost com janela de {window} dias.")
        return None, None, None
    
    X_flat = X_seq.reshape(X_seq.shape[0], -1)
    X_train, X_test, y_train, y_test = train_test_split(X_flat, y_target, test_size=0.2, shuffle=False, random_state=42)

    if len(X_train) == 0 or len(y_train) == 0:
        st.error(f"Não há dados suficientes para treinar o modelo XGBoost com janela de {window} dias após o split.")
        return None, None, None

    model = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    model.fit(X_train, y_train)
    
    os.makedirs("modelos", exist_ok=True)
    joblib.dump(model, f"modelos/{key_prefix}_model_{window}d.pkl")
    joblib.dump(price_scaler_base, f"modelos/{key_prefix}_price_scaler_{window}d.pkl")
    joblib.dump(ma_scaler_base, f"modelos/{key_prefix}_ma_scaler_{window}d.pkl")
    return model, price_scaler_base, ma_scaler_base

# --- Random Forest ---
def load_rf_model(window, key_prefix="rf"):
    """
    Carrega um modelo Random Forest treinado, juntamente com seus scalers de preço e média móvel.

    Args:
        window (int): A janela (em dias) para a qual o modelo foi treinado.
        key_prefix (str): Prefixo usado na nomeação dos arquivos de modelo e scalers.

    Returns:
        tuple: Contendo (model, price_scaler, ma_scaler) se os arquivos existirem.
               Caso contrário, retorna (None, None, None).
    """
    model_path = f"modelos/{key_prefix}_model_{window}d.pkl"
    price_scaler_path = f"modelos/{key_prefix}_price_scaler_{window}d.pkl"
    ma_scaler_path = f"modelos/{key_prefix}_ma_scaler_{window}d.pkl"
    if os.path.exists(model_path) and os.path.exists(price_scaler_path) and os.path.exists(ma_scaler_path):
        model = joblib.load(model_path)
        price_scaler = joblib.load(price_scaler_path)
        ma_scaler = joblib.load(ma_scaler_path)
        return model, price_scaler, ma_scaler
    return None, None, None

def train_rf_model(window, df_features, price_scaler_base, ma_scaler_base, key_prefix="rf"):
    """
    Treina um modelo RandomForestRegressor para prever o preço do petróleo.

    Os dados de entrada são transformados em sequências, e o modelo é treinado
    com essas sequências. O modelo treinado e os scalers são salvos em disco.

    Args:
        window (int): O tamanho da janela (número de passos temporais anteriores)
                      para criar as sequências de entrada.
        df_features (pd.DataFrame): DataFrame contendo as features normalizadas
                                    ('normalized_price', 'normalized_moving_average_7d').
        price_scaler_base (sklearn.preprocessing.MinMaxScaler): Scaler ajustado para 'price'.
        ma_scaler_base (sklearn.preprocessing.MinMaxScaler): Scaler ajustado para 'moving_average_7d'.
        key_prefix (str): Prefixo usado na nomeação dos arquivos de modelo e scalers.

    Returns:
        tuple: Contendo (model, price_scaler, ma_scaler) se o treinamento for bem-sucedido.
               Caso contrário, ou se não houver dados suficientes, retorna (None, None, None).
    """
    sequence_input_data = df_features[['normalized_price', 'normalized_moving_average_7d']].values
    sequence_target_data = df_features['normalized_price'].values
    X_seq, y_target = create_sequences_multivariate(sequence_input_data, sequence_target_data, window)

    if X_seq.shape[0] == 0:
        st.error(f"Não há dados suficientes para treinar o modelo Random Forest com janela de {window} dias.")
        return None, None, None

    X_flat = X_seq.reshape(X_seq.shape[0], -1)
    X_train, X_test, y_train, y_test = train_test_split(X_flat, y_target, test_size=0.2, shuffle=False, random_state=42)

    if len(X_train) == 0 or len(y_train) == 0:
        st.error(f"Não há dados suficientes para treinar o modelo Random Forest com janela de {window} dias após o split.")
        return None, None, None
        
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    os.makedirs("modelos", exist_ok=True)
    joblib.dump(model, f"modelos/{key_prefix}_model_{window}d.pkl")
    joblib.dump(price_scaler_base, f"modelos/{key_prefix}_price_scaler_{window}d.pkl")
    joblib.dump(ma_scaler_base, f"modelos/{key_prefix}_ma_scaler_{window}d.pkl")
    return model, price_scaler_base, ma_scaler_base

# --- LightGBM ---
def load_lgbm_model(window, key_prefix="lgbm"):
    """
    Carrega um modelo LightGBM treinado, seus scalers e a lista de nomes de features.

    Args:
        window (int): A janela (em dias) para a qual o modelo foi treinado.
        key_prefix (str): Prefixo usado na nomeação dos arquivos de modelo e scalers.

    Returns:
        tuple: Contendo (model, price_scaler, ma_scaler, feature_names) se os arquivos existirem.
               Caso contrário, retorna (None, None, None, None).
    """
    model_path = f"modelos/{key_prefix}_model_{window}d.pkl"
    price_scaler_path = f"modelos/{key_prefix}_price_scaler_{window}d.pkl"
    ma_scaler_path = f"modelos/{key_prefix}_ma_scaler_{window}d.pkl"
    features_path = f"modelos/{key_prefix}_features_{window}d.pkl"
    if os.path.exists(model_path) and os.path.exists(price_scaler_path) and os.path.exists(ma_scaler_path) and os.path.exists(features_path):
        model = joblib.load(model_path)
        price_scaler = joblib.load(price_scaler_path)
        ma_scaler = joblib.load(ma_scaler_path)
        features = joblib.load(features_path)
        return model, price_scaler, ma_scaler, features
    return None, None, None, None

def train_lgbm_model(window, df_features, price_scaler_base, ma_scaler_base, key_prefix="lgbm"):
    """
    Treina um modelo LGBMRegressor para prever o preço do petróleo.

    Os dados de entrada são transformados em sequências com nomes de features explícitos,
    e o modelo é treinado. O modelo, scalers e nomes de features são salvos.

    Args:
        window (int): Tamanho da janela para criar sequências.
        df_features (pd.DataFrame): DataFrame com 'normalized_price' e 'normalized_moving_average_7d'.
        price_scaler_base (MinMaxScaler): Scaler para 'price'.
        ma_scaler_base (MinMaxScaler): Scaler para 'moving_average_7d'.
        key_prefix (str): Prefixo usado na nomeação dos arquivos de modelo e scalers.

    Returns:
        tuple: (model, price_scaler, ma_scaler, feature_names) ou (None, None, None, None) em caso de falha.
    """
    sequence_input_data = df_features[['normalized_price', 'normalized_moving_average_7d']].values
    sequence_target_data = df_features['normalized_price'].values
    X_seq, y_target = create_sequences_multivariate(sequence_input_data, sequence_target_data, window)

    if X_seq.shape[0] == 0:
        st.error(f"Não há dados suficientes para treinar o modelo LightGBM com janela de {window} dias.")
        return None, None, None, None

    feature_names = []
    for i in range(window):
        feature_names.append(f'price_lag_{window-i}')
        feature_names.append(f'ma_lag_{window-i}')
    
    X_flat = X_seq.reshape(X_seq.shape[0], -1)
    X_df = pd.DataFrame(X_flat, columns=feature_names)
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_target, test_size=0.2, shuffle=False, random_state=42)

    if len(X_train) == 0 or len(y_train) == 0:
        st.error(f"Não há dados suficientes para treinar o modelo LightGBM com janela de {window} dias após o split.")
        return None, None, None, None

    model = LGBMRegressor(n_estimators=100, random_state=42, verbosity=-1)
    model.fit(X_train, y_train)

    os.makedirs("modelos", exist_ok=True)
    joblib.dump(model, f"modelos/{key_prefix}_model_{window}d.pkl")
    joblib.dump(price_scaler_base, f"modelos/{key_prefix}_price_scaler_{window}d.pkl")
    joblib.dump(ma_scaler_base, f"modelos/{key_prefix}_ma_scaler_{window}d.pkl")
    joblib.dump(feature_names, f"modelos/{key_prefix}_features_{window}d.pkl")
    return model, price_scaler_base, ma_scaler_base, feature_names
