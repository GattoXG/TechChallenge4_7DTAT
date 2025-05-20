import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
import threading
from src.data_utils import get_brent_oil_data, create_sequences_multivariate
from src.plot_utils import create_main_chart, create_context_chart
from src.model_utils import evaluate_model, load_xgb_model, train_xgb_model, load_rf_model, train_rf_model, load_lgbm_model, train_lgbm_model

st.set_page_config(layout="wide", page_title="Dashboard Petróleo Brent")

NUM_INPUT_FEATURES = 2

def setup_sidebar(oil_df):
    """
    Configura a barra lateral do Streamlit com filtros de data e estatísticas gerais.

    Args:
        oil_df (pd.DataFrame): DataFrame contendo os dados do preço do petróleo,
                                    deve incluir uma coluna 'date' para filtragem
                                    e uma coluna 'price' para estatísticas.

    Returns:
        tuple: Uma tupla contendo:
            - filtered_df (pd.DataFrame): DataFrame filtrado pelo intervalo de datas selecionado.
            - start_date (datetime.date): A data de início selecionada.
            - end_date (datetime.date): A data de fim selecionada.
    """
    st.sidebar.subheader("Filtro de Data")
    min_date = oil_df['date'].min().date()
    max_date = oil_df['date'].max().date()

    if 'start_date' not in st.session_state:
        st.session_state.start_date = min_date
    if 'end_date' not in st.session_state:
        st.session_state.end_date = max_date

    start_date_slider, end_date_slider = st.sidebar.slider(
        "Selecione o intervalo de datas",
        min_value=min_date,
        max_value=max_date,
        value=(st.session_state.start_date, st.session_state.end_date),
        format="DD/MM/YYYY"
    )
    st.session_state.start_date = start_date_slider
    st.session_state.end_date = end_date_slider

    date_col1, date_col2 = st.sidebar.columns(2)
    with date_col1:
        start_date_input = st.date_input("De:", st.session_state.start_date, min_value=min_date, max_value=max_date, key="date_input_start")
    with date_col2:
        end_date_input = st.date_input("Até:", st.session_state.end_date, min_value=min_date, max_value=max_date, key="date_input_end")

    if start_date_input != st.session_state.start_date:
        st.session_state.start_date = start_date_input
    if end_date_input != st.session_state.end_date:
        st.session_state.end_date = end_date_input

    filtered_df = oil_df[(
        oil_df['date'].dt.date >= st.session_state.start_date
    ) & (oil_df['date'].dt.date <= st.session_state.end_date)]

    st.sidebar.subheader("Estatísticas Gerais")
    st.sidebar.metric("Preço Médio", f"{filtered_df['price'].mean():.2f} USD")
    st.sidebar.metric("Preço Máximo", f"{filtered_df['price'].max():.2f} USD")
    st.sidebar.metric("Preço Mínimo", f"{filtered_df['price'].min():.2f} USD")
    if 'date' in filtered_df.columns and not filtered_df['date'].empty:
        st.sidebar.metric("Última Atualização", f"{filtered_df['date'].max().strftime('%d/%m/%Y')}")
    else:
        st.sidebar.metric("Última Atualização", "N/A")
    return filtered_df, st.session_state.start_date, st.session_state.end_date

def display_historical_analysis(oil_df, filtered_df, contexts, extra_days, start_date, end_date):
    """
    Exibe a seção de análise histórica do dashboard.

    Inclui um gráfico geral da visão geral dos preços e gráficos específicos para
    contextos históricos definidos (ex: crises, decisões da OPEP).

    Args:
        oil_df (pd.DataFrame): O DataFrame completo dos preços do petróleo.
        filtered_df (pd.DataFrame): DataFrame filtrado pelo intervalo de datas selecionado pelo usuário.
        contexts (list): Uma lista de dicionários, cada um definindo um contexto histórico
                          (name, start_date, end_date, description, color).
        extra_days (int): Número de dias extras para mostrar antes e depois de cada período de contexto.
        start_date (datetime.date): A data de início para o intervalo do eixo x do gráfico de visão geral.
        end_date (datetime.date): A data de fim para o intervalo do eixo x do gráfico de visão geral.
    """
    st.header("Visão Geral do Preço do Petróleo Brent")
    general_fig = create_main_chart(filtered_df, contexts)
    general_fig.update_layout(xaxis=dict(range=[
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    ]))
    st.plotly_chart(general_fig, use_container_width=True)

    st.header("Análise por Contexto Histórico")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(contexts[0]['name'])
        fig_crisis_2008 = create_context_chart(oil_df, contexts[0], extra_days)
        if fig_crisis_2008:
            st.plotly_chart(fig_crisis_2008, use_container_width=True)
    with col2:
        st.subheader(contexts[1]['name'])
        fig_opec = create_context_chart(oil_df, contexts[1], extra_days)
        if fig_opec:
            st.plotly_chart(fig_opec, use_container_width=True)
    col3, col4 = st.columns(2)
    with col3:
        st.subheader(contexts[2]['name'])
        fig_covid = create_context_chart(oil_df, contexts[2], extra_days)
        if fig_covid:
            st.plotly_chart(fig_covid, use_container_width=True)
    with col4:
        st.subheader(contexts[3]['name'])
        fig_war = create_context_chart(oil_df, contexts[3], extra_days)
        if fig_war:
            st.plotly_chart(fig_war, use_container_width=True)

def display_forecast_section(model_name, load_model_func, train_model_func, oil_df, model_features_df, global_price_scaler, global_ma_scaler, num_input_features, key_prefix):
    """
    Exibe a seção de previsão para um determinado modelo de aprendizado de máquina.

    Esta função lida com:
    - Seleção do usuário para o tamanho da janela de previsão.
    - Carregamento de um modelo pré-treinado ou acionamento do treinamento se não estiver disponível para a janela selecionada.
    - Iniciação do treinamento em segundo plano para outros tamanhos de janela se seus modelos não forem encontrados.
    - Geração de previsões de preços futuros.
    - Plotagem de dados históricos juntamente com as previsões.
    - Exibição de métricas de desempenho do modelo e informações relevantes de preços atuais/previstos.

    Args:
        model_name (str): Nome do modelo (ex: "XGBoost", "Random Forest", "LightGBM").
        load_model_func (callable): Função para carregar o modelo (ex: load_xgb_model).
                                    Espera-se que retorne (model, price_scaler, ma_scaler, *optional_feature_names).
        train_model_func (callable): Função para treinar o modelo (ex: train_xgb_model).
                                     Espera-se que seja chamável com (window, model_features_df, global_price_scaler, global_ma_scaler).
        oil_df (pd.DataFrame): DataFrame com preços históricos do petróleo e datas.
        model_features_df (pd.DataFrame): DataFrame com features para treinamento/previsão do modelo.
                                          Deve incluir 'normalized_price' e 'normalized_moving_average_7d'.
        global_price_scaler (sklearn.preprocessing.MinMaxScaler): Scaler para a feature 'price'.
        global_ma_scaler (sklearn.preprocessing.MinMaxScaler): Scaler para a feature 'moving_average_7d'.
        num_input_features (int): Número de features de entrada que o modelo espera (ex: 2 para preço e média móvel).
        key_prefix (str): Prefixo único para as chaves dos widgets Streamlit nesta seção para evitar conflitos.
    """
    st.header(f"Previsão de Preço do Petróleo - {model_name} Regressor")
    model_col1, model_col2 = st.columns([2,1])

    with model_col1:
        windows = {"7 dias":7, "14 dias":14, "21 dias":21, "28 dias":28}
        window_text = st.selectbox(f"Janela {model_name}:", list(windows.keys()), index=0, key=f'{key_prefix}_window')
        window = windows[window_text]

        model, price_scaler, ma_scaler, *feature_names_list = load_model_func(window, key_prefix=key_prefix)
        feature_names = feature_names_list[0] if feature_names_list else None

        if model is None:
            with st.spinner(f"Treinando {model_name} {window} dias..."):
                if model_name == "LightGBM":
                    model, price_scaler, ma_scaler, feature_names = train_model_func(window, model_features_df, global_price_scaler, global_ma_scaler, key_prefix=key_prefix)
                else:
                    model, price_scaler, ma_scaler = train_model_func(window, model_features_df, global_price_scaler, global_ma_scaler, key_prefix=key_prefix)
            if model:
                 st.success(f"{model_name} {window} dias pronto.")
            else:
                 st.error(f"Falha ao treinar {model_name} {window} dias.")
                 return

        for background_window_value in windows.values():
            if background_window_value != window:
                model_for_bg_window, _, _, *_ = load_model_func(background_window_value, key_prefix=key_prefix)

                if model_for_bg_window is None:
                    train_args = (background_window_value, model_features_df.copy(), global_price_scaler, global_ma_scaler, key_prefix)
                    thread = threading.Thread(target=train_model_func, args=train_args, daemon=True)
                    thread.start()

        pred_days = 15
        if len(model_features_df) < window:
            st.error(f"Não há dados suficientes em model_features_df ({len(model_features_df)}) para a janela de {window} dias do {model_name}.")
            current_window_features = np.zeros((window, num_input_features))
        else:
            current_window_features = model_features_df[['normalized_price', 'normalized_moving_average_7d']].values[-window:]
        
        if len(oil_df['price']) < 7:
             st.error(f"Não há dados de preço suficientes para iniciar a previsão {model_name} com cálculo de MA.")
             recent_original_prices = []
        else:
            recent_original_prices = list(oil_df['price'].values[-(max(window, 7)):])
        
        norm_preds, pred_dates = [], []
        original_preds = []
        last_date_from_df = oil_df['date'].iloc[-1]
        last_original_price_from_df = oil_df['price'].iloc[-1]

        if model and price_scaler and ma_scaler:
            for i in range(pred_days):
                current_date = last_date_from_df + timedelta(days=i + 1)
                pred_dates.append(current_date)
                flat_model_input = current_window_features.reshape(1, -1)
                
                predicted_norm_price = 0
                if model_name == "LightGBM" and feature_names:
                    df_pred_input = pd.DataFrame(flat_model_input, columns=feature_names)
                    predicted_norm_price = model.predict(df_pred_input)[0]
                else:
                    predicted_norm_price = model.predict(flat_model_input)[0]

                norm_preds.append(predicted_norm_price)
                
                input_for_price_scaler = [[predicted_norm_price]]
                if hasattr(price_scaler, 'feature_names_in_') and price_scaler.feature_names_in_ is not None and isinstance(price_scaler.feature_names_in_, np.ndarray):
                    input_for_price_scaler = pd.DataFrame([[predicted_norm_price]], columns=price_scaler.feature_names_in_)
                predicted_original_price = price_scaler.inverse_transform(input_for_price_scaler)[0][0]
                original_preds.append(predicted_original_price)
                
                recent_original_prices.append(predicted_original_price)
                if len(recent_original_prices) > 50: 
                    recent_original_prices.pop(0)
                
                next_original_moving_average_7d = pd.Series(recent_original_prices).rolling(window=7).mean().iloc[-1]
                if pd.isna(next_original_moving_average_7d):
                    last_ma_val = oil_df['moving_average_7d'].iloc[-1] if not oil_df.empty else predicted_original_price
                    next_original_moving_average_7d = last_ma_val
                
                input_for_ma_scaler = [[next_original_moving_average_7d]]
                if hasattr(ma_scaler, 'feature_names_in_') and ma_scaler.feature_names_in_ is not None and isinstance(ma_scaler.feature_names_in_, np.ndarray):
                    input_for_ma_scaler = pd.DataFrame([[next_original_moving_average_7d]], columns=ma_scaler.feature_names_in_)
                next_norm_moving_average_7d = ma_scaler.transform(input_for_ma_scaler)[0][0]
                
                new_feature_pair = np.array([predicted_norm_price, next_norm_moving_average_7d])
                current_window_features = np.roll(current_window_features, -1, axis=0)
                current_window_features[-1, :] = new_feature_pair
        else:
            original_preds = [np.nan] * pred_days
            pred_dates = [last_date_from_df + timedelta(days=i+1) for i in range(pred_days)]

        current_date_actual = datetime.now().date()
        days_diff = (current_date_actual - last_date_from_df.date()).days 
        pred_idx = days_diff -1 if days_diff > 0 else 0
        pred_idx = max(0, min(pred_idx, len(original_preds) - 1 if original_preds else 0))
        pred_today = original_preds[pred_idx] if original_preds and pred_idx < len(original_preds) else np.nan
        pred_label_date = pred_dates[pred_idx] if pred_dates and pred_idx < len(pred_dates) else current_date_actual
        
        all_sequence_input_data = model_features_df[['normalized_price', 'normalized_moving_average_7d']].values
        all_sequence_target_data = model_features_df['normalized_price'].values
        X_seq_all, y_target_all = create_sequences_multivariate(all_sequence_input_data, all_sequence_target_data, window)
        
        metrics = None
        if X_seq_all.shape[0] > 0 and model and price_scaler:
            X_flat_all = X_seq_all.reshape(X_seq_all.shape[0], -1)
            if len(X_flat_all) == len(y_target_all) and len(y_target_all) > 0:
                X_test_data_for_metrics = X_flat_all
                if model_name == "LightGBM" and feature_names:
                     X_test_data_for_metrics = pd.DataFrame(X_flat_all, columns=feature_names)
                
                _, X_test_final, _, y_test_final = train_test_split(X_test_data_for_metrics, y_target_all, test_size=0.2, shuffle=False, random_state=42)
                if len(X_test_final) > 0 :
                     metrics = evaluate_model(model, X_test_final, y_test_final, price_scaler)
                else: metrics = {}
            else: metrics = {}
        else: metrics = {}
        
        hist_df = oil_df.copy(); hist_df['type']='Histórico'
        fut_df = pd.DataFrame({'date':pred_dates,'price':original_preds,'type':'Previsão'})
        vis_df = pd.concat([hist_df.tail(50), fut_df])
        pred_fig = go.Figure()
        pred_fig.add_trace(go.Scatter(x=vis_df[vis_df['type']=='Histórico']['date'], y=vis_df[vis_df['type']=='Histórico']['price'], mode='lines', name='Histórico'))
        pred_fig.add_trace(go.Scatter(x=vis_df[vis_df['type']=='Previsão']['date'], y=vis_df[vis_df['type']=='Previsão']['price'], mode='lines', name=f'Previsão {pred_days}d'))
        pred_fig.update_layout(title=f'{model_name} Previsão (janela {window}d)', xaxis_title='Data', yaxis_title='Preço', template='plotly_dark', height=400)
        st.plotly_chart(pred_fig, use_container_width=True)

    with model_col2:
        st.subheader(f'Dados Gerais {model_name}')
        price_info_col1, price_info_col2 = st.columns(2)
        pred_var = ((pred_today - last_original_price_from_df) / last_original_price_from_df) * 100 if pd.notna(pred_today) and last_original_price_from_df != 0 else 0
        with price_info_col1:
            st.metric('Último preço', f"US$ {last_original_price_from_df:.2f}")
        with price_info_col2:
            st.metric(
                f"Previsão para {pred_label_date.strftime('%d/%m/%Y') if isinstance(pred_label_date, (datetime, pd.Timestamp)) else pred_label_date}",
                f"US$ {pred_today:.2f}" if pd.notna(pred_today) else "N/A",
                f"{pred_var:.2f}%"
            )
        st.subheader(f'Métricas {model_name}')
        if metrics:
            metrics_col1, metrics_col2 = st.columns(2)
            with metrics_col1:
                st.metric('MAE', f"{metrics.get('MAE', 0):.4f}")
                st.metric('MSE', f"{metrics.get('MSE', 0):.4f}")
                st.metric('RMSE (USD)', f"US$ {metrics.get('RMSE (USD)', 0):.2f}")
            with metrics_col2:
                st.metric('RMSE', f"{metrics.get('RMSE', 0):.4f}")
                st.metric('R²', f"{metrics.get('R²', 0):.4f}")


def main():
    """
    Função principal para executar o dashboard Streamlit para análise e previsão do preço do petróleo Brent.

    Configura a página do Streamlit, carrega os dados iniciais, define contextos históricos,
    e então chama funções auxiliares para exibir a barra lateral, análise histórica,
    e seções de previsão para os modelos XGBoost, Random Forest e LightGBM.
    """
    st.title("Dashboard de Análise do Preço do Petróleo Brent")
    st.sidebar.header("Informações")
    extra_days = 30
    with st.spinner('Carregando dados do petróleo...'):
        oil_df, model_features_df, global_price_scaler, global_ma_scaler = get_brent_oil_data()
    
    if oil_df is None or oil_df.empty or model_features_df is None or model_features_df.empty:
        st.error("Não foi possível carregar os dados ou os dados estão vazios. Tente novamente mais tarde.")
        return

    filtered_df, start_date_sidebar, end_date_sidebar = setup_sidebar(oil_df)

    contexts = [
        {
            "name": "Crise de 2008",
            "start_date": "2008-06-01",
            "end_date": "2009-01-01",
            "description": "Crise financeira global",
            "color": "red"
        },
        {
            "name": "Efeito da OPEP",
            "start_date": "2016-01-01",
            "end_date": "2016-12-01",
            "description": "Cortes de produção pela OPEP",
            "color": "green"
        },
        {
            "name": "COVID-19",
            "start_date": "2020-03-01",
            "end_date": "2022-03-01",
            "description": "Colapso da demanda devido à pandemia",
            "color": "orange"
        },
        {
            "name": "Guerra da Ucrânia",
            "start_date": "2022-02-01",
            "end_date": "2022-12-01",
            "description": "Aumento dos preços devido ao conflito",
            "color": "purple"
        }
    ]

    display_historical_analysis(oil_df, filtered_df, contexts, extra_days, start_date_sidebar, end_date_sidebar)

    st.markdown("---")
    display_forecast_section(
        model_name="XGBoost", 
        load_model_func=load_xgb_model, 
        train_model_func=train_xgb_model, 
        oil_df=oil_df, 
        model_features_df=model_features_df, 
        global_price_scaler=global_price_scaler, 
        global_ma_scaler=global_ma_scaler, 
        num_input_features=NUM_INPUT_FEATURES,
        key_prefix="xgb"
    )

    st.markdown("---")
    with st.expander("Modelos Alternativos"):
        display_forecast_section(
            model_name="Random Forest", 
            load_model_func=load_rf_model, 
            train_model_func=train_rf_model, 
            oil_df=oil_df, 
            model_features_df=model_features_df, 
            global_price_scaler=global_price_scaler, 
            global_ma_scaler=global_ma_scaler, 
            num_input_features=NUM_INPUT_FEATURES,
            key_prefix="rf"
        )
        st.markdown("---")
        display_forecast_section(
            model_name="LightGBM", 
            load_model_func=load_lgbm_model, 
            train_model_func=train_lgbm_model, 
            oil_df=oil_df, 
            model_features_df=model_features_df, 
            global_price_scaler=global_price_scaler, 
            global_ma_scaler=global_ma_scaler, 
            num_input_features=NUM_INPUT_FEATURES,
            key_prefix="lgbm"
        )

    st.caption("Fonte dos dados: IPEA Data - Série Histórica do Preço do Petróleo Brent")
    
if __name__ == "__main__":
    main()