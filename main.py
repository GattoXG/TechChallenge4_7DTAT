import requests
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import joblib
import os
import threading

st.set_page_config(layout="wide", page_title="Dashboard Petróleo Brent")

@st.cache_data(ttl=3600)
def get_dados_petroleo_brent():
    codigo_serie = 'EIA366_PBRENT366'
    url = f"http://www.ipeadata.gov.br/api/odata4/ValoresSerie(SERCODIGO='{codigo_serie}')"

    response = requests.get(url)

    if response.status_code == 200:
        dados = response.json()['value']
        df = pd.DataFrame(dados)
        df = df[['VALDATA', 'VALVALOR']]
        df.columns = ['data', 'preco']

        # Converter para datetime explicitamente, tratando possíveis erros
        try:
            # Tenta converter, forçando UTC se houver fuso horário
            df['data'] = pd.to_datetime(df['data'], utc=True)
            # Remove o fuso horário para tornar tz-naive
            df['data'] = df['data'].dt.tz_localize(None)
        except Exception as e:
            st.warning(f"Aviso: Não foi possível converter datas para datetime: {e}")
            # Se falhar, tenta converter sem UTC e remove fuso se existir
            df['data'] = pd.to_datetime(df['data'], errors='coerce')
            if pd.api.types.is_datetime64_any_dtype(df['data']) and df['data'].dt.tz is not None:
                 df['data'] = df['data'].dt.tz_localize(None)

        df.dropna(subset=['data', 'preco'], inplace=True) # Garante que não há NaT em data
        df = df.sort_values('data').reset_index(drop=True)

        # Calcula a média móvel de 7 dias
        df['media_movel_7d'] = df['preco'].rolling(window=7).mean()
        df.dropna(subset=['media_movel_7d'], inplace=True) # Garante que não há NaN na média móvel

        # Normaliza os preços entre 0 e 1
        scaler = MinMaxScaler()
        df_normalizado = df.copy()
        df_normalizado.set_index('data', inplace=True)
        df_normalizado['preco_normalizado'] = scaler.fit_transform(df_normalizado[['preco']])
        df_normalizado['media_movel_7d_normalizado'] = scaler.fit_transform(df_normalizado[['media_movel_7d']])
        df_normalizado.drop(columns=['preco', 'media_movel_7d'], inplace=True)

        return df, df_normalizado
    else:
        st.error(f"Erro ao acessar API do IPEA: {response.status_code}")
        return None, None

def criar_grafico_geral(df, contextos):
    # Encontra o valor máximo
    max_preco = df['preco'].max()
    max_data = df[df['preco'] == max_preco]['data'].iloc[0]

    # Encontra o valor mínimo
    min_preco = df['preco'].min()
    min_data = df[df['preco'] == min_preco]['data'].iloc[0]

    fig = go.Figure()

    # Gráfico de linha original
    fig.add_trace(go.Scatter(
        x=df['data'],
        y=df['preco'],
        mode='lines',
        name='Preço do Petróleo',
        line=dict(color='blue')
    ))

    # Linha de média móvel de 7 dias
    fig.add_trace(go.Scatter(
        x=df['data'],
        y=df['media_movel_7d'],
        mode='lines',
        name='Média Móvel 7 Dias',
        line=dict(color='red', dash='dash')
    ))

    # Inserindo retângulos para cada contexto histórico
    for contexto in contextos:
        fig.add_vrect(
            x0=contexto['inicio'], x1=contexto['fim'],
            fillcolor=contexto['cor'], opacity=0.2, layer="below", line_width=0
        )

    # Anotação para o maior preço
    fig.add_annotation(
        x=max_data,
        y=max_preco,
        text=f'Máximo: {max_preco:.2f} USD',
        showarrow=True,
        arrowhead=2,
        ax=20,
        ay=-40 # Ajustado para evitar sobreposição
    )

    # Anotação para o menor preço
    fig.add_annotation(
        x=min_data,
        y=min_preco,
        text=f'Mínimo: {min_preco:.2f} USD',
        showarrow=True,
        arrowhead=2,
        ax=-20, # Ajustado para apontar de outra direção
        ay=40   # Ajustado para evitar sobreposição
    )


    # Atualizando layout
    fig.update_layout(
        title="Preço do Petróleo Brent - Visão Geral com Contextos Históricos",
        xaxis_title="Data",
        yaxis_title="Preço (USD)",
        template="plotly_dark",
        height=600,
        showlegend=True
    )

    return fig

def criar_grafico_contexto(df, contexto, dias_extra=30): # Default 30 dias
    # Converte as datas de string para datetime e garante que são tz-naive
    data_inicio = pd.to_datetime(contexto['inicio']).tz_localize(None) - timedelta(days=dias_extra)
    data_fim = pd.to_datetime(contexto['fim']).tz_localize(None) + timedelta(days=dias_extra)

    # Garante que as datas no DataFrame também não têm fuso horário
    # Cria uma cópia para evitar modificar o DataFrame original
    df_temp = df.copy()
    # Garante que a coluna 'data' é tz-naive
    if pd.api.types.is_datetime64_any_dtype(df_temp['data']) and df_temp['data'].dt.tz is not None:
        df_temp['data'] = df_temp['data'].dt.tz_localize(None)

    # Filtra o dataframe para o período desejado (agora ambos devem ser tz-naive)
    try:
        df_periodo = df_temp[(df_temp['data'] >= data_inicio) & (df_temp['data'] <= data_fim)]
    except TypeError as e:
         st.error(f"Erro de tipo ao filtrar datas: {e}. Verifique os tipos de 'data', 'data_inicio', 'data_fim'.")
         st.write(f"Tipo df_temp['data']: {df_temp['data'].dtype}")
         st.write(f"Tipo data_inicio: {type(data_inicio)}")
         st.write(f"Tipo data_fim: {type(data_fim)}")
         return None # Retorna None se houver erro na filtragem

    if len(df_periodo) == 0:
        # st.warning(f"Nenhum dado encontrado para o período de {contexto['nome']}") # Opcional: avisar se não há dados
        return None

    fig = go.Figure()

    # Gráfico de linha do preço
    fig.add_trace(go.Scatter(
        x=df_periodo['data'],
        y=df_periodo['preco'],
        mode='lines',
        name='Preço do Petróleo',
        line=dict(color='blue')
    ))

    # Linha de média móvel de 7 dias
    fig.add_trace(go.Scatter(
        x=df_periodo['data'],
        y=df_periodo['media_movel_7d'],
        mode='lines',
        name='Média Móvel 7 Dias',
        line=dict(color='red', dash='dash')
    ))

    # Destacando o período do evento
    fig.add_vrect(
        x0=contexto['inicio'], x1=contexto['fim'],
        fillcolor=contexto['cor'], opacity=0.3, layer="below", line_width=0
    )

    # Datas para cálculo de estatísticas (garantindo que são tz-naive)
    inicio_evento = pd.to_datetime(contexto['inicio']).tz_localize(None)
    fim_evento = pd.to_datetime(contexto['fim']).tz_localize(None)

    # Estatísticas do período - com verificação segura
    variacao = 0 # Valor padrão
    try:
        # Encontra os preços no início e fim do período dentro do df_periodo já filtrado
        df_inicio = df_periodo[df_periodo['data'] >= inicio_evento]
        df_fim = df_periodo[df_periodo['data'] <= fim_evento]

        if not df_inicio.empty and not df_fim.empty:
            preco_inicio = df_inicio.iloc[0]['preco']
            preco_fim = df_fim.iloc[-1]['preco']
            if preco_inicio != 0: # Evita divisão por zero
                 variacao = ((preco_fim - preco_inicio) / preco_inicio) * 100
            else:
                 variacao = float('inf') if preco_fim > 0 else 0 # Ou outra lógica para preço inicial zero
        # else: # Opcional: avisar se não encontrou início/fim exato
            # st.warning(f"Não foi possível encontrar preço exato de início/fim para {contexto['nome']}")

    except Exception as e:
        st.warning(f"Erro ao calcular variação para {contexto['nome']}: {e}")
        variacao = 0 # Reseta para 0 em caso de erro

    # Calculate the maximum price in the period for y-axis scaling
    max_preco_periodo = df_periodo['preco'].max() if not df_periodo.empty else 0
    y_axis_upper_limit = max_preco_periodo * 1.5


    # Atualizando layout
    fig.update_layout(
        title=f"{contexto['nome']}: {contexto['descricao']} (Variação: {variacao:.2f}%)",
        xaxis_title="Data",
        yaxis_title="Preço (USD)",
        template="plotly_dark",
        height=400
    )

    fig.update_yaxes(range=[0, y_axis_upper_limit])

    return fig

def separar_dados_treino_teste(df, variavel_alvo='preco_normalizado', test_size=0.2, random_state=42):
    """
    Separa os dados em conjuntos de treino e teste respeitando a sequência temporal.
    
    Parameters:
    -----------
    df_normalizado : pandas.DataFrame
        DataFrame normalizado contendo os dados
    variavel_alvo : str, default='preco_normalizado'
        Nome da coluna que será usada como variável alvo
    test_size : float, default=0.2
        Proporção dos dados a ser usada como conjunto de teste (0.0 a 1.0)
    random_state : int, default=42
        Semente para reprodutibilidade
    
    Returns:
    --------
    X_train, X_test, y_train, y_test : numpy.ndarray
        Arrays contendo os conjuntos de treino e teste para features e alvo
    """
    # Garante que o dataframe está ordenado pelo índice (que deve ser a data)
    df = df.sort_index()
    
    # Define as features e o alvo
    features = [col for col in df.columns if col != variavel_alvo]
    X = df[features].values
    y = df[variavel_alvo].values
    
    # Divide os dados respeitando a sequência temporal
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test

def create_sequences(data, window_size):
    """
    Cria sequências de dados para uso em modelos de previsão.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Array contendo os dados para criar sequências
    window_size : int
        Tamanho da janela deslizante para criar as sequências
    
    Returns:
    --------
    X : numpy.ndarray
        Array contendo as sequências criadas
    """
    X = []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
    return np.array(X)

def avaliar_modelo(modelo, X_test, y_test, scaler=None):
    """
    Avalia o desempenho do modelo usando métricas comuns.
    
    Parameters:
    -----------
    modelo : sklearn.ensemble.RandomForestRegressor
        Modelo treinado para avaliar
    X_test : numpy.ndarray
        Conjunto de teste (features)
    y_test : numpy.ndarray
        Conjunto de teste (valores reais)
    scaler : sklearn.preprocessing.MinMaxScaler, optional
        Scaler usado para desnormalizar os valores, se necessário
    
    Returns:
    --------
    metricas : dict
        Dicionário contendo as métricas calculadas
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    # Faz previsões no conjunto de teste
    y_pred = modelo.predict(X_test)
    
    # Calcula as métricas
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Se um scaler foi fornecido, desnormaliza os valores para mostrar na escala original
    if scaler is not None:
        # Converte para o formato esperado pelo scaler
        y_test_reshaped = y_test.reshape(-1, 1)
        y_pred_reshaped = y_pred.reshape(-1, 1)
        
        # Desnormaliza
        y_test_original = scaler.inverse_transform(y_test_reshaped).flatten()
        y_pred_original = scaler.inverse_transform(y_pred_reshaped).flatten()
        
        # Recalcula as métricas com os valores desnormalizados
        mse_original = mean_squared_error(y_test_original, y_pred_original)
        rmse_original = np.sqrt(mse_original)
        
        # Retorna métricas normalizadas e RMSE em escala original (USD)
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

def treinar_modelo(X_train, y_train):
    """
    Treina um modelo de Random Forest Regressor.
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Conjunto de dados de treino (features)
    y_train : numpy.ndarray
        Conjunto de dados de treino (variável alvo)
    
    Returns:
    --------
    modelo : sklearn.ensemble.RandomForestRegressor
        Modelo treinado
    """
    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)
    
    return modelo
    
# Principal
def main():
    st.title("Dashboard de Análise do Preço do Petróleo Brent")

    # Sidebar
    st.sidebar.header("Informações")
    # Removido o slider e definido valor fixo de 30 dias
    dias_extra = 30  # 30 dias antes e 30 depois do contexto
    
    # Carrega os dados
    with st.spinner('Carregando dados do petróleo...'):
        df_petroleo, df_modelo = get_dados_petroleo_brent()

    if df_petroleo is None or df_petroleo.empty: # Verifica se o df não é None e não está vazio
        st.error("Não foi possível carregar os dados ou os dados estão vazios. Tente novamente mais tarde.")
        return # Interrompe a execução se não houver dados
    
    print(df_modelo.head())  # Debug: imprime as primeiras linhas do dataframe
        
    # Adicionando filtro de data
    st.sidebar.subheader("Filtro de Data")
    
    # Obtendo data mínima e máxima do dataframe
    data_min = df_petroleo['data'].min().date()
    data_max = df_petroleo['data'].max().date()
    
    # Implementação simplificada sem usar callbacks
    # Inicializar valores de state se necessário
    if 'data_inicio' not in st.session_state:
        st.session_state.data_inicio = data_min
    if 'data_fim' not in st.session_state:
        st.session_state.data_fim = data_max
    
    # Slider para ajuste interativo do período
    data_inicio_slider, data_fim_slider = st.sidebar.slider(
        "Selecione o intervalo de datas",
        min_value=data_min,
        max_value=data_max,
        value=(data_min, data_max),
        format="DD/MM/YYYY"
    )
    
    # Usando date_input simples
    col_date1, col_date2 = st.sidebar.columns(2)
    with col_date1:
        data_inicio = st.date_input("De:", data_inicio_slider, min_value=data_min, max_value=data_max)
    with col_date2:
        data_fim = st.date_input("Até:", data_fim_slider, min_value=data_min, max_value=data_max)
    
    # Filtrando o dataframe usando as datas do slider que são sempre atualizadas
    df_filtrado = df_petroleo[(df_petroleo['data'].dt.date >= data_inicio_slider) & 
                             (df_petroleo['data'].dt.date <= data_fim_slider)]

    # Dados estatísticos (usando o dataframe filtrado)
    st.sidebar.subheader("Estatísticas Gerais")
    st.sidebar.metric("Preço Médio", f"{df_filtrado['preco'].mean():.2f} USD")
    st.sidebar.metric("Preço Máximo", f"{df_filtrado['preco'].max():.2f} USD")
    st.sidebar.metric("Preço Mínimo", f"{df_filtrado['preco'].min():.2f} USD")
    # Verifica se a coluna 'data' existe e não está vazia antes de chamar max()
    if 'data' in df_filtrado.columns and not df_filtrado['data'].empty:
        st.sidebar.metric("Última Atualização", f"{df_filtrado['data'].max().strftime('%d/%m/%Y')}")
    else:
        st.sidebar.metric("Última Atualização", "N/A")


    # Define os contextos históricos
    contextos = [
        {
            "nome": "Crise de 2008",
            "inicio": "2008-06-01",
            "fim": "2009-01-01",
            "descricao": "Crise financeira global",
            "cor": "red"
        },
        {
            "nome": "Efeito da OPEP",
            "inicio": "2016-01-01",
            "fim": "2016-12-01",
            "descricao": "Cortes de produção pela OPEP",
            "cor": "green"
        },
        {
            "nome": "COVID-19",
            "inicio": "2020-03-01",
            "fim": "2022-03-01",
            "descricao": "Colapso da demanda devido à pandemia",
            "cor": "orange"
        },
        {
            "nome": "Guerra da Ucrânia",
            "inicio": "2022-02-01",
            "fim": "2022-12-01",
            "descricao": "Aumento dos preços devido ao conflito",
            "cor": "purple"
        }
    ]

    # Gráfico Geral
    st.header("Visão Geral do Preço do Petróleo Brent")
    fig_geral = criar_grafico_geral(df_filtrado, contextos)
    # Ajusta o eixo x para exibir apenas o período selecionado
    fig_geral.update_layout(xaxis=dict(range=[
        data_inicio.strftime('%Y-%m-%d'),
        data_fim.strftime('%Y-%m-%d')
    ]))
    st.plotly_chart(fig_geral, use_container_width=True)

    # Gráficos de cada contexto histórico
    st.header("Análise por Contexto Histórico")

    # Layout de 2 colunas
    col1, col2 = st.columns(2)

    # Gráfico da Crise de 2008
    with col1:
        st.subheader(contextos[0]['nome'])
        fig_crise_2008 = criar_grafico_contexto(df_petroleo, contextos[0], dias_extra)
        if fig_crise_2008:
            st.plotly_chart(fig_crise_2008, use_container_width=True)

    # Gráfico da OPEP
    with col2:
        st.subheader(contextos[1]['nome'])
        fig_opep = criar_grafico_contexto(df_petroleo, contextos[1], dias_extra)
        if fig_opep:
            st.plotly_chart(fig_opep, use_container_width=True)

    col3, col4 = st.columns(2)

    # Gráfico do COVID-19
    with col3:
        st.subheader(contextos[2]['nome'])
        fig_covid = criar_grafico_contexto(df_petroleo, contextos[2], dias_extra)
        if fig_covid:
            st.plotly_chart(fig_covid, use_container_width=True)

    # Gráfico da Guerra da Ucrânia
    with col4:
        st.subheader(contextos[3]['nome'])
        fig_guerra = criar_grafico_contexto(df_petroleo, contextos[3], dias_extra)
        if fig_guerra:
            st.plotly_chart(fig_guerra, use_container_width=True)

    # Informações adicionais
    st.markdown("---")
    
    # Seção de Previsão do Modelo - XGBoost
    st.markdown("---")
    st.header("Previsão de Preço do Petróleo - XGBoost Regressor")
    # Layout para seleção e visualização XGBoost
    col_xgb1, col_xgb2 = st.columns([2,1])
    with col_xgb1:
        # Seleção de janela
        janelas = {"7 dias":7, "14 dias":14, "21 dias":21, "28 dias":28}
        janela_text = st.selectbox("Janela XGBoost:", list(janelas.keys()), index=0, key='xgb_janela')
        window_xgb = janelas[janela_text]
        
        def carregar_xgb(w):
            pm = f"modelos/xgb_model_{w}d.pkl"
            ps = f"modelos/xgb_scaler_{w}d.pkl"
            if os.path.exists(pm) and os.path.exists(ps):
                return joblib.load(pm), joblib.load(ps)
            return None, None
        def treinar_xgb(w):
            sc = MinMaxScaler()
            df_t = df_petroleo.copy()
            df_t['scaled'] = sc.fit_transform(df_t[['preco']])
            seq = create_sequences(df_t['scaled'].values, w)
            y = df_t['scaled'].values[w:]
            Xtr, Xte, ytr, yte = train_test_split(seq, y, test_size=0.2, shuffle=False)
            mdl = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
            mdl.fit(Xtr, ytr)
            os.makedirs("modelos", exist_ok=True)
            joblib.dump(mdl, f"modelos/xgb_model_{w}d.pkl")
            joblib.dump(sc, f"modelos/xgb_scaler_{w}d.pkl")
            return mdl, sc
        model_xgb, scaler_xgb = carregar_xgb(window_xgb)
        if model_xgb is None:
            with st.spinner(f"Treinando XGBoost {window_xgb} dias..."):
                model_xgb, scaler_xgb = treinar_xgb(window_xgb)
            st.success(f"XGBoost {window_xgb} dias pronto.")
        # background train
        for w in janelas.values():
            if w!=window_xgb and not os.path.exists(f"modelos/xgb_model_{w}d.pkl"):
                threading.Thread(target=treinar_xgb, args=(w,), daemon=True).start()
        # Previsão XGB 15 dias
        dias_f = 15
        tmp = df_petroleo.copy()
        tmp['scaled'] = scaler_xgb.transform(tmp[['preco']])
        seq_all = create_sequences(tmp['scaled'].values, window_xgb)
        inp_x = seq_all[-1].copy()
        preds_x, dates_x = [], []
        ld = df_petroleo['data'].iloc[-1]
        lp = df_petroleo['preco'].iloc[-1]
        for i in range(dias_f):
            d2 = ld + timedelta(days=i+1)
            dates_x.append(d2)
            p2 = model_xgb.predict(inp_x.reshape(1,-1))[0]
            r2 = scaler_xgb.inverse_transform([[p2]])[0][0]
            preds_x.append(r2)
            inp_x = np.append(inp_x[1:], p2)
        
        # Após gerar preds_x e dates_x, ajusta previsão para data atual
        data_atual = datetime.now().date()
        days_diff_x = (data_atual - ld.date()).days
        idx_x = days_diff_x - 1 if days_diff_x > 0 else 0
        idx_x = max(0, min(idx_x, len(preds_x) - 1))
        pred_today_x = preds_x[idx_x]
        label_date_x = data_atual

        # Avaliação XGB
        _, Xtx, _, ytx = train_test_split(seq_all, tmp['scaled'].values[window_xgb:], test_size=0.2, shuffle=False)
        metrics_x = avaliar_modelo(model_xgb, Xtx, ytx, scaler_xgb)
        # Gráfico XGB
        hist = df_petroleo.copy(); hist['tipo']='Histórico'
        fut = pd.DataFrame({'data':dates_x,'preco':preds_x,'tipo':'Previsão'})
        vis_x = pd.concat([hist.tail(50), fut])
        fig_x = go.Figure()
        fig_x.add_trace(go.Scatter(x=vis_x[vis_x['tipo']=='Histórico']['data'], y=vis_x[vis_x['tipo']=='Histórico']['preco'], mode='lines', name='Histórico'))
        fig_x.add_trace(go.Scatter(x=vis_x[vis_x['tipo']=='Previsão']['data'], y=vis_x[vis_x['tipo']=='Previsão']['preco'], mode='lines', name=f'Previsão {dias_f}d'))
        fig_x.update_layout(title=f'XGBoost Previsão (janela {window_xgb}d)', xaxis_title='Data', yaxis_title='Preço', template='plotly_dark', height=400)
        st.plotly_chart(fig_x, use_container_width=True)
    with col_xgb2:
        st.subheader('Dados Gerais XGBoost')
        c1, c2 = st.columns(2)
        # Variação percentual para data atual
        var_x = ((pred_today_x - lp) / lp) * 100
        with c1:
            st.metric('Último preço', f"US$ {lp:.2f}")
        with c2:
            st.metric(
                f"Previsão para {label_date_x.strftime('%d/%m/%Y')}",
                f"US$ {pred_today_x:.2f}",
                f"{var_x:.2f}%"
            )
        st.subheader('Métricas XGBoost')
        if metrics_x:
            m1, m2 = st.columns(2)
            with m1:
                st.metric('MAE', f"{metrics_x['MAE']:.4f}")
                st.metric('MSE', f"{metrics_x['MSE']:.4f}")
                st.metric('RMSE (USD)', f"US$ {metrics_x['RMSE (USD)']:.2f}")
            with m2:
                st.metric('RMSE', f"{metrics_x['RMSE']:.4f}")
                st.metric('R²', f"{metrics_x['R²']:.4f}")
    
    st.markdown("---")

    # Modelos Alternativos agrupados em expander
    with st.expander("Modelos Alternativos"):
        # Seção de Previsão do Modelo - Random Forest
        st.header("Previsão de Preço do Petróleo - Random Forest Regressor")
        col_prev1, col_prev2 = st.columns([2, 1])
        
        with col_prev1:
            # Seleção da janela de tempo
            janelas = {"7 dias":7, "14 dias":14, "21 dias":21, "28 dias":28}
            janela_text = st.selectbox("Janela para previsão:", list(janelas.keys()), index=0)
            window_size = janelas[janela_text]
            
            # Funções para carregamento e treino
            def carregar(window):
                path_model = f"modelos/forecast_model_{window}d.pkl"
                path_scaler = f"modelos/scaler_{window}d.pkl"
                if os.path.exists(path_model) and os.path.exists(path_scaler):
                    mdl = joblib.load(path_model)
                    scl = joblib.load(path_scaler)
                    return mdl, scl
                else:
                    return None, None
            
            def treinar(window):
                scaler = MinMaxScaler()
                df_t = df_petroleo.copy()
                df_t['scaled'] = scaler.fit_transform(df_t[['preco']])
                X = create_sequences(df_t['scaled'].values, window)
                y = df_t['scaled'].values[window:]
                Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, shuffle=False)
                mdl = RandomForestRegressor(n_estimators=100, random_state=42)
                mdl.fit(Xtr, ytr)
                os.makedirs("modelos", exist_ok=True)
                joblib.dump(mdl, f"modelos/forecast_model_{window}d.pkl")
                joblib.dump(scaler, f"modelos/scaler_{window}d.pkl")
                return mdl, scaler
            
            # Tenta carregar ou treinar o modelo escolhido
            modelo_rf, scaler_rf = carregar(window_size)
            if modelo_rf is None:
                with st.spinner(f"Treinado modelo {window_size} dias..."):
                    modelo_rf, scaler_rf = treinar(window_size)
                st.success(f"Modelo de {window_size} dias pronto.")
            
            # Inicia background training para outras janelas
            for w in janelas.values():
                if w!=window_size:
                    if not os.path.exists(f"modelos/forecast_model_{w}d.pkl"):
                        threading.Thread(target=treinar, args=(w,), daemon=True).start()
            
            # Previsão (15 dias fixo)
            dias_previsao=15
            df_tmp = df_petroleo.copy()
            df_tmp['scaled']=scaler_rf.transform(df_tmp[['preco']])
            seq= create_sequences(df_tmp['scaled'].values, window_size)
            inp = seq[-1].copy()
            preds=[]; dates=[]
            last_date=df_petroleo['data'].iloc[-1]
            last_price=df_petroleo['preco'].iloc[-1]
            for i in range(dias_previsao):
                d= last_date + timedelta(days=i+1)
                dates.append(d)
                p = modelo_rf.predict(inp.reshape(1,-1))[0]
                real= scaler_rf.inverse_transform([[p]])[0][0]
                preds.append(real)
                inp = np.append(inp[1:], p)
            
            # Define previsão correspondente à data atual
            data_atual = datetime.now().date()
            # Calcula índice baseado na diferença de dias entre data_atual e último dado
            days_diff = (data_atual - last_date.date()).days
            idx = days_diff - 1 if days_diff > 0 else 0
            idx = max(0, min(idx, len(preds) - 1))
            pred_today = preds[idx]
            label_date = data_atual

            # Avaliação
            Xeval= seq; Yeval=df_tmp['scaled'].values[window_size:]
            _, Xte, _, yte = train_test_split(Xeval, Yeval, test_size=0.2, shuffle=False)
            metricas = avaliar_modelo(modelo_rf, Xte, yte, scaler_rf)
            
            # Gráfico
            df_hist = df_petroleo.copy(); df_hist['tipo']='Histórico'
            df_fut = pd.DataFrame({ 'data':dates, 'preco':preds, 'tipo':'Previsão'})
            df_vis = pd.concat([df_hist.tail(50), df_fut])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_vis[df_vis['tipo']=='Histórico']['data'], y=df_vis[df_vis['tipo']=='Histórico']['preco'], mode='lines', name='Histórico'))
            fig.add_trace(go.Scatter(x=df_vis[df_vis['tipo']=='Previsão']['data'], y=df_vis[df_vis['tipo']=='Previsão']['preco'], mode='lines', name=f'Previsão {dias_previsao}d'))
            fig.update_layout(title=f'Previsão (janela {window_size}d)', xaxis_title='Data', yaxis_title='Preço', template='plotly_dark', height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col_prev2:
            st.subheader("Dados Gerais")
            col_info1, col_info2 = st.columns(2)
            # Variação percentual para data atual
            var_percent = ((pred_today - last_price) / last_price) * 100
            with col_info1:
                st.metric("Último preço conhecido", f"US$ {last_price:.2f}")
            with col_info2:
                # Exibe previsão para a data atual (delta color padrão)
                st.metric(
                    f"Previsão para {label_date.strftime('%d/%m/%Y')}",
                    f"US$ {pred_today:.2f}",
                    f"{var_percent:.2f}%"
                )
            
            # Métricas de desempenho
            st.subheader('Métricas de Modelo')
            if metricas:
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.metric("MAE", f"{metricas['MAE']:.4f}")
                    st.metric("MSE", f"{metricas['MSE']:.4f}")
                    st.metric("RMSE (USD)", f"US$ {metricas['RMSE (USD)']:.2f}")
                with col_m2:
                    st.metric("RMSE", f"{metricas['RMSE']:.4f}")
                    st.metric("R²", f"{metricas['R²']:.4f}")
        
        st.markdown("---")
        
        # Seção de Previsão do Modelo - LightGBM
        st.header("Previsão de Preço do Petróleo - LightGBM Regressor")
        col_lgb1, col_lgb2 = st.columns([2,1])
        with col_lgb1:
            # Seleção de janela LightGBM
            janelas = {"7 dias":7, "14 dias":14, "21 dias":21, "28 dias":28}
            janela_l = st.selectbox("Janela LightGBM:", list(janelas.keys()), index=0, key='lgbm_janela')
            window_l = janelas[janela_l]
            # Funções de carregamento e treino
            def carregar_lgbm(win):
                pm = f"modelos/lgbm_model_{win}d.pkl"
                ps = f"modelos/lgbm_scaler_{win}d.pkl"
                pf = f"modelos/lgbm_features_{win}d.pkl"
                if os.path.exists(pm) and os.path.exists(ps):
                    model = joblib.load(pm)
                    scaler = joblib.load(ps)
                    if os.path.exists(pf):
                        features = joblib.load(pf)
                    else:
                        features = [f'f{i}' for i in range(win)]
                    return model, scaler, features
                return None, None, None
            def treinar_lgbm(win):
                sc = MinMaxScaler()
                df_t = df_petroleo.copy()
                df_t['scaled'] = sc.fit_transform(df_t[['preco']])
                seq = create_sequences(df_t['scaled'].values, win)
                features = [f'f{i}' for i in range(win)]
                seq_df = pd.DataFrame(seq, columns=features)
                y = df_t['scaled'].values[win:]
                Xtr, Xte, ytr, yte = train_test_split(seq_df, y, test_size=0.2, shuffle=False)
                mdl = LGBMRegressor(n_estimators=100, random_state=42)
                mdl.fit(Xtr, ytr)
                os.makedirs("modelos", exist_ok=True)
                pm = f"modelos/lgbm_model_{win}d.pkl"
                ps = f"modelos/lgbm_scaler_{win}d.pkl"
                pf = f"modelos/lgbm_features_{win}d.pkl"
                joblib.dump(mdl, pm)
                joblib.dump(sc, ps)
                joblib.dump(features, pf)
                return mdl, sc, features
            # Carrega ou treina
            model_l, scaler_l, feat_names = carregar_lgbm(window_l)
            if model_l is None:
                with st.spinner(f"Treinando LightGBM {window_l} dias..."):
                    model_l, scaler_l, feat_names = treinar_lgbm(window_l)
                st.success(f"LightGBM {window_l} dias pronto.")
            # Treino em background para outras janelas
            for w in janelas.values():
                if w!=window_l and not os.path.exists(f"modelos/lgbm_model_{w}d.pkl"):
                    threading.Thread(target=treinar_lgbm, args=(w,), daemon=True).start()
            # Previsão LightGBM (15 dias)
            dias_l = 15
            tmp_l = df_petroleo.copy()
            tmp_l['scaled'] = scaler_l.transform(tmp_l[['preco']])
            seq_l = create_sequences(tmp_l['scaled'].values, window_l)
            inp_l = seq_l[-1].copy()
            preds_l, dates_l = [], []
            ld_l = df_petroleo['data'].iloc[-1]
            lp_l = df_petroleo['preco'].iloc[-1]
            for i in range(dias_l):
                d_p = ld_l + timedelta(days=i+1)
                dates_l.append(d_p)
                df_pred = pd.DataFrame([inp_l], columns=feat_names)
                p_l = model_l.predict(df_pred)[0]
                r_l = scaler_l.inverse_transform([[p_l]])[0][0]
                preds_l.append(r_l)
                inp_l = np.append(inp_l[1:], p_l)
            
            # Ajusta previsão para data atual
            data_atual_l = datetime.now().date()
            days_diff_l = (data_atual_l - ld_l.date()).days
            idx_l = days_diff_l - 1 if days_diff_l > 0 else 0
            idx_l = max(0, min(idx_l, len(preds_l) - 1))
            pred_today_l = preds_l[idx_l]
            label_date_l = data_atual_l

            # Avaliação LightGBM
            seq_df_all = pd.DataFrame(seq_l, columns=feat_names)
            _, X_lte_df, _, y_lte = train_test_split(seq_df_all, tmp_l['scaled'].values[window_l:], test_size=0.2, shuffle=False)
            # Use DataFrame with feature names for prediction to avoid warnings
            metrics_l = avaliar_modelo(model_l, X_lte_df, y_lte, scaler_l)
            # Gráfico LightGBM
            hist_l = df_petroleo.copy(); hist_l['tipo']='Histórico'
            fut_l = pd.DataFrame({'data':dates_l,'preco':preds_l,'tipo':'Previsão'})
            vis_l = pd.concat([hist_l.tail(50), fut_l])
            fig_l = go.Figure()
            fig_l.add_trace(go.Scatter(x=vis_l[vis_l['tipo']=='Histórico']['data'], y=vis_l[vis_l['tipo']=='Histórico']['preco'], mode='lines', name='Histórico'))
            fig_l.add_trace(go.Scatter(x=vis_l[vis_l['tipo']=='Previsão']['data'], y=vis_l[vis_l['tipo']=='Previsão']['preco'], mode='lines', name=f'Previsão {dias_l}d'))
            fig_l.update_layout(title=f'LightGBM Previsão (janela {window_l}d)', xaxis_title='Data', yaxis_title='Preço', template='plotly_dark', height=400)
            st.plotly_chart(fig_l, use_container_width=True)
        with col_lgb2:
            st.subheader('Dados Gerais LightGBM')
            l1, l2 = st.columns(2)
            # Variação percentual para data atual
            var_l = ((pred_today_l - lp_l) / lp_l) * 100
            with l1:
                st.metric('Último preço', f"US$ {lp_l:.2f}")
            with l2:
                st.metric(
                    f"Previsão para {label_date_l.strftime('%d/%m/%Y')}",
                    f"US$ {pred_today_l:.2f}",
                    f"{var_l:.2f}%"
                )
            st.subheader('Métricas LightGBM')
            if metrics_l:
                lcol1, lcol2 = st.columns(2)
                with lcol1:
                    st.metric('MAE', f"{metrics_l['MAE']:.4f}")
                    st.metric('MSE', f"{metrics_l['MSE']:.4f}")
                    st.metric('RMSE (USD)', f"US$ {metrics_l['RMSE (USD)']:.2f}")
                with lcol2:
                    st.metric('RMSE', f"{metrics_l['RMSE']:.4f}")
                    st.metric('R²', f"{metrics_l['R²']:.4f}")

    st.caption("Fonte dos dados: IPEA Data - Série Histórica do Preço do Petróleo Brent")


if __name__ == "__main__":
    main()