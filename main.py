import requests
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta

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
        return df
    else:
        st.error(f"Erro ao acessar API do IPEA: {response.status_code}")
        return None

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

# Principal
def main():
    st.title("Dashboard de Análise do Preço do Petróleo Brent")

    # Sidebar
    st.sidebar.header("Informações")
    # Removido o slider e definido valor fixo de 30 dias
    dias_extra = 30  # 30 dias antes e 30 depois do contexto
    
    # Carrega os dados
    with st.spinner('Carregando dados do petróleo...'):
        df_petroleo = get_dados_petroleo_brent()

    if df_petroleo is None or df_petroleo.empty: # Verifica se o df não é None e não está vazio
        st.error("Não foi possível carregar os dados ou os dados estão vazios. Tente novamente mais tarde.")
        return # Interrompe a execução se não houver dados
        
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
    st.caption("Fonte dos dados: IPEA Data - Série Histórica do Preço do Petróleo Brent")

if __name__ == "__main__":
    main()