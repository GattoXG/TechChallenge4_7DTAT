\
# src/plot_utils.py
import plotly.graph_objects as go
import pandas as pd
from datetime import timedelta
import streamlit as st

def create_main_chart(df, contexts):
    """
    Cria um gráfico geral da série histórica do preço do petróleo Brent,
    incluindo a média móvel de 7 dias e destacando contextos históricos.

    Args:
        df (pd.DataFrame): DataFrame contendo os dados do petróleo, incluindo
                           as colunas 'date', 'price' e 'moving_average_7d'.
        contexts (list): Uma lista de dicionários, onde cada dicionário representa
                          um contexto histórico e deve conter as chaves 'start_date' (data de início),
                          'end_date' (data de fim) e 'color' (cor para destacar o período).

    Returns:
        plotly.graph_objects.Figure: Um objeto Figure do Plotly contendo o gráfico gerado.
    """
    max_price = df['price'].max()
    max_date = df[df['price'] == max_price]['date'].iloc[0]
    min_price = df['price'].min()
    min_date = df[df['price'] == min_price]['date'].iloc[0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['price'],
        mode='lines',
        name='Preço do Petróleo',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['moving_average_7d'],
        mode='lines',
        name='Média Móvel 7 Dias',
        line=dict(color='red', dash='dash')
    ))
    for context in contexts:
        fig.add_vrect(
            x0=context['start_date'], x1=context['end_date'],
            fillcolor=context['color'], opacity=0.2, layer="below", line_width=0
        )
    fig.add_annotation(
        x=max_date,
        y=max_price,
        text=f'Máximo: {max_price:.2f} USD',
        showarrow=True,
        arrowhead=2,
        ax=20,
        ay=-40
    )
    fig.add_annotation(
        x=min_date,
        y=min_price,
        text=f'Mínimo: {min_price:.2f} USD',
        showarrow=True,
        arrowhead=2,
        ax=-20,
        ay=40 
    )
    fig.update_layout(
        title="Preço do Petróleo Brent - Visão Geral com Contextos Históricos",
        xaxis_title="Data",
        yaxis_title="Preço (USD)",
        template="plotly_dark",
        height=600,
        showlegend=True
    )
    return fig

def create_context_chart(df, context, extra_days=30):
    """
    Cria um gráfico focado em um contexto histórico específico do preço do petróleo.

    Exibe o preço do petróleo e sua média móvel de 7 dias para um período
    em torno do contexto especificado, destacando o período do evento e
    mostrando a variação percentual do preço durante o evento.

    Args:
        df (pd.DataFrame): DataFrame completo com os dados históricos do petróleo.
                           Deve conter as colunas 'date', 'price', 'moving_average_7d'.
        context (dict): Um dicionário definindo o contexto histórico. Deve conter:
                         'name' (str): Nome do contexto (para o título do gráfico).
                         'start_date' (str or datetime): Data de início do contexto.
                         'end_date' (str or datetime): Data de fim do contexto.
                         'color' (str): Cor para destacar o período do evento no gráfico.
                         'description' (str): Descrição do evento (para o título).
        extra_days (int, optional): Número de dias a serem exibidos antes do início
                                    e após o fim do contexto. Padrão é 30.

    Returns:
        plotly.graph_objects.Figure or None: Um objeto Figure do Plotly com o gráfico
                                             do contexto, ou None se não houver dados
                                             suficientes para o período.
    """
    start_date_val = pd.to_datetime(context['start_date']).tz_localize(None) - timedelta(days=extra_days)
    end_date_val = pd.to_datetime(context['end_date']).tz_localize(None) + timedelta(days=extra_days)
    temp_df = df.copy()
    if pd.api.types.is_datetime64_any_dtype(temp_df['date']) and temp_df['date'].dt.tz is not None:
        temp_df['date'] = temp_df['date'].dt.tz_localize(None)
    try:
        period_df = temp_df[(temp_df['date'] >= start_date_val) & (temp_df['date'] <= end_date_val)]
    except TypeError as e:
         st.error(f"Erro de tipo ao filtrar datas: {e}. Verifique os tipos de 'date', 'start_date_val', 'end_date_val'.")
         return None
    if len(period_df) == 0:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=period_df['date'],
        y=period_df['price'],
        mode='lines',
        name='Preço do Petróleo',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=period_df['date'],
        y=period_df['moving_average_7d'],
        mode='lines',
        name='Média Móvel 7 Dias',
        line=dict(color='red', dash='dash')
    ))
    fig.add_vrect(
        x0=context['start_date'], x1=context['end_date'],
        fillcolor=context['color'], opacity=0.3, layer="below", line_width=0
    )
    event_start = pd.to_datetime(context['start_date']).tz_localize(None)
    event_end = pd.to_datetime(context['end_date']).tz_localize(None)
    variation = 0
    try:
        df_event_start = period_df[period_df['date'] >= event_start]
        df_event_end = period_df[period_df['date'] <= event_end]
        if not df_event_start.empty and not df_event_end.empty:
            start_price = df_event_start.iloc[0]['price']
            end_price = df_event_end.iloc[-1]['price']
            if start_price != 0:
                 variation = ((end_price - start_price) / start_price) * 100
            else:
                 variation = float('inf') if end_price > 0 else 0
    except Exception as e:
        print(f"Warning: Erro ao calcular variação para {context['name']}: {e}")
        variation = 0
    max_price_period = period_df['price'].max() if not period_df.empty else 0
    y_axis_upper_limit = max_price_period * 1.5
    fig.update_layout(
        title=f"{context['name']}: {context['description']} (Variação: {variation:.2f}%)",
        xaxis_title="Data",
        yaxis_title="Preço (USD)",
        template="plotly_dark",
        height=400
    )
    fig.update_yaxes(range=[0, y_axis_upper_limit])
    return fig
