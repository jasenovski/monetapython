import os
from funcoes.moneta import moneta_ag
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff

if __name__ == "__main__":
    
    st.title("Modelo Moneta")
    st.markdown("Este modelo ajuda a selecionar uma carteira de ações que otimiza o par risco-retorno")
    st.divider()

    country = st.sidebar.radio(
        label="Selecione o país desejado:",
        options=["US", "BR"]
    )
    st.sidebar.divider()

    check = st.sidebar.checkbox(
        label=f"Selecione se quiser rodar o modelo para todas as ações ({country})",
        value=False
    )

    stocks = sorted(pd.read_csv(os.path.join("tickers", f"tickers_{country.lower()}.csv"))["Tickers"].tolist())
    if check is True:
        stocks_filtered = stocks[:]
    else:
        stocks_filtered = stocks[:5]

    stocks_selections = st.sidebar.multiselect(
        label="Acoes Disponiveis:", 
        options=stocks,
        default=stocks_filtered
        )
    st.sidebar.divider()

    currency = "R$" if country == "BR" else "US$"
    investment = st.sidebar.text_input(
        label=f"Valor (base) do investimento na carteira ({currency}):",
        value="1000"
    )
    investment = float(investment.replace(",", "."))
    st.sidebar.divider()


    minimum_percentage = st.sidebar.text_input(
        label="Insira o valor mínimo para o percentual (%) aceitável de uma ação na carteira final:",
        value="5"
    )
    minimum_percentage = float(minimum_percentage.replace(",", ".")) / 100 if minimum_percentage != "" else 0.05
    st.sidebar.divider()

    dias_cotacoes = st.sidebar.slider(
        label="Selecione o número de cotações anteriores por ação:",
        min_value=34,
        max_value=200,
        value=72,
        step=1
    )
    st.sidebar.divider()

    top_averages = st.sidebar.slider(
        label="Maiores médias:",
        min_value=0,
        max_value=20,
        value=15,
        step=1
    )
    st.sidebar.divider()

    solucao = st.sidebar.button(
        label="Gerar Carteira",
        on_click=moneta_ag,
        kwargs=
        {
        "tickers_list": stocks_selections,
        "dp_final": 0.0001,
        "valor_investimento": investment,
        "percentual_corte": minimum_percentage,
        "country": country,
        "fixar_seed": True,
        "tolerancia_hurst": 0.01,
        "dias_cots": dias_cotacoes,
        "qtd_maiores_medias": top_averages
        }
    )

    if solucao:
        df_solucao = pd.read_csv(os.path.join("resultado.csv"), index_col=0)
        df_solucao = df_solucao.reset_index().rename(columns={"index": "Ações"})
        df_solucao = df_solucao.rename(columns={"precos": "Preços", 
                                                "qtd_comprar": "Qtd. Comprar", 
                                                "valor_total_formatado": f"Valor Total ({currency})"})

        st.subheader(
            body="Solução Obtida"
        )

        st.dataframe(
            data=df_solucao[["Ações", "Preços", "Qtd. Comprar", f"Valor Total ({currency})"]]
        )

        df_solucao["%"] = df_solucao["valor_total"] / df_solucao["valor_total"].sum()

        fig = px.pie(df_solucao, values='%', names='Ações', title='Portfolio Otimizado')
        st.plotly_chart(fig)

        valores_finais = pd.read_pickle(os.path.join("resultados.pkl"))
        col1, col2, col3 = st.columns(3)
        col1.metric("Valor Total", f"{currency} {df_solucao['valor_total'].sum():_.2f}".replace(".", ",").replace("_", "."))
        col2.metric("Retorno a.m.", f"{((1 + valores_finais['retorno']) ** 22 - 1) * 100:.2f}%".replace(".", ","))
        col3.metric("Risco a.m.", f"{((1 + valores_finais['risco']) ** 22 - 1) * 100:.2f}%".replace(".", ","))


        st.warning(
            body="O retorno mencionado acima é baseado em desempenho passado. Retorno passado não garante retorno futuro!!!",
            icon="⚠️"
        )
        try:
            cotacoes = pd.read_pickle(os.path.join("cotacoes", "cotacoes.pkl"))

            st.divider()
            st.subheader(f"Cotações (adj. close) para a geração da carteira ({currency}):")

            st.dataframe(
                data=cotacoes
            )
        except FileNotFoundError:
            pass
