import numpy as np
import pandas as pd
import pandas_datareader.data as web
import yfinance as yf
from funcoes import calculo_hurst
import os

def escolher_seed(tolerancia: float) -> tuple:

    """
    parametro: tolerancia = faixa tolerável para o hurst calculado de um determinado seed de números aleatórios
    Esta função é feita para determinar um seed que, para mil números, o aleatório respeita a faixa tolerável do Hurst
    """

    seed = 0
    distancia = np.inf
    while distancia > tolerancia:
        seed += 1
        np.random.seed(seed)
        aleatorios = np.random.random(size=1000)
        hurst = calculo_hurst.hurst(aleatorios)
        distancia = np.abs(0.5 - hurst)

    return seed, hurst


def buscar_cotacoes(tickers_list: list, dias_cotacoes: int, country: str) -> tuple:

    """
    arquivo_txt: endereco do caminho parcial do arquivo dos tickers
    dias_cotacoes: quantidade de dias para download dos dados de fechamento
    country: país de origem das ações
    Esta função realiza a busca dos dados de fechamento das cotações para cada dia
    """

    # tickers_df = pd.read_csv(arquivo_txt, header=None)
    tickers = []
    casas_arred = 0
    if country.upper() == "US":
        casas_arred = 4     # é possível comprar uma ação americana com casas decimais
        for ticker in tickers_list:
            tickers.append(ticker)
    elif country.upper() == "BR":
        for ticker in tickers_list:
            tickers.append(ticker + ".SA")  # ação BR deve terminar com '.SA'

    yf.pdr_override()
    cotations = web.get_data_yahoo(tickers, period=str(dias_cotacoes) + 'd', threads=1)['Adj Close']

    return cotations, casas_arred


def calcular_variacoes(cotations, tickers):
    n_lins = cotations.shape[0]
    n_cols = cotations.shape[1]

    cotations_var = np.zeros(shape=(n_lins - 1, n_cols), dtype=float)
    cotations_var = pd.DataFrame(data=cotations_var, columns=tickers)
    for ticker in tickers:
        for j in range(1, n_lins):
            cotations_var[ticker][j - 1] = (cotations[ticker][j] - cotations[ticker][j - 1]) / cotations[ticker][j - 1]

    return cotations_var


def calc_retornos_riscos_carteira(cromos, medias, mat_cov):
    retornos = cromos.dot(medias)

    num_cromossomos = len(cromos)

    riscos = np.zeros(shape=num_cromossomos)
    for i in range(num_cromossomos):
        riscos[i] = cromos[i].dot(mat_cov).dot(cromos[i].T)

    return retornos, riscos


def exportar_df(valor_inv, arr, names_indexes, perc_corte, casas_arred, cotacoes, country):

    df = pd.DataFrame(data=(np.round(arr, 2)).T, index=names_indexes, columns=["%"])

    moeda = "R$" if country.upper() == "BR" else "US$"

    for ticker in names_indexes:
        if df["%"][ticker] < perc_corte:
            df.drop(labels=[ticker], inplace=True)

    cotations = []
    for ticker in df.index:
        cotation = cotacoes[ticker][-1]
        cotations.append(cotation)

    df["precos"] = cotations

    if country.upper() == "US":
        df["qtd_comprar"] = np.round((df["%"] * valor_inv) / df["precos"], casas_arred)
    else:
        df["qtd_comprar"] = np.floor((df["%"] * valor_inv) / df["precos"])

    df["valor_total"] = df["qtd_comprar"] * df["precos"]
    df["valor_total_formatado"] = df["valor_total"].apply(lambda x: f"{moeda} {x:_.2f}".replace(".", ",").replace("_", "."))

    df["qtd_comprar"] = df["qtd_comprar"].apply(lambda x: f"{float(x):_.4f}".replace(".", ",").replace("_", "."))

    df["%"] = df["%"] * 100

    df["precos"] = df["precos"].apply(lambda x: f"{moeda} {x:_.2f}".replace(".", ",").replace("_", "."))

    df.to_csv(os.path.join("resultado.csv"))