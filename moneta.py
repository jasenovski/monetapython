# Manipulação de Dados
import numpy as np
import pandas as pd
import os
# import argparse
import json
import datetime as dtm
from funcoes import calculo_hurst

# Obtençaõ das cotações
import pandas_datareader.data as web
import yfinance as yf


def escolher_seed(tolerancia):
    seed = 0
    np.random.seed(seed)
    aleatorios = np.random.random(size=1000)
    hurst = calculo_hurst.hurst(aleatorios)
    while np.abs(0.5 - hurst) > tolerancia:
        aleatorios = np.random.random(size=1000)
        hurst = calculo_hurst.hurst(aleatorios)
        seed += 1

    return seed, hurst


def buscar_cotacoes(arquivo_txt, dias_cotacoes, country):
    tickers_df = pd.read_csv(arquivo_txt, header=None)
    tickers = []
    casas_arred = 0
    if country == "US":
        casas_arred = 4
        for ticker in tickers_df[0]:
            tickers.append(ticker)
    elif country == "BR":
        for ticker in tickers_df[0]:
            tickers.append(ticker + ".SA")

    yf.pdr_override()
    cotations = web.get_data_yahoo(tickers, period=str(dias_cotacoes) + 'd', threads=1)['Adj Close']

    return cotations, tickers, casas_arred


def calcular_variacoes(cotations, tickers):
    n_lins = cotations.shape[0]
    n_cols = cotations.shape[1]

    cotations_var = np.zeros(shape=(n_lins - 1, n_cols), dtype=float)
    cotations_var = pd.DataFrame(data=cotations_var, columns=tickers)
    for ticker in tickers:
        for j in range(1, n_lins):
            cotations_var[ticker][j - 1] = (cotations[ticker][j] - cotations[ticker][j - 1]) / cotations[ticker][j - 1]

    return cotations_var


def gerar_cromos(num_cols, num_cromossomos=6):

    cromossomos = np.random.rand(num_cromossomos, num_cols)

    for i in range(cromossomos.shape[0]):
        cromossomos[i] = cromossomos[i] / cromossomos[i].sum()

    return cromossomos


def calc_retornos_riscos_carteira(cromos, medias, mat_cov):
    retornos = cromos.dot(medias)

    num_cromossomos = len(cromos)

    riscos = np.zeros(shape=num_cromossomos)
    for i in range(num_cromossomos):
        riscos[i] = cromos[i].dot(mat_cov).dot(cromos[i].T)

    return retornos, riscos


def roda_acaso(fitness_acumulado):
    indices_sorteados = [np.inf]
    while len(indices_sorteados) < 3:
        alfa = np.random.rand()
        for i in range(fitness_acumulado.shape[0]):
            if alfa <= fitness_acumulado[i] and indices_sorteados[-1] != i:
                indices_sorteados.append(i)
                break

    return indices_sorteados[1:]


def mutacao_um(cromo, posicoes):
    cromossomo_final = cromo.copy()
    valor_um = cromossomo_final[posicoes[0]]
    valor_dois = cromossomo_final[posicoes[1]]
    cromossomo_final[posicoes[0]] = valor_dois
    cromossomo_final[posicoes[1]] = valor_um
    return cromossomo_final


def mutacao_dois(cromo, posicoes):
    cromo1 = cromo.copy()
    cromo2 = cromo.copy()
    valor_um = cromo[posicoes[0]]
    valor_dois = cromo[posicoes[1]]
    cromo1[posicoes[0]] = 0
    cromo1[posicoes[1]] = valor_um + valor_dois
    cromo2[posicoes[0]] = valor_um + valor_dois
    cromo2[posicoes[1]] = 0
    return cromo1, cromo2


def corrigir_fitnesses(fitnesses):
    fitnesses_corrigidos = []
    for fitness in fitnesses:
        if fitness < -1:
            fitnesses_corrigidos.append(1 / np.abs(fitness))
        elif fitness < 0:
            fitnesses_corrigidos.append(np.abs(fitness))
        else:
            fitnesses_corrigidos.append(fitness)

    return np.array(fitnesses_corrigidos)


def substituir_geracoes(fit_pais, fit_filhos, cromos_pais, cromos_filhos):
    args_min_pais = fit_pais.argsort()[:2]
    args_max_filhos = fit_filhos.argsort()[-2:]
    cromos_finais = cromos_pais.copy()

    if fit_filhos[args_max_filhos[-1]] > fit_pais[args_min_pais[0]]:
        cromos_finais[args_min_pais[0]] = cromos_filhos[args_max_filhos[-1]]
    elif fit_filhos[args_max_filhos[-1]] > fit_pais[args_min_pais[1]]:
        cromos_finais[args_min_pais[1]] = cromos_filhos[args_max_filhos[-1]]
    else:
        return cromos_finais

    if fit_filhos[args_max_filhos[-2]] > fit_pais[args_min_pais[1]]:
        cromos_finais[args_min_pais[1]] = cromos_filhos[args_max_filhos[-2]]

    return cromos_finais


def exportar_df(valor_inv, arr, names_indexes, path_name, perc_corte, casas_arred, cotacoes):
    df = pd.DataFrame(data=(np.round(arr, 2)).T, index=names_indexes, columns=["%"])

    for ticker in names_indexes:
        if df["%"][ticker] < perc_corte:
            df.drop(labels=[ticker], inplace=True)

    cotations = []
    for ticker in df.index:
        cotation = np.round(cotacoes[ticker][-1], 2)
        cotations.append(cotation)

    df["precos"] = cotations

    df["qtd_comprar"] = np.round((df["%"] * valor_inv) / df["precos"], casas_arred)

    df["valor_total"] = df["qtd_comprar"] * df["precos"]

    df.to_excel(path_name)


def moneta_ag(arq_txt, dp_final, valor_investimento, percentual_corte, country, exportar_cotacoes, fixar_seed, tolerancia_hurst, dias_cots):

    if fixar_seed is True:
        seed_bom, hurst_seed = escolher_seed(tolerancia=tolerancia_hurst)
        np.random.seed(seed_bom)

    cotacoes, tickers, casas_arred = buscar_cotacoes(arquivo_txt=arq_txt, dias_cotacoes=dias_cots, country=country)

    cotacoes = cotacoes.dropna()

    if exportar_cotacoes is True:
        cotacoes.to_excel(os.path.join("cotacoes", "cotacoes.xlsx"))

    cotations_var = calcular_variacoes(cotations=cotacoes, tickers=tickers)

    num_cotacoes = cotacoes.shape[0]
    num_genes = cotacoes.shape[1]

    mat_cov = np.cov(cotations_var.T)

    medias = np.average(cotations_var, axis=0)

    cromossomos = gerar_cromos(num_cols=num_genes, num_cromossomos=6)

    retornos_carteiras, riscos_carteiras = calc_retornos_riscos_carteira(cromos=cromossomos, medias=medias, mat_cov=mat_cov)

    fitness_carteiras = retornos_carteiras / riscos_carteiras

    num_filhos = 8
    cromossomos_filhos = np.zeros(shape=(num_filhos, num_genes), dtype=float)
    desvio_padrao_carteiras = np.inf

    iteracoes = 0
    while desvio_padrao_carteiras > dp_final:
        fitnesses_corrigidos = corrigir_fitnesses(fitnesses=fitness_carteiras)
        fitness_acum = np.cumsum(fitnesses_corrigidos) / fitnesses_corrigidos.sum()
        cromo_sorteados = roda_acaso(fitness_acumulado=fitness_acum)
        beta = np.random.rand()
        cromossomos_filhos[0] = cromossomos[cromo_sorteados[0]] * beta + cromossomos[cromo_sorteados[1]] * (1 - beta)
        cromossomos_filhos[1] = cromossomos[cromo_sorteados[0]] * (1 - beta) + cromossomos[cromo_sorteados[1]] * beta

        while True:
            genes_mutacao_um = np.random.choice(a=range(num_genes), size=2)
            genes_mutacao_dois = np.random.choice(a=range(num_genes), size=2)
            if genes_mutacao_um[0] != genes_mutacao_um[1] and genes_mutacao_dois[0] != genes_mutacao_dois[1]:
                cromossomos_filhos[2] = mutacao_um(cromo=cromossomos_filhos[0], posicoes=genes_mutacao_um)
                cromossomos_filhos[3] = mutacao_um(cromo=cromossomos_filhos[1], posicoes=genes_mutacao_dois)
                break

        while True:
            genes_mutacao_um = np.random.choice(a=range(num_genes), size=2)
            genes_mutacao_dois = np.random.choice(a=range(num_genes), size=2)
            if genes_mutacao_um[0] != genes_mutacao_um[1] and genes_mutacao_dois[0] != genes_mutacao_dois[1]:
                cromossomos_filhos[4], cromossomos_filhos[5] = mutacao_dois(cromo=cromossomos_filhos[0],
                                                                            posicoes=genes_mutacao_um)

                cromossomos_filhos[6], cromossomos_filhos[7] = mutacao_dois(cromo=cromossomos_filhos[1],
                                                                            posicoes=genes_mutacao_dois)
                break

        retornos_filhos, riscos_filhos = calc_retornos_riscos_carteira(cromos=cromossomos_filhos,
                                                                       medias=medias, mat_cov=mat_cov)

        fitness_filhos = retornos_filhos / riscos_filhos

        cromossomos = substituir_geracoes(fit_pais=fitness_carteiras, fit_filhos=fitness_filhos,
                                          cromos_pais=cromossomos, cromos_filhos=cromossomos_filhos)

        retornos_carteiras, riscos_carteiras = calc_retornos_riscos_carteira(cromos=cromossomos,
                                                                             medias=medias, mat_cov=mat_cov)

        fitness_carteiras = retornos_carteiras / riscos_carteiras

        desvio_padrao_carteiras = np.std(fitness_carteiras)

        iteracoes += 1

    if (np.round(cromossomos.sum(axis=1), 0) == 1).all():
        path_export = ""
        cromossomo_final = cromossomos[0]
        retorno_final, risco_final = calc_retornos_riscos_carteira(cromos=np.array([cromossomo_final]), medias=medias, mat_cov=mat_cov)
        fitness_final = np.round(retorno_final / risco_final, 2)
        for i in range(100):
            nome_txt = arq_txt.split(os.path.sep)[1].split('.')[0]
            dia = dtm.datetime.now().strftime("%d-%m-%y")
            name_file = os.path.join("resultados", f"resultado_{dia}_{dias_cots}d_{country}_{nome_txt}_{fitness_final[0]}_{i}.xlsx")
            if not os.path.exists(name_file):
                path_export = name_file
                break

        exportar_df(valor_inv=valor_investimento, arr=cromossomo_final, names_indexes=tickers,
                    path_name=path_export, perc_corte=percentual_corte, casas_arred=casas_arred, cotacoes=cotacoes)
        print(f"[INFO] O resultado foi obtido com {iteracoes} iteracoes.")
        print(f"[INFO] O resultado final foi exportado com sucesso para: {path_export}")
        print(f"[INFO] O fitness obtido foi de: {round(np.average(fitness_carteiras), 2)}")
        print(f"[INFO] O retorno esperado é de: {round(np.average(retornos_carteiras), 5) * 100} %")
        print(f"[INFO] O risco esperado é de: {round(np.average(riscos_carteiras), 5) * 100} %")
    else:
        print(f"[INFO] A soma dos percentuais não resulta 100% para todos os cromossomos\n {cromossomos.sum(axis=1)}")


if __name__ == "__main__":

    endereco_configuracoes = "configs/configuracoes.json"

    with open(file=endereco_configuracoes, mode="r") as json_file:
        configuracoes = json.load(json_file)

    moneta_ag(arq_txt=os.path.join("tickers", configuracoes["arquivo_txt"]),
              dp_final=configuracoes["desvio_padrao_final"],
              valor_investimento=configuracoes["valor_investimento"],
              percentual_corte=configuracoes["percentual_corte"],
              country=configuracoes["country"],
              exportar_cotacoes=configuracoes["exportar_cotacoes"],
              fixar_seed=configuracoes["fixar_seed"],
              tolerancia_hurst=configuracoes["tolerancia_hurst"],
              dias_cots=configuracoes["dias_cots"])
