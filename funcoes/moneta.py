import numpy as np
import datetime as dtm
import os
import glob
from funcoes.utils import escolher_seed, buscar_cotacoes, calcular_variacoes, calc_retornos_riscos_carteira, exportar_df
from funcoes.ag import gerar_cromos, corrigir_fitnesses, roda_acaso, mutacao_um, mutacao_dois, substituir_geracoes
import pandas as pd

def moneta_ag(tickers_list, dp_final, valor_investimento, percentual_corte, country, fixar_seed, tolerancia_hurst, dias_cots, qtd_maiores_medias):

    if fixar_seed is True:
        seed_bom, coef_hurst = escolher_seed(tolerancia=tolerancia_hurst)
        np.random.seed(seed_bom)

    cotacoes, casas_arred = buscar_cotacoes(tickers_list=tickers_list, dias_cotacoes=dias_cots, country=country)

    cotacoes = cotacoes.dropna(axis=1)
    # cotacoes.columns = [acao.replace(".SA", "") for acao in cotacoes.columns if  acao[-2:] == "SA"]
    tickers = cotacoes.columns
    # tickers = cotacoes.columns

    # if exportar_cotacoes is True:
    #     cotacoes.to_excel(os.path.join("cotacoes", f"cotacoes{country.upper()}_{dias_cots}d_{dtm.datetime.now().strftime('%d-%m-%y')}.xlsx"))

    cotations_var = calcular_variacoes(cotations=cotacoes, tickers=tickers)

    means = cotations_var.mean(axis=0)
    tickers = list(means.nlargest(qtd_maiores_medias).index)
    cotations_var = cotations_var[tickers]
    
    num_cotacoes = cotations_var.shape[0]
    num_genes = cotations_var.shape[1]

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

    cromossomo_final = cromossomos[0]
    retornos_finais, riscos_finais = calc_retornos_riscos_carteira(cromos=np.array([cromossomo_final]), medias=medias, mat_cov=mat_cov)
    fitnesses_finais = np.round(retornos_finais / riscos_finais, 2)
    # nome_txt = arq_txt.split(os.path.sep)[1].split('.')[0]
    # dia = dtm.datetime.now().strftime("%d-%m-%y")

    # qtd_files = len(glob.glob(os.path.join("resultados", f"resultado_{dia}_{dias_cots}d_{country.upper()}_{nome_txt}_*.xlsx")))
    # name_file = os.path.join("resultados", f"resultado_{dia}_{dias_cots}d_{country.upper()}_{nome_txt}_fitness{np.mean(fitnesses_finais)}_{qtd_files + 1}.xlsx")

    pd.to_pickle({"fitness final": np.average(fitnesses_finais), 
                  "retorno": np.average(retornos_finais), 
                  "risco": np.average(riscos_finais)},
                  os.path.join("resultados.pkl"))

    exportar_df(valor_inv=valor_investimento, arr=cromossomo_final, names_indexes=tickers, 
                            perc_corte=percentual_corte, casas_arred=casas_arred, 
                            cotacoes=cotacoes, country=country)
    
    return True

    #     if fixar_seed is True:
    #         print(f"[INFO] O resultado foi obtido com {iteracoes} iteracoes. (Coeficiente de Hurst: {coef_hurst:.2f})")
    #     else:
    #         print(f"[INFO] O resultado foi obtido com {iteracoes} iteracoes.")
        
    #     print(f"[INFO] O resultado final foi exportado com sucesso para: {name_file}")
    #     print(f"[INFO] O fitness obtido foi de: {round(np.average(fitnesses_finais), 5):.2f}")
    #     print(f"[INFO] O retorno esperado é de: {round(np.average(retornos_finais), 5) * 100:.5f}%")
    #     print(f"[INFO] O risco esperado é de: {round(np.average(riscos_finais), 5) * 100:.5f}%")
    #     print(f"------------------------ Resultado Final ------------------------")
    #     print(df_final[["%", "precos", "qtd_comprar", "valor_total_formatado"]].rename(columns={"valor_total_formatado": "valor_total"}))
    #     print(f"[INFO] O valor total do investimento é de: {moeda} {df_final['valor_total'].sum():,.2f}")
    #     print(f"[INFO] O percentual investido será de: {(df_final['valor_total'].sum() / valor_investimento * 100):.2f}%")
    #     print(f"-----------------------------------------------------------------")
    # else:
    #     print(f"[INFO] A soma dos percentuais não resulta 100% para todos os cromossomos\n {cromossomos.sum(axis=1)}")