import numpy as np

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


def gerar_cromos(num_cols, num_cromossomos=6):

    cromossomos = np.random.rand(num_cromossomos, num_cols)

    for i in range(cromossomos.shape[0]):
        cromossomos[i] = cromossomos[i] / cromossomos[i].sum()

    return cromossomos