import numpy as np


def mmq_reta(entradas: np.array, saidas: np.array):

    """
    os parâmetros 'entradas' e 'saidas' devem estar no formato de array com apenas uma dimensão
    """

    a = 0
    b = 0
    c = 0
    d = 0
    e = 0
    f = 0

    tamanho_entradas = entradas.shape[0]

    for i in range(0, tamanho_entradas, 1):
        a = a + (2 * np.square(entradas[i]))

    for i in range(0, tamanho_entradas, 1):
        b = b + (2 * entradas[i])

    for i in range(0, tamanho_entradas, 1):
        c = c + (-2 * entradas[i] * saidas[i])

    c = c * (-1)

    for i in range(0, tamanho_entradas, 1):
        d = d + (2 * entradas[i])

    e = 2.0 * tamanho_entradas

    for i in range(0, tamanho_entradas, 1):
        f = f + (2 * (-1) * saidas[i])
    f = f * (-1)

    valor_coef_ang = (-(f * b) + (c * e)) / ((a * e) - (d * b))

    valor_coef_lin = ((f * a) - (c * d)) / ((a * e) - (d * b))

    return valor_coef_ang, valor_coef_lin
