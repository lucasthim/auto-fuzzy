__author__ = 'jparedes'

import numpy as np
from itertools import compress


def similarity(b, An):
    """
    Se calcula la similaridad de un vector columna con respecto a las columnas de una matriz
    :param b: premissa de un atributo
    :param An: premissas de atributos diferentes a b
    :return: lista de simidaridades [0.2 0.9 0.12 .....]1xAn.shape[1]
    """
    B = np.tile(b, (1, An.shape[1]))

    dif = An - b
    ind_min = dif < 0
    ind_max = dif > 0

    An1, An2 = An.copy(), An.copy()
    An1[ind_min] = B[ind_min]  # resultado de minimos
    An2[ind_max] = B[ind_max]  # resultado de maximos

    s = An2.sum(axis=0) / (An1.sum(axis=0) + np.spacing(0))
    return s


def splitListFromSizes(lista, sizes):
    """
    Agrupa elementos de una lista acorde a unos tamanhos dados
    :param lista:   [8, 9, 2, 7, 5]
    :param sizes:   [2, 3]
    :return:        [[8,9], [2,7,5]]
    """
    sl = []
    ref = 0
    for i in sizes:
        sl.append(lista[ref:ref + i])
        ref += i
    return sl


def calculateNewSizes(mask, sizes):
    """
    From a binary list are filtered premises of each attribute
    :param mask:    [0,1,1,0,1]
    :param sizes:   [2,3]
    :return:        [1,2]
    """
    l = splitListFromSizes(mask, sizes)  # [[0,1],[1,0,1]]
    return [sum(i) for i in l]           # [1, 2]


def XOR(v1, v2):
    """
    XOR operation element by element from 2 lists
    :param v1: [1, 0, 1, 0, 0, 1]
    :param v2: [1, 1, 0, 0, 1, 1]
    :return:   [0, 1, 1, 0, 1, 0]
    """
    return [a ^ b for a, b in zip(v1, v2)]


def compressPremises(premises_by_attribute, list_filters):
    """
    :param premises_by_attribute: [(2, 4, 6, 7), (13, 15)]
    :param list_filters:          [[1, 1, 1, 0], [1, 1]]
    :return:                      [(2,4,6), (13,15)]
    """
    new_premises_by_attribute = []
    for i in range(len(list_filters)):
        new_premises_by_attribute.append(tuple(compress(premises_by_attribute[i], list_filters[i])))
    return new_premises_by_attribute


def similarity_basic_premises(ref_attrib, premises_by_attrib, num_prem_by_attrib, ux, threshold_similarity=0.95):
    """
    Eliminacion de premisas altamente correlacionadas de atributos diferentes,
    en caso se detecte premisas redundantes se mantienen las de mayor area.
    :param ref_attrib: [0, 1, 2]
    :param premises_by_attrib: [(0, 1), (2, 3, 4, 5), (6, 7, 8)]
    :param num_prem_by_attrib: [2,4,3]
    :param ux:  Matriz correspondiente a las premisas de los atributos fuzificados
    :return:     premisas sobrevivientes de cada atributo
                    [(0, 1), (7, 8)]
    """
    ind_A, A, ref_attrib_A = premises_by_attrib, ux, ref_attrib
    j = 0  # Referencia para atributo cero

    new_premises = []
    new_ux = []
    new_ref_attrib = []
    while len(ind_A) > 1:
        # Split A into B and A(A-B)
        B = A[:, 0:num_prem_by_attrib[j]]  # A[:,0:3] --> columns 0,1,2
        A = A[:, num_prem_by_attrib[j]:]  # A[:,3:]

        p_B = ind_A.pop(0)  # p_B = ind_A[0] | # ind_A = ind_A[1:] .... indA[:,3:]
        ref_B = ref_attrib_A.pop(0)
        r = A.shape[1] * [1]
        p_sub = []

        for k in range(B.shape[1]):
            b = B[:, [k]]  # column reference 0
            s = similarity(b, A)  # [0.3 0.4 0.96 0.955 .....]
            suspects = A[:, np.array(s > threshold_similarity)]

            if suspects.shape[1] != 0:
                aux = b.sum(axis=0) > suspects.sum(axis=0)  # area comparison between column reference and suspects
                if aux.all():  # suspects have a minor area that reference premise
                    p_sub.append(p_B[k])
                    new_ux.append(b)
                    r = XOR(r, s > 0.95)  # [1,1,1,0,1,1]
                    r_tam = splitListFromSizes(r, num_prem_by_attrib[j + 1:])  # [ [0,0,0,0], [1,1,1], [0,0] ]

                    log_r_tam = [1 if sum(d) != 0 else 0 for d in r_tam]  # [0,1,0]

                    if sum(log_r_tam) == len(log_r_tam):  # Any attribute died
                        num_prem_by_attrib[j + 1:] = calculateNewSizes(r, num_prem_by_attrib[j + 1:])  # [3,2]
                        ind_A = compressPremises(ind_A, r_tam)
                        ref_attrib_A = list(compress(ref_attrib_A, log_r_tam))

                    else:  # At least 1 attribute died
                        aux1 = num_prem_by_attrib[:j + 1]
                        aux2 = list(compress(num_prem_by_attrib[j + 1:], log_r_tam))  # compress([4,3,2], [0,1,0])
                        num_prem_by_attrib = aux1 + aux2
                        ind_A = list(compress(ind_A, log_r_tam))
                        ref_attrib_A = list(compress(ref_attrib_A, log_r_tam))

                # else: #print 'se elimina la referencia'

                R = [True if k2 == 1 else False for k2 in r]
                A = A[:, np.array(R)]
                r = A.shape[1] * [1]
            else:
                p_sub.append(p_B[k])
                new_ux.append(b)

        if len(p_sub) > 0:
            new_premises.append(tuple(p_sub))
            new_ref_attrib.append(ref_B)
        j += 1

    if len(ind_A) != 0:
        new_ux.append(A)
        new_premises.append(ind_A[0])
        new_ref_attrib.append(ref_attrib_A[0])
    new_num_prem_by_attrib = [len(k) for k in new_premises]

    return new_ref_attrib, new_premises, new_num_prem_by_attrib, np.hstack(new_ux)


def similarity_derived_premises(ref_comb_attrib, premises, ux, threshold_similarity=0.95):
    A, ind_A, ref_combA = ux, premises, ref_comb_attrib

    new_ref_comb = []
    new_premises = []
    new_ux = []
    cont = len(premises)

    while cont > 1:  # cont != 1 and cont!=0
        b = A[:, [0]]
        A = A[:, 1:]
        ref = ind_A.pop(0)  # ind_A = ind_A[1:]
        ref_comb = ref_combA.pop(0)

        r = A.shape[1] * [1]

        s = similarity(b, A)  # [0.2, 0.97, 0.36]
        suspects = A[:, np.array(s > threshold_similarity)]

        if suspects.shape[1] != 0:  # Hay al menos un sospechoso
            aux = b.sum(axis=0) >= suspects.sum(axis=0)  # <cardinalidade> comparison between reference and suspects
            if aux.all():  # suspects are deleted and reference is saved
                new_premises.append(ref)
                new_ref_comb.append(ref_comb)
                new_ux.append(b)
                r = XOR(r, s > 0.95)  # [1,1,1,0,1,1]
                # R = [True if k2 == 1 else False for k2 in r]  # [True, True, True, False, True, True]
                A = A[:, np.array(r) == 1]
                ind_A = list(compress(ind_A, r))
                ref_combA = list(compress(ref_combA, r))
            # else: # reference is not considered, es decir, no se hace nada
        else:
            new_premises.append(ref)
            new_ref_comb.append(ref_comb)
            new_ux.append(b)

        cont = A.shape[1]  # con len(ind_A)

    if len(ind_A) != 0:
        new_premises.append(ind_A[0])
        new_ref_comb.append(ref_combA[0])
        new_ux.append(A)

    return new_ref_comb, new_premises, np.hstack(new_ux)


def main():
    ao = np.array([[0.20, 0.70, 0.1900, 0.199, 0.18, 0.200, 0.65, 0.25, 0.10],
                   [0.50, 0.40, 0.5010, 0.490, 0.50, 0.470, 0.39, 0.42, 0.19],
                   [0.05, 0.40, 0.0505, 0.048, 0.05, 0.050, 0.42, 0.23, 0.35],
                   [0.01, 0.87, 0.0060, 0.009, 0.02, 0.018, 0.88, 0.11, 0.01]])

    print ('1:', similarity(ao[:, [0]], ao))
    print (ao.sum(axis=0))
    print ('2:', similarity(ao[:, [1]], ao))
    print ('3:', similarity(ao[:, [2]], ao))

    ref_ao = [0, 1, 2]
    ind_ao = [(0, 1, 2,), (3, 4, 5,), (6, 7, 8,)]
    aux = similarity_basic_premises(ref_ao, ind_ao, [3, 3, 3], ao)  # (ind_Ao, [3, 3, 3], Ao)
    print (aux[0])
    print (aux[1])
    print (aux[2])

    print (5 * '----')

    # A = np.array([[0.20, 0.70, 0.1900, 0.199, 0.18, 0.200, 0.65, 0.25, 0.10],
    #               [0.50, 0.40, 0.5010, 0.490, 0.50, 0.470, 0.39, 0.42, 0.19],
    #               [0.05, 0.40, 0.0505, 0.048, 0.05, 0.050, 0.42, 0.23, 0.35],
    #               [0.01, 0.87, 0.0060, 0.009, 0.02, 0.018, 0.88, 0.11, 0.01]])
    ref_a1 = [(0, 1), (0, 1), (0, 1), (1, 2), (1, 2), (1, 2), (0, 2), (0, 2), (0, 2)]
    ind_a1 = [(0, 3), (1, 4), (2, 5), (3, 7), (4, 8), (5, 8), (0, 6), (2, 7), (2, 8)]
    aux2 = similarity_derived_premises(ref_a1, ind_a1, ao)
    print ('7:', similarity(ao[:, [7]], ao))
    print (aux2[0])
    print (aux2[1])
    print (aux2[2])

if __name__ == '__main__':
    main()