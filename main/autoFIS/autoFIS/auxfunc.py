__author__ = 'jparedes'

import numpy as np
from scipy.sparse.linalg import cgs
from scipy.sparse import csr_matrix
from itertools import combinations, chain


def get_sol(P, indexes, c):
    R = np.ones((1, P.shape[1]))  # equality restrictions
    xm = P[:, indexes]

    # Building A
    a11 = xm.T.dot(xm)  # np.dot(xm.transpose(), xm)
    a21 = R[:, indexes]
    a12 = a21.transpose()
    a22 = np.zeros((1, 1))

    af1 = np.concatenate((a11, a12), axis=1)
    af2 = np.concatenate((a21, a22), axis=1)
    A = np.concatenate((af1, af2), axis=0)

    # Building B
    b11 = xm.T.dot(c)  # np.dot(xm.transpose(), c)
    b12 = np.ones((1, 1))  # a11.shape[0]+a21.shape[0]-b11.shape[0], b11.shape[1]))
    B = np.concatenate((b11, b12), axis=0)

    A[A<0.01] = 0
    spA = csr_matrix(A)

    xs = cgs(spA, B, maxiter=200)
    xs = xs[0]  # getting solutions
    xs = xs.reshape(len(xs), 1)
    xs = xs[0:-1, :]  # deleting lagangre multiplicator

    return xs


def iter_beta(P, c):
    # Calculating solutions of beta_vector using Minimos Quadrados Restrictos
    indexes_0 = np.array([True] * P.shape[1])
    beta = get_sol(P, indexes_0, c)

    while (sum(beta < 0) != 0):  # when a negative solution exits
        beta[np.where(beta < 0)[0]] = 0  # se pone a cero las soluciones negativas
        ind = (beta > 0).transpose()[0]
        xs = get_sol(P, ind, c)  # calcula soluciones referentes para los que index positivos del paso anterior
        beta[ind, :] = xs

    return beta


def get_CD(P, C):
    conf_deg = P.T.dot(C)  # np.dot(P.T,C)
    a1 = P.sum(axis=0)  # np.sum(P, axis=0) # suma x columnas de P: output 1 fila
    a2 = a1.reshape(P.shape[1], 1)  # = a1'
    B = np.tile(a2, (1, conf_deg.shape[1]))
    CD = 1. * conf_deg / (B + np.spacing(0))
    return CD


def calculate_cd_transpose(P,C):
    """
    Calculation of Confidence Degree, this function was defined to reduce
    some operations with respect of get_CD
    :param P: Matrix of uX
    :param C: Binary classes
    :return: CD matrix transpose <[number of classes, number of premises]>
    """
    conf_deg_t = C.T.dot(P)
    a1 = P.sum(axis=0)
    B = np.tile(a1, (C.shape[1], 1))
    CD_t = 1. * conf_deg_t / (B + np.spacing(0))
    return CD_t

def list_duplicates(seq):
    # http://stackoverflow.com/questions/23645433/list-indexes-of-duplicate-values-in-a-list-with-python
    seen = set()
    seen_add = seen.add
    return [idx for idx, item in enumerate(seq) if item in seen or seen_add(item)]


def int_prem_nivel_m(ind_P, P, m, tnorm='prod'):
    # construyendo la nueva P:
    head_associates = ind_P
    nA = P.shape[1]

    l1 = []
    l2 = []

    for i in combinations(range(nA), m):
        arrays = [head_associates[item] for item in i]
        l1.append(tuple(chain(*arrays)))  # mapea desde el origen
        l2.append(i)
        ##  Falta implementar deteccion de repeticion
        # premisas:  [(28,), (36,), (8, 20), (27, 37), (8, 27), (27, 37)]
        # pesos [[ 0.14446904  0.29625484  0.03160199  0.00291711  0.52183991  0.00291711]]

    ind2remove = list_duplicates(l1)
    # print "jejeje:D", ind2remove ## repeticion ocasional

    for j in sorted(ind2remove, reverse=True):
        del l1[j]
        del l2[j]

    A = np.zeros((P.shape[0], len(l2)))

    for k in range(len(l2)):
        temp = P[:, l2[k]]
        if tnorm == 'prod':
            temp2 = temp.prod(axis=1)  # np.prod(temp, axis=1)
        else:  # 'min'
            temp2 = temp.min(axis=1)  # np.min(temp, axis=1)
        A[:, k] = temp2

    return l1, A


def int_premises(origin_P, P, m, t_norm='prod'):
    new_indP = []
    new_P = []
    for i in range(1, m + 1):
        objetos = int_prem_nivel_m(origin_P, P, i, t_norm)
        new_indP.append(objetos[0])
        new_P.append(objetos[1])

    nuevo_indP = list(chain(*new_indP))
    ind2remove = list_duplicates(nuevo_indP)

    for j in sorted(ind2remove, reverse=True):
        del nuevo_indP[j]

    BB = np.delete(np.hstack(new_P), ind2remove, 1)  # delete repeated columns

    return nuevo_indP, BB  # list(chain(*new_indP)), np.hstack(new_P)


def dummies2int(cBin):
    """
    Conversion of Binary Classes to Integers Classes
    :param cBin:  [1 0 0] ==> 1
    (numpy array) [0 0 1] ==> 3
                  [0 1 0] ==> 2
    :return: [1,3,2]
    """
    c = []
    for i in cBin.tolist():
        c.append(i.index(1) + 1)
    return c


'''
indice_P = [(18,), (19,), (2, 13)]
matrix_P = np.array([[0.1, 0.2, 0.4], [0.4, 0.5, 0.2]])

#print int_premisas(indice_P, matrix_P, 1)
print int_premisas(indice_P, matrix_P, 2)
#print int_premisas(indice_P, matrix_P, 3)
'''

'''
Por si acaso:
prediction = [[1,0,1],[0,1,1]]
for i in prediction:
    print int("".join(map(str, i)),2)
salida: [[5],[3]]
'''

'''
A = np.array([[ 0.88525932,  0.11474068], [0.17474463, 0.82525537],
              [0.75132891, 0.24867109],[0,0]])
print 'A:\n', A
B = (1. * A.T)/  (np.max(A.T,0) + np.spacing(0))
B[B<1] = 0
print 'win:\n',B.T
'''


def main():
    P = np.random.rand(5, 3)
    C = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1]])
    print (get_CD(P, C))


if __name__ == '__main__':
    main()