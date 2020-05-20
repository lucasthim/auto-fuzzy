__author__ = 'jparedes'

import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, csc_matrix
from scipy.sparse.linalg import cgs
import timeit


def main():
    """
    try:
        for i in range(5):
            if i == 3:
                raise ValueError("CV error " + str(i))
            print 'ojala', i
    except ValueError as e:
        print e
    """
    A = np.array([[1,2,0],[0,0,3],[1,0,4]])
    spA = csr_matrix(A)
    spB = spA * spA

    B = spB.todense()
    print B
    print B.shape

    # data = sp.concatenate(spA.data,spB.data) # arreglar aqui
    # rows = sp.concatenate((spA.row,spB.row))
    # cols = sp.concatenate((spA.col,spB.col))
    # nueva = sparse.coo_matrix((data,(rows,cols)), shape=(3,6))
    # print nueva.todense()

    #########

    n = 1250
    A1 = np.random.rand(n,n)
    B1 = np.random.rand(n,1)

    index = A1 < 0.9
    A1[index] = 0
    B1[B1<0.98]=0

    sp_A1 = csr_matrix(A1) # coo_matrix(A1)
    #print sp_A1

    t0 = timeit.default_timer()
    x = cgs(sp_A1,B1)
    tf = timeit.default_timer()

    x = cgs(A1,B1)
    tf1 = timeit.default_timer()

    print 'modo sparse(seg)', tf- t0
    print 'modo normal(seg)', tf1- tf

    #########

    # http://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html#scipy.sparse.csr_matrix
    # concatenate: http://stackoverflow.com/questions/6844998/is-there-an-efficient-way-of-concatenating-scipy-sparse-matrices

    ## implementar get_sol de auxfunc.py en modo sparse_matrix


'''
    for j in range(2):

        lista = [[(15,), (7, 8, 9)], [(6, 9), (11, 13)]]

        try:
            # Division
            for i in lista:
                if not i:  # Si hay algun elemento vacio
                    raise ValueError("Error in Module Division. Some classes did not get premises. "
                                     "Sorry, you can not continue in the next step."
                                     "\nTry to change the configuration")
            print "<Module Division was successful>"
            print "Premisas obetenidas: ", lista

            # Aggregation
            lista2 = [[(5,), (8, 11)], []]
            for i in lista2:
                if not i:
                    raise ValueError("Error in Module Aggregation. Some classes did not get premises. "
                                     "Sorry, you can not continue in the next step."
                                     "\nTry to change the configuration")
            print "<Module Aggregation was successful>"
            print lista2

            # -------------------------------------------

        except ValueError as e:
            print e

        print "------------------------"
'''


if __name__ == '__main__':
    main()