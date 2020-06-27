__author__ = 'jparedes'


from itertools import compress, chain
from numpy import tile, spacing, array
from .similarity_filter import splitListFromSizes


def get_CD(P, C):
    """
    Calculation of Confidence Degree
    :param P: Matrix of uX
    :param C: Binary classes
    :return: CD matrix <[number of premises, number of classes]>
    """
    conf_deg = P.T.dot(C)  # np.dot(P.T,C)
    a1 = P.sum(axis=0)
    a2 = a1.reshape(P.shape[1], 1)  # = a1'
    B = tile(a2, (1, conf_deg.shape[1]))
    CD = 1. * conf_deg / (B + spacing(0))
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
    B = tile(a1, (C.shape[1], 1))
    CD_t = 1. * conf_deg_t / (B + spacing(0))
    return CD_t


def pcd_basic_premises(ref_attribute, p, np_xA, ux, cbin):
    """
    PCD to evaluate premises in the seed process
    :param p: [(0,3) (5,6,8,9)]
    :param np_xA: [2, 4]
    :param ux: [0.2, 0.5 0.01 0.20 0.7 0.39;
                0.1  0.3 0.08 0.09 0.6 0.29]
    :param Cbin: Binary output
    :return: [(0,3), (5,8,9)]
             [2, 3]
             new_ux
    """
    PCD_t = calculate_cd_transpose(ux, cbin)
    logica = 2 * PCD_t.max(axis=0) - PCD_t.sum(axis=0)
    PCD_t[:, logica < 0] = 0
    # Binarization of PCD
    max_def = PCD_t / (PCD_t.max(axis=0) + spacing(0))
    max_def[max_def < 1] = 0
    bool_ux = max_def.sum(axis=0) != 0  # array([True, True, ...., False, True])

    bool_ux_number = 1 * bool_ux  # array([1, 1, ...., 0, 1])
    bool_premises = splitListFromSizes(bool_ux_number.tolist(), np_xA)  # [[1,1,1], [1,1], [1,1,0,1]]

    dupla = list(zip(p, bool_premises))  # [([1, 1, 0], (0, 1, 2)), .....]
    new_premises = []
    new_num_prem_by_attrib = []
    new_ref_attribute = []
    for i in range(len(dupla)):
        if sum(dupla[i][1]) != 0:  # verifica que al menos una premisa de un atributo sobrevivio
            aux = tuple(compress(*dupla[i]))
            new_premises.append(aux)
            new_num_prem_by_attrib.append(len(aux))
            new_ref_attribute.append(ref_attribute[i])

    new_ux = ux[:, bool_ux]  # list(chain(*new_p))
    return new_ref_attribute, new_premises, new_num_prem_by_attrib, new_ux


def pcd_derived_premises(ref_comb, premises, ux, cbin):
    """
    PCD to evaluate premises in the seed process
    :param ref_comb: [(0,1), (0,1), (1,2)]
    :param premises: [(0,5), (1,7), (6,9)]
    :param ux: [0.2, 0.01 0.39;
                0.1  0.08 0.29]
    :param cbin: Binary output
    :return: [(0,1), (1,2)] --> new_ref_comb
             [(0,5), (6,9)] --> new_premises
             new_ux
    """
    pcd_t = calculate_cd_transpose(ux, cbin)
    logica = 2 * pcd_t.max(axis=0) - pcd_t.sum(axis=0)
    pcd_t[:, logica < 0] = 0
    # Binarization of PCD
    max_def = pcd_t / (pcd_t.max(axis=0) + spacing(0))
    max_def[max_def < 1] = 0
    bool_ux = max_def.sum(axis=0) != 0  # array([True, True, ...., False, True])

    bool_ux_number = 1 * bool_ux  # array([1, 1, ...., 0, 1])
    bool_premises = bool_ux_number.tolist()  # [1, 1, ...., 0, 1]

    new_premises = list(compress(premises, bool_premises))
    new_ref_comb = list(compress(ref_comb, bool_premises))

    new_ux = ux[:, bool_ux]
    return new_ref_comb, new_premises, new_ux


def main():

    ind = [(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,)]
    A = array([[0.20, 0.70, 0.1900, 0.199, 0.18, 0.200, 0.65, 0.25, 0.10],
               [0.50, 0.40, 0.5010, 0.490, 0.50, 0.470, 0.39, 0.42, 0.19],
               [0.05, 0.40, 0.0505, 0.048, 0.05, 0.050, 0.42, 0.23, 0.35],
               [0.01, 0.87, 0.0060, 0.009, 0.02, 0.018, 0.88, 0.11, 0.01]])
    cbin = array ([[1, 0, 0],
                   [0, 0, 1],
                   [0, 1, 0],
                   [1, 0, 0]])
    x = pcd_f2(ind, A, cbin)
    print (x[0])
    print (x[1])

    # #####################################
    ind1 = [(0, 1, 2), (3, 4), (5, 6, 7, 8)]
    x1 = pcd_f1(ind1, [3, 2, 4], A, cbin)
    print (x1[0])
    print (x1[1])
    print (x1[2])


if __name__ == '__main__':
    main()