__author__ = 'jparedes'

# -*- coding: utf-8 -*-
from numpy import tile, spacing, zeros, concatenate
from itertools import compress

from .auxfunc import iter_beta

class Association:
    def __init__(self, tree, target_class_one_hot):
        self.tree = tree
        self.target_class_one_hot = target_class_one_hot

        self.association_method_mapper = {"CD": self.div_CD,
                                 "PCD": self.div_PCD,
                                 "MQR": self.div_MQR,
                                 "PMQR": self.div_PMQR,
                                 "freq": self.div_freqmax}

    def build_association_rules(self, method='MQR'):
        
        # Ordenando para tener la siguiente estructura:
        # [[premisas_p1 clase1], [premisas_p2 clase2], ....]
        premises_u_by_size = []  # agrupa premisas por tamanho, en cada tamanho estan contenidas por clase
        for i in range(len(self.tree)):
            premises_u_by_size.append(self.get_partial_premises(self.tree[i], self.target_class_one_hot, method))

        premises_u_by_class = []
        for i in range(self.target_class_one_hot.shape[1]):
            premises_class_i = []
            M_clase = []
            for j in range(len(premises_u_by_size)):
                premises_class_i = premises_class_i + (premises_u_by_size[j][i][0])
                M_clase.append(premises_u_by_size[j][i][1])
            premises_u_by_class.append([premises_class_i, concatenate(M_clase, axis=1)])

        return premises_u_by_class

    @staticmethod
    def calculate_cd_transpose(P, C):

        """
        Calculation of Confidence Degree, this function was defined to reduce
        some operations with respect of get_CD
        :param P: Matrix of uX
        :param C: Binary classe
        :return: CD matrix transpose <[number of classes, number of premises]>
        """

        conf_deg_t = C.T.dot(P)
        a1 = P.sum(axis=0)
        B = tile(a1, (C.shape[1], 1))
        cd_t = 1. * conf_deg_t / (B + spacing(0))
        return cd_t

    def div_CD(self, u_premises, c_bin):
        CD_t = self.calculate_cd_transpose(u_premises, c_bin)
        # Binarization of CD
        max_def = CD_t / (CD_t.max(axis=0) + spacing(0))
        max_def[max_def < 1] = 0
        # Filtro para caso de empate o seja 1 premisa que se activa para mais de uma classe.
        # tal vez index_conflito.sum > 0:
        conflict_index = max_def.sum(axis=0) > 1
        max_def[:, conflict_index] = 0
        return max_def.T

    def div_PCD(self, P, C):
        PCD_t = self.calculate_cd_transpose(P, C)
        logic = 2 * PCD_t.max(axis=0) - PCD_t.sum(axis=0)
        PCD_t[:, logic < 0] = 0
        # Binarization of PCD
        max_def = PCD_t / (PCD_t.max(axis=0) + spacing(0))
        max_def[max_def < 1] = 0
        return max_def.T

    @staticmethod
    def div_MQR(u_premises, c_bin):
        number_of_premises = u_premises.shape[1]  # numero de premisas asociadas al orden de la matriz P(u_premises)
        number_of_classes = c_bin.shape[1]
        betas = zeros((number_of_premises, number_of_classes))
        for j in range(number_of_classes):
            betas[:, [j]] = iter_beta(u_premises, c_bin[:, [j]])
        return betas

    def div_PMQR(self, P, C):
        PMQR = self.div_MQR(P, C)
        logic = 2 * PMQR.max(axis=1) - PMQR.sum(axis=1)
        indexes = logic < 0
        PMQR[indexes, :] = 0
        return PMQR

    def div_freqmax(self, P, C):
        freq = P > 0
        return self.div_CD(freq, C)

    def get_partial_premises(self, tree_i, c_bin, method):
        premises, u_premises = tree_i

        betas_selectors = self.association_method_mapper[method](u_premises, c_bin)
        premises_and_grouped_membership_premises = []

        for i in range(betas_selectors.shape[1]):
            enables_premises = betas_selectors[:, i] > 0
            new_premises = list(compress(premises, enables_premises))
            premises_and_grouped_membership_premises.append([new_premises, u_premises[:, enables_premises]])
        return premises_and_grouped_membership_premises


def main():
    print ('Module 4 <<Splitting>>')

if __name__ == '__main__':
    main()


# ?????????? Paredes comentou isso
# Revisar freq_max: Me parece raro
# freq = P>0
# self.div_CV(freq, C)