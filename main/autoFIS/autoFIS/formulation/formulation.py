__author__ = 'jparedes'

import numpy as np
from itertools import combinations, product, chain

from .support_filter import support_basic_premises, support_premises_derived
from .similarity_filter import similarity_basic_premises, similarity_derived_premises
from .pcd_filter import pcd_basic_premises, pcd_derived_premises


class Formulation:
    def __init__(self, ux, c_bin, ref_attributes, p_by_attribute, np_by_attribute, attributes_contain_negation):
        
        # ------- Received parameters from Fuzzification --------------
        self.ux = ux
        self.c_bin = c_bin
        self.tnorm = []
        self.np_by_attribute = np_by_attribute
        self.p_by_attribute = p_by_attribute
        self.attributes_contain_negation = attributes_contain_negation
        self.ref_attributes = ref_attributes
        
        # ------- Parameters given by user -----------------------
        self.premise_max_size = []  # premise_max_size
        # Parameters of area filter
        self.criteria_support = []  # 'cardinalidade relativa', 'frequencia relativa'
        self.threshold_support = []  # tolerancia da area
        # Parameters of overlapping filter
        self.isEnableSimilarityPremisesBase = []
        self.isEnableSimilarityPremisesDerived = []
        self.threshold_similarity = []
        # Parameters of PCD filter
        self.isEnablePCDpremisesBase = []
        self.isEnablePCDpremisesDerived = []

    def load_filter_parameters(self, par_area, par_overlapping, par_pcd):
        # Load area parameters
        self.criteria_support = par_area[0]
        self.threshold_support = par_area[1]
        # Load overlapping parameters
        self.isEnableSimilarityPremisesBase = par_overlapping[0][0]
        self.isEnableSimilarityPremisesDerived = par_overlapping[0][1]
        self.threshold_similarity = par_overlapping[1]
        # Load logic enable of PCD
        self.isEnablePCDpremisesBase = par_pcd[0]
        self.isEnablePCDpremisesDerived = par_pcd[1]

    def get_basic_premises(self):
        """
        Se obtiene las premisas de orden 1 que han de generar posteriomente las premisas compuestas
        :return: premisas sobrevivientes agrupadas en los atributos que pertenecen
                [(0,1,2), (5,6), (9,11,13)]
        """

        # Filter area
        aux = support_basic_premises(self.ref_attributes, self.p_by_attribute, self.np_by_attribute, self.ux,
                                       self.criteria_support, self.threshold_support, self.attributes_contain_negation)
        new_ref_attribute, new_premises, new_num_premises_by_attrib, new_ux = aux

        # Filter PCD
        if self.isEnablePCDpremisesBase:
            aux1 = pcd_basic_premises(new_ref_attribute, new_premises, new_num_premises_by_attrib, new_ux, self.c_bin)
            new_ref_attribute, new_premises, new_num_premises_by_attrib, new_ux = aux1

        # Filter overlapping
        if self.isEnableSimilarityPremisesBase and len(new_num_premises_by_attrib) > 1:
            aux2 = similarity_basic_premises(new_ref_attribute, new_premises, new_num_premises_by_attrib,
                                              new_ux, self.threshold_similarity)
            new_ref_attribute, new_premises, new_num_premises_by_attrib, new_ux = aux2

        return new_ref_attribute, new_premises, new_ux  # [(0, 1), (3,), (7, 8)]

    def generate_premises(self, premise_max_size, tnorm, par_area, par_overlapping, par_pcd):
        """
        Generate premises of order 1 to m.
        :param premise_max_size:  2,3,4,5
        :param tnorm:           'min' or 'prod'
        :param par_area:        [criteria, threshold_area]
        :param par_overlapping: [isEnableOverlapping, threshold_similarity]
        :param par_pcd:         [isEnablePCD]
        :return: premisas y sus correspondientes uX
                arbol = [ [p1, uXp1], [p2, uXp2], ...[pm, uXpm]]
        """
        self.premise_max_size = premise_max_size  # 1,2,3,4,5....
        self.tnorm = tnorm
        self.load_filter_parameters(par_area, par_overlapping, par_pcd)

        # Getting seed of premises
        ref_premises, basic_premises, u_base = self.get_basic_premises()  # [(0,1,2), (5,6), (9,11,13)]

        # Save single premises:
        p1 = [(x,) for x in list(chain(*basic_premises))]  # [(0,), (1,), (2,), (5,), (6,), (9,), (11,), (13,)]
        arbol = [[p1, u_base]]  # Seed

        # Generate premises of length 2 and their references
        ref_comb = list(combinations(ref_premises, 2))
        ref_p2 = []
        p2 = []
        for i in ref_comb:
            i2 = [basic_premises[ref_premises.index(j)] for j in i]  # [basic_premises[j] for j in i]
            p2_i = list(product(*i2))

            p2 += p2_i
            ref_p2 += [i] * len(p2_i)

        ref_p2_survivors, p2_survivors, ux_p2_survivors = self.premises_validation(ref_p2, p2)
        arbol.append([p2_survivors, ux_p2_survivors])

        if self.premise_max_size > 2:
            p_i, ref_i = p2_survivors[:], ref_p2_survivors[:]
            for i in range(2, self.premise_max_size):
                ref_next, p_next = self.premises_next_level(basic_premises, ref_premises, p_i, ref_i)  # ---
                # new_premises, ref_next
                ref_pi_survivors, pi_survivors, ux_pi_survivors = self.premises_validation(ref_next, p_next)
                arbol.append([pi_survivors, ux_pi_survivors])
                p_i, ref_i = p_next[:], ref_next[:]

        return arbol

    def premises_validation(self, ref_premises, premises):
        ref_valid_premises = []
        valid_premises = []
        new_ux_prev = []
        number_rows = self.ux.shape[0]

        for i in range(len(premises)):
            # Building the new premise
            temp = self.ux[:, premises[i]]
            if self.tnorm == 'prod':
                temp2 = temp.prod(axis=1)
            else:  # 'min'
                temp2 = temp.min(axis=1)

            # Area validation
            if support_premises_derived(temp2, self.threshold_support, self.criteria_support):
                new_ux_prev.append(temp2.reshape(number_rows, 1))
                valid_premises.append(premises[i])
                ref_valid_premises.append(ref_premises[i])

        if valid_premises:
            new_ux = np.hstack(new_ux_prev)  # np.array(new_ux)

            # PCD validation
            if self.isEnablePCDpremisesDerived and ref_valid_premises:
                ref_valid_premises, valid_premises, new_ux = pcd_derived_premises(ref_valid_premises[:],
                                                                                  valid_premises, new_ux, self.c_bin)

            # Overlapping validation
            if self.isEnableSimilarityPremisesDerived and len(ref_valid_premises) != 0:
                aux = similarity_derived_premises(ref_valid_premises[:], valid_premises,
                                                   new_ux, self.threshold_similarity)
                ref_valid_premises, valid_premises, new_ux = aux
        else:
            new_ux = []

        return ref_valid_premises, valid_premises, new_ux

    @staticmethod
    def premises_next_level(p1, ref_p1, pn, ref_pn):
        next_premises = []
        ref_next_premises = []
        for i in range(len(pn)):
            atributos_x_misturar = set(ref_p1) - set(ref_pn[i])
            for j in atributos_x_misturar:
                prem = p1[ref_p1.index(j)]
                for k in prem:
                    bar = pn[i]
                    candidate = tuple(sorted(bar + (k,)))
                    if candidate not in next_premises:
                        next_premises.append(candidate)
                        ref_next_premises.append(ref_pn[i] + (j,))  # tuple(sorted(apn[i] + (j,)))
        return ref_next_premises, next_premises  # next_premises, ref_next_premises ---


def main():
    print ("Module 3 <<Formulation>>")
    print ('==========')


if __name__ == '__main__':
    main()
