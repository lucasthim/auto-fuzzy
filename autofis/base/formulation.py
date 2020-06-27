import numpy as np
from itertools import combinations, product, chain

from .filter.support_filter import support_basic_premises, support_premises_derived
from .filter.similarity_filter import similarity_basic_premises, similarity_derived_premises
from .filter.pcd_filter import pcd_basic_premises, pcd_derived_premises


class Formulation:
    def __init__(self, target_class, ux, attribute_is_binary, antecedents_by_attribute, num_of_antecedents_by_attribute, 
    enable_negation,t_norm, premise_max_size, criteria_support, threshold_support, enable_similarity_premises_bases, 
    enable_similarity_premises_derived, threshold_similarity, enable_pcd_premises_base, enable_pcd_premises_derived):

        # ------- Received parameters from Fuzzification --------------
        self.attribute_is_binary = attribute_is_binary
        self.negation_enabled = enable_negation

        self.ux = ux
        self.target_class = target_class
        self.tnorm = t_norm
        self.antecedents_by_attribute = antecedents_by_attribute
        self.num_of_antecedents_by_attribute = num_of_antecedents_by_attribute
        self.premise_max_size = premise_max_size 

        # Parameters of area filter
        self.criteria_support = criteria_support  # 'cardinalidade relativa', 'frequencia relativa'
        self.threshold_support = threshold_support 
        
        # Parameters of overlapping filter
        self.enable_similarity_premises_bases = enable_similarity_premises_bases
        self.enable_similarity_premises_derived = enable_similarity_premises_derived
        self.threshold_similarity = threshold_similarity
        
        # Parameters of PCD filter
        self.enable_pcd_premises_base = enable_pcd_premises_base
        self.enable_pcd_premises_derived = enable_pcd_premises_derived


    def generate_premises(self):
        """

        Generate premises of order 1 to m.
        
        Parameters:
            premise_max_size:  2,3,4,5
            tnorm:             'min' or 'prod'
            criteria:
            threshold_area
            enable_overlapping
            threshold_similarity
            enable_pcd:

        :return: premises and their corresponding uX; tree = [ [p1, uXp1], [p2, uXp2], ...[pm, uXpm]]
        """

        # Getting seed of premises
        ref_premises, basic_premises, u_base = self.get_basic_premises()  # [(0,1,2), (5,6), (9,11,13)]

        # Save single premises:
        single_premises = [(x,) for x in list(chain(*basic_premises))]  # [(0,), (1,), (2,), (5,), (6,), (9,), (11,), (13,)]
        tree = [[single_premises, u_base]]  # Seed

        # Generate premises of length 2 and their references
        ref_comb = list(combinations(ref_premises, 2))
        ref_double_premises = []
        double_premises = []
        for i in ref_comb:
            i2 = [basic_premises[ref_premises.index(j)] for j in i]  # [basic_premises[j] for j in i]
            double_premises_i = list(product(*i2))

            double_premises += double_premises_i
            ref_double_premises += [i] * len(double_premises_i)

        ref_double_premises_survivors, double_premises_survivors, ux_double_premises_survivors = self.premises_validation(ref_double_premises, double_premises)
        tree.append([double_premises_survivors, ux_double_premises_survivors])

        if self.premise_max_size > 2:
            p_i, ref_i = double_premises_survivors[:], ref_double_premises_survivors[:]
            for i in range(2, self.premise_max_size):
                ref_next, p_next = self.premises_next_level(basic_premises, ref_premises, p_i, ref_i)  # ---
                # new_premises, ref_next
                ref_pi_survivors, pi_survivors, ux_pi_survivors = self.premises_validation(ref_next, p_next)
                tree.append([pi_survivors, ux_pi_survivors])
                p_i, ref_i = p_next[:], ref_next[:]

        return tree


    def get_basic_premises(self):
        """
        
        Return only the premises (of order 1) that pass the overlapping and PCD filters.
        These will then generate more complex premises.
        return: [(0,1,2), (5,6), (9,11,13)]
        
        """

        attributes_num_list = range(0,len(self.antecedents_by_attribute))
        attributes_negation_mask =  np.invert(attribute_is_binary) if self.negation_enabled else len(self.antecedents_by_attribute) * [False]
        # Filter area
        aux = support_basic_premises(attributes_num_list, self.antecedents_by_attribute, self.num_of_antecedents_by_attribute, self.ux,
                                       self.criteria_support, self.threshold_support, attributes_negation_mask)
        new_ref_attribute, new_premises, new_num_premises_by_attrib, new_ux = aux

        # Filter PCD
        if self.enable_pcd_premises_base:
            aux1 = pcd_basic_premises(new_ref_attribute, new_premises, new_num_premises_by_attrib, new_ux, self.target_class)
            new_ref_attribute, new_premises, new_num_premises_by_attrib, new_ux = aux1

        # Filter overlapping
        if self.enable_similarity_premises_bases and len(new_num_premises_by_attrib) > 1:
            aux2 = similarity_basic_premises(new_ref_attribute, new_premises, new_num_premises_by_attrib,
                                              new_ux, self.threshold_similarity)
            new_ref_attribute, new_premises, new_num_premises_by_attrib, new_ux = aux2

        return new_ref_attribute, new_premises, new_ux  # [(0, 1), (3,), (7, 8)]

        
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
            if self.enable_pcd_premises_derived and ref_valid_premises:
                ref_valid_premises, valid_premises, new_ux = pcd_derived_premises(ref_valid_premises[:],
                                                                                  valid_premises, new_ux, self.target_class)

            # Overlapping validation
            if self.enable_similarity_premises_derived and len(ref_valid_premises) != 0:
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
