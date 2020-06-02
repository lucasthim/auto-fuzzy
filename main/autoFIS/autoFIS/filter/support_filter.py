__author__ = 'Jorge Paredes, Adriano Koshiyama'

import numpy as np  # creo no es necesario en casos como np.sum(x)
from itertools import compress, chain


def support_by_attribute(u_attribute, perc_area=0.1, criteria='cardinalidade_relativa'):
    # Se evalua no un vector sino una matriz o sea varias columnas
    # u_attribute es un numpy
    if criteria == 'cardinalidade_relativa':
        logica = u_attribute.sum(axis=0) > u_attribute.shape[0] * perc_area
    else:  # criteria == 'frequencia_relativa'
        aux = u_attribute > 0
        logica = aux.sum(axis=0) > aux.shape[0] * perc_area
    return logica.tolist()


def support_negated_attribute(u_attribute, perc_area=0.1, criteria='cardinalidade_relativa'):
    # Se evalua no un vector sino una matriz o sea varias columnas
    # u_attribute: numpy, not included part negated

    val_min = u_attribute.shape[0] * perc_area
    val_max = u_attribute.shape[0] * (1 - perc_area)
    num_membership_functions = u_attribute.shape[1]
    logica = 2 * num_membership_functions * [0]

    if criteria == 'cardinalidade_relativa':
        support_premises = u_attribute.sum(axis=0)
    else:  # criteria == 'frequencia_relativa'
        aux = u_attribute > 0
        support_premises = aux.sum(axis=0)
    for i in range(len(support_premises)):
        support_premise = support_premises[i]
        if val_min < support_premise < val_max:
            logica[i] = 1
            logica[i + num_membership_functions] = 1
        else:
            if val_min < support_premise:
                logica[i] = 1
            else:
                logica[i + num_membership_functions] = 1
    return logica


def support_premises_base_without_neg(ref_attrib, premises, num_premises_by_attrib, ux, criteria, tolerance):
    new_ref_attribute = []
    new_premises = []
    new_num_premises_by_attrib = []
    index_prev_ux = []
    ref = 0
    for j in range(len(num_premises_by_attrib)):  # len(np_xA) = numero de atributos # Evaluate each attribute
        tam = num_premises_by_attrib[j]
        attribute_ux = ux[:, ref:ref+tam]
        ref += tam
        # indicator: indexes which indicates who membership functions survived. Example: [1, 0, 1, ...]
        indexes_survivors = support_by_attribute(attribute_ux, tolerance, criteria)
        num_survivors = sum(indexes_survivors)  # num_survivors = indicator.sum(axis=0)
        index_prev_ux.append(indexes_survivors)
        if num_survivors != 0:
            new_num_premises_by_attrib.append(num_survivors)
            new_premises.append(tuple(compress(premises[j], indexes_survivors)))
            new_ref_attribute.append(ref_attrib[j])  # In most cases could be: .append(j)

    new_ux = ux[:, np.array(list(chain(*index_prev_ux)))]  # implementar un warning o msg caso no haya nadie vivo
    return new_ref_attribute, new_premises, new_num_premises_by_attrib, new_ux


def support_premises_base_with_neg(ref_attrib, premises, num_premises_by_attrib, ux,
                                   criteria, tolerance, attributes_contain_negation):
    new_ref_attribute = []
    new_premises = []
    new_num_premises_by_attrib = []
    ref = 0
    for j in range(len(num_premises_by_attrib)):  # len(np_xA) = numero de atributos # Evaluate each attribute
        tam = num_premises_by_attrib[j]

        if attributes_contain_negation[j]:  # this attribute was negated
            attribute_ux = ux[:, ref:ref+tam//2]
            # indicator: indexes which indicates who membership functions survived. Example: [1, 0, 1, ...]
            indexes_survivors = support_negated_attribute(attribute_ux, tolerance, criteria)
            ref += tam//2
        else:
            attribute_ux = ux[:, ref:ref+tam]
            # indicator: indexes which indicates who membership functions survived. Example: [1, 0, 1, ...]
            indexes_survivors = support_by_attribute(attribute_ux, tolerance, criteria)
            ref += tam

        num_survivors = sum(indexes_survivors)  # num_survivors = indicator.sum(axis=0)

        if num_survivors != 0:
            new_num_premises_by_attrib.append(num_survivors)
            new_premises.append(tuple(compress(premises[j], indexes_survivors)))
            new_ref_attribute.append(ref_attrib[j])  # In most cases could be: .append(j)

    index_prev_ux = tuple(chain(*new_premises))
    new_ux = ux[:, index_prev_ux]
    return new_ref_attribute, new_premises, new_num_premises_by_attrib, new_ux


def support_basic_premises(ref_attrib, premises, num_premises_by_attrib, ux,
                             criteria, tolerance, attributes_contain_negation):
    
    """
    Se eliminan las premisas que no pasen el criterio del area
    :param premises: [(0,1,2)(3,4,5),(6,7,8)]
    :param tolerance:       0.1
    :param criteria: 'cardinalidade' ou 'frequencia'
    :return:         [(1,2),(3,4,5),(8,)]
    """
    
    if sum(attributes_contain_negation) != 0:  # At least some attribute was negated
        return support_premises_base_with_neg(ref_attrib, premises, num_premises_by_attrib, ux,
                                                criteria, tolerance, attributes_contain_negation)
    else:
        return support_premises_base_without_neg(ref_attrib, premises, num_premises_by_attrib, ux,
                                                   criteria, tolerance)


def support_premises_derived(x, perc_area=0.1, criteria='cardinalidade_relativa'):
    # Retorna True si el vectox cumple con el criterio de crop_area
    # ejemplo: x1 = [0,0,0.2,0.3,0] ===> return 1: si cumple
    #          x2 = [0,0,0.01,0,0] ====> return 0: no cumple
    if criteria == 'cardinalidade_relativa':
        logica = x.sum(axis=0) < x.shape[0] * perc_area
        return 0 if logica else 1
    else:  # criteria == 'frequencia_relativa'
        logica = (x > 0).sum(axis=0) < x.shape[0] * perc_area
        return 0 if logica else 1


def main():
    x1 = np.array([[0.], [0.], [0.21], [0.3], [0.]])  # (5L, 1L)
    x2 = np.array([[0.], [0.], [0.01], [0.], [0.]])   # (5L, 1L)
    #
    # a1 = crop_areaf2(x1, 0.1, 'cardinalidade_relativa')
    # a2 = crop_areaf2(x2, 0.1, 'cardinalidade_relativa')
    #
    # if a1 == 1:
    #     print "x1 paso el criterio del area"
    # else:
    #     print "x1 no paso"
    #
    # if a2 == 1:
    #     print "x2 paso el criterio del area"
    # else:
    #     print "x2 no paso"


if __name__ == '__main__':
    main()
