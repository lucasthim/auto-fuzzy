__author__ = 'jparedes'

from nose.tools import *
from BRFIS.autoFIS.autoFIS.formulation.similarity_filter import similarity, similarity_basic_premises, similarity_derived_premises
import numpy as np


def test0_similarity_value():
    ux = np.array([[0.20, 0.70, 0.1900, 0.199, 0.18, 0.200, 0.65, 0.25, 0.10],
                  [0.50, 0.40, 0.5010, 0.490, 0.50, 0.470, 0.39, 0.42, 0.19],
                  [0.05, 0.40, 0.0505, 0.048, 0.05, 0.050, 0.42, 0.23, 0.35],
                  [0.01, 0.87, 0.0060, 0.009, 0.02, 0.018, 0.88, 0.11, 0.01]])
    valor_similarity = similarity(ux[[0], :], ux)
    tolerance = 0.01

    ref = [1.,  0.2672,  0.9796, 0.9816, 0.9610, 0.9505, 0.2653, 0.6239, 0.3302]

    num_elements = len(ref)
    error = [(ref[i] - valor_similarity[i])**2/num_elements for i in range(num_elements)]
    assert_equal(sum(error), tolerance)


def test1_similarity_premises():
    ref_attribute = [0, 1, 2]

    premises_by_attrib_case1 = [(0, 1, 2), (3, 4, 5,), (6, 7, 8)]
    num_prem_by_attrib_case1 = [3, 3, 3]

    premises_by_attrib_case2 = [(0, 1), (2, 3, 4, 5,), (6, 7, 8)]
    num_prem_by_attrib_case2 = [2, 4, 3]

    ux = np.array([[0.20, 0.70, 0.1900, 0.199, 0.18, 0.200, 0.65, 0.25, 0.10],
                  [0.50, 0.40, 0.5010, 0.490, 0.50, 0.470, 0.39, 0.42, 0.19],
                  [0.05, 0.40, 0.0505, 0.048, 0.05, 0.050, 0.42, 0.23, 0.35],
                  [0.01, 0.87, 0.0060, 0.009, 0.02, 0.018, 0.88, 0.11, 0.01]])

    case1 = similarity_basic_premises(ref_attribute, premises_by_attrib_case1, num_prem_by_attrib_case1, ux)
    case2 = similarity_basic_premises(ref_attribute, premises_by_attrib_case2, num_prem_by_attrib_case2, ux)

    assert_equal(case1[0], [(0, 1, 2), (7, 8)])
    assert_equal(case2[0], [(0, 1), (7, 8)])

    # Caso Bobo
    ux2 = np.array([[0.2, 0.5, 0.01, 0.20, 0.7, 0.39],
                  [0.1, 0.3, 0.08, 0.09, 0.6, 0.29]])
    premises_by_attrib_case3 = [(0, 3), (5, 6, 8, 9)]
    num_prem_by_attrib_case3 = [2, 4]
    case3 = similarity_basic_premises([0, 1], premises_by_attrib_case3, num_prem_by_attrib_case3, ux2)
    assert_equal(case3[0], [(0, 3), (5, 8, 9)])


def test_overlapping_derived_premises():
    ind_a0 = [(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,)]
    ind_a1 = [(1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,)]
    ind_a2 = [(2,), (3,), (4,), (5,), (6,), (7,), (8,)]

    ux = np.array([[0.20, 0.70, 0.1900, 0.199, 0.18, 0.200, 0.65, 0.25, 0.10],
                  [0.50, 0.40, 0.5010, 0.490, 0.50, 0.470, 0.39, 0.42, 0.19],
                  [0.05, 0.40, 0.0505, 0.048, 0.05, 0.050, 0.42, 0.23, 0.35],
                  [0.01, 0.87, 0.0060, 0.009, 0.02, 0.018, 0.88, 0.11, 0.01]])

    a0 = similarity_derived_premises(range(9), ind_a0, ux)
    a1 = similarity_derived_premises(range(8), ind_a1, ux[:, 1:])
    a2 = similarity_derived_premises(range(7), ind_a2, ux[:, 2:])

    assert_equal(a0[0], [(0,), (1,), (7,), (8,)])
    assert_equal(a1[0], [(1,), (3,), (4,), (7,), (8,)])
    assert_equal(a2[0], [(3,), (4,), (6,), (7,), (8,)])
