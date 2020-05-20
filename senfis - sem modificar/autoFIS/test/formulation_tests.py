__author__ = 'jparedes'

from nose.tools import *
from itertools import combinations, product, chain
from BRFIS.autoFIS.autoFIS.lecture import Lecture
from BRFIS.autoFIS.autoFIS.fuzzification import Fuzzification
from BRFIS.autoFIS.autoFIS.formulation.formulation import Formulation
import os

folder_data = os.path.join(os.path.dirname(__file__), "datas")

filezip_name = os.path.join(folder_data, "iris-10-fold_csv.zip")
obj1 = Lecture()
obj1.read_1cv(filezip_name, 'iris-10-1tra.csv', 'iris-10-1tst.csv')

cat_bool = [0, 0, 0, 0]
obj2 = Fuzzification(obj1.X, cat_bool)
obj2.build_uX("normal", 3)

# -------------------------
# Formulation parameters
# -------------------------
ordem_max_premises = 2
# Area filter parameters:
criteria_area = "cardinalidade_relativa"  # "cardinalidade_relativa", "frequencia_relativa"
area_threshold = 0.050
# PCD filter parameter:
is_enable_pcd = [1, 1]
# Overlapping filter parameters:
is_enable_overlapping = [1, 1]
overlapping_threshold = 0.95


def test_premises_generated():
    assert_equal(obj2.num_prem_by_attribute, [3, 3, 3, 3])

    ref_comb = list(combinations(obj2.premises_attributes, 2))
    aux = [list(product(*i)) for i in ref_comb]
    assert_equal(len(aux), 6)
    assert_equal(len(list(chain(*aux))), 54)


def test_premises_survivors():
    par_area, par_overlapping, par_pcd = [criteria_area, area_threshold], [is_enable_pcd, 1], [is_enable_pcd, 1]
    ux = obj2.uX[:135, :]
    c_bin = obj1.cBin[:135, :]
    obj3 = Formulation(ux, c_bin, obj2.ref_attributes, obj2.premises_attributes,
                       obj2.num_prem_by_attribute, obj2.indexes_premises_contain_negation)
    obj3.load_filter_parameters(par_area, par_overlapping, par_pcd)
    arbol = obj3.gen_ARB(ordem_max_premises, "prod", par_area, par_overlapping, par_pcd)
    assert_equal(len(arbol[0][0]), 10)
    assert_equal(len(arbol[1][0]), 18)
