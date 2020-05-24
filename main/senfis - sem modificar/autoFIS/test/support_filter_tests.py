__author__ = 'jparedes'

from nose.tools import *
from numpy import array
from BRFIS.autoFIS.autoFIS.formulation.support_filter import support_premises_derived


def test_support_one_premise():
    x1 = array([[0.], [0.], [0.21], [0.3], [0.]])
    x2 = array([[0.], [0.], [0.01], [0.], [0.]])

    a1 = support_premises_derived(x1, 0.1, 'cardinalidade_relativa')
    a2 = support_premises_derived(x2, 0.1, 'cardinalidade_relativa')

    assert_equal(a1, 1)
    assert_equal(a2, 0)
