__author__ = 'jparedes'

from nose.tools import *
from numpy import array
from BRFIS.autoFIS.autoFIS.decisions import Decisions


def decision_maxi_relevance_test_1():
    fuzzy_prediction = array([[0.80, 0.15, 0.05], [0.03, 0.90, 0.07], [0.20, 0.60, 0.20], [0.20, 0.10, 0.70]])
    frequency_classes = [0.5, 0.15, 0.35]

    obj3 = Decisions(fuzzy_prediction, frequency_classes)
    predicted_output = obj3.dec_max_pert()

    expected_output = array([[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1]])
    error = (expected_output - predicted_output).sum(0).sum(0) < 0.00001
    assert_equal(error, True)


def decision_maxi_relevance_test_2():
    fuzzy_prediction = array([[0.80, 0.15, 0.05], [0.03, 0.90, 0.07], [0.20, 0.40, 0.40], [0.20, 0.10, 0.70]])
    frequency_classes = [0.5, 0.15, 0.35]

    obj3 = Decisions(fuzzy_prediction, frequency_classes)
    predicted_output = obj3.dec_max_pert()

    expected_output = array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]])
    error = (expected_output - predicted_output).sum(0).sum(0) < 0.00001
    assert_equal(error, True)


def decision_maxi_relevance_test_3():
    fuzzy_prediction = array([[0.80, 0.15, 0.05], [0.03, 0.90, 0.07], [0.20, 0.40, 0.40], [0.45, 0.10, 0.45]])
    frequency_classes = [0.5, 0.15, 0.35]

    obj3 = Decisions(fuzzy_prediction, frequency_classes)
    predicted_output = obj3.dec_max_pert()

    expected_output = array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])
    error = (expected_output - predicted_output).sum(0).sum(0) < 0.00001
    assert_equal(error, True)
