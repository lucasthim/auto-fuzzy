__author__ = 'jparedes'

from nose.tools import *
from BRFIS.autoFIS.autoFIS.lecture import Lecture
from BRFIS.autoFIS.autoFIS.fuzzification import Fuzzification, trimf
import numpy as np

import os

folder_data = os.path.join(os.path.dirname(__file__), "datas")


def test_fuzzy_triangular():
    x = np.array([[0.2], [0.9], [0.42], [0.74], [0.24], [0.28], [0.34]])
    params = [0.3, 0.4, 0.5]
    ux_ref = np.array([[0.], [0.], [0.8], [0.], [0.], [0.], [0.4]])
    dif = trimf(x, params).sum(0) - ux_ref.sum(0) < 0.00000000001  # error tolerance
    assert_equal(dif, 1)


def test_non_categorical_data():
    filezip_name = os.path.join(folder_data, "iris-10-fold_csv.zip")
    L = Lecture()
    L.read_1cv(filezip_name, "iris-10-7tra.csv", "iris-10-7tst.csv")
    assert_equal(L.train_instances, 135)
    info = L.info_data()

    X = info[0]
    cat_bool = [0, 0, 0, 0, 0, 0, 0, 0, 0]  # Funciona con al menos: [0]
    F = Fuzzification(X, cat_bool)
    F.build_uX('normal', 3)
    assert_equal(F.uX.shape[1], 12)


def test_categorical_data():
    filezip_name = folder_data + '\\' + 'saheart-10-fold_csv.zip'
    L = Lecture()
    L.read_1cv(filezip_name, 'saheart-10-7tra.csv', 'saheart-10-7tst.csv')
    info = L.info_data()

    X = info[0]
    cat_bool = [0, 0, 0, 0, 1, 0, 0, 0, 0]
    F = Fuzzification(X, cat_bool)
    F.build_uX('normal', 3)
    assert_equal(F.uX.shape[1], 26)


def test_categorical_negation():
    filezip_name = folder_data + '\\' + 'saheart-10-fold_csv.zip'
    L = Lecture()
    L.read_1cv(filezip_name, 'saheart-10-7tra.csv', 'saheart-10-7tst.csv')
    info = L.info_data()

    X = info[0]
    cat_bool = [0, 0, 0, 0, 1, 0, 0, 0, 0]
    F = Fuzzification(X, cat_bool)
    F.build_uX('normal', 3)
    F.add_negation()
    assert_equal(F.uX.shape[1], 52)
