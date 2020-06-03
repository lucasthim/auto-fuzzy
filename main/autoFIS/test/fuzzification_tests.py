# # from nose.tools import *
# from main.autoFIS.autoFIS.fuzzification import Fuzzification
# import numpy as np

# import os

# folder_data = os.path.join(os.path.dirname(__file__), "data")

# def test_fuzzy_triangular():
#     x = np.array([[0.2], [0.9], [0.42], [0.74], [0.24], [0.28], [0.34]])
#     params = [0.3, 0.4, 0.5]
#     ux_ref = np.array([[0.], [0.], [0.8], [0.], [0.], [0.], [0.4]])
#     dif = trimf(x, params).sum(0) - ux_ref.sum(0) < 0.00000000001  # error tolerance
#     assert_equal(dif, 1)