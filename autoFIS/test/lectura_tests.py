__author__ = "Jorge Paredes"

# Verificar cantidad de premisas base al aplicar los filtros de area, overlapping y PCD
# Proporcion: Caso ideal sin ser eliminadas las premisas / premisas base

from nose.tools import *
from BRFIS.autoFIS.autoFIS.lecture import Lecture
import os

folder_data = os.path.join(os.path.dirname(__file__), "datas")


def test_1file():
    file_path = os.path.join(folder_data, "iris-10-7tra.csv")
    obj1 = Lecture()
    obj1.read_1file(file_path)
    assert_equal(obj1.train_instances, 'Disable')
    

def test_1cv():
    filezip_name = os.path.join(folder_data, "iris-10-fold_csv.zip")
    obj2 = Lecture()
    obj2.read_1cv(filezip_name, 'iris-10-7tra.csv', 'iris-10-7tst.csv')
    assert_equal(obj2.train_instances, 135)
