import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd


from autofis.classification.senfis_classifier import SENFISClassifier

iris = datasets.load_iris()
df_iris = pd.DataFrame(iris['data'])
df_iris['target'] = iris['target']

X = iris['data']
y = iris['target']

X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.30, random_state=42)

senfis = SENFISClassifier().fit(X_train,y_train,[False,False,False,False],verbose=1)