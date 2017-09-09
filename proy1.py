# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import glob
from sklearn import linear_model
from sklearn.metrics import r2_score

"""
    AUTOR:
        Team Oyentes
"""

"""
-- Lectura de los datos de entrenamiento
"""
df = pd.read_csv(
    '/home/urb/PycharmProjects/Machine_Learning/Data/P1/BlogFeedback/blogData_train.csv',
    header=None)
DATA = df.as_matrix()
X = DATA[:, :-1]
Y = DATA[:, -1]

"""
-- Instancias de los modelos de regresión, utilizando los dos tipos de regularización
    y elasticNet que es la combinación de los dos
"""
model = linear_model.Ridge(alpha=0.01, normalize=True)
model2 = linear_model.Lasso(alpha=0.01, normalize=True)
model3 = linear_model.ElasticNet(alpha=0.01, normalize=True)

"""
-- Entrenamiento de los modelos utilizando el dataset de training
"""
model.fit(X, Y)
model2.fit(X, Y)
model3.fit(X, Y)

dir_test = glob.glob(
    '/home/urb/PycharmProjects/Machine_Learning/Data/P1/BlogFeedback/blogData_test*')

"""
-- Prediccion del testing 
"""
for path in dir_test:
    df_test = pd.read_csv(path, header=None)
    DATA_T = df_test.as_matrix()
    X_test = DATA_T[:, :-1]
    Y_test = DATA_T[:, -1]

    y_predict1 = model.predict(X_test)
    y_predict2 = model2.predict(X_test)
    y_predict3 = model3.predict(X_test)

    """
    Medida  R cuadrada,  la más óptima es 1. 
    """
    print("Model 1; metric R square, ", r2_score(Y_test, y_predict1))
    print("Model 2; metric R square, ", r2_score(Y_test, y_predict2))
    print("Model 3; metric R square, ", r2_score(Y_test, y_predict3))
    print("")
