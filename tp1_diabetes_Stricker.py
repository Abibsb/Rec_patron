# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 18:05:25 2022

@author: A. Stricker
"""
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------

# Trabajo Practico 1: Regresion lineal - California Housing dataset

#1. Implementar Regresión Lineal utilizando el California Housing dataset para los
#siguientes métodos:
#a) Ridge Regression
#b) LASSO
#c) Elastic Net

#2. Para cada uno de los métodos del punto anterior utilizar Cross Validation para
#elegir los mejores valores de los hiper-parámetros.

#3. Aplicar lo implementado en los dos puntos anteriores al Diabetes dataset

##En este script esta resuelto el punto 3, donde se cambia el dataset de California Housing por el de Diabetes
print("En este script esta resuelto el punto 3, donde se cambia el dataset de California Housing por el de Diabetes")

#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
#%%
# carga de librerias necesarias y dataset
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV
import numpy as np
import pandas as pd
import math
#%%
# ---------
#%%
# Leer dataset California Housing (cal_housing.data, cal_housing.target)
diabetes = load_diabetes()
# Creacion de matriz de datos X_cal como DataFrame de Pandas
X_diabetes = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
# Creacion del array de valores del target y_cal
y_diabetes = diabetes.target

# Separar los datos de entrada en test y train, tanto para X_cal como para y_cal.
# Se reserva el 20% para el test set (lo tipico para la mayoria de los datasets)
X_diabetes_train, X_diabetes_test, y_diabetes_train, y_diabetes_test = train_test_split(
    X_diabetes, y_diabetes, test_size=0.2, random_state=0)
# Hay que escalar los datos
X_diabetes_train = scale(X_diabetes_train)
X_diabetes_test = scale(X_diabetes_test)
#%%
# -------------------

# Ejercicio 1
#%%
# Regresion lineal: para comparar con las demas regresiones
clasificador_LR = LinearRegression()
clasificador_LR.fit(X_diabetes_train, y_diabetes_train)
predict_LR = clasificador_LR.predict(X_diabetes_test)
LR_score = clasificador_LR.score(X_diabetes_test, y_diabetes_test)
RMSE_LR = math.sqrt(mean_squared_error(y_diabetes_test, predict_LR))
#%%
# ----
#%%
#a) Ridge regression. Cuanto mas chico es alpha, Ridge se vuelve mas similar a regresion lineal.
#Si alpha crece mucho, Ridge se vuelve una linea horizonal que pasa por el promedio del dataset
ridge_sinCV = Ridge(alpha=0.8, normalize=True)
ridge_sinCV.fit(X_diabetes_train, y_diabetes_train)
ridge_sinCV_score = ridge_sinCV.score(X_diabetes_test, y_diabetes_test)
ridge_sinCV_pred = ridge_sinCV.predict(X_diabetes_test)
MSE_sinCV_RR = math.sqrt(mean_squared_error(y_diabetes_test, ridge_sinCV_pred))
RMSE_sinCV_RR = math.sqrt(mean_squared_error(y_diabetes_test, ridge_sinCV_pred))
#%%
# ----
#%%
# b)LASSO. Este hiperparametro funciona igual que en Ridge. Lasso se diferencia de Ridge
#en que elimina mas facilmente aquellas variables del dataset que tengan menor importancia en la ecuacion final.
lasso_sinCV = Lasso(alpha=0.8)
lasso_sinCV.fit(X_diabetes_train, y_diabetes_train)
lasso_sinCV_score = lasso_sinCV.score(X_diabetes_test, y_diabetes_test)
lasso_sinCV_pred = lasso_sinCV.predict(X_diabetes_test)
RMSE_sinCV_LS = math.sqrt(mean_squared_error(y_diabetes_test, lasso_sinCV_pred))
#%%
# ----
#%%
#c) Elastic Net. Este modelo es un punto medio entre las dos regresiones previamente usadas.
#Un nuevo hiperparametro r (en python definido como l1) define cuan parecido se comportara Elastic Net
#a Ridge o Lasso. Si l1=0 Elastic net se comporta como Ridge; con l1=1 se comporta como LASSO.
elasticnet = ElasticNet(alpha=0.8, l1_ratio=0.5)
elasticnet.fit(X_diabetes_train, y_diabetes_train)
enet_sinCV_pred = elasticnet.predict(X_diabetes_test)
enet_sinCV_score = elasticnet.score(X_diabetes_test, y_diabetes_test)
RMSE_sinCV_enet = math.sqrt(mean_squared_error(y_diabetes_test, enet_sinCV_pred))
#%%

#%%
#Comparacion de scores de las diferentes regresiones. Estos scores dan una idea de cuan bien un modelo
#se desempeña con el dataset dado. 
print("Valores de score para cada tipo de regresion:\n")
print("El score correspondiente a la regresion lineal es", format(LR_score))
print("Utilizando un alpha de 0.8, el score correspondiente a Ridge es", format(ridge_sinCV_score))
print("Utilizando un alpha de 0.8, el score correspondiente a LASSO es", format(lasso_sinCV_score))
print("Utilizando un alpha de 0.8 y un l1 de 0.5, el score correspondiente a ElasticNet es", format(enet_sinCV_score))
print("\n")
print("\n")
#%%


#%%
#RMSE es un valor util para evaluar la performance de las regresiones. Cuanto mas bajo da, mejor fiteada esta la
#el modelo de regresion al dataset
print("Valores de RMSE para cada tipo de regresion:\n")
print("El RMSE correspondiente a la regresion lineal es", format(RMSE_LR))
print("Utilizando un alpha de 0.8, el RMSE correspondiente a Ridge es", format(RMSE_sinCV_RR))
print("Utilizando un alpha de 0.8, el RMSE correspondiente a LASSO es", format(RMSE_sinCV_LS))
print("Utilizando un alpha de 0.8 y un l1 de 0.5, el RMSE correspondiente a ElasticNet es", format(RMSE_sinCV_enet))
print("\n")
print("\n")
#%%
#----


#Cross-validation: es uno de los metodos mas usados para poner a punto los hiperparametros.
#En nuestro caso esto se traslada a encontrar el valor mas optimo posible para alpha y r en nuestros modelos.
#%%
print("Se utilizará Cross validation con el 20% del dataset reservado para test para determinar los hiperparametros\n")

#Ridge Cross-validation

#Se elige valores random de alpha entre 0 y 10 para la Ridge regression del punto a)
#La funcion de Cross-validation se utilizara la estrategia 5-fold.
alphas_ridge = 10 * np.random.random_sample((1000))
ridgecv = RidgeCV(alphas=alphas_ridge, gcv_mode='auto', scoring='neg_mean_squared_error', normalize=True, cv=5)
ridgecv.fit(X_diabetes_train, y_diabetes_train)

#Aplicamos Cross-validation sobre alpha para seleccionar su valor optimo para Ridge regression.
alpha_r = ridgecv.alpha_

#Aplicamos el alpha validado a Ridge Regresion.
ridge = Ridge(alpha=alpha_r, normalize=True)
ridge.fit(X_diabetes_train, y_diabetes_train)
ridge_score = ridge.score(X_diabetes_test, y_diabetes_test)
ridge_pred = ridge.predict(X_diabetes_test)
RMSE_RR = math.sqrt(mean_squared_error(y_diabetes_test, ridge_pred))
print(" Ridge score antes de CV es ", ridge_sinCV_score, "con valor de alpha 0.8", '\n', "Ridge score luego de CV es ", ridge_score ,"y el valor Alpha elegido es ", alpha_r)
print(" RMSE para regresión Ridge Regression sin CV = ", RMSE_sinCV_RR, "\n","RMSE para regresión Ridge Regression con mejor valor alpha es ", RMSE_RR, '\n')
#%%
#---
#%%
# Lasso Cross-validation


#Se elige valores random de alpha para la regresion Lasso del punto b). Debido al error en ElasticNet que genera un rango entre 10 y 0 en elastic net restringo el rango de 0 a 1.
#Aplicamos Cross Validation sobre alpha para seleccionar valor optimo de alpha para Lasso.
alphas_lasso = np.random.random_sample((1000))

#Cross-validation se deja default para mantener la estrategia de 5-fold validation.
lassocv = LassoCV(alphas=alphas_lasso, normalize=True, max_iter=1000)
lassocv.fit(X_diabetes_train, y_diabetes_train)
lasso_alpha = lassocv.alpha_
lassocv_pred = lassocv.predict(X_diabetes_test)
lasso = Lasso(alpha=lasso_alpha, max_iter=1000000, normalize = True)
lasso.fit(X_diabetes_train, y_diabetes_train)
lasso_pred = lasso.predict(X_diabetes_test)
lasso_score = lasso.score(X_diabetes_test, y_diabetes_test)
RMSE_LS = math.sqrt(mean_squared_error(y_diabetes_test, lasso_pred))
print(" Lasso score antes de CV es ", lasso_sinCV_score, "con valor de alpha 0.8", '\n', "Lasso score luego de CV es ", lasso_score ,"y el valor Alpha elegido es ", lasso_alpha)
print(" RMSE para regresión Lasso Regression sin CV = ", RMSE_sinCV_LS, "\n","RMSE para regresión Lasso Regression con mejor valor alpha es ", RMSE_LS, '\n')
#%%
#---
#%%
#Elastic Net Cross-validation
#Cross-validation se deja default para mantener la estrategia de 5-fold validation.
elastic_alpha =  np.random.random_sample((1000))
elastic_l1_ratio = np.random.random_sample((1000))
elasticnetcv = ElasticNetCV(alphas=elastic_alpha, l1_ratio=elastic_l1_ratio, normalize=True, max_iter=1000)
elasticnetcv.fit(X_diabetes_train, y_diabetes_train)
elasticnetcv_seleccion_alpha = elasticnetcv.alpha_
elasticnetcv_seleccion_l1_ratio = elasticnetcv.l1_ratio_
elasticnetcv_prediccion = elasticnetcv.predict(X_diabetes_test)
elasticnet = ElasticNet(alpha=elasticnetcv_seleccion_alpha, l1_ratio=elasticnetcv_seleccion_alpha)

elasticnet.fit(X_diabetes_train, y_diabetes_train)
enet_pred = elasticnet.predict(X_diabetes_test)
enet_score = elasticnet.score(X_diabetes_test, y_diabetes_test)
RMSE_enet = math.sqrt(mean_squared_error(y_diabetes_test, enet_sinCV_pred))
#%%


#%%
print(" Elastic Net score antes de CV es ", enet_sinCV_score, "con valor de alpha 0.8 y l1_ratio 0.5",'\n', "Elastic Net score luego de CV es", enet_score, " y los valores Alpha",elasticnetcv_seleccion_alpha," y ratio l1 son ", elasticnetcv_seleccion_l1_ratio)
print(" RMSE para regresión Elastic Net Regression sin CV = ", RMSE_sinCV_enet, '\n', "RMSE para regresión Elastic Net Regression con mejor valor alpha es ", RMSE_enet)

print("\n\n El score de Regresion Lineal es ", LR_score,'\n',"El score de Regresion Ridge es", ridge_score, '\n',"El score de Regresion Lasso es {}".format(lasso_score), '\n',"El score de Regresion ElasticNet es ", enet_score, '\n')
print(" RMSE para Regresión Lineal es ",(RMSE_LR), '\n', "RMSE para Regresión Lasso es ",(RMSE_LS), '\n', "RMSE para Regresión Ridge es ",(RMSE_RR), '\n', "RMSE para Regresión ElasticNet es ",(RMSE_enet), '\n')

print("Si se observa el score, todos los modelos tienen un desempeño similar entre sí, ya que todos tienen un valor de score cercano a 0.3.\n")
print("De acuerdo al RMSE, la regresion elastic net es el mejor modelo para este dataset, ya que su valor es 2 unidades menor al resto de los modelos, que mantienen un RMSE cercano a 60.", "\n")


print('w0 LR =', clasificador_LR.intercept_, '\n', 'w0 Ridge =', ridgecv.intercept_, '\n', 'w0 Lasso =', lasso.intercept_, '\n', 'w0 ElasticNet =', elasticnet.intercept_, '\n')
print('Los coeficientes encontrados usando Regresion Lineal son [w1 w2 w3 w4 w5 w6 w7 w8] =', clasificador_LR.coef_, '\n')
print('Los coeficientes encontrados usando Regresion Ridge son [w1 w2 w3 w4 w5 w6 w7 w8] =', ridgecv.coef_, '\n')
print('Los coeficientes encontrados usando Regresion Lasso son [w1 w2 w3 w4 w5 w6 w7 w8] =', lasso.coef_, '\n')
print('Los coeficientes encontrados usando Regresion ElasticNet son [w1 w2 w3 w4 w5 w6 w7 w8] =', elasticnet.coef_, '\n')
#%%

