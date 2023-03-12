# @author: A. Stricker

# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------
# Trabajo Práctico 2: Support Vector Machines (SVMs)
# 1. Implementar separación lineal utilzando SVM aplicandolo a 2 clases del Iris plants
# dataset en los siguientes casos:
# a) Dos clases perfectamente separables.
# b) Dos clases no perfectamente separables Estudiar el comportamiento para
# distintos valores del hiperparametro C.

# 2. Implementar SVM utilizando RBF functions para dos clases no perfectamente
# separables del mismo dataset. Estudiar el comportamiento para distintos valores de
# los hiperparametros C y γ (gama).

# En todos los casos, graficar resultados.

# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------
#%%
# carga de librerias necesarias y dataset
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import svm, datasets
from matplotlib.colors import Normalize
from sklearn.preprocessing import scale
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
iris = datasets.load_iris()
#%%
# ---------
#Ejercicio 1
#punto a)
#Es necesario identificar clases en el dataset que sean perfectamente separables y las que no lo son.
#Para esto se exploran los datos
 #%%
iris = datasets.load_iris()
df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                   columns=iris['feature_names'] + ['target'])
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
df = df.drop("target", 1)
sns.pairplot(df, hue="species", palette="pastel", markers=["o", "s", "D"])
plt.savefig('iris_distribucion.png')
 
 #%%

#%%
#Las dos clases que pueden separarse mas facilmente entre ella son setosa y virginia. Seran usadas en el 
#punto a). Las clases no perfectamente separables entre si son virginia y versicolor; seran
#definidas como el subset para el punto b)
X = iris.data[:, [0, 3]]

X_separables = np.delete(X, slice(50, 100), axis=0)
y = iris.target

y_separables = np.delete(y, slice(50, 100), None)
h = .02
#Se prueban diferentes valores del hiperparametro C para determinar el mejor valor
svc = svm.SVC(kernel='linear', C=1).fit(X_separables, y_separables)
svc_mayor = svm.SVC(kernel='linear', C=100).fit(X_separables, y_separables)
svc_menor = svm.SVC(kernel='linear', C=0.01).fit(X_separables, y_separables)
svc_medio = svm.SVC(kernel='linear', C=10).fit(X_separables, y_separables)
# Se genera un mesh para plotear todo
x_min, x_max = X_separables[:, 0].min() - 1, X_separables[:, 0].max() + 1
y_min, y_max = X_separables[:, 1].min() - 1, X_separables[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
titles = ['SVC usando C=1', 'SVC usando C=100', 'SVC usando C=0.01', 'SVC usando C=10']
for i, clf in enumerate((svc, svc_mayor, svc_menor, svc_medio)):
     plt.subplot(2, 2, i + 1)
     plt.subplots_adjust(wspace=0.4, hspace=0.4)
     Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
     Z = Z.reshape(xx.shape)
     plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.5)
     plt.scatter(X_separables[:, 0], X_separables[:, 1],
                 c=y_separables, cmap=plt.cm.coolwarm)
     plt.xlabel('Sepal length')
     plt.ylabel('Petal width')
     plt.xlim(xx.min(), xx.max())
     plt.ylim(yy.min(), yy.max())
     plt.xticks(())
     plt.yticks(())
     plt.title(titles[i])
     plt.show()
# Se guarda el plot
     plt.savefig('clases_separables.png')

#Los mejores valores para el hiperparametro C parecen estar entre 1 y 100. La amplitud
#de posibles valores de C exitoso se debe a que las clases son facilmente separables

#-------
#%%
#punto b)

X_noseparables = np.delete(X, slice(0, 50), axis=0)
y_noseparables = np.delete(y, slice(0, 50), None)
svc = svm.SVC(kernel='linear', C=1).fit(X_noseparables, y_noseparables)
svc_mayor = svm.SVC(kernel='linear', C=100).fit(X_noseparables, y_noseparables)
svc_menor = svm.SVC(kernel='linear', C=0.01).fit(X_noseparables, y_noseparables)
svc_medio = svm.SVC(kernel='linear', C=10).fit(X_noseparables, y_noseparables)
# Se genera un mesh para plotear todo
x_min, x_max = X_noseparables[:, 0].min() - 1, X_noseparables[:, 0].max() + 1
y_min, y_max = X_noseparables[:, 1].min() - 1, X_noseparables[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
titles = ['SVC con C=1', 'SVC con C=100', 'SVC con C=0.01', 'SVC con C=10']
for i, clf in enumerate((svc, svc_mayor, svc_menor, svc_medio)):
    plt.subplot(2, 2, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.5)
    plt.scatter(X_noseparables[:, 0], X_noseparables[:, 1],
                c=y_noseparables, cmap=plt.cm.coolwarm)
    plt.xlabel('Sepal length')
    plt.ylabel('Petal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])
    plt.show()
    plt.savefig('clases_NO_separables.png')
    
#Los diferentes valores de C no logra separar exitosamente las clases

#%%

#-----------------------------------
#Ejercicio 2
#%%
#Se usan el subset del punto b) (no separables)
X_noseparables = np.delete(X, slice(0, 50), axis=0)
y_noseparables = np.delete(y, slice(0, 50), None)
## Se escalan los datos en este caso                                          #
X_scaled = scale(X_noseparables)
#%%
#%%
#Entrenamiento del clasificador
C_range = np.logspace(-3, 1, 5)
gamma_range = np.logspace(-3, 1, 5)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
grid.fit(X_scaled, y_noseparables)
print("Los mejores parametros son: %s , con un score de %0.2f" %
      (grid.best_params_, grid.best_score_))
x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

#De acuerdo al entrenamiento del clasificador, C=  0.1 junto con gamma = 0.1 da 
#la mejor separacion posible de las clases establecidas en el punto b)
#%%

#%%
#Entrenamiento de clasificador para todos los valores de los hiperparametros C y gamma
classifiers = []
for C in C_range:
    for gamma in gamma_range:
        clf = SVC(C=C, gamma=gamma)
        clf.fit(X_scaled, y_noseparables)
        classifiers.append((C, gamma, clf))
print(classifiers)
# Graficos para visualizar efecto de parametros
plt.figure(figsize=(8, 6))
xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
for (k, (C, gamma, clf)) in enumerate(classifiers):
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.subplot(len(C_range), len(gamma_range), k + 1)
    plt.title("gamma=10^%d | C=10^%d" %
              (np.log10(gamma), np.log10(C)), size="small")
    plt.pcolormesh(xx, yy, -Z, cmap=plt.cm.RdGy_r, shading="auto")
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1],
                c=y_noseparables, cmap=plt.cm.RdGy, edgecolors="k")
    plt.xticks(())
    plt.yticks(())
    plt.axis("tight")
#%%

#Calculo de los scores de combinacion de hiperparametros
scores = grid.cv_results_["mean_test_score"].reshape(
    len(C_range), len(gamma_range))
#%%

#%%
#Grafico de heatmap con resultados de los hiperparametros C y gamma

plt.figure(figsize=(8, 6))
plt.subplots_adjust(left=0.2, right=0.95, bottom=0.15, top=0.95)
plt.imshow(scores, interpolation="nearest", cmap=plt.cm.autumn, norm=None)
plt.xlabel("gamma")
plt.ylabel("C")
plt.colorbar()
plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
plt.yticks(np.arange(len(C_range)), C_range)
plt.title("Validation accuracy")
plt.show()

#De acuerdo al heatmap, la mejor combinacion de valores de C y gamma se encuentra cerca de 
#0.1 para ambos hiperparametros
#%%