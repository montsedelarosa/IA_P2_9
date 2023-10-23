# _*_ coding: utf-8 _*_
#!/usr/bin/env python
# _*_ coding: cp1252 _*_
# _*_ cdoing: 850 _*_

from sklearn import datasets
from sklearn import svm
import matplotlib.pyplot as plt

# Cargar un conjunto de datos de ejemplo (Iris dataset)
iris = datasets.load_iris()
X = iris.data[:, :2]  # Usar solo las dos primeras características
y = iris.target

# Crear un modelo de SVM
clf = svm.SVC(kernel='linear')
clf.fit(X, y)

# Graficar los datos y las regiones de decisión
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
plt.title('SVM con kernel lineal')
plt.show()
