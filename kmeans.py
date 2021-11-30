#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

data = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8 ],
              [8, 8],
              [1, 0.6],
              [9,11]])

k = 2
tol = 0.001
max_iter = 300
centroids = {}
classif = {}

# Se obtienen los primeros puntos como centroides
for i in range(k):
    centroids[i] = data[i]

def predict(data, centroids):
    distances = [np.linalg.norm(data - centroids[centroid]) for centroid in centroids]
    cl = distances.index(min(distances))
    return cl

# comenzamos a iterar
for i in range(max_iter):
    # Diccionario que almacena las clasificaciones
    classif = {}
    # Se guarda un array donde se almacenará la posición de cada punto clasificado
    for i in range(k):
        classif[i] = []

    # Iteramos por el set de datos y los clasificamos. Comparamos la diferencia
    # de cada dato con los centroides y aquel centroide que cuente con la menor
    # diferencia será el elegido.
    for fs in data:
        distances = [np.linalg.norm(fs - centroids[c]) for c in centroids]
        cl = distances.index(min(distances))
        classif[cl].append(fs) # Se añade el dato del dataset a su respectivo
    prev_centroids = dict(centroids)

    # Se obtiene el promedio de cada clasificación y así se generan nuevos centroides
    for cl in classif:
        centroids[cl] = np.average(classif[cl], axis=0)

    # Si los centroides se han movido menos del índice de tolerancia se termina la
    # ejecución
    optimized = True
    for c in centroids:
        original_centroid = prev_centroids[c]
        current_centroid = centroids[c]
        if np.sum((current_centroid - original_centroid) / original_centroid * 100) > tol:
            optimized = False
    if optimized:
        for c in centroids:
            plt.scatter(centroids[c][0], centroids[c][1], marker='o', color='k', s=150, linewidths=5)

        colors = ['r', 'b']
        for c in classif:
            color = colors[c]
            for fs in classif[c]:
                plt.scatter(fs[0], fs[1], marker='x', color=color, linewidths=5)
        plt.show()
        break
