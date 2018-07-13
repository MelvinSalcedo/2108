import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import csv

num_centroildes=10
I = Image.open("imagenes_entrada/imagenA.bmp")
#size = 50,50
#I .thumbnail(size, Image.ANTIALIAS)
##I .save(outfile, "JPEG")

plt.figure(figsize=(4,4))
plt.imshow(I)
plt.axis('off')
#plt.show()
##Para simplificar el problema, convertimos la imagen de color a blanco y negro
I1 = I.convert('L')
I2 = np.asarray(I1,dtype=np.float)
plt.imshow(I2,cmap='gray')
plt.axis('off')
#plt.show()
##Preparamos la matriz para aplicar k-means.
##Ahora tendra tantas filas como pixeles pero solo
#_____________________________________________________________
A=[]
B=[]
COORDENADA=[]
csvfile = 'input.csv'

try:
    f = open(csvfile, 'r')
    reader = csv.reader(f, delimiter=';')
    for row in reader:
    	A.append(float(row[0]))
    	B.append(float(row[1]))
    	COORDENADA.append([float(row[0]),float(row[1])])
        #print(row[0],row[1])
finally:
    f.close()

coo=np.asarray(COORDENADA,dtype=np.float)
x=np.asarray(A,dtype=np.float)
y=np.asarray(B,dtype=np.float)

M = x.reshape(-1,1)
N = y.reshape(-1,1)

k_means_T = KMeans(n_clusters=num_centroildes)
k_means_T.fit(M,N)

centroidesT = k_means_T.cluster_centers_

#_____________________________________________________________
## una columna, la intensidad de gris.
X = I2.reshape(-1,1)
##Agrupamos los pixeles en tres clusteres con kmeans
k_means = KMeans(n_clusters=num_centroildes)
k_means.fit(X)
etiquetas = k_means.labels_

print("_______________")
print(centroidesT)
print(etiquetas)
print("_______________")
"""

centroidesT[0]=0.99993698
centroidesT[1]=0.06131988
centroidesT[2]=0.13271872
centroidesT[3]=0.01454356
centroidesT[4]=0.41036414

[[0.99993698]
 [0.06121941]
 [0.01453339]
 [0.13198147]
 [0.43053677]]

"""

##Reconstruimos la imagen utilizando las tres intensidades de los centroides
I2_compressed = np.choose(etiquetas, centroidesT)
I2_compressed.shape = I2.shape

#print(I2_compressed)
##Visualizamos la foto reconstruida
plt.figure(figsize=(4,4))
plt.imshow(I2_compressed,cmap='gray')
plt.savefig('salida.png')
plt.axis('off')
plt.show()









