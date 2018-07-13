import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
from time import time
from skimage import data, io, segmentation, color
from skimage.future import graph
import numpy as np


#creando poblacion inicial:
k = 10
tam_cromosoma=k
tam_poblacion=10
poblacion = []


def crear_etiquetas():
	global K
	I = Image.open("imagenes_entrada/imagenD.bmp")
	#size = 50,50
	#I .thumbnail(size, Image.ANTIALIAS)
	##I .save(outfile, "JPEG")
	"""plt.figure(figsize=(8,8))
	plt.imshow(I)
	plt.axis('off')"""
	#plt.show()
	##Para simplificar el problema, convertimos la imagen de color a blanco y negro
	I1 = I.convert('L')
	I2 = np.asarray(I1,dtype=np.float)

	plt.figure(figsize=(4,4))
	plt.imshow(I2,cmap='gray')
	plt.axis('off')
	#plt.show()
	##Preparamos la matriz para aplicar k-means.
	##Ahora tendra tantas filas como pixeles pero solo

	## una columna, la intensidad de gris.
	X = I2.reshape(-1,1)

	##Agrupamos los pixeles en clusteres con kmeans
	
	k_means = KMeans(n_clusters=k)
	k_means.fit(X)
	#_____________________________________________
	#________________________________________________
	##Extraemos el valor de los centroides y las etiquetas de cada pixel
	centroides = k_means.cluster_centers_
	etiquetas = k_means.labels_

	#print(centroides[0],"----")

	#print(centroides)
	#print(etiquetas,len(etiquetas)) Reconstruimos la imagen utilizando las tres intensidades de los centroides
	I2_compressed = np.choose(etiquetas, centroides)
	I2_compressed.shape = I2.shape

	return (X,etiquetas,centroides,I2_compressed,I2)
	##Visualizamos la foto reconstruida
	
	#___________

DDC=[]
def poblacion_inicial():
	for i in range(tam_poblacion):
		m=random.sample(range(k), k)
		poblacion.append(m)
def Dissimilarity_Distance_Computation(array):
	sumatoria_grises=0
	punto_medio_grises=0
	variansa_region=0

	for i in array:
		punto_medio_grises+=i;
	
	punto_medio_grises=punto_medio_grises/len(array)

	for i in array:
		sumatoria_grises+=(i-punto_medio_grises)
	#print("pmg=",punto_medio_grises)
	if(punto_medio_grises==0):
		return 0
	variansa_region=sumatoria_grises/punto_medio_grises

	return variansa_region

def hallar_region_etiqueta(num,etiquetas,X):
	Et_c=[]
	for i in range(66564):
		if(etiquetas[i]==num):
			Et_c.append(X[i][0])
	#print(Et_c)
	return Et_c
def distancia_euclidiana(a ,b):
	di=abs(pow(a,2)-pow(b,2))
	#print("----",distancia_euclidiana)
	r=math.sqrt(di)
	return r

region_removida=[]
def procedimiento_fusion(E,X):
	umbral=0.00001
	for x in range (tam_poblacion):
		inicial=0
		rm=[]
		for i in range(1,tam_cromosoma-1):
			if(inicial!=i):
				etiqueta0= hallar_region_etiqueta(poblacion[x][i-1],E,X)
				etiqueta1= hallar_region_etiqueta(poblacion[x][i],E,X)
				etiqueta2= hallar_region_etiqueta(poblacion[x][i+1],E,X)
				varianza_a=Dissimilarity_Distance_Computation(etiqueta0)
				varianza_b=Dissimilarity_Distance_Computation(etiqueta1)
				varianza_c=Dissimilarity_Distance_Computation(etiqueta2)
				dist_1=distancia_euclidiana(varianza_b,varianza_a)
				dist_2=distancia_euclidiana(varianza_b,varianza_c)
				distancia_minima=0
				conquien=0
				d=0
				if(dist_2<=dist_1):
					distancia_minima=dist_2
					conquien=poblacion[x][i+1]
					d=dist_2
				else:
					distancia_minima=dist_1
					conquien=poblacion[x][i-1]
					d=dist_1

				if(distancia_minima<umbral):
					rm.append((poblacion[x][i],d))
					poblacion[x][i]=poblacion[x][conquien]
		region_removida.append(rm)

		inicial+=1	
region_removida2=[]		
def procedimiento_fusion_hijo(E,X,hijo):
	umbral=0.00001
	for x in range (1):
		inicial=0
		rm=[]
		for i in range(1,tam_cromosoma-1):
			if(inicial!=i):
				etiqueta0= hallar_region_etiqueta(hijo[i-1],E,X)
				etiqueta1= hallar_region_etiqueta(hijo[i],E,X)
				etiqueta2= hallar_region_etiqueta(hijo[i+1],E,X)
				varianza_a=Dissimilarity_Distance_Computation(etiqueta0)
				varianza_b=Dissimilarity_Distance_Computation(etiqueta1)
				varianza_c=Dissimilarity_Distance_Computation(etiqueta2)
				dist_1=distancia_euclidiana(varianza_b,varianza_a)
				dist_2=distancia_euclidiana(varianza_b,varianza_c)
				distancia_minima=0
				conquien=0
				d=0
				if(dist_2<=dist_1):
					distancia_minima=dist_2
					conquien=poblacion[x][i+1]
					d=dist_2
				else:
					distancia_minima=dist_1
					conquien=poblacion[x][i-1]
					d=dist_1

				if(distancia_minima<umbral):
					rm.append((poblacion[x][i],d))
					poblacion[x][i]=poblacion[x][conquien]
		region_removida2.append(rm)

		inicial+=1	
			
fitness_=[]

def fitness_global(e,X):
	
	for x in range(tam_poblacion):
		suma=0
		for i in range(len(region_removida[x])):
			poss=region_removida[i][0][0]
			region_rm=hallar_region_etiqueta(poss,e,X)
			suma+=1/(len(region_rm)*region_removida[i][0][1])
			#print(poss," ",suma,"  ",(len(region_rm),"  ",region_removida[i][0][1]))
		fitness_.append(suma)

	#print(fitness_)
def fitness(e,X,ia):
	suma=0
	for x in range(tam_poblacion):
		if(x==ia):
			for i in range(len(region_removida[x])):
				poss=region_removida[i][0][0]
				region_rm=hallar_region_etiqueta(poss,e,X)
				suma+=1/(len(region_rm)*region_removida[i][0][1])
				#print(poss," ",suma,"  ",(len(region_rm),"  ",region_removida[i][0][1]))
	return suma

def fitnes_2(e,x):
	suma=0
	for i in range(len(region_removida2)):
		poss=region_removida2[i][0][0]
		region_rm=hallar_region_etiqueta(poss,e,x)
		if(len(region_rm)*region_removida2[i][0][1]==0):
			return 0
		suma+=1/(len(region_rm)*region_removida2[i][0][1])
		#print(poss," ",suma,"  ",(len(region_rm),"  ",region_removida[i][0][1]))
	return suma

def mostrar_poblacion(m):
	for i in range (tam_poblacion):
		print(m[i]," ")


def cruzamiento_1_punto(a,b):
	hijo=[]
	medio =int(tam_cromosoma/2)
	for i in range(medio):
		hijo.append(poblacion[a][i])
	for i in range(medio,tam_cromosoma):
		hijo.append(poblacion[b][i])
	return hijo

mejor_global=0
def mejor_global_():
	for i in range (len(fitnes_)-1):
		if i >= i+1:
			mejor_global=i
		else:
			mejor_global=i+1
def AG(X,e,c,i2_C):
	poblacion_inicial()
	mostrar_poblacion(poblacion);
	print("\n")
	
	for i in range(2):
		print("iteracion ",i+1)
		procedimiento_fusion(e,X);
		m=random.randint(0,tam_poblacion-1)
		n=random.randint(0,tam_poblacion-1)
		print(m," ",n)
		hijo=cruzamiento_1_punto(m,n)
		procedimiento_fusion_hijo(e,X,hijo)
		if(fitnes_2(e,X)>fitness(e,X,m)):
			for i in range(tam_cromosoma):
				poblacion[m][i]=hijo[i]
		region_removida=[]
		region_removida2=[]
	mostrar_poblacion(poblacion)
	fitness_global(e,X)

def generar_imagen(e,centroides,I2):

	
	ii=poblacion[mejor_global]
	for i in range(tam_cromosoma):
		for x in range(len(e)):
			if(e[x]==i):
				e[x]=ii[i]
	print("_________________________________")
	print(ii)
	I2_compressed = np.choose(e, centroides)
	I2_compressed.shape = I2.shape

	#print(e)
	plt.figure(figsize=(4,4))
	plt.imshow(I2_compressed,cmap='gray')
	plt.axis('off')
	plt.show()


if __name__=="__main__":
	X,e,c,i2_C,I2=crear_etiquetas()
	tiempo_inicial = time() 
	AG(X,e,c,i2_C);
	tiempo_final = time() 
	tiempo_ejecucion = tiempo_final - tiempo_inicial
	print ('El tiempo de ejecucion fue:',tiempo_ejecucion)
	generar_imagen(e,c,I2);

	
