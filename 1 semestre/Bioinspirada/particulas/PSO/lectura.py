import matplotlib.pyplot as plt
import numpy as np
archivo = open("salida.txt", "r")
x=[]
for linea in archivo.readlines():
    x.append(linea.split("\n"))
a=[]
i=[]
j=[]
t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)
for m in x:
    a.append(m[0])

for m in range(0,len(a),2):
    i.append(float(a[m]))
    j.append(float(a[m+1]))

print(len (i)," ",len(j))
plt.plot(i,j,'ro')
plt.show()
