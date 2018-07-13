import numpy as np
import random
import copy
import matplotlib.pyplot as plt

limite_inferior_X = 0.
limite_superior_X = 5.

limite_inferior_Y = 0.
limite_superior_Y = 3.

Datos = []

n_functions = 2
numero_iteraciones = 2
tam_poblacion = 50
tam_hijo = 10
n_adversaries = 8
beta = 0.5
alpha = 1.
mutation_prob = 0.5   # .0 - 1.

distancia_ciudades=[ [0, 12, 3, 23, 1, 5, 23, 56, 12, 11],
				   [12, 0, 9, 18, 3, 41, 45, 5, 41, 27],
				   [3, 9, 0, 89, 56, 21, 12, 48, 14, 29],
				   [23, 18, 89, 0, 87, 46, 75, 17, 50, 42],
				   [1, 3, 56, 87, 0, 55, 22, 86, 14, 33],
				   [5, 41, 21, 46, 55, 0, 21, 76, 54, 81],
				   [23, 45, 12, 75, 22, 21, 0, 11, 57, 48],
				   [56, 5, 48, 17, 86, 76, 11, 0, 63, 24],
				   [12, 41, 14, 50, 14, 54, 57, 63, 0, 9],
				   [11, 27, 29, 42, 33, 81, 48, 24, 9, 0] ]

Costo = [               [0, 22, 47, 15, 63, 21, 23, 16, 11, 9],
						[22, 0, 18, 62, 41, 52, 13, 11, 26, 43],
						[47, 18, 0, 32, 57, 44, 62, 20, 8, 36],
						[15, 62, 32, 0, 62, 45, 75, 63, 14, 12],
						[63, 41, 57, 62, 0, 9, 99, 42, 56, 23],
						[21, 52, 44, 45, 9, 0, 77 ,58, 22, 14],
						[23, 13, 62, 75, 99, 77, 0, 30, 25, 60],
						[16, 11, 20, 63, 42, 58, 30, 0, 66, 85],
						[11, 26, 8, 14, 56, 22, 25, 66, 0, 54],
						[9, 43, 36, 12, 23, 14, 60, 85, 54, 0]]


n_cities = len( distancia_ciudades )

def generate_individual_tsp():
	return { "cm": np.random.permutation( n_cities ), "f_distance": np.inf, \
					"f_cost": np.inf }

def genrar_poblacion_tsp():
	for i in range( tam_poblacion ):
		Datos.append( generate_individual_tsp() )

def evaluate_population_tsp():
	for i in Datos:
		i["f_distance"] = fitness_distancia( i )
		i["f_cost"] = fitness_costo( i )

def fitness_distancia( individual ):
	total_distance = 0
	for i in range( n_cities -1 ):
		total_distance += distancia_ciudades[ individual["cm"][i] ]\
										[ individual["cm"][i+1] ]
	return total_distance

def fitness_costo( individual ):
	total_cost = 0
	for i in range( n_cities -1 ):
		total_cost += Costo[ individual["cm"][i] ]\
										[ individual["cm"][i+1] ]
	return total_cost

def cruzamiento_PBX(individual_1, individual_2):
	offspring = []

	mother_cromosome = individual_1["cm"]
	father_cromosome = individual_2["cm"]

	son_1, son_2 = [], []

	mask = np.random.randint(2, size=len(mother_cromosome))

	for i, j in zip(range(len(mother_cromosome)), mask):

		if(j == 1):
			son_1.append(father_cromosome[i])
			son_2.append(mother_cromosome[i])
		else:
			son_1.append(-1)
			son_2.append(-1)

	# print(son_1)
	# print(son_2)

	for i in mother_cromosome:
		if (i not in son_1):
			tmp = son_1.index(-1)
			son_1[tmp] = i

	for i in father_cromosome:
		if (i not in son_2):
			tmp = son_2.index(-1)
			son_2[tmp] = i

	# print(*son_1, sep='')
	# print(*son_2, sep='')

	new_individual_1 = { "cm": son_1 }
	new_individual_2 = { "cm": son_2 }

	new_individual_1["f_distance"] = fitness_distancia( new_individual_1 )
	new_individual_2["f_distance"] = fitness_distancia( new_individual_2 )

	new_individual_1["f_cost"] = fitness_costo( new_individual_1 )
	new_individual_2["f_cost"] = fitness_costo( new_individual_2 )

	offspring.append(new_individual_1)
	offspring.append(new_individual_2)

	return offspring

def mutacion_tsp( individual_1 ):
	perm_tmp = list(np.random.permutation( n_cities ))
	individual_1["cm"][perm_tmp[1]], individual_1["cm"][perm_tmp[0]] = \
	 	individual_1["cm"][perm_tmp[0]], individual_1["cm"][perm_tmp[1]]

def seleccion_torneo():
	adversaries = np.random.permutation( list( range( tam_poblacion ) ) )
	tmp = [ Datos[i] for i in adversaries[:n_adversaries]]
	sums = [ i["f_distance"]+i["f_cost"] for i in tmp ]
	m_min = min( sums )
	return Datos[adversaries[sums.index(m_min)]]

def function_1( x, y):
	return 4*(x**2) + 4*(y**2)

def function_2( x, y):
	return (x-5)**2 + (y-5)**2

def generate_individual():
	return {"x": random.uniform(limite_inferior_X, limite_superior_X), \
			"y": random.uniform(limite_inferior_Y, limite_superior_Y), \
			"fitness_1": np.inf, \
			"fitness_2": np.inf }

def generate_population():
	for i in range( tam_poblacion ):
		Datos.append( generate_individual() )

def evaluate_population():
	for i in Datos:
		i["fitness_1"] = function_1( i["x"], i["y"] )
		i["fitness_2"] = function_2( i["x"], i["y"] )

def tournament_selection():
	adversaries = np.random.permutation( list( range( tam_poblacion ) ) )
	tmp = [ Datos[i] for i in adversaries[:n_adversaries]]
	sums = [ i["fitness_1"]+i["fitness_2"] for i in tmp ]
	m_min = min( sums )
	return Datos[adversaries[sums.index(m_min)]]

def BLX_crossover( parent_1, parent_2 ):
	m_beta = random.uniform( beta - alpha, beta + alpha )

	m_x = parent_1["x"] + m_beta*( parent_2["x"] - parent_1["x"] )
	m_y = parent_1["y"] + m_beta*( parent_2["y"] - parent_1["y"] )
	m_1 = function_1( m_x, m_y )
	m_2 = function_2( m_x, m_y )

	return {"x": m_x, "y": m_y, "fitness_1": m_1, "fitness_2": m_2 }

def uniform_mutation( individual ):
	if( random.randint(0, 1) ):
		individual["x"] = random.uniform(limite_inferior_X, limite_superior_X)
	else:
		individual["y"] = random.uniform(limite_inferior_Y, limite_superior_Y)
	individual["fitness_1"] = function_1( individual["x"], individual["y"] )
	individual["fitness_2"] = function_2( individual["x"], individual["y"] )

def valid_individual( individual ):
	if( limite_inferior_X <= individual["x"] <= limite_superior_X and \
		limite_inferior_Y <= individual["y"] <= limite_superior_Y ):
		return True
	return False

def Dominado( individual_1, individual_2 ):
		if( ( individual_1["fitness_1"] < individual_2["fitness_1"] and \
			  individual_1["fitness_2"] < individual_2["fitness_2"] ) or \
			( individual_1["fitness_1"] <= individual_2["fitness_1"] and \
			  individual_1["fitness_2"] < individual_2["fitness_2"] ) or \
			( individual_1["fitness_1"] < individual_2["fitness_1"] and \
			  individual_1["fitness_2"] <= individual_2["fitness_2"] ) ):
			return True
		return False

def Dominado_tsp( individual_1, individual_2 ):
		if( ( individual_1["f_distance"] < individual_2["f_distance"] and \
			  individual_1["f_cost"] < individual_2["f_cost"] ) or \
			( individual_1["f_distance"] <= individual_2["f_distance"] and \
			  individual_1["f_cost"] < individual_2["f_cost"] ) or \
			( individual_1["f_distance"] < individual_2["f_distance"] and \
			  individual_1["f_cost"] <= individual_2["f_cost"] ) ):
			return True
		return False



def ordenar_tipo_no_dominado():
	S = []
	N = []
	rank = []
	Frontera = [[]]

	for i in range( len( Datos ) ):
		S.append([])
		N.append( 0 )
		rank.append( 0 )

	for p in  range(len( Datos )) :
		for q in range(len( Datos )):
			if( Dominado(Datos[p], Datos[q]) ):
				S[p].append(q)
			elif( Dominado( Datos[q], Datos[p]) ):
				N[p] += 1
		if(N[p] == 0):
			rank[p] = 0
			Frontera[0].append(p)
	i = 0
	while( Frontera[i] != [] ):
		Q = []
		for p in Frontera[i]:
			for q in S[p]:
				N[q] -= 1
				if( N[q] == 0 ):
					rank[q] = i+1
					Q.append(q)
		i += 1
		Frontera.append( Q )
	print(Frontera)
	del Frontera[len(Frontera)-1]
	return Frontera

def ordenar_tipo_no_dominado_tsp():
	S = []
	N = []
	rank = []
	Frontera = [[]]

	for i in range( len( Datos ) ):
		S.append([])
		N.append( 0 )
		rank.append( 0 )

	for p in  range(len( Datos )) :
		for q in range(len( Datos )):
			if( Dominado_tsp(Datos[p], Datos[q]) ):
				S[p].append(q)
			elif( Dominado_tsp( Datos[q], Datos[p]) ):
				N[p] += 1
		if(N[p] == 0):
			rank[p] = 0
			Frontera[0].append(p)
	i = 0
	while( Frontera[i] != [] ):
		Q = []
		for p in Frontera[i]:
			for q in S[p]:
				N[q] -= 1
				if( N[q] == 0 ):
					rank[q] = i+1
					Q.append(q)
		i += 1
		Frontera.append( Q )
	del Frontera[len(Frontera)-1]
	return Frontera

def crowding_distance( Frontera ):
	Distancias = dict()
	for f in Frontera:
		distance = [ 0. ] * len(f)
		for m in range( n_functions ):
			m_sorted = [ [i, Datos[i]["fitness_"+str(m+1)]] for i in f ]
			m_sorted = sorted( m_sorted, key=lambda x: x[1] )
			if( len(m_sorted) > 1):
				distance[0] = np.inf
				distance[ len(f)-1 ] = np.inf

				if( len(m_sorted) > 2):
					m_max = max(m_sorted, key=lambda item: item[1])[1]
					m_min = min(m_sorted, key=lambda item: item[1])[1]
					if( m_max - m_min == 0):
						divisor = 10e-5
					else:
						divisor = m_max - m_min
					for k in range(1, len(f)-1 ):
						distance[k] += (m_sorted[k+1][1] - m_sorted[k-1][1]) / \
									   (divisor)
			else:
				distance[0] = 0
		for i in range( len(f) ):
			Distancias[f[i]] = distance[i]
	return Distancias

def agrupamiento_distancia( Frontera ):
	Distancias = dict()
	for f in Frontera:
		distance = [ 0. ] * len(f)
		for m in range( n_functions ):
			if( m == 0):
				m_sorted = [ [i, Datos[i]["f_distance"]] for i in f ]
			elif( m == 1):
				m_sorted = [ [i, Datos[i]["f_cost"]] for i in f ]
			m_sorted = sorted( m_sorted, key=lambda x: x[1] )
			if( len(m_sorted) > 1):
				distance[0] = np.inf
				distance[ len(f)-1 ] = np.inf

				if( len(m_sorted) > 2):
					m_max = max(m_sorted, key=lambda item: item[1])[1]
					m_min = min(m_sorted, key=lambda item: item[1])[1]
					if( m_max - m_min == 0):
						divisor = 10e-5
					else:
						divisor = m_max - m_min
					for k in range(1, len(f)-1 ):
						distance[k] += (m_sorted[k+1][1] - m_sorted[k-1][1]) / \
									   (divisor)
			else:
				distance[0] = 0
		for i in range( len(f) ):
			Distancias[f[i]] = distance[i]
	return Distancias

def crowded_tournament_selection(Frontera, Distancias):
	global Datos
	Datos_nuevos = []
	front = dict()
	c = 0
	for i in range( len( Frontera) ):
		for j in range( len(Frontera[i]) ):
			front[c] = Frontera[i][j]
			c+=1
	tmp = list( range( len(Datos) ) )
	while( len(Datos_nuevos) < tam_poblacion ):
		perm_tmp = list(np.random.permutation( tmp ))
		# print( perm_tmp )
		rand_1 = perm_tmp[0]
		rand_2 = perm_tmp[1]
		# print( rand_1 )
		# print( rand_2 )
		if( front[rand_1] > front[rand_2] ):
			Datos_nuevos.append( Datos[rand_1] )
			tmp.remove(rand_1)
		elif( front[rand_1] < front[rand_2] ):
			Datos_nuevos.append( Datos[rand_2] )
			tmp.remove(rand_2)
		elif( front[rand_1] == front[rand_2] ):
			if( Distancias[rand_1] >= Distancias[rand_2] ):
				Datos_nuevos.append( Datos[rand_1] )
				tmp.remove(rand_1)
			else:
				Datos_nuevos.append( Datos[rand_2] )
				tmp.remove(rand_2)

	return Datos_nuevos

def seleccion_agrupamiento( Frontera, Distancias):
	global Datos
	Datos_nuevos = []

	while( True ):
		for i in Frontera:
			for j in i:
				Datos_nuevos.append( Datos[j] )
				# print(len(Datos_nuevos))
				if( len(Datos_nuevos) >= tam_poblacion ):
					# print("asd")
					return Datos_nuevos

def minimize_F():
	global Datos
	iteracion = 0

	generate_population()
	evaluate_population()

	while( iteracion <= numero_iteraciones ):

		print("iteracion #", iteracion)

		for i in range(tam_hijo):

			while(True):
				m_individual = BLX_crossover( tournament_selection(), \
											  tournament_selection() )
				if( random.random() <= mutation_prob ):
					uniform_mutation( m_individual )
				if( valid_individual( m_individual ) ):
					break

			Datos.append( m_individual )
			Datos.append( m_individual )

		print("Datos\t", len(Datos))

		Frontera = ordenar_tipo_no_dominado()
		Distancias = crowding_distance( Frontera )

		# Datos_nuevos = crowded_tournament_selection( Frontera, Distancias )
		Datos_nuevos = seleccion_agrupamiento( Frontera, Distancias )
		Datos = []
		Datos = copy.deepcopy( Datos_nuevos )


		iteracion += 1

		for i in Datos:
			print("fitness_1  ", i["fitness_1"], "\tfitness_2  ", i["fitness_2"])

	print(Datos)
	plt.plot([ i["fitness_1"] for i in Datos ], \
				[i["fitness_2"] for i in Datos], 'ro')
	# plt.axis([0, 6, 0, 20])
	plt.show()


array_no_dominados=[]
def guardar_array_no_dominantes():
    array_=ordenar_tipo_no_dominado_tsp()
    array_no_dominados.append(array_)


def minimisar_tsp():
	global Datos
	iteracion = 0
    #w=random.random()
	genrar_poblacion_tsp()
	evaluate_population_tsp()

	while( iteracion <= numero_iteraciones ):

		print("iteracion #", iteracion)

		for i in range( tam_hijo ):
			m_individual = cruzamiento_PBX( seleccion_torneo(), \
										  seleccion_torneo() )
			if( random.random() <= mutation_prob ):
				mutacion_tsp( m_individual[0] )
			if( random.random() <= mutation_prob ):
				mutacion_tsp( m_individual[1] )

			Datos.append( m_individual[0] )
			Datos.append( m_individual[1] )

		# print("Datos\t", len(Datos))

		Frontera = ordenar_tipo_no_dominado_tsp()
        #temporal=guardar_array_no_dominantes()
		Distancias = agrupamiento_distancia( Frontera )

		Datos_nuevos = seleccion_agrupamiento( Frontera, Distancias )
		Datos = []
		Datos = copy.deepcopy( Datos_nuevos )

		print(Frontera)

		iteracion += 1

		#for i in Datos:
		#	print("distance  ", i["f_distance"], "\tcost  ", i["f_cost"])

	print(Datos)
	plt.plot([ i["f_distance"] for i in Datos ],[i["f_cost"] for i in Datos], 'ro')
	#plt.show()



if __name__ == "__main__":
	minimisar_tsp()
