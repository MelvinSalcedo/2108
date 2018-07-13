#include <iostream>
#include <vector>
#include <string>
#include <iostream>   // std::cout
#include <string>
#include <cmath>
#include <time.h>
#include <algorithm>
#include <list>
#include <iterator>
#include<string.h>
#include <fstream>
#include<string>
#include <cstdio>
#include <sstream>

using namespace std;


static int Dim=2;
typedef struct{
	double Valor[2];
	double Fitness;
	double fit;
	int cont;
	double prob;
}Sol;

bool compare(Sol a, Sol b)
{
	if(a.fit<b.fit) return false;
	return true;
}

typedef pair<int,int> tupla;
typedef vector<vector<double> > mat;
typedef vector<Sol> Soluciones;
vector<Sol> Memorisados;
double genRandom(double li, double ls){
	return li + static_cast <double> (rand()) /( static_cast <double> (RAND_MAX/(ls-li)));
}

void printSol(string s,Sol a){
	cout<<s;
	cout<<a.Valor[0]<<"  "<<a.Valor[1]<<"  "<<a.Fitness<<"  "<<a.fit<<" "<<a.cont<<endl;
}

void printSolProb(string s,Sol a){
	cout<<s;
	cout<<a.Valor[0]<<"  "<<a.Valor[1]<<"  "<<a.Fitness<<"  "<<a.fit<<"  Prob: "<< a.prob<<" cont: "<<a.cont<<endl;
}

void printSolucionesProb(string s,Soluciones E){
	cout<<s;
	for(int i=0;i<E.size();i++){
		printSolProb(to_string(i)+")   ",E[i]);
	}
}
void printSoluciones(string s,Soluciones E){
	cout<<s;
	for(int i=0;i<E.size();i++){
		printSol(to_string(i)+")   ",E[i]);
	}
}
void fitness(Sol &a){
	 a.Fitness=pow(a.Valor[0],2)+pow(a.Valor[1],2);
}

void Fit(Sol &a){
	if(a.Fitness>=0)
		a.fit=1/(1+a.Fitness);
	else
		a.fit=1+abs(a.Fitness);
}
void fitness_soluciones(Soluciones &E){
	for(auto &e:E)
		fitness(e);
}

void calcular_fit_soluciones(Soluciones &E){
	for(auto &e:E)
		Fit(e);
}

void Iniciar_poblacion(Soluciones &A, int tam_colmena){
	for(int i=0;i<tam_colmena;i++){
		Sol n_a;
		n_a.Valor[0]= genRandom(-5,5);
		n_a.Valor[1]= genRandom(-5,5);
		n_a.Fitness=0;
		n_a.fit=0;
		n_a.cont=0;
		A.push_back(n_a);
	}
}
void genCandidatas(Soluciones & A,Soluciones & cand,int tam_colmena, int D, int num_soluciones){
	/// v(i,j) = x(i,j) + fi* (x ɸ i,j − xk,j),
	for(int i=0;i<tam_colmena;i++){
		double fi=genRandom(-1,1);
		int j = round((genRandom(0,D-1)));///1,D
		int k = round(genRandom(0,num_soluciones-1));///k=1,sn
        cout<<i<<")"<<"K="<<k<<" j=" <<j<<endl;
		while(k==i)
			 k = round(genRandom(0,num_soluciones-1));
		if(k!=i){
			Sol n;
            n.Valor[j]=A[i].Valor[j]+ fi * (A[i].Valor[j]-A[k].Valor[j]);
			n.Valor[(j+1)%2]=A[i].Valor[(j+1)%2];
			n.Fitness=0;
			n.fit=0;
			n.cont=0;
			cand.push_back(n);
		}

	}
}

void Generando_N_obser(Soluciones & A, int pos,int D, int num_soluciones){
	/// vi,j = xi,j + ɸ · (xi,j − xk,j),

	/**
        Al igual que con el paso de las abejas empleadas, en el paso
    de las observadoras también es necesario calcular soluciones
    candidatas y compararla con la solución en cuestión.
    **/
	double fi=genRandom(-1,1);
	int j = round(genRandom(0,D-1));
	int k = round(genRandom(0,num_soluciones-1));
	while(k==pos)
		k = round(genRandom(0,num_soluciones-1));
    cout<<" \n ---->  fuente  ------> "<<pos<<" K="<<k<<endl;
	if(k!=pos){
		Sol n;
		n.Valor[0]= A[pos].Valor[0];
		n.Valor[1]= A[pos].Valor[1]+ fi * (A[pos].Valor[1]-A[k].Valor[1]);
		n.Fitness=0;
		n.fit=0;
		n.cont=0;
		fitness(n);
		Fit(n);
        /// al compara las apitudes la soluciones no mejro entonces el
        ///contador aumenta en 1
		if(A[pos].fit> n.fit){
			n.cont++;
			A[pos].cont++;
		}
		else{
			n.cont=0; A[pos].cont=0;
			A[pos]=n;
		}
		/// calcular soluciones candidatas  y compararla con la solución
		printSol("\nsolucion candidata de la observadora \n\n",n);
		printSoluciones("\nNuevos Mejores \n\n", A);
	}
}

/**
Se evalúa la aptitud de las soluciones candidatas y se
comparan con las soluciones actuales y se mantienen las
mejores soluciones, cabe destacar que si una solución
candidata mejora una solución actual su contador se hace
igual a 0 de lo contrario si se mantiene la solución entonces
se incrementa en 1 la solución, por ejemplo:
**/
void Comparar_Soluciones(Soluciones & A, Soluciones & cand){
	for(int i=0;i<A.size();i++){
		if(A[i].fit>=cand[i].fit){
			cand[i].cont++;
		}
	}
}

Soluciones Mejore_Solucione(Soluciones &A , Soluciones &cand){
	Soluciones B;
	for(int i=0;i<A.size();i++){
		if(cand[i].fit < A[i].fit){
			A[i].cont++;
			B.push_back(A[i]);
		}
		else
			B.push_back(cand[i]);
	}
	return B;
}

int probSeleccion(Soluciones & B){
	double ruleta=0;
	float suma_total=0;
	///sumatoria de todos los fits
	for(int i=0;i<B.size();i++){
		ruleta += B[i].fit;
	}

	for(int i=0;i<B.size();i++){
		B[i].prob=B[i].fit/ruleta;
		suma_total+=B[i].prob;
	}cout<<"SUMA_TOTAL_PROB = "<<suma_total<<endl;

	double acumulado_probavilidad=0;
	double r=genRandom(0,1);
	for(int i=0;i<B.size();i++){
		acumulado_probavilidad+=B[i].prob;
		if(r<acumulado_probavilidad)
			return i;
	}

	return B.size()-1;

}
Sol CalcBest(Soluciones & A)
{
	sort(A.begin(), A.end(),compare);
	return A[0];
}

void excedLimt(Soluciones & A, int L ){

	for(auto &y: A){
		if(y.cont > L){
            cout<<"(reiniciando)"<<endl;
			for(int j=0;j<2;j++){
				y.Valor[j]= min(y.Valor[0],y.Valor[1])+ genRandom(0,1)*(max(y.Valor[0],y.Valor[1])-min(y.Valor[0],y.Valor[1]));
			}
			y.Fitness=0;
			y.fit=0;
			y.cont=0;
			fitness(y);Fit(y);
		}

	}
}
int main(){
	srand(time(0));
	int tam_colmena=3; ///tamaño de colmena
	int D=2 ; ///Numero de variables
	int L=6; ///num de iteracciones antes de abandonar la fuente=C_S*D/2=6
	int MCN=5; ///Maximo numero de ciclos que itera el algorimo
	int num_soluciones=3; ///numero de soluciones
	Soluciones A;
	Soluciones cand, best;
	Iniciar_poblacion(A,tam_colmena);
	cout<<"size"<<A.size()<<endl;
	//printSoluciones("\n Poblacacion inicial \n\n", A);
	fitness_soluciones(A);
	//printSoluciones("\n Fitness \n\n", A);
	calcular_fit_soluciones(A);
	printSoluciones("\n Fuente de alimentos iniciales \n\n", A);

	int ite=0;

    Sol nn;
	while(ite<1){
        cout<<"%%%%%%%%%%%%%%%%%%%%%%%%ITERACION "<<ite+1<<"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"<<endl;
        printSoluciones("\n poblacion\n", A);

		cand.clear();
		//best.clear();
		//cand=best;s
		//printSoluciones("\n Calculando datos iniciales \n\n", A);
		cout<<"Soluciones Candidatas"<<endl;
		genCandidatas(A,cand,tam_colmena,D,num_soluciones);
		//printSoluciones("\n Soluciones Candidatas \n\n", cand);
		fitness_soluciones(cand);
		calcular_fit_soluciones(cand);
		Comparar_Soluciones(A,cand);
		printSoluciones("\n", cand);
		A=Mejore_Solucione(A,cand);
		printSoluciones("\n Mejores Soluciones \n\n", A);

		int obs=0;
		while(obs<num_soluciones){
			cout<<"\n ==========Observadora "<<obs+1<<"========="<<endl;
			//probSeleccion(A);
			double solEle =probSeleccion(A);
			printSolucionesProb("\nProbabilidad \n\n", A);
			Generando_N_obser(A,solEle,D,num_soluciones);
			obs++;
		}

		excedLimt(A,L);
		printSolucionesProb("\n Ciclo Terminado , Comprobando si se acabo recursos \n \n ", A);
		sort(A.begin(),A.end(),compare);
		if(nn.fit< A[0].fit){
			nn= A[0];
			Memorisados.push_back(nn);
		}
		printSol("Memorisando\n", nn);
		ite++;
		//break;
	}
	return 0;
}
