#include <map>
#include <iostream>
#include <vector>
#include <time.h>
#include <cmath>
#include <stdlib.h>
#include <string>
#include <cstdio>
#include <algorithm>
using  namespace std;
char letra_F='A';
char letra_C='A';

typedef float num;
num mejor_Camino=1000;
const int maximo=13;
num   D[maximo][maximo] = {{ 0,12, 3,23, 1, 5,23,56,12,11,89,97,52},
                            {12, 0, 9,18, 3,41,45, 5,41,27,16,76,56},
                            { 3, 9, 0,89,56,21,12,48,14,29, 5,91, 8},
                            {23,18,89,0,87,46,75,17,50,42,100,70,15},
                            { 1, 3,56,87, 0,55,22,86,14,33,31,84,21},
                            { 5,41,21,46,55, 0,21,76,54,81,92,37,22},
                            {23,45,12,75,22,21, 0,11,57,48,39,59,22},
                            {56, 5,48,17,86,76,11, 0,63,24,55,58,98},
                            {12,41,14,50,14,54,57,63, 0, 9,44,18,52},
                            {11,27,29,42,33,81,48,24, 9, 0,64,65,82},
                            {89,16,5,100,31,92,39,55,44,64, 0, 9,70},
                            {97,76,91,70,84,37,59,58,18,65, 9, 0,50},
                            {52,56, 8,15,21,22,22,98,52,82,70,50,0}};

num   V[maximo][maximo];
num   F[maximo][maximo];


int cantidad_hormigas=10;
int cantidad_iteraciones=50;

num Ro=0.099;
num alfa=1,beta=1;
num Q=1.0;
num feromona_inicial=0.1;

vector<pair<pair<num,num>,num > > v_provavilidad_hormigas;
vector<pair<num,num>> v_guardadps;
vector<vector<int> > caminos_rrecorridos_hormigas;

void mostrar_poblacion_siguiente(){
    cout<<"CAMINOS "<<endl;
    for(int x=0;x<v_provavilidad_hormigas.size();x++){
        cout<<"["<<v_provavilidad_hormigas[x].first.first<<"] -> "<<
        "["<<v_provavilidad_hormigas[x].first.second<<"] = "<<
        "["<<v_provavilidad_hormigas[x].second<<"]"<<endl;
    }
}

void generar_pesos_grafo(){
    for(int i = 0; i < maximo; i++){
		for(int j = 0; j < maximo; j++){
		    num vi = 1.0/D[i][j];
			if(i==j){
                //D[i][j]=0;
                V[i][j]=0;
                F[i][j]=0;
            }
			else{
                //D[i][j]=r;
                V[i][j]=vi;
                F[i][j]=0.1;
            }
		}
	}

}

void mostrar_poblacion_inicial(){
    cout<<"MATRIZ DISTANCIA"<<endl;

    for(int i = 0; i < maximo; i++){
        cout <<"\t["<<letra_F<<"]";
        letra_F++;
    }cout <<endl;

	for(int i = 0; i < maximo; i++){
        cout <<"["<<letra_C<<"]";
		for(int j = 0; j < maximo; j++)
			cout <<"\t"<<D[i][j];
		cout <<" "<<endl;
		letra_C++;
	}
	cout << "\n";

	cout<<"MATRIZ VISIBILIDAD"<<endl;
    letra_F='A';
    letra_C='A';
    for(int i = 0; i < maximo; i++){
        cout <<"\t["<<letra_F<<"]";
        letra_F++;
    }cout <<endl;

	for(int i = 0; i < maximo; i++){
        cout <<"["<<letra_C<<"]";
		for(int j = 0; j < maximo; j++)
			cout <<"\t"<<V[i][j];
		cout <<" "<<endl;
		letra_C++;
	}
	cout << "\n";
	cout<<"MATRIZ FEROMONA"<<endl;
    letra_F='A';
    letra_C='A';
    for(int i = 0; i < maximo; i++){
        cout <<"\t["<<letra_F<<"]";
        letra_F++;
    }cout <<endl;

	for(int i = 0; i < maximo; i++){
        cout <<"["<<letra_C<<"]";
		for(int j = 0; j < maximo; j++)
			cout <<"\t"<<F[i][j];
		cout <<" "<<endl;
		letra_C++;
	}
	cout << "\n";

}

bool buscar_repetidos(pair<num,num> temporal){
    for(int x=0;x<v_guardadps.size();x++){
        if((v_guardadps[x].first==temporal.first &&
           v_guardadps[x].second==temporal.second) ||
           (v_guardadps[x].first==temporal.second &&
           v_guardadps[x].second==temporal.first)){
                return true;
           }
    }
    return false;
}

bool buscar_camino_hormiga(int z,int a,int b){
    for(int x=0;x<caminos_rrecorridos_hormigas[0].size()-1;x++){

        if((caminos_rrecorridos_hormigas[z][x]==a && caminos_rrecorridos_hormigas[z][x+1]==b) ||
            (caminos_rrecorridos_hormigas[z][x]==b && caminos_rrecorridos_hormigas[z][x+1]==a)){
            return true;
        }

    }
    return false;
}

int fitness(vector<int> n){
    int sum=0;
    for(int x=0;x<n.size()-1;x++){
        //cout<<"--> "<<D[n[x]][n[x+1]]<<endl;
        sum+=D[n[x]][n[x+1]];
    }
    return sum;
}

void cambiar_feromonas(){
    for(int x=0;x<maximo;x++){
        for(int y=0;y<maximo;y++){
            if(x!=y){
                num suma=Ro;
                suma*=F[x][y];
                for(int z=0;z<cantidad_hormigas;z++){
                    if(buscar_camino_hormiga(z,x,y)==true){
                        suma += Q/fitness(caminos_rrecorridos_hormigas[z]);
                    }
                }
                F[x][y]=suma;
            }else{F[x][y]=0;}
        }
    }
}

bool comprovar_resultado_optimo(){
    int suma=1;
    num valor=fitness(caminos_rrecorridos_hormigas[0]);
    for(int x=1;x<caminos_rrecorridos_hormigas.size();x++){
        if(fitness(caminos_rrecorridos_hormigas[x])==valor){
            suma++;
        }
    }
    if(suma==cantidad_hormigas){return true;}
    return false;
}

void actualizar_Arcos_escalada_primera_opcion(){
    for(int x=0;x<cantidad_hormigas;x++){
        for(int y=0;y<maximo-1;y++){
            ///__________________________________________
            num suma=Ro;
            suma*=F[caminos_rrecorridos_hormigas[x][y]][caminos_rrecorridos_hormigas[x][y+1]];
            suma += Q/fitness(caminos_rrecorridos_hormigas[x]);
            ///__________________________________________
            F[caminos_rrecorridos_hormigas[x][y]][caminos_rrecorridos_hormigas[x][y+1]]=suma;
        }
    }
}
void busqueda_local_ESCALADA_PRIMERA_OPCION(){
    for(int x=0;x<cantidad_hormigas;x++){
        vector<int> camino_h;
        int sum=0;
        int fitness_=0;
        for(int m=0;m<5;m++){
            do{
                sum++;
                camino_h=caminos_rrecorridos_hormigas[x];

                int pos_a=rand()%(maximo);
                int pos_b=rand()%(maximo);

                int temporal=camino_h[pos_a];
                camino_h[pos_a]=camino_h[pos_b];
                camino_h[pos_b]=temporal;
                fitness_=fitness(camino_h);
            }while(fitness(camino_h) > fitness(caminos_rrecorridos_hormigas[x]));
            caminos_rrecorridos_hormigas[x]=camino_h;
        }


    }
    actualizar_Arcos_escalada_primera_opcion();
    ///
}
void caminos_hormigas(){
    cout<<"ciudad nueva = 0"<<endl;
    vector<int>camino;
    for(int p=0;p<cantidad_iteraciones;p++){
        for(int c=0;c<cantidad_hormigas;c++){
            int ciudad_inicial=0;
            for(int z=0;z<maximo;z++){
                camino.push_back(ciudad_inicial);
                for(int x=0;x<maximo;x++){
                   if(x!=ciudad_inicial && buscar_repetidos(make_pair(ciudad_inicial,x))==false){
                        float t=pow(F[ciudad_inicial][x],alfa);
                        float n=pow(V[ciudad_inicial][x],beta);

                        v_provavilidad_hormigas.push_back(make_pair(make_pair(ciudad_inicial,x),n*t));
                        v_guardadps.push_back(make_pair(ciudad_inicial,x));
                   }
                }
                /// hallar la suma
                float suma_total=0;
                float summm=0;
                for(int x=0;x<v_provavilidad_hormigas.size();x++){
                    suma_total+=v_provavilidad_hormigas[x].second;
                }
                ///hallando la divicion
                for(int x=0;x<v_provavilidad_hormigas.size();x++){
                    v_provavilidad_hormigas[x].second = v_provavilidad_hormigas[x].second/suma_total;
                }
                ///hallando la provavilidad
                for(int x=0;x<v_provavilidad_hormigas.size();x++){
                    if(x==0){
                        v_provavilidad_hormigas[x].second = v_provavilidad_hormigas[x].second;
                    }
                    else{
                        v_provavilidad_hormigas[x].second += v_provavilidad_hormigas[x-1].second;
                    }
                }
                double prob = ((double) rand() / ((double)RAND_MAX + 1));
                /*for(int x=0;x<v_provavilidad_hormigas.size();x++){
                    cout<<v_provavilidad_hormigas[x].first.first<<" -> "<<
                    v_provavilidad_hormigas[x].first.second<<" = "<<
                    v_provavilidad_hormigas[x].second<<endl;
                }
                cout<<"random = "<<prob<<endl;cout<<endl;*/
                //mostrar_poblacion_siguiente();
                ///escojuedo la nueva ciudad
                //cout<<"prob "<<prob<<endl;
                for(int x=0;x<v_provavilidad_hormigas.size();x++){
                    if(x==0){
                        if(prob<=v_provavilidad_hormigas[0].second){
                            ciudad_inicial=v_provavilidad_hormigas[0].first.second;
                            break;
                        }
                    }
                    else{
                        if(prob>v_provavilidad_hormigas[x-1].second &&
                           prob<=v_provavilidad_hormigas[x].second){
                            ciudad_inicial=v_provavilidad_hormigas[x].first.second;
                            break;
                        }
                    }
                }
               // cout<<"ciudad nueva = "<<ciudad_inicial<<endl;
                v_provavilidad_hormigas.clear();
                //cout<<"------"<<endl;
            }

            caminos_rrecorridos_hormigas.push_back(camino);
            camino.clear();
            v_guardadps.clear();
            v_provavilidad_hormigas.clear();
        }
        cout<<"\nMEJOR MAXIMO_GLOBLAL"<<mejor_Camino<<endl;
        cout<<"\nPOBLACION ANTES DE LA BUSQUEDA LOCAL = "<<p<<endl;
        for(int x=0;x<caminos_rrecorridos_hormigas.size();x++){
            cout<<"hormiga "<<x+1<<" = [ ";
            for(int y=0;y<caminos_rrecorridos_hormigas[0].size();y++){
                char le=caminos_rrecorridos_hormigas[x][y]+65;
                cout<<le<<" ";
            }cout<<"] peso = "<<fitness(caminos_rrecorridos_hormigas[x])<<endl;
        }
        busqueda_local_ESCALADA_PRIMERA_OPCION();

        cout<<"\nPOBLACION DESPUES DE LA BUSQUEDA LOCAL = "<<p<<endl;
        for(int x=0;x<caminos_rrecorridos_hormigas.size();x++){
            cout<<"hormiga "<<x+1<<" = [ ";
            for(int y=0;y<caminos_rrecorridos_hormigas[0].size();y++){
                char le=caminos_rrecorridos_hormigas[x][y]+65;
                cout<<le<<" ";
            }cout<<"] peso = "<<fitness(caminos_rrecorridos_hormigas[x])<<endl;

            if(fitness(caminos_rrecorridos_hormigas[x])<mejor_Camino){
                mejor_Camino=fitness(caminos_rrecorridos_hormigas[x]);
            }
        }
        cambiar_feromonas();
        //if(comprovar_resultado_optimo()==true){break;}
        caminos_rrecorridos_hormigas.clear();

        cout<<"MATRIZ FEROMONA"<<endl;
        letra_F='A';
        letra_C='A';
        for(int i = 0; i < maximo; i++){
            cout <<"\t["<<letra_F<<"]";
            letra_F++;
        }cout <<endl;

        for(int i = 0; i < maximo; i++){
            cout <<"["<<letra_C<<"]";
            for(int j = 0; j < maximo; j++)
                cout <<"\t"<<F[i][j];
            cout <<" "<<endl;
            letra_C++;
        }
        cout << "\n";
    }
}

int main(){
    srand(time(0));
    generar_pesos_grafo();
    mostrar_poblacion_inicial();
    cout<<"___________________________________"<<endl;
    caminos_hormigas();
    //mostrar_poblacion_inicial();
    return 0;
}
