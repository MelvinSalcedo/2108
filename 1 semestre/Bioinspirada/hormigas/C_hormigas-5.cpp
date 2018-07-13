#include <map>
#include <iostream>
#include <vector>
#include <time.h>
#include <cmath>
#include <stdlib.h>
#include <string>
#include <cstring>
#include <cstdio>
#include <algorithm>
using  namespace std;
char letra_F='A';
char letra_C='A';

typedef float num;
const int maximo=5;
num   D[maximo][maximo] = {{ 0,50,50,94,50},
                           {50, 0,22,50,36},
                           {50,22, 0,44,14},
                           {94,50,44, 0,50},
                           {50,36,14,50, 0}};

num   V[maximo][maximo];
num   F[maximo][maximo];

num Fer[maximo][maximo]={{ 0, 0, 2, 0, 3},
                         { 0, 0, 0, 3, 0},
                         { 2, 0, 0, 0, 0},
                         { 0, 3, 0, 0, 1},
                         { 3, 0, 0, 1, 0}};
int cantidad_hormigas=3;
int cantidad_iteraciones=30;

num Ro=0.99;
num alfa=1,beta=1;
num Q=1.0;
num e=5;
num feromona_inicial=0.1;
num w=10.0;
num q0=0.7;
num phy=0.05;


num feromona_maxima=0.1;
num feromana_minima=0.11;
num maximo_local=1110;
vector<pair<pair<num,num>,num > > v_provavilidad_hormigas;
vector<pair<pair<num,num>,num > > v_provavilidad_hormigas_T;
vector<pair<num,num>> v_guardadps;
vector<vector<int> > caminos_rrecorridos_hormigas;



void mostrar_poblacion_siguiente2(){
    cout<<"CAMINOS "<<endl;
    for(int x=0;x<v_provavilidad_hormigas.size();x++){
        cout<<"["<<v_provavilidad_hormigas[x].first.first<<"] -> "<<
        "["<<v_provavilidad_hormigas[x].first.second<<"] = "<<
        "["<<v_provavilidad_hormigas[x].second<<"]"<<endl;
    }
}

void mostrar_poblacion_siguiente(){
    cout<<"CAMINOS "<<endl;
    for(int x=0;x<v_provavilidad_hormigas_T.size();x++){
        cout<<"["<<v_provavilidad_hormigas_T[x].first.first<<"] -> "<<
        "["<<v_provavilidad_hormigas_T[x].first.second<<"] = "<<
        "["<<v_provavilidad_hormigas_T[x].second<<"]"<<endl;
    }
}

void generar_pesos_grafo(){
    for(int i = 0; i < maximo; i++){
		for(int j = 0; j < maximo; j++){
			if(i==j){
                //D[i][j]=0;
                V[i][j]=0;
                F[i][j]=0;
            }
			else{
                int r=rand()%10;
                //D[i][j]=r+3;
                num vi = 1.0/D[i][j]*Fer[i][j];
                V[i][j]=vi;
                F[i][j]=feromona_inicial;
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

int fitness2(vector<int> n){
    int sum=0;
    for(int x=0;x<n.size()-1;x++){
        //cout<<"--> "<<D[n[x]][n[x+1]]<<endl;
        sum+=D[n[x]][n[x+1]];
    }
    return sum;
}

int fitness(vector<int> S){
    int sum=0;
    for(int x=0;x<maximo;x++){
        for(int y=0;y<maximo;y++){
            if(x!=y){
                int s1=S[x];
                int s2=S[y];
                sum+=Fer[x][y]*D[s1][s2];
            }
        }
    }
    return sum;
}

bool compare(const std::pair<double, vector<int>>& firstElem, const std::pair<double, vector<int>>& secondElem) {
  return firstElem.first < secondElem.first;

}

void maximo_global(int &posicion,int &maxmi_global){
vector<pair<int,vector<int> > > orden;
    vector<int>tem;
    int pos=0;
    num fitness_=fitness(caminos_rrecorridos_hormigas[0]);
    for(int x=0;x<caminos_rrecorridos_hormigas.size()-1;x++){
        if(fitness_>fitness(caminos_rrecorridos_hormigas[x+1])){
            pos=x+1;
            fitness_=fitness(caminos_rrecorridos_hormigas[x+1]);
        }
    }
    for(int x=0;x<caminos_rrecorridos_hormigas.size();x++){
        for(int y=0;y<caminos_rrecorridos_hormigas[0].size();y++){
            tem.push_back(caminos_rrecorridos_hormigas[x][y]);
        }
        orden.push_back(make_pair(fitness(caminos_rrecorridos_hormigas[x]),tem));
        tem.clear();
    }
    sort(orden.begin(),orden.end(),compare);
    if(maximo_local>=orden[0].first){
        maximo_local=orden[0].first;
    }

    for(int x=0;x<1;x++){
        cout<<"MEJOR H. LOCAL  "<<orden[x].first<<" = [";
        for(int y=0;y<orden[0].second.size();y++){
            char m=orden[x].second[y]+65;
            cout<<m<<" ";
        }cout<<"] posicion= "<<pos<<"\n"<<endl;
    }
    posicion=pos;
    maxmi_global=orden[0].first;
}

void cambiar_feromonas(){
    int posicion=0;
    int maxmi_global=0;
    maximo_global(posicion,maxmi_global);
    for(int x=0;x<maximo;x++){
        for(int y=0;y<maximo;y++){
            if(x!=y){
                num uno_menos_ro=1-Ro;
                num suma=F[x][y];
                if(buscar_camino_hormiga(posicion,x,y)==true){
                    num maximo_iteracion=((1.0/maxmi_global));
                    suma += (maximo_iteracion*uno_menos_ro);
                }
                F[x][y]=suma;
            }else{F[x][y]=0;}
        }
    }
}

void caminos_hormigas(){
    cout<<"ciudad nueva = 0"<<endl;
    vector<int>camino;
    for(int p=0;p<cantidad_iteraciones;p++){
        for(int c=0;c<cantidad_hormigas;c++){
            int ciudad_inicial=rand()%4;
            for(int z=0;z<maximo;z++){
                camino.push_back(ciudad_inicial);
                for(int x=0;x<maximo;x++){
                   if(x!=ciudad_inicial && buscar_repetidos(make_pair(ciudad_inicial,x))==false){
                        float t=pow(F[ciudad_inicial][x],alfa);
                        float n=pow(V[ciudad_inicial][x],beta);
                        v_provavilidad_hormigas.push_back(make_pair(make_pair(ciudad_inicial,x),n*t));
                        v_provavilidad_hormigas_T.push_back(make_pair(make_pair(ciudad_inicial,x),n*t));
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
                double rand_= ((double) rand() / ((double)RAND_MAX + 1));

                ///escojuedo la nueva ciudad
                if(rand_>=q0){
                    //cout<<"DIVERSIFICACION"<<endl;
                    int temporal=0;
                    //mostrar_poblacion_siguiente2();
                    for(int x=0;x<v_provavilidad_hormigas.size();x++){
                        if(x==0){
                            if(prob<=v_provavilidad_hormigas[0].second){
                                temporal=ciudad_inicial;
                                ciudad_inicial=v_provavilidad_hormigas[0].first.second;
                                break;
                            }
                        }
                        else{
                            if(prob>v_provavilidad_hormigas[x-1].second &&
                               prob<=v_provavilidad_hormigas[x].second){
                                temporal=ciudad_inicial;
                                ciudad_inicial=v_provavilidad_hormigas[x].first.second;
                                break;
                            }
                        }
                    }
                    ///actualizamos el arco
                   // cout<<"next cifitness(camino_h)ty -> ["<<temporal<<" "<<ciudad_inicial<<"]"<<endl;
                    F[temporal][ciudad_inicial]=(1-phy)*0.1+phy*0.1;
                }
                else{
                   // cout<<"iNTENSIFICACION"<<endl;
                    //mostrar_poblacion_siguiente();
                    int pos_x=0;
                    int pos_y=0;
                    num fitnes=0;
                    for(int x=0;x<v_provavilidad_hormigas_T.size();x++){
                        if(v_provavilidad_hormigas_T[x].second>fitnes){
                            pos_x=v_provavilidad_hormigas_T[x].first.first;
                            pos_y=v_provavilidad_hormigas_T[x].first.second;
                            fitnes=v_provavilidad_hormigas_T[x].second;
                        }
                    }
                    ///actualizamos el arco
                     ciudad_inicial=pos_y;
                    //cout<<"["<<pos_x<<" "<<pos_y<<"]"<<endl;
                    F[pos_x][pos_y]=(1-phy)*0.1 + phy*0.1;
                  //  cout<<"hormona -> "<<F[pos_x][pos_y]<<endl;

                }
                v_provavilidad_hormigas.clear();
                v_provavilidad_hormigas_T.clear();
            }

            caminos_rrecorridos_hormigas.push_back(camino);
            camino.clear();
            v_guardadps.clear();
            v_provavilidad_hormigas.clear();
            v_provavilidad_hormigas_T.clear();
        }
        cout<<"\nPOBLACION = "<<p<<endl;
        for(int x=0;x<caminos_rrecorridos_hormigas.size();x++){
            cout<<"hormiga "<<x+1<<" = [ ";
            for(int y=0;y<caminos_rrecorridos_hormigas[0].size();y++){
                char le=caminos_rrecorridos_hormigas[x][y]+65;
                cout<<le<<" ";
            }cout<<"] fittnes = "<<fitness(caminos_rrecorridos_hormigas[x])<<" maximo global = "<<
            maximo_local<<endl;
        }cout<<endl;
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
    caminos_hormigas();
    return 0;
}

