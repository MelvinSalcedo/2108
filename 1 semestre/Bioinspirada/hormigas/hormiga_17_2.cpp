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
num maximo_global_=1000;
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

num Fer[maximo][maximo]={{ 0, 0, 2, 0, 3},
                         { 0, 0, 0, 3, 0},
                         { 2, 0, 0, 0, 0},
                         { 0, 3, 0, 0, 1},
                         { 3, 0, 0, 1, 0}};
int cantidad_hormigas=3;
int cantidad_iteraciones=50;

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
                num vi = 1.0/D[i][j];
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

int fitness(vector<int> n){
    int sum=0;
    for(int x=0;x<n.size()-1;x++){
        //cout<<"--> "<<D[n[x]][n[x+1]]<<endl;
        sum+=D[n[x]][n[x+1]];
    }
    return sum;
}

int fitness1(vector<int> S){
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
    if(orden[0].first<maximo_global_){
        maximo_global_=orden[0].first;
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

void actualizar_Arcos_escalada_primera_opcion(){
    int maxmi_global=0;
    int posicion=0;
    maximo_global(posicion,maxmi_global);
    for(int x=0;x<cantidad_hormigas;x++){
        for(int y=0;y<maximo-1;y++){
            ///__________________________________________
            num uno_menos_ro=1-Ro;
            num suma=F[caminos_rrecorridos_hormigas[x][y]][caminos_rrecorridos_hormigas[x][y+1]];

            num maximo_iteracion=((1.0/maxmi_global));
            suma += (maximo_iteracion*uno_menos_ro);
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

                int pos_a=rand()%maximo;
                int pos_b=rand()%maximo;

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
        cout<<"\nPOBLACION ANTES DE LA BUSQUEDA LOCAL = "<<p<<endl;
        for(int x=0;x<caminos_rrecorridos_hormigas.size();x++){
            cout<<"hormiga "<<x+1<<" = [ ";
            for(int y=0;y<caminos_rrecorridos_hormigas[0].size();y++){
                char le=caminos_rrecorridos_hormigas[x][y]+65;
                cout<<le<<" ";
            }cout<<"] fittnes = "<<fitness(caminos_rrecorridos_hormigas[x])<<" maximo global = "<<
            maximo_global_<<endl;
        }cout<<endl;

        busqueda_local_ESCALADA_PRIMERA_OPCION();

        cout<<"\nPOBLACION DESPUES DE LA BUSQUEDA LOCAL = "<<p<<endl;
        for(int x=0;x<caminos_rrecorridos_hormigas.size();x++){
            cout<<"hormiga "<<x+1<<" = [ ";
            for(int y=0;y<caminos_rrecorridos_hormigas[0].size();y++){
                char le=caminos_rrecorridos_hormigas[x][y]+65;
                cout<<le<<" ";
            }cout<<"] fittnes = "<<fitness(caminos_rrecorridos_hormigas[x])<<" maximo global = "<<
            maximo_global_<<endl;
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

