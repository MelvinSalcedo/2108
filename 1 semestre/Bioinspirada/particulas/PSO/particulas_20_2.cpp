#include <map>
#include <fstream>
#include <iostream>
#include <vector>
#include <time.h>
#include <cmath>
#include <stdlib.h>
#include <string>
#include <cstdio>
#include <algorithm>
using  namespace std;
typedef int num;
num w = ((double) rand() / (RAND_MAX));
///_____________________
num poblacion=8;
int z=0;

vector<vector<num>> repositorio_global;


int tam_cromosomas=10;
num distancia[10][10] =
                  {{0, 12, 3, 23, 1, 5, 23, 56, 12, 11},
				   {12, 0, 9, 18, 3, 41, 45, 5, 41, 27},
				   {3, 9, 0, 89, 56, 21, 12, 48, 14, 29},
				   {23, 18, 89, 0, 87, 46, 75, 17, 50, 42},
				   {1, 3, 56, 87, 0, 55, 22, 86, 14, 33},
				   {5, 41, 21, 46, 55, 0, 21, 76, 54, 81},
				   {23, 45, 12, 75, 22, 21, 0, 11, 57, 48},
				   {56, 5, 48, 17, 86, 76, 11, 0, 63, 24},
				   {12, 41, 14, 50, 14, 54, 57, 63, 0, 9},
				   {11, 27, 29, 42, 33, 81, 48, 24, 9, 0}};

num Costo[10][10] =
                        {{0, 22, 47, 15, 63, 21, 23, 16, 11, 9},
						{22, 0, 18, 62, 41, 52, 13, 11, 26, 43},
						{47, 18, 0, 32, 57, 44, 62, 20, 8, 36},
						{15, 62, 32, 0, 62, 45, 75, 63, 14, 12},
						{63, 41, 57, 62, 0, 9, 99, 42, 56, 23},
						{21, 52, 44, 45, 9, 0, 77 ,58, 22, 14},
						{23, 13, 62, 75, 99, 77, 0, 30, 25, 60},
						{16, 11, 20, 63, 42, 58, 30, 0, 66, 85},
						{11, 26, 8, 14, 56, 22, 25, 66, 0, 54},
						{9, 43, 36, 12, 23, 14, 60, 85, 54, 0}};


struct particula{
    vector<num> P;
    pair<num,num>fitness;
    vector<pair<num,num>> velocidad;
    vector<vector<num>>r_local;
};

vector<particula>individuos;
num fitness_distancia(vector<num>P){
    num suma=0;
    for(int n=0;n<P.size()-1;n++){
        int x=P[n];
        int y=P[n+1];
        suma+=distancia[x][y];
    }
    return suma;
}
num fitness_costo(vector<num>P){
    int suma=0;
    for(int n=0;n<P.size()-1;n++){
        int x=P[n];
        int y=P[n+1];
        suma+=Costo[x][y];
    }
    return suma;
}

bool V_repetidos(vector<int>temporal,int x){
    for(int n=0;n<temporal.size();n++){
        if(temporal[n]==x){
            return true;
        }
    }
    return false;
}

void crear_poblacion_inicial(){
    for(int n=0;n<poblacion;n++){
        particula part;
        vector<num>a;
        for(int n=0;n<tam_cromosomas;n++){
            int ran=rand()%tam_cromosomas;
            do{
                ran=rand()%tam_cromosomas;

            }while(V_repetidos(a,ran)==true);
            a.push_back(ran);
        }
        part.r_local.push_back(a);
        part.P=a;
        individuos.push_back(part);
        part.fitness.first=fitness_distancia(a);
        part.fitness.second=fitness_costo(a);
    }
}


void mostrar_poblacion(){
    cout<<"POBLACION\n"<<endl;
    for(int x=0;x<poblacion;x++){
        cout<<"P"<<x+1<<"[";
        for(int n=0;n<tam_cromosomas;n++){
            cout<<individuos[x].P[n]<<" ";
        }cout<<"]"<<endl;;
    }cout<<endl;
}


void evaluar_poblacion(){
    for(int x=0;x<poblacion;x++){
       individuos[x].fitness.first=fitness_distancia(individuos[x].P);
       individuos[x].fitness.second=fitness_costo(individuos[x].P);
    }
    for(int x=0;x<poblacion;x++){
        cout<<"P"<<x+1<<" : "<<"Fitness distancia = "<<individuos[x].fitness.first<<" "
        <<"Fitness costo = "<<individuos[x].fitness.second<<endl;
    }
}

num verificar(vector<int>  p1,int nu){
    int m=0;
    for(int x=0;x<p1.size();x++){
        if(nu==p1[x]){m=x;break;}
    }
   return m;
}

vector<pair<int,int>> permutacion(vector<int> P1,vector<int> P2){
    vector<pair<int,int>> SS;
    for(int i=0;i<P1.size();i++){
        if(P1[i]!=P2[i]){
            int index=0;

            for(int m=0;m<P1.size();m++){
                if(P1[i]==P2[m]){
                    index=m;
                }
            }

            SS.push_back(make_pair(i,index));

            int temp=P2[i];
            P2[i]=P2[index];
            P2[index]=temp;
        }
    }
    return SS;
}


bool dominado(vector<num> P1,vector<num>P2){
    if((fitness_distancia(P1)<fitness_distancia(P2) &&
       fitness_costo(P1)<fitness_costo(P2))
       ||
       (fitness_distancia(P1)<=fitness_distancia(P2) &&
       fitness_costo(P1)<fitness_costo(P2))
       ||
       (fitness_distancia(P1)<fitness_distancia(P2) &&
       fitness_costo(P1)<=fitness_costo(P2))

       ){return true;}
    return false;
}

vector<vector<num>> sort_no_dominado(vector<particula> from_data ){
    vector<vector<num>> S;
    vector<num>N,Rank;
    vector<vector<num>> frontera_;
    for(int x=0;x<from_data.size();x++){
        vector<num>a;
        S.push_back(a);
        N.push_back(0);
        Rank.push_back(0);
    }
    vector<num>a;
    frontera_.push_back(a);

    for(int x=0;x<from_data.size();x++){

        for(int y=0;y<from_data.size();y++){
            //if(x!=y){
                if(dominado(from_data[x].P,from_data[y].P)){
                    S[x].push_back(y);
                }
                else if(dominado(from_data[y].P,from_data[x].P)){
                    N[x]+=1;
                }
            //}
        }
        if(N[x]==0){
            Rank[x]=0;
            frontera_[0].push_back(x);
        }
    }
    int i=0;
    while(!frontera_[i].empty()){
        vector<num>Q;
        for(int p=0;p<frontera_[i].size();p++){
            for(int q=0;q<S[p].size();q++){
                N[S[p][q]]--;
                if(N[S[p][q]]==0){
                    Rank[S[p][q]]=i+1;
                    Q.push_back(S[p][q]);
                }
            }
        }
        i++;

        frontera_.push_back(Q);
    }
    frontera_.erase (frontera_.begin()+frontera_.size()-1);
    return frontera_;
}



void Actualizar_respositorio_global(vector<num> pareto, vector<particula>from_data){
    /*cout<<"**********************************************"<<endl;
    for(int x=0;x<pareto.size();x++){
        cout<<pareto[x]<<" ";
    }cout<<endl;
    for(int x=0;x<from_data.size();x++){
        for(int y=0;y<tam_cromosomas;y++){
            cout<<from_data[x].P[y]<<" ";
        }cout<<endl;

    }cout<<endl;*/

    repositorio_global.clear();
    for(int x=0;x<pareto.size();x++){
        repositorio_global.push_back(from_data[pareto[x]].P);
    }
    /*cout<<"***********"<<endl;
    for(int x=0;x<repositorio_global.size();x++){
        for(int y=0;y<tam_cromosomas;y++){
            cout<<repositorio_global[x][y]<<" ";
        }cout<<endl;

    }cout<<endl;*/

    //cout<<"**********************************************"<<endl;
}

void Actualizar_respositorio_local(){
    vector<particula> new_Data;
    for(int x=0;x<poblacion;x++){
        int temp=0;
        for(int y=0;y<individuos[x].r_local.size();y++){
            particula a;
            new_Data.push_back(a);
            new_Data[y].P=(individuos[x].r_local[y]);
            temp++;
        }
        particula a;
        new_Data.push_back(a);
        new_Data[temp].P=(individuos[x].P);

        vector<vector<num>> pareto=sort_no_dominado(new_Data);
        individuos[x].r_local.clear();

        for(int y=0;y<pareto[0].size();y++){
            individuos[x].r_local.push_back(new_Data[pareto[0][y]].P);
        }
        //cout<<"****"<<individuos[x].r_local.size()<<endl;
        new_Data.clear();

    }

}

vector<num> mejor_local_particula(num x){
    num pos =rand()%individuos[x].r_local.size();
    return individuos[x].r_local[pos];

}

vector<num> mejor_global_particula(){
    num pos =rand()%repositorio_global.size();
    return repositorio_global[pos];
}

void actualizar_particulas(int t,vector<pair<int,int>>ss){
    for(int x=0;x<ss.size();x++){
        int posx=ss[x].first;
        int posy=ss[x].second;

        int temporal=individuos[t].P[posx];
        individuos[t].P[posx]=individuos[t].P[posy];
        individuos[t].P[posy]=temporal;

    }
}

vector<pair<int,int>> sumar_SS(vector<pair<int,int>> a,vector<pair<int,int>> b){
    vector<pair<int,int>> nuevo;
    for(int x=0;x<a.size();x++){
        nuevo.push_back(a[x]);
    }
    for(int x=0;x<b.size();x++){
        nuevo.push_back(b[x]);
    }
    return nuevo;
}

bool igual(vector<int>a,vector<num>b){
    for(int y=0;y<tam_cromosomas;y++){
        if(a[y]!=b[y]){
            return true;
        }
    }
    return false;
}
void PSO(){

    crear_poblacion_inicial();
    mostrar_poblacion();
    evaluar_poblacion();
    vector<vector<num>> pareto=sort_no_dominado(individuos);
    Actualizar_respositorio_global(pareto[0],individuos);
    Actualizar_respositorio_local();
    for(int x=0;x<30;x++){
        cout<<"________________________POBLACION "<<x+1<<"___________________"<<endl;

        for(int y=0;y<poblacion;y++){

            vector<num> Plocal=mejor_local_particula(y);
            vector<num> PGlobal=mejor_global_particula();


            vector<pair<int,int>>a= permutacion(individuos[y].P,PGlobal);
            vector<pair<int,int>>b= permutacion(individuos[y].P,Plocal);
            vector<pair<int,int>> temporal=sumar_SS(a,b);
            actualizar_particulas(y,temporal);
        }
        mostrar_poblacion();
        evaluar_poblacion();
        vector<particula> new_S_;
        for(int y=0;y<poblacion;y++){
            particula new_;
            new_.P=individuos[y].P;
            new_S_.push_back(new_);
        }
        for(int y=0;y<repositorio_global.size();y++){
            particula new_;
            new_.P=repositorio_global[y];
            new_S_.push_back(new_);
        }
        vector<vector<num>> Npareto=sort_no_dominado(new_S_);
        Actualizar_respositorio_global(Npareto[0],new_S_);
        Actualizar_respositorio_local();
        cout<<"REPOSITORIO LOCAL DE PARTICULA "<<endl;
        for(int i=0;i<poblacion;i++){
            cout<<"P"<<i+1<<"[";
            for(int j=0;j<individuos[i].r_local.size();j++){
                for(int l=0;l<tam_cromosomas;l++){
                    cout<<individuos[i].r_local[j][l]<<" ";
                }cout<<"] [";

            }cout<<endl;/*cout<<"] fitness_D = "<<fitness_distancia(respositorio_local[i])<<
            "\tfitness_C = "<<fitness_costo(respositorio_local[i])<<endl;*/

        }

    }
    cout<<"REPOSITORIO GLOBAL DE PARTICULAS "<<endl;
        for(int i=0;i<repositorio_global.size();i++){
            cout<<"[";
            for(int j=0;j<tam_cromosomas;j++){
                cout<<repositorio_global[i][j]<<" ";
            }cout<<"] fitness_D = "<<fitness_distancia(repositorio_global[i])<<
            "\tfitness_C = "<<fitness_costo(repositorio_global[i])<<endl;

        }
    ofstream fs("salida.txt");
    for(int i=0;i<repositorio_global.size();i++){
            fs <<fitness_distancia(repositorio_global[i])<<"\n"<<
            fitness_costo(repositorio_global[i])<<endl;
    }fs.close();
}

int main(){
    srand(time(0));
    PSO();
    return 0;
}
