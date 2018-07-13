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
typedef double num;

num numero_particulas=10;
const num tam_cro=5;
num phy1,phy2=1;
vector<vector<pair<int,int>>> velocidad;
vector<vector<int>> Mejor_Local;


num pesos[5][5]={
    {0,1,3,4,5},
    {1,0,1,4,8},
    {3,1,0,5,1},
    {4,4,5,0,2},
    {5,8,1,2,0},
};

num mejor_global;
vector<vector<int>> particulas;
num fitness(vector<int > temporal){
    int suma=0;
    for(int y=0;y<temporal.size()-1;y++){
        suma+=pesos[temporal[y]][temporal[y+1]];
    }
    return suma;
}

bool verificar_repetidos(vector<int> temp,int numro){
    for(int n=0;n<temp.size();n++){
        if(numro==temp[n]){
            return true;
        }
    }
    return false;
}

void generar_Particulas(){
    for(int y=0;y<numero_particulas;y++){
        vector<int> p;
        for(int n=0;n<tam_cro;n++){
            int x=rand()%5;
            do{
                x=rand()%5;
            }while(verificar_repetidos(p,x));
            p.push_back(x);
        }
        particulas.push_back(p);
    }
}

void mostrar_particulas(){
    for(int y=0;y<numero_particulas;y++){
        cout<<"[ ";
        for(int n=0;n<tam_cro;n++){
             char m=particulas[y][n]+65;
            cout<<m<<" ";
        }
        cout<<" ] Fitness = "<<fitness(particulas[y])<<endl;
    }
}



vector<pair<int,int>> actualisar_velocidades(vector<int> P1,vector<int> P2){
    vector<pair<int,int>> SS;

        for(int j=0;j<P1.size();j++){
            pair<int,int>SO;
            for(int x=0;x<P1.size();x++){
                if(P2[j]==P1[x]){SO.second=x;break;}
            }
            SO.first=j;

            if(SO.first!=SO.second){
                int temporal=P1[SO.first];
                P1[SO.first]=P1[SO.second];
                P1[SO.second]=temporal;
                SS.push_back(SO);
            }
        }


    return SS;
}

void actualisar_posiciones(int a,int b){

}

void inicialisar_ML(){
    for(int x=0;x<numero_particulas;x++){
        Mejor_Local.push_back(particulas[x]);
    }
}

void imprimir_Datos(vector<int> dato){
    cout<<"[ ";
    for(int x=0;x<dato.size();x++){
        char m=dato[x]+65;
        cout<<m<<" ";
    }cout<<"]\tfitness = "<<fitness(dato)<<endl;
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

void PSO(){
    vector<int> Mejor_Global=particulas[0];

    for(int x=0;x<5;x++){
        cout<<"\nITERACION :" <<x+1<<"--------------------"<<endl;
        for(int i=0;i<numero_particulas;i++){
            if(fitness(particulas[i])<fitness(Mejor_Global)){
                Mejor_Global=particulas[i];
            }
            if(fitness(particulas[i])<fitness(Mejor_Local[i])){
                Mejor_Local[i]=particulas[i];
            }
        }
        cout<<"MEJOR GLOBAL "<<endl;
        imprimir_Datos(Mejor_Global);
        cout<<"\n"<<"MEJOR LOCAL"<<endl;
        for(int i=0;i<numero_particulas;i++){
            imprimir_Datos(Mejor_Local[i]);
        }cout<<endl;
        ///________________________________________________actualizar velocidades
        for(int i=0;i<numero_particulas;i++){
            vector<pair<int,int>> a=actualisar_velocidades(Mejor_Local[i],particulas[i]);
            vector<pair<int,int>> b=actualisar_velocidades(Mejor_Global,particulas[i]);
            vector<pair<int,int>> temporal=sumar_SS(a,b);
            velocidad.push_back(temporal);
            a.clear();b.clear();
        }
        cout<<"VELOCIDADES ACTUALIZADAS"<<endl;
        for(int x=0;x<numero_particulas;x++){
            cout<<"velocidad de P"<<x+1<<" = [";
            for(int y=0;y<velocidad[x].size();y++){
                cout<<"("<<velocidad[x][y].first<<","<<velocidad[x][y].second<<")";
            }
            cout<<"]"<<endl;
        }
        cout<<endl;
        ///________________________________________________actualizar posiciones
        for(int x=0;x<numero_particulas;x++){
            for(int y=0;y<velocidad[x].size();y++){
                int temporal=particulas[x][velocidad[x][y].first];
                particulas[x][velocidad[x][y].first]=particulas[x][velocidad[x][y].second];
                particulas[x][velocidad[x][y].second]=temporal;
            }
        }
        cout<<"PARTICULAS ACTUALIZADAS"<<endl;
        mostrar_particulas();
        velocidad.clear();
    }
}

int main(){
    srand(time(0));
    generar_Particulas();
    mostrar_particulas();
    inicialisar_ML();

    PSO();

    actualisar_velocidades(particulas[0],particulas[1]);
    return 0;
}
