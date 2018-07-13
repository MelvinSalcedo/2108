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

vector<pair<num,num>> pos_particula,vel_particula;
vector <num> F;


num poblacion=6;
num w = ((double) rand() / (RAND_MAX));
num py_1=2.0;
num py_2=2.0;
num rand1=((double) rand() / (RAND_MAX));
num rand2=((double) rand() / (RAND_MAX));


pair<pair<num,num>,num> mejor_it;
pair<pair<num,num>,num> best_global;
vector<pair<pair<num,num>,num>> mejor_posicion_cada_prticula;

void iniiciar_mejor_posicion_cada_prticula(){
     for(int n=0;n<poblacion;n++){
            mejor_posicion_cada_prticula.push_back(make_pair(make_pair(1,1),1000));

     }
}

void generar_poblacion_inicial(){
    cout<<"w =" <<w<<" "<<rand1<<" "<<rand2<<endl;
    for(int n=0;n<poblacion;n++){
        int m=rand()%10;
        double a,b,x,y;
        if(m%2==0){
            a=(double)rand()/(RAND_MAX + 1)+1+(rand()%4);
            b=(double)rand()/(RAND_MAX + 1)+1+(rand()%4);
            x=((double) rand() / (RAND_MAX));
            y=((double) rand() / (RAND_MAX));
        }
        else{
            a=-(double)rand()/(RAND_MAX + 1)+-1+(rand()%4);
            b=-(double)rand()/(RAND_MAX + 1)+-1+(rand()%4);
            x=-((double) rand() / (RAND_MAX));
            y=-((double) rand() / (RAND_MAX));
        }
        pos_particula.push_back(make_pair(a,b));
        vel_particula.push_back(make_pair(x,y));
    }
}

void mostrar_poblacion_inicial(){
     cout<<"_________________________________________"<<endl;
    cout<<"MATRIZ DISTANCIA"<<endl;
    for(int x=0;x<poblacion;x++){
        cout<<"X:"<<pos_particula[x].first<<" \tY: "<<pos_particula[x].second
            <<"\tV1: "<<vel_particula[x].first<<"\tV2: "<<vel_particula[x].second<<endl;
    }
}

void fitness(){

    num temporal=1110;
    int pos=0;
    for(int x=0;x<poblacion;x++){
        num f=pos_particula[x].first*pos_particula[x].first + pos_particula[x].second*pos_particula[x].second;
        if(f<temporal){
            temporal=f;
            pos=x;
        }
        F.push_back(f);
        if(F[x]<mejor_posicion_cada_prticula[x].second){
            mejor_posicion_cada_prticula[x].second=F[x];
            mejor_posicion_cada_prticula[x].first.first=x;
            mejor_posicion_cada_prticula[x].first.second=x;
        }
    }
    mejor_it.first.first=pos_particula[pos].first;
    mejor_it.first.second=pos_particula[pos].second;
    mejor_it.second=temporal;
    cout<<endl;

    for(int x=0;x<poblacion;x++){
        cout<<"Fitness "<<": "<<F[x]<<endl;
    }

    cout<<"\nMEJOR FITNESS LOCAL DE CADA PARTICULA\n";
    for(int x=0;x<poblacion;x++){
        cout<<mejor_posicion_cada_prticula[x].second<<endl;
    }


    if(mejor_it.second<best_global.second){
        best_global.first.first=mejor_it.first.first;
        best_global.first.second=mejor_it.first.second;
        best_global.second=mejor_it.second;
    }
    cout<<"\nEL QUE TIENE FITNESS GLOBAL\n\n";
    cout<<"X:"<< best_global.first.first<<" \tY: "<<best_global.first.second
            <<"\t fitness: "<<best_global.second<<endl;
    F.clear();
}

void actualisar_velocidades(){
     for(int x=0;x<poblacion;x++){
        mejor_it.first.first=pos_particula[mejor_posicion_cada_prticula[x].first.first].first;
        mejor_it.first.second=pos_particula[mejor_posicion_cada_prticula[x].first.second].first;

        num anterior_a=mejor_it.first.first -pos_particula[x].first;
        num anterior_b=mejor_it.first.second-pos_particula[x].second;

        vel_particula[x].first=w*vel_particula[x].first + py_1*rand1*anterior_a+py_2*rand2*(best_global.first.first -pos_particula[x].first);
        vel_particula[x].second=w*vel_particula[x].second+py_1*rand1*anterior_b+py_2*rand2*(best_global.first.second-pos_particula[x].second);
     }
}

void actualisar_posiciones(){
    for(int x=0;x<poblacion;x++){
        pos_particula[x].first =pos_particula[x].first +vel_particula[x].first;
        pos_particula[x].second=pos_particula[x].second+vel_particula[x].second;
    }
}
void PSO(){
    for(int x=0;x<1;x++){
        cout<<"_________________________________________"<<endl;
        cout<<"ITERACION : "<<x+1<<endl;
        mostrar_poblacion_inicial();
        fitness();
        actualisar_velocidades();
        actualisar_posiciones();
        w = ((double) rand() / (RAND_MAX));
    }


}
int main(){
    srand(time(0));
    iniiciar_mejor_posicion_cada_prticula();
    best_global.second=1111;
    generar_poblacion_inicial();
    PSO();
    return 0;
}
