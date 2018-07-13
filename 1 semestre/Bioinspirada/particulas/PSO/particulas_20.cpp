#include <map>
#include <iostream>
#include <vector>
#include <time.h>
#include <cmath>
#include <stdlib.h>
#include <string>
#include <cstdio>
#include <algorithm>
#include <fstream>
using  namespace std;
typedef double num;

vector<pair<num,num>> pos_particula,vel_particula,fitnes_2F;
vector <num> F;


num poblacion=4;
num w = ((double) rand() / (RAND_MAX));
num py_1=2.0;
num py_2=2.0;
num rand1=((double) rand() / (RAND_MAX));
num rand2=((double) rand() / (RAND_MAX));

num min_x = 0;
num max_x = 5;

num min_y = 0;
num max_y = 3;

pair<num,num> mejor_it;
pair<num,num> best_global;

vector<pair<num,num>>repositorio_best_global;
vector<pair<num,num>>RG_v;
vector<pair<num,num>> mejor_posicion_cada_prticula;
vector<vector<pair<num,num>>> respositorio_local;
vector<vector<pair<num,num>>> RL_v;

void iniiciar_mejor_posicion_cada_prticula(){
     for(int n=0;n<poblacion;n++){
        mejor_posicion_cada_prticula.push_back(make_pair(1,1));
        fitnes_2F.push_back(make_pair(0,0));
        vector<pair<num,num>>a;
        respositorio_local.push_back(a);
        RL_v.push_back(a);
     }
}
void generar_poblacion_inicial(){
    cout<<"w =" <<w<<" "<<rand1<<" "<<rand2<<endl;
    for(int n=0;n<poblacion;n++){
        int m=rand()%10;
        double a,b,x,y;
        //if(m%2==0){
            a=(double)rand()/(RAND_MAX + 1)+1+(rand()%4);
            b=(double)rand()/(RAND_MAX + 1)+(rand()%3);
            x=((double) rand() / (RAND_MAX));
            y=((double) rand() / (RAND_MAX));
       // }
        /*else{
            a=-(double)rand()/(RAND_MAX + 1)+-1+(rand()%4);
            b=-(double)rand()/(RAND_MAX + 1)+-1+(rand()%4);
            x=-((double) rand() / (RAND_MAX));
            y=-((double) rand() / (RAND_MAX));
        }*/
        pos_particula.push_back(make_pair(a,b));
        vel_particula.push_back(make_pair(x,y));
    }
}

void mostrar_poblacion_inicial(){

    cout<<"PARTICULAS\n"<<endl;
    for(int x=0;x<poblacion;x++){
        cout<<"X:"<<pos_particula[x].first<<" \tY: "<<pos_particula[x].second
            <<"\tV1: "<<vel_particula[x].first<<"\tV2: "<<vel_particula[x].second<<endl;
    }cout<<endl;
}

num fitness_1(num x,num y){
    return 4*pow(x,2)+4*pow(y,2);
}

num fitness_2(num x,num y){
    return pow(x-5,2)+pow(y-5,2);
}

bool dominado(pair<num,num> P1,pair<num,num>P2){
    if((fitness_1(P1.first,P1.second)<fitness_1(P2.first,P2.second) &&
       fitness_2(P1.first,P1.second)<fitness_2(P2.first,P2.second)) ||(
       fitness_1(P1.first,P1.second)<=fitness_1(P2.first,P2.second) &&
       fitness_2(P1.first,P1.second)<fitness_2(P2.first,P2.second))||(
       fitness_1(P1.first,P1.second)<fitness_1(P2.first,P2.second) &&
       fitness_2(P1.first,P1.second)<=fitness_2(P2.first,P2.second))
       ){return true;}
    return false;
}

void evaluando_particula(){
    vector<num>indices_para_eliminar;

    for(int x=0;x<poblacion;x++){
        if((pos_particula[x].first<min_x || pos_particula[x].first>max_x )||
           (pos_particula[x].second<min_y || pos_particula[x].second>max_y)){
            cout<<"sobrepasas los limites "<<endl;
            indices_para_eliminar.push_back(x);
        }
        else{
            fitnes_2F[x].first=fitness_1(pos_particula[x].first,pos_particula[x].second);
            fitnes_2F[x].second=fitness_2(pos_particula[x].first,pos_particula[x].second);
        }
    }
    pair<num,num>t;
    pair<num,num>v;

    vector<pair<num,num>>t1;
    vector<pair<num,num>>v1;
    for(int n=0;n<poblacion;n++){
        int m=rand()%10;
        double a,b,x,y;
        //if(m%2==0){
            a=(double)rand()/(RAND_MAX + 1)+1+(rand()%4);
            b=(double)rand()/(RAND_MAX + 1)+(rand()%3);
            x=((double) rand() / (RAND_MAX));
            y=((double) rand() / (RAND_MAX));
       // }
        /*else{
            a=-(double)rand()/(RAND_MAX + 1)+-1+(rand()%4);
            b=-(double)rand()/(RAND_MAX + 1)+-1+(rand()%4);
            x=-((double) rand() / (RAND_MAX));
            y=-((double) rand() / (RAND_MAX));
        }*/
        pos_particula.push_back(make_pair(a,b));
        vel_particula.push_back(make_pair(x,y));
    }

    for(int x=0;x<indices_para_eliminar.size();x++){
        int m=rand()%10;
        double a,b,xx,y;
        //if(m%2==0){
            a=(double)rand()/(RAND_MAX + 1)+1+(rand()%4);
            b=(double)rand()/(RAND_MAX + 1)+(rand()%3);
            xx=((double) rand() / (RAND_MAX));
            y=((double) rand() / (RAND_MAX));

        pos_particula[indices_para_eliminar[x]].first=a;
        pos_particula[indices_para_eliminar[x]].second=b;
        vel_particula[indices_para_eliminar[x]].first=xx;
        vel_particula[indices_para_eliminar[x]].first=y;
    }
}

vector<vector<num>> sort_no_dominado(vector<pair<num,num>> from_data ){
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
            if(dominado(from_data[x],from_data[y])){
                S[x].push_back(y);
            }
            else if(dominado(from_data[y],from_data[x])){
                N[x]++;
            }
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

    /*for(int x=0;x<frontera_.size();x++){
           cout<<"Frontera ["<<x<<"] = ";
        for(int y=0;y<frontera_[x].size();y++){
            cout<<frontera_[x][y]<<" ";
        }
        cout<<endl;
    }*/

    return frontera_;
}

void Actualizar_respositorio_global(vector<num> pareto,vector<pair<num,num>>P){
    repositorio_best_global.clear();
    for(int x=0;x<pareto.size();x++){
        repositorio_best_global.push_back(make_pair(P[pareto[x]].first,
                                            P[pareto[x]].second));

        /*RG_v.push_back(make_pair(vel_particula[x].first,
                                     vel_particula[x].second));*/
    }
}

void Actualizar_respositorio_local(){
    for(int x=0;x<poblacion;x++){
        if(respositorio_local[x].size()==0){
            respositorio_local[x].push_back(make_pair(pos_particula[x].first,
                                            pos_particula[x].second));
            RL_v[x].push_back(make_pair(vel_particula[x].first,
                                        vel_particula[x].second));
        }
        vector<pair<num,num>>new_s;
        for(int i=0;i<respositorio_local[x].size();i++){
            new_s.push_back(make_pair(respositorio_local[x][i].first,
                                      respositorio_local[x][i].second));
        }
        new_s.push_back(make_pair(pos_particula[x].first,pos_particula[x].second));
        vector<vector<num>>pareto=sort_no_dominado(new_s);
        respositorio_local[x].clear();
        for(int j=0;j<pareto[0].size();j++){
            respositorio_local[x].push_back(make_pair(new_s[pareto[0][x]].first,
                                                      new_s[pareto[0][x]].second));
        }
    }
}

pair<num,num> mejor_local_particula(num x){
    num pos =rand()%respositorio_local[x].size();
    return respositorio_local[x][pos];
}
pair<num,num> mejor_global_particula(){
    num pos =rand()%repositorio_best_global.size();
    //cout<<"->>>"<<repositorio_best_global.size()<<endl;
    return repositorio_best_global[pos];
}


void actualisar_velocidades( pair<num,num>pl, pair<num,num> pg,int x){
    mejor_it.first =pos_particula[pl.first].first;
    mejor_it.second=pos_particula[pl.second].first;

    num anterior_a=mejor_it.first -pos_particula[x].first;
    num anterior_b=mejor_it.second-pos_particula[x].second;

    vel_particula[x].first=w*vel_particula[x].first + py_1*rand1*anterior_a+py_2*rand2*(pg.first -pos_particula[x].first);
    vel_particula[x].second=w*vel_particula[x].second+py_1*rand1*anterior_b+py_2*rand2*(pg.second-pos_particula[x].second);
    num a=pos_particula[x].first +vel_particula[x].first;
    num b=pos_particula[x].second+vel_particula[x].second;
    //if(a>=min_x && a<=max_x ){
        //if (b>=min_y && b<=max_y){
            pos_particula[x].first =a;
            pos_particula[x].second=b;
//}
   // }

}

void mostrar_RL(){
    cout<<"REPOSITORIO LOCAL DE CADA PARTICULA\n";
    cout<<"posiciones\n";
    for(int x=0;x<poblacion;x++){
        cout<<"P"<<x+1;
        for(int y=0;y<respositorio_local[x].size();y++){
            cout<<"[ X = "<<respositorio_local[x][y].first<<"\Y = "<<respositorio_local[x][y].second<<"],";
        }cout<<endl;
    }
    /*cout<<"\nvelocidades"<<endl;
    for(int x=0;x<poblacion;x++){
        cout<<"V"<<x+1;
        for(int y=0;y<RL_v[x].size();y++){
            cout<<"[ Vx = "<<RL_v[x][y].first<<"\Vy = "<<RL_v[x][y].second<<"],";
        }cout<<endl;
    }*/
}

struct sort_pred {
    bool operator()(const std::pair<num,num> &left, const std::pair<num,num> &right) {
        return right.first > left.first;
    }
};

void PSO(){
    generar_poblacion_inicial();
    mostrar_poblacion_inicial();
    evaluando_particula();
    vector<vector<num>>pareto=sort_no_dominado(pos_particula);
    Actualizar_respositorio_global(pareto[0],pos_particula);
    Actualizar_respositorio_local();
    for(int x=0;x<10;x++){
        cout<<"\n_______________________________ITERACION_________________________ : "<<x+1<<endl;
        for(int y=0;y<poblacion;y++){
            pair<num,num> Plocal=mejor_local_particula(y);
            pair<num,num> PGlobal=mejor_global_particula();
            //cout<<Plocal.first<<" "<<Plocal.second<<" |||| "<<PGlobal.first<<" "<<PGlobal.second<<endl;
            actualisar_velocidades(Plocal,PGlobal,y);
        }

        evaluando_particula();
        cout<<"***********"<<endl;
        vector<pair<num,num>>new_s;
        for(int i=0;i<poblacion;i++){
            new_s.push_back(make_pair(pos_particula[i].first,
                                      pos_particula[i].second));
        }cout<<"***********"<<endl;
        for(int i=0;i<repositorio_best_global.size();i++){
            new_s.push_back(make_pair(repositorio_best_global[i].first,
                                      repositorio_best_global[i].second));
        }
        vector<vector<num>>pareto=sort_no_dominado(new_s);

        Actualizar_respositorio_global(pareto[0],new_s);
        Actualizar_respositorio_local();
        mostrar_poblacion_inicial();
        mostrar_RL();
        w = ((double) rand() / (RAND_MAX));
    }
    cout<<"REPOSITORIO GLOBAL DE PARTICULAS "<<endl;
    for(int x=0;x<repositorio_best_global.size();x++){
        cout<<"X = "<<repositorio_best_global[x].first<<"\t Y = "<<repositorio_best_global[x].second
        <<"\tFitness_1 = "<<fitness_1(repositorio_best_global[x].first,repositorio_best_global[x].second)<<"\tFitness_2 = "<<
        fitness_2(repositorio_best_global[x].first,repositorio_best_global[x].second)<<endl;
    }
    char cadena[128];
   ofstream fs("salida.txt");
   for(int x=0;x<repositorio_best_global.size();x++){
        fs <<fitness_1(repositorio_best_global[x].first,repositorio_best_global[x].second)<<"\n"<<fitness_2(repositorio_best_global[x].first,repositorio_best_global[x].second)<<endl;
   }fs.close();
}

int main(){
    srand(time(0));
    iniiciar_mejor_posicion_cada_prticula();
    PSO();
    return 0;
}
