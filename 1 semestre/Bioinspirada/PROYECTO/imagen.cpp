#include "CImg.h"
#include <fstream>
#include <vector>
#include<iostream>
#include <time.h>
#include <fstream>
#include <string>
using namespace cimg_library;
using namespace std;
typedef int num;
//g++ -o imagen.exe imagen.cpp -O4 -lgdi32

/// Transforma una imagen a escala de grises, con depth cantidad de canales (rgb=3, else 1)
template<typename T>
CImg<T> transform_to_grayscale(CImg<T> imagen, bool rgb = false) {

    unsigned int spectrum = imagen.spectrum();

    // Si ya tiene un solo canal, no tiene sentido procesarla
    if (spectrum > 1) {

        // Si no es rgb, solo trabajo con un canal
        if (!rgb) {
            spectrum = 1;
        }

        // La transformamos en escala de grises
        cimg_forXY(imagen, x, y) {
            // Sumo los valores de los canales y lo promedio
            unsigned int promedio = 0;

            cimg_forC(imagen, c) {
                promedio += imagen(x, y, c);
            }

            promedio /= imagen.spectrum();

            for (int v = 0 ; v < spectrum ; v++) {
                imagen(x, y, v) = promedio;
            }
        }

        // Nuevamente, si no es rgb, me quedo con un unico canal
        if (!rgb) {
            imagen = imagen.get_channel(0);
        }

    }

    return imagen;
}

struct hormiga{
    num pos_x;
    num pos_y;
    float s_G;
    vector<pair<num,num>> camino;
};

num vecindad=8;
num tam_poblacion=10;
num feromona_inicial=0.1;
ofstream outfile;

float Ro=0.7;
num alfa=1,beta=1;
float Q=1.0;
float w=0.6;
num A=500;
num B=10;

num cluster=3;
vector<hormiga> colonia;
int dim_D=0;
const int tam_I=258;
float F[tam_I][tam_I];
float D[tam_I][tam_I];

vector<num> mejor_global;

template<typename T>
void rellenar_matriz_D(CImg<T> img){
    for (int x = 0; x < dim_D; x++){
        for (int y = 0; y < dim_D   ; y++){
            float n=(num)img(x,y,0,0);
            F[x][y]=0.1;
            D[x][y]=n;
        }
    }

}

template<typename T>
void asginar_pixel_hormiga(CImg<T> img){
    for (int x = 0; x < dim_D; x++){
        for (int y = 0; y < dim_D   ; y++){
            hormiga n;
            n.pos_x=x;
            n.pos_y=y;
            n.s_G=(float)img(x,y,0,0);
            colonia.push_back(n);
        }
    }
}

bool buscar_repetidos(vector<pair<num,num>> temporal,pair<num,num> key){
    for(int x=0;x<temporal.size();x++){
        if(temporal[x].first==key.first &&temporal[x].second==key.second){
            return true;
        }
    }
    return false;
}

float genRandom(double li, double ls){
	return li + static_cast <double> (rand()) /( static_cast <double> (RAND_MAX/(ls-li)));
}

float suma_v(vector<float> a){
    float s=0;
    for(int c=0;c<a.size();c++){
        s+=a[c];
    }
    return s;
}

void camino_pixel(int x,vector<pair<int,int>>temporal,pair<num,num>ciudad_i){

    vector<float> sumas_provavilidad;
    //c-out<<"--"<<ciudad_i.first<<" "<<ciudad_i.second<<endl;
    for(int y=0;y<temporal.size();y++){
        //cout<<","<<temporal[y].first<<" "<<temporal[y].second;
    }//cout<<endl;
    for(int y=0;y<temporal.size();y++){
        if(ciudad_i!=temporal[y]){
            ////cout<<","<<temporal[y].first<<" "<<temporal[y].second;
            float t=F[ciudad_i.first][ciudad_i.first];
            float n=D[temporal[y].first][temporal[y].second];
            sumas_provavilidad.push_back(n*t);
        ////cout<<n<<"*"<<t<<" ";
        }
    }////cout<<endl;
    ////cout<<sumas_provavilidad.size()<<endl;
    float suma = suma_v(sumas_provavilidad);

    for(int y=0;y<sumas_provavilidad.size();y++){
        if(suma==0){
            float numm=1.0/8;
            ////cout<<"(---)"<<numm<<endl;
            sumas_provavilidad[y]=numm;
        }
        else{
            sumas_provavilidad[y]=sumas_provavilidad[y]/suma;
        }
    }

    float suma_total=0;
    /*for(int y=0;y<sumas_provavilidad.size();y++){
        suma_total+=sumas_provavilidad[y];
    }*/

    for(int y=1;y<sumas_provavilidad.size();y++){
        sumas_provavilidad[y]+=sumas_provavilidad[y-1];
    }
    /*for(int y=0;y<sumas_provavilidad.size();y++){
        //cout<<sumas_provavilidad[y]<<" , ";
    }*/

    ////cout<<"====================="<<suma_total<<endl;

    float random_= genRandom(0,1);
    ///aqui esta esl error arregla las probavilidades
    for(int y=0;y<sumas_provavilidad.size();y++){
        if(y==0){
            if(random_<sumas_provavilidad[0]){
                colonia[x].camino.push_back(temporal[y]);
                break;
            }
        }
        else{
            if(random_>sumas_provavilidad[y-1]&& random_<=sumas_provavilidad[y]){
                colonia[x].camino.push_back(temporal[y]);
                break;
            }
        }
    }
    sumas_provavilidad.clear();
}

void hallar_camino(){
    for(int x=0;x<colonia.size();x++){
        pair<num,num>ciudad_i;
        ciudad_i.first=colonia[x].pos_x;
        ciudad_i.second=colonia[x].pos_y;
        vector<pair<int,int>> temporal;
        //cout<<"\n___________________________________________________"<<endl;

        colonia[x].camino.push_back(ciudad_i);

        while(colonia[x].camino.size()<8){
            //cout<<"camino = "<<ciudad_i.first<<","<<ciudad_i.second<<" = "<<colonia[x].camino.size()<<endl;
            ///esquina superior izquierda
            if(ciudad_i.first==0.0 && ciudad_i.second==0.0){
                    ////cout<<"esquina superior izquierda"<<endl;
                num suma_total=0;
                temporal.clear();
                for(int i=0;i<=2;i++){
                    for(int j=0;j<=2;j++){
                        //if(i!=j){
                            //if(buscar_repetidos(colonia[x].camino,make_pair(i,j))==false){
                                temporal.push_back(make_pair(i,j));
                          //  }
                        //}
                    }
                }
                camino_pixel(x,temporal,ciudad_i);
            }

            ///centro
            else if(ciudad_i.first<dim_D-1 && ciudad_i.second<dim_D-1
                    && ciudad_i.first>0 && ciudad_i.second>0){
                //cout<<"centro"<<endl;
                num suma_total=0;
                temporal.clear();
                for(int i=ciudad_i.first-1;i<=ciudad_i.first+1;i++){
                    for(int j=ciudad_i.second-1;j<=ciudad_i.second+1;j++){
                        //if(i!=ciudad_i.first && j!=ciudad_i.second){
                            //if(buscar_repetidos(colonia[x].camino,make_pair(i,j))==false){
                                temporal.push_back(make_pair(i,j));
                            //}
                        //}
                    }
                }

                camino_pixel(x,temporal,ciudad_i);
            }

            ///pared superior
            else if(ciudad_i.first==0.0 && ciudad_i.second<=dim_D-2 &&ciudad_i.second>=1){
                //cout<<"pared superior********"<<endl;
                num suma_total=0;
                temporal.clear();
                for(int i=0;i<=2;i++){
                    for(int j=ciudad_i.second-1;j<=ciudad_i.second+1;j++){
                        //if(i!=ciudad_i.first && j!=ciudad_i.second){
                            //if(buscar_repetidos(colonia[x].camino,make_pair(i,j))==false){
                                temporal.push_back(make_pair(i,j));
                            //}
                        //}
                    }
                }

                /*for(int i=0;i<temporal.size();i++){
                        //cout<<"temporal "<<temporal[i].first<<" "<<
                        temporal[i].second<<"="<<colonia[x].camino.size()cl<<endl;
                }*/
                camino_pixel(x,temporal,ciudad_i);
                ////cout<<"**************"<<endl;
            }

            ///esquina superior derecha
            else if(ciudad_i.first==0.0 && ciudad_i.second==dim_D-1){
                //cout<<"esquina superior derecha"<<endl;
                num suma_total=0;
                temporal.clear();
                for(int i=0;i<2;i++){
                    for(int j=dim_D-3;j<dim_D;j++){
                        //if(i!=j){
                            //if(buscar_repetidos(colonia[x].camino,make_pair(i,j))==false){
                                temporal.push_back(make_pair(i,j));
                            //}
                        //}
                    }
                }
                camino_pixel(x,temporal,ciudad_i);
            }
            ///pared  derecha

            else if(ciudad_i.first<=dim_D-2 && ciudad_i.first>=1 && ciudad_i.second==0){

                //cout<<"pared  derecha"<<endl;
                num suma_total=0;
                temporal.clear();
                for(int i=ciudad_i.first-1;i<=ciudad_i.first+1;i++){
                    for(int j=dim_D-3;j<=dim_D-1;j++){
                        //if(i!=ciudad_i.first && j!=ciudad_i.second){
                            //if(buscar_repetidos(colonia[x].camino,make_pair(i,j))==false){
                                temporal.push_back(make_pair(i,j));
                            //}
                        //}
                    }
                }
                camino_pixel(x,temporal,ciudad_i);
            }




             ///pared izquierda
            else if(ciudad_i.first<=dim_D-1 && ciudad_i.first>=1 && ciudad_i.second==dim_D-1){
                //cout<<"pared izquierda"<<endl;

                num suma_total=0;
                temporal.clear();
                for(int i=ciudad_i.first-1;i<ciudad_i.first+1;i++){
                    for(int j=0;j<=2;j++){
                        //if(i!=ciudad_i.first && j!=ciudad_i.second){
                            //if(buscar_repetidos(colonia[x].camino,make_pair(i,j))==false){
                        temporal.push_back(make_pair(i,j));
                            //}
                        //}
                    }
                }
                camino_pixel(x,temporal,ciudad_i);
            }

             ///esquina inferior izquierda
            else if(ciudad_i.first==dim_D-1 && ciudad_i.second==0.0){
                temporal.clear();
                //cout<<"esquina inferior izquierda"<<endl;
                num suma_total=0;
                for(int i=dim_D-3;i<dim_D;i++){
                    for(int j=0;j<=2;j++){
                        //if(i!=j){
                            //if(buscar_repetidos(colonia[x].camino,make_pair(i,j))==false){
                        temporal.push_back(make_pair(i,j));
                            ///}
                        //}
                    }
                }
                camino_pixel(x,temporal,ciudad_i);

            }
             ///pared inferior

            else if(ciudad_i.first==dim_D-1 && ciudad_i.second<dim_D-1 &&ciudad_i.second>=1){
                temporal.clear();
                //cout<<"pared inferior"<<endl;
                num suma_total=0;
                for(int i=dim_D-3;i<dim_D;i++){
                    for(int j=ciudad_i.second-1;j<=ciudad_i.second+1;j++){
                        //if(i!=ciudad_i.first && j!=ciudad_i.second){
                        //if(buscar_repetidos(colonia[x].camino,make_pair(i,j))==false){
                        temporal.push_back(make_pair(i,j));
                            //}
                        //}
                    }
                }

                camino_pixel(x,temporal,ciudad_i);
            }

            ///esquina inferior derecha
            else if(ciudad_i.first==dim_D-1 && ciudad_i.second==dim_D-1){
                temporal.clear();
                //cout<<"esquina inferior derecha"<<endl;
                num suma_total=0;
                for(int i=dim_D-3;i<dim_D;i++){
                    for(int j=dim_D-3;j<dim_D;j++){
                        //if(i!=j){
                            //if(buscar_repetidos(colonia[x].camino,make_pair(i,j))==false){
                                temporal.push_back(make_pair(i,j));
                            //}
                        //}
                    }
                }
                camino_pixel(x,temporal,ciudad_i);

            }

            //////cout<<"-------="<<colonia[x].camino.size()-1<<endl;
            ciudad_i.first =colonia[x].camino[colonia[x].camino.size()-1].first;
            ciudad_i.second=colonia[x].camino[colonia[x].camino.size()-1].second;
            temporal.clear();

        }
    }

}

float fitness(vector<pair<num,num>> vec){
    float Uk_g=0;
    for(int x=0;x<vec.size()-1;x++){
        int posx=vec[x].first;
        int posy=vec[x].second;
        Uk_g+=D[posx][posy];
        //////cout<<D[posx][posy]<<", ";
    }float dato = 1/(A+(B*(Uk_g)));
    //////cout<<"="<<dato<<"\n"<<endl;

    return dato;
}

void actualzar_feromonas(){
    //////cout<<"__________________________________________"<<endl;
    for(int x=0;x<colonia.size();x++){
        for(int y=0;y<colonia[x].camino.size();x++){
            int Px=colonia[x].camino[y].first;
            int Py=colonia[x].camino[y].second;
            ///faltaria el filtrado de pixeles analogos
            float numero=fitness(colonia[x].camino);
            F[Px][Py]=F[Px][Py]*w+numero;
            //////cout<<F[Px][Py]<<","<<fitness(colonia[x].camino)<<"\t";
        }//////cout<<endl;
        colonia[x].camino.clear();
    }
    ///reduccion de feromonas
    for(int x=0;x<dim_D;x++){
        for(int y=0;y<dim_D;y++){
            F[x][y]=F[x][y]*(1-Ro);
            //////cout<<"["<<F[x][y]<<"] ";
        }//////cout<<"\n"<<endl;
    }
    //////cout<<"__________________________________________"<<endl;
}
void mostrar_matriz_feremonas(){

}
void normalisar_matriz_de_feromonas(){
    float valor_minimo=F[0][0];
    float valor_maximo=F[0][0];

    for(int x=0;x<dim_D;x++){
        for(int y=0;y<dim_D;y++){
            if(valor_minimo>F[x][y]){
                valor_minimo=F[x][y];
            }
            if(valor_maximo<F[x][y]){
                valor_maximo=F[x][y];
            }
        }
    }
    //////cout<<"V_m= "<<valor_minimo<<" V_M= "<<valor_maximo<<endl;
    for(int x=0;x<dim_D;x++){
        for(int y=0;y<dim_D;y++){
            F[x][y]=(F[x][y]-valor_minimo)/(valor_maximo-valor_minimo);
            //////cout<<F[x][y]<<" ";
        }//////cout<<endl;
    }

}

void guardar_fichero(){
    cout<<"guardando fichero"<<endl;

    ofstream fs("input.csv");
    //fs<<"feromonas"<<";"<<"colores"<<endl;
    for(int x=0;x<dim_D;x++){
        for(int y=0;y<dim_D;y++){
            fs<<F[x][y]<<";"<<D[x][y]<<endl;
        }//////cout<<endl;
    }fs.close();
}

int main() {
    srand(time(0));
    CImg<float> img("imagenes_entrada/imagenA.bmp");  // Image definitions.

    //img.resize(15,15);
    img=transform_to_grayscale(img);
    unsigned int spectrum = img.spectrum();
    CImg<float> img2(img);
    img2.save("salida.bmp");
    int tam_imagen = img2.height();
    ////cout<<"tam_imagen"<<tam_imagen<<endl;
    dim_D=tam_imagen;
    ///____________________________________________
    asginar_pixel_hormiga(img2);
    rellenar_matriz_D(img2);
    for(int x=0;x<5;x++){///5
        cout<<"iteracion "<<x+1<<endl;
        hallar_camino();
        actualzar_feromonas();
    }
    cout<<"normalisando matriz de feromonas"<<endl;
    normalisar_matriz_de_feromonas();
    guardar_fichero();
    ///____________________________________________

    //img2.sqrt();
    //CImgDisplay vent1(img,"original"), vent2(img2,"procesada");
    //while(!vent1.is_closed()){}
    return 0;

}
