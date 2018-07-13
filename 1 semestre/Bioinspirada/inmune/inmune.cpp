#include <iostream>
#include <bitset>
#include <string>
#include <algorithm>    // std::max
#include <vector>
#include <math.h>
#include <time.h>
using namespace std;
typedef int num;
num generacion = 100;
num tam_poblacion = 4;
num tam_seleccion = 4;
num n_random_celulas = 2;
double clone_rate = 0.5;
double fator_mutacion = -2.5;
num n_clones = int(tam_poblacion * clone_rate);
//__________________________________________________________________________--
int binary_to_decimal(const std::vector<int>& bits){
    int result = 0;
    int base = 1;

    //Supposing the MSB is at the begin of the bits vector:
    for(int i = bits.size()-1 ; i >= 0 ; --i){
        result += bits[i]*base;
        base *= 2;
    }

    return result;
}
std::vector<int> get_decimal_digits(int number)
{
    //Allocate the vector with the number of digits of the number:
    std::vector<int> digits( std::log10(number) - 1 );

    while(number / 10 > 0)
    {
        digits.insert( digits.begin() , number % 10);
        number /= 10;
    }

    return digits;
}

std::vector<int> binary_digits_to_decimal_digits(const std::vector<int>& bits){
    int n=binary_to_decimal(bits);
    cout<<n<<endl;
    //return get_decimal_digits(n);
}
///_________________________________________________________
num min_x = -5;
num max_x = 5;

num min_y = -5;
num max_y = 5;
struct indi{
    double x;
    double y;
    double funcion;
    double a;
    vector<num> lista;
};
vector<indi> individuos;
indi mejor_solucion;


void mostrar_poblacion_incial(){
     for(int i=0;i<tam_poblacion;i++){
        cout<<"X:"<<individuos[i].x<<"\tY:"<<individuos[i].y<<" \tF:"
        <<individuos[i].funcion<<"\tA:"<<individuos[i].a<<"\tL:";
        for(int x=0;x<individuos[i].lista.size();x++){
            cout<<individuos[i].lista[x]<<" ";
        }cout<<endl;
     }
}
double problema( double x, double y ){
	return (pow(x,2) + pow(y,2));

}
void rrellenar(vector<num>&a,vector<num>lista,num limite){
    /*cout<<"----------------------------"<<limite<<endl;
    for(int i=0;i<lista.size();i++){
        cout<<lista[i]<<" ";
    }cout<<endl;*/
    if(limite==16){
        for(int i=0;i<16;i++){
            a.push_back(lista[i]);
        }
    }
    if(limite==32){
        for(int i=16;i<32;i++){
            a.push_back(lista[i]);
        }
    }
}

void convertir_binario_decimal( vector<num>binary_list ,double&x,double&y ){
	vector<num>x1,y1;
	rrellenar(x1,binary_list,16);
	rrellenar(y1,binary_list,32);

    /*for (int x=0;x<x1.size();x++){
        cout<<x1[x]<<" ";
    }cout<<endl;
    for (int x=0;x<x1.size();x++){
        cout<<y1[x]<<" ";
    }cout<<endl;*/

	x=binary_to_decimal(x1);
	y=binary_to_decimal(y1);

	x = min_x + (( max_x - min_x ) * x ) / (( pow(2.,16) - 1. ));
	y = min_y + (( max_y - min_y ) * y ) / (( pow(2.,16) - 1. ));
    //cout<<x<<endl;
}

indi creacion_solucion(){
    indi temporal;
    vector<num> lista_b;
    for(int x=0;x<32;x++){
        lista_b.push_back(rand() % 2);
    }
    double x,y;
	convertir_binario_decimal(lista_b,x,y);

	temporal.x=x;
	temporal.y=y;
	temporal.funcion=problema(x,y);
	temporal.a=100000000;
	temporal.lista=lista_b;
	//sol = {"x": x, "y": y, "func": problema(x,y), "a": np.inf, "binary": lista_b }
	return temporal;
}

num valor_limite(vector<num>a,string m){
    num valor=a[0];
    for(int i=1 ;i<a.size();i++){
        if(m=="M"){
            if(valor<a[i]){
                valor=a[i];
            }
        }
        if(m=="m"){
            if(valor>a[i]){
                valor=a[i];
            }
        }
    }
    return valor;
}
void afinidad(){
    ///vector<pair<double,vector<num>>>
	vector<num>fs;
	for(int i=0 ;i<individuos.size();i++){
		fs.push_back(individuos[i].funcion);
	}
	num max_f = valor_limite(fs,"M");
	num min_f = valor_limite(fs,"m");

	for(int i=0 ;i<individuos.size();i++){
		if (max_f - min_f == 0){
			individuos[i].a=1;
        }
		else{
			individuos[i].a=1-((individuos[i].funcion)/( max_f - min_f ));
		}
	}
}
void poblacion_inicial(){
	for(int i=0;i<tam_poblacion;i++){
		individuos.push_back( creacion_solucion() );
	}
}

vector<indi> clon_hipermutacion(){
	vector<indi> tmp;
	for(int i=0;i<tam_poblacion;i++){
		for(int j=0;j<n_clones;j++){
            indi clon;
			clon =individuos[i];
			double taza_mutacion;
			try{
				 taza_mutacion=exp(fator_mutacion * clon.a);
			}
			catch( double ){
				taza_mutacion = 0;
			}

			for(int k=0;k<clon.lista.size();k++){

				double rand_ = static_cast <double> (rand()) /static_cast <double> (RAND_MAX);
				if( rand_ < taza_mutacion ){
					if (clon.lista[k] == 0){
						clon.lista[k] = 1;
					}
					else{
						clon.lista[k] = 0;
					}
				}
            }
			double x_tmp, y_tmp;
			convertir_binario_decimal( clon.lista,x_tmp,y_tmp);
			clon.x = x_tmp;
			clon.y = y_tmp;
			clon.funcion = problema( clon.x, clon.y );
			tmp.push_back(clon);
		}
	}
    return tmp;
}

vector<indi> crear_celulas_random(){
	vector<indi> tmp;
	for(int i=0;i<tam_poblacion;i++){
        indi a;
		tmp.push_back( creacion_solucion() );
	}
	return tmp;
}

bool compareByLength(const indi &a, const indi &b){
    return a.funcion < b.funcion;
}

void clonalg(){
	num iteracion = 0;
	poblacion_inicial();
	cout<<"________POBLACION INICIAL_________"<<endl;
    mostrar_poblacion_incial();
	for(int i=0;i<generacion;i++){
		//cout<<"+++++++ iteracion "<<i+1<<endl;
		afinidad();

		//cout<<"actualizando afinida\n";
		//mostrar_poblacion_incial();

		vector<indi> tmp;
		//cout<<"hallando clones"<<endl;
		tmp = clon_hipermutacion();
		vector<indi>rnd_cells = crear_celulas_random();
		for(int x=0;x<tmp.size();x++){
            individuos.push_back(tmp[x]);
		}
		for(int x=0;x<rnd_cells.size();x++){
            individuos.push_back(rnd_cells[x]);
		}
		std::sort(individuos.begin(), individuos.end(), compareByLength);
		//vector<indi>sorted_individuos = sorted(individuos, key=lambda k: k["func"] )
        vector<indi>temporal=individuos;
		individuos.clear();
		for(int j=0;j<tam_poblacion;j++){
			individuos.push_back( temporal[j] );
		}
		temporal.clear();

		if (mejor_solucion.funcion > individuos[0].funcion){
			mejor_solucion=individuos[0];
		}
	}
	cout<<"________POBLACION FINAL_________"<<endl;
    mostrar_poblacion_incial();
}
int main(){
    srand(time(0));
    mejor_solucion.funcion=1000000000;
    clonalg();
    cout<<"MEJOR_SOLUCION"<<endl;
    cout<<"x:"<< mejor_solucion.x<<"y: "<<mejor_solucion.y<<" f: "<<
    mejor_solucion.funcion<<" a:"<<mejor_solucion.a<<"\tL:";
    for(int x=0;x<mejor_solucion.lista.size();x++){
        cout<<mejor_solucion.lista[x]<<" ";
    }cout<<endl;
}
