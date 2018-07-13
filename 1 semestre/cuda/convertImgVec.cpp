#include <stdio.h>
#include "CImg.h"

using namespace cimg_library;

int main(int argc,char *argv[]){
	CImg<unsigned char> img(argv[1]);
	img.resize(500,500);

	printf("%d %d\n",img.width(),img.height());

	for(int i=0;i<img.height();i++){
		for(int j=0;j<img.width();j++){
			printf("%d ",img(j,i,0));
			printf("%d ",img(j,i,1));
			printf("%d ",img(j,i,2));
		}
	}
	
	return 0;
}
