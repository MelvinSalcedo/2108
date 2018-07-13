#include <stdio.h>
#include "CImg.h"

using namespace cimg_library;

int main( ){
	int w,h;
	scanf("%d",&w);
	scanf("%d",&h);
	
	int total = w * h;

	unsigned char * pixels = new unsigned char[total]();
	int val;
	for(int i=0;i<total;i++){
		scanf("%d",&val);
		pixels[i]=val;
	}
	
	CImg<unsigned char> imgR(w,h,1,3,0);
	int pos;	

	for(int i=0;i<h;i++){
		for(int j=0;j<w;j++){
			pos=i*w + j;
			imgR(j,i,0)=pixels[pos];
			imgR(j,i,1)=pixels[pos];
			imgR(j,i,2)=pixels[pos];
		}
	}

	imgR.display();
	return 0;
}
