#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CHANNELS 3
#define BLOCK_SIZE 16

__global__ void colorToGreyscale(unsigned char * Pin,unsigned char * Pout, int width,int height)
{
	int col = blockIdx.x*blockDim.x+threadIdx.x;
	int row = blockIdx.y*blockDim.y+threadIdx.y;
	
	if (col < width && row < height){
		int greyOffset = row*width+col;
		int rgbOffset = greyOffset * CHANNELS;

		unsigned char r = Pin[rgbOffset];
		unsigned char g = Pin[rgbOffset+1];
		unsigned char b = Pin[rgbOffset+2];

		Pout[greyOffset] = (0.21f*float(r)) + (0.71f*float(g)) + (0.07f*float(b));	
	}
}


int main( )
{
	int Iwidth,Iheight;

	scanf("%d",&Iwidth);
	scanf("%d",&Iheight);
	
	int total_pixel=Iwidth*Iheight;	
	
	unsigned char * pixels = new unsigned char[total_pixel*3]();	
	int val;
	for(int i=0;i<total_pixel*3;i++){
		scanf("%d",&val);
		pixels[i]=val;
	}
	unsigned char * h_out;
	unsigned char * d_out;
	unsigned char * d_pixels;

	size_t bytes = total_pixel*sizeof(unsigned char);
	
	h_out = (unsigned char*)malloc(bytes);
	cudaMalloc((void**)&d_pixels,bytes*3);
	cudaMalloc((void**)&d_out,bytes);

	cudaMemcpy(d_pixels,pixels,bytes*3,cudaMemcpyHostToDevice);	

	dim3 dimGrid(ceil(float(Iwidth)/BLOCK_SIZE),ceil(float(Iheight)/BLOCK_SIZE),1);
	dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE,1);
	
	colorToGreyscale<<<dimGrid,dimBlock>>>(d_pixels,d_out,Iwidth,Iheight);

	cudaMemcpy(h_out,d_out,bytes,cudaMemcpyDeviceToHost);	
	
	printf("%d %d\n",Iwidth,Iheight);

	for(int i=0;i<total_pixel;i++){
		printf("%d ",h_out[i]);
	}

	cudaFree(d_pixels);
	cudaFree(d_out);
	free(pixels);
	free(h_out);
        return 0;
}
