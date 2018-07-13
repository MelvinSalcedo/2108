#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CHANNELS 3
#define BLOCK_SIZE 16
#define BLURSIZE 1

__global__ void blurFunction(unsigned char * Pin,unsigned char * Pout, int width,int height)
{
	int col = blockIdx.x*blockDim.x+threadIdx.x;
	int row = blockIdx.y*blockDim.y+threadIdx.y;
	
	if (col < width && row < height){
		int pixValR,pixValG,pixValB;
	    	pixValR = pixValB = pixValG= 0;
		int pixels = 0;
	
		for(int Brow = -BLURSIZE;Brow<BLURSIZE+1;Brow++){
			for(int Bcol=-BLURSIZE;Bcol<BLURSIZE+1;Bcol++){
				int curRow = row + Brow;
				int curCol = col + Bcol;

				if(curRow > -1 && curCol>-1 && curRow<height && curCol<width){
					int curPos = (curRow*width+curCol)*3;
					pixValR+=Pin[curPos];
					pixValG+=Pin[curPos+1];
					pixValB+=Pin[curPos+2];
					pixels++;
				}
			}
		}
		int pos = (row*width+col)*3;
		Pout[pos] = (unsigned char)(pixVal/pixels);
		Pout[pos+1] = (unsigned char)(pixval/pixels);
		Pout[pos+1] = (unsigned char)(pixVal/pixels);
	}
}


int main( )
{
	int Iwidth,Iheight;

	scanf("%d",&Iwidth);
	scanf("%d",&Iheight);
	
	int total_pixel=Iwidth*Iheight*3;
	
	unsigned char * pixels = new unsigned char[total_pixel]();	
	int val;
	for(int i=0;i<total_pixel;i++){
		scanf("%d",&val);
		pixels[i]=val;
	}
	unsigned char * h_out;
	unsigned char * d_out;
	unsigned char * d_pixels;

	size_t bytes = total_pixel*sizeof(unsigned char);
	
	h_out = (unsigned char*)malloc(bytes);
	cudaMalloc((void**)&d_pixels,bytes);
	cudaMalloc((void**)&d_out,bytes);

	cudaMemcpy(d_pixels,pixels,bytes,cudaMemcpyHostToDevice);	

	dim3 dimGrid(ceil(float(Iwidth)/BLOCK_SIZE),ceil(float(Iheight)/BLOCK_SIZE),1);
	dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE,1);
	
	blurFunction<<<dimGrid,dimBlock>>>(d_pixels,d_out,Iwidth*3,Iheight*3);

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
