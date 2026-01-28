#include<iostream>
#include<cuda_runtime.h>
#include<cuda_profiler_api.h>
#include<cmath>
#include<sys/time.h>
#include"cuda_utility.h"
#include"main.h"

#define PARAMETER 10

#define BLOCK_X 32	
#define BLOCK_Y 4
#define GRID_Y 16

__global__ void NLmeansOnGPU_shift_2(unsigned char *data,float *out_image,float *weight,int W, int H, int sr_size, int nb_size, int IMAGE_SIZE_X, int IMAGE_SIZE_Y)
{	
	unsigned int col = IMAGE_SIZE_Y/GRID_Y;	
	unsigned int col_thread = col/BLOCK_Y;																																//
	unsigned int iy = blockIdx.y;																											
	unsigned int wrapID = (threadIdx.x +blockIdx.x*32)/32, wrapLane=(threadIdx.x +blockIdx.x*32) % 32, wrap_y = threadIdx.y;											//
	unsigned int ix = wrapID*30+wrapLane, ix_GM = wrapID*32+wrapLane;																									////#
	unsigned int m=(sr_size - 1) / 2,  n=(nb_size - 1) / 2;									////

	float g1 = 0;
	float g3 = 0;
	float gmid = 0;
	float result = 0;
//out_image

	for (int i = 0; i < sr_size; i++)
	{
		for (int j = 0; j < sr_size; j++)
		{
			for(int k = 0; k < nb_size; k++)
			{
				g1 = data[m + ix + ( k + m +iy*col+wrap_y*col_thread) * W] - data[ix + j + ( k + i +iy*col+wrap_y*col_thread) * W];
				g1 = g1 * g1;
				g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
				g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
				
				g1 = data[m + ix + 3 + ( k + m +iy*col+wrap_y*col_thread) * W] - data[ix + j + 3 + ( k + i +iy*col+wrap_y*col_thread) * W];
				g1 = g1 * g1;
				g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
				g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
				
				g1 = data[m + ix + 6 + (k + m +iy*col+wrap_y*col_thread) * W] - data[ix + j + 6 + ( k + i +iy*col+wrap_y*col_thread) * W];
				g1 = g1 * g1;
				g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
				g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
				
				g1 = data[m + ix + 9 + (k + m +iy*col+wrap_y*col_thread) * W] - data[ix + j + 9 + ( k + i +iy*col+wrap_y*col_thread) * W];
				g1 = g1 * g1;
				g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
				
			} 
			weight[ix_GM + j *(int)gridDim.x*32  + i * sr_size * (int)gridDim.x*32+(iy*BLOCK_Y+wrap_y)*(int)gridDim.x*32*sr_size*sr_size] = g3;			////#
			g3 = g3 / (float)(nb_size * nb_size);
			g3 = exp(-g3 / (PARAMETER * PARAMETER));
			gmid += g3;
			result += g3 * data[ix +n+ j + (i +n+iy*col+wrap_y*col_thread)* W];
			g3 = 0;
       	}
	}
	if(wrapLane<30 && ix<IMAGE_SIZE_X)out_image[ix+(iy*col+wrap_y*col_thread)*IMAGE_SIZE_Y] = result / gmid;
	result = 0;
	gmid = 0;

///*
	for (int itra = 1; itra < col_thread; itra++)
	{
		for (int i = 0; i < sr_size; i++)
		{
			for (int j = 0; j < sr_size; j++)
			{
				g3 = weight[ix_GM + j * (int)gridDim.x*32 + i * sr_size * (int)gridDim.x*32 +(iy*BLOCK_Y+wrap_y)*(int)gridDim.x*32 *sr_size*sr_size];				////#
				g1 = data[m + ix + (itra-1 + m+iy*col+wrap_y*col_thread) * W] - data[ix + j + (itra-1 + i+iy*col+wrap_y*col_thread) * W];
				g1 = g1 * g1;
				g3 = g3-(g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
				g3 = g3-__shfl_down_sync(0xffffffff, g1, 2, 32);
				g1 = data[m + ix + 3 + (itra-1 + m+iy*col+wrap_y*col_thread) * W] - data[ix + j + 3 + (itra-1 + i+iy*col+wrap_y*col_thread) * W];
				g1 = g1 * g1;
				g3 = g3-(g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
				g3 = g3-__shfl_down_sync(0xffffffff, g1, 2, 32);
				g1 = data[m + ix + 6 + (itra-1 + m+iy*col+wrap_y*col_thread) * W] - data[ix + j + 6 + (itra-1 + i+iy*col+wrap_y*col_thread) * W];
				g1 = g1 * g1;
				g3 = g3-(g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
				g3 = g3-__shfl_down_sync(0xffffffff, g1, 2, 32);
				g1 = data[m + ix + 9 + (itra-1 + m+iy*col+wrap_y*col_thread) * W] - data[ix + j + 9 + (itra-1 + i+iy*col+wrap_y*col_thread) * W];
				g1 = g1 * g1;
				g3 = g3-(g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));





				g1 = data[m + ix + (itra + 10 + m+iy*col+wrap_y*col_thread) * W] - data[ix + j + (itra + 10 + i+iy*col+wrap_y*col_thread) * W];
				g1 = g1 * g1;
				g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
				g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
				g1 = data[m + ix + 3 + (itra + 10 + m+iy*col+wrap_y*col_thread) * W] - data[ix + j + 3 + (itra + 10 + i+iy*col+wrap_y*col_thread) * W];
				g1 = g1 * g1;
				g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
				g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
				g1 = data[m + ix + 6 + (itra + 10 + m+iy*col+wrap_y*col_thread) * W] - data[ix + j + 6 + (itra + 10 + i+iy*col+wrap_y*col_thread) * W];
				g1 = g1 * g1;
				g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
				g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
				g1 = data[m + ix + 9 + (itra + 10 + m+iy*col+wrap_y*col_thread) * W] - data[ix + j + 9 + (itra + 10 + i+iy*col+wrap_y*col_thread) * W];
				g1 = g1 * g1;
				g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
				
				weight[ix_GM + j * (int)gridDim.x*32 + i * sr_size * (int)gridDim.x*32+(iy*BLOCK_Y+wrap_y)*(int)gridDim.x*32*sr_size*sr_size] = g3;			////#
				g3 = g3 / (float)(nb_size * nb_size);
				g3 = exp(-g3 / (PARAMETER * PARAMETER));
				gmid += g3;
				result += g3 * data[ix +n+ j + (i +n+ itra+iy*col+wrap_y*col_thread)* W];
				g3 = 0;
			}
		}
		if(wrapLane<30 && ix<IMAGE_SIZE_X)out_image[ix+(itra+iy*col+wrap_y*col_thread)* IMAGE_SIZE_Y] = result / gmid;
		result = 0;
		gmid = 0;
	}

//*/	
}

__global__ void NLmeansOnGPU_shift_1(unsigned char *data,float *out_image,float *weight,int W, int H, int sr_size, int nb_size, int IMAGE_SIZE_X, int IMAGE_SIZE_Y)
{	
	unsigned int col = IMAGE_SIZE_Y/GRID_Y;	
	unsigned int col_thread = col/BLOCK_Y;																																//
	unsigned int iy = blockIdx.y;																											
	unsigned int wrapID = (threadIdx.x +blockIdx.x*32)/32, wrapLane=(threadIdx.x +blockIdx.x*32) % 32, wrap_y = threadIdx.y;											//
	unsigned int ix = wrapID*31+wrapLane, ix_GM = wrapID*32+wrapLane;																									////#
	unsigned int m=(sr_size - 1) / 2,  n=(nb_size - 1) / 2;									////

	float g1 = 0;
	float g3 = 0;
	float gmid = 0;
	float result = 0;
//out_image

	for (int i = 0; i < sr_size; i++)
	{
		for (int j = 0; j < sr_size; j++)
		{
			for(int k = 0; k < nb_size; k++)
			{
				g1 = data[m + ix + ( k + m +iy*col+wrap_y*col_thread) * W] - data[ix + j + ( k + i +iy*col+wrap_y*col_thread) * W];
				g1 = g1 * g1;
				g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);

				g1 = data[m + ix + 2 + ( k + m +iy*col+wrap_y*col_thread) * W] - data[ix + j + 2 + ( k + i +iy*col+wrap_y*col_thread) * W];
				g1 = g1 * g1;
				g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);


				g1 = data[m + ix + 4 + ( k + m +iy*col+wrap_y*col_thread) * W] - data[ix + j + 4 + ( k + i +iy*col+wrap_y*col_thread) * W];
				g1 = g1 * g1;
				g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);

				g1 = data[m + ix + 6 + ( k + m +iy*col+wrap_y*col_thread) * W] - data[ix + j + 6 + ( k + i +iy*col+wrap_y*col_thread) * W];
				g1 = g1 * g1;
				g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);

				g1 = data[m + ix + 8 + ( k + m +iy*col+wrap_y*col_thread) * W] - data[ix + j + 8 + ( k + i +iy*col+wrap_y*col_thread) * W];
				g1 = g1 * g1;
				g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);

				g1 = data[m + ix + 10 + ( k + m +iy*col+wrap_y*col_thread) * W] - data[ix + j + 10 + ( k + i +iy*col+wrap_y*col_thread) * W];
				g1 = g1 * g1;
				g3 += g1;						
			} 
			weight[ix_GM + j *(int)gridDim.x*32  + i * sr_size * (int)gridDim.x*32+(iy*BLOCK_Y+wrap_y)*(int)gridDim.x*32*sr_size*sr_size] = g3;			////#
			g3 = g3 / (float)(nb_size * nb_size);
			g3 = exp(-g3 / (PARAMETER * PARAMETER));
			gmid += g3;
			result += g3 * data[ix + n+j + (i +n+iy*col+wrap_y*col_thread)* W];
			g3 = 0;
       	}
	}
	if(wrapLane<31 && ix<IMAGE_SIZE_X)out_image[ix+(iy*col+wrap_y*col_thread)*IMAGE_SIZE_Y] = result / gmid;
	result = 0;
	gmid = 0;

///*
	for (int itra = 1; itra < col_thread; itra++)
	{
		for (int i = 0; i < sr_size; i++)
		{
			for (int j = 0; j < sr_size; j++)
			{
				g3 = weight[ix_GM + j * (int)gridDim.x*32 + i * sr_size * (int)gridDim.x*32 +(iy*BLOCK_Y+wrap_y)*(int)gridDim.x*32 *sr_size*sr_size];				////#
				
				g1 = data[m + ix + (itra-1 + m+iy*col+wrap_y*col_thread) * W] - data[ix + j + (itra-1 + i+iy*col+wrap_y*col_thread) * W];
				g1 = g1 * g1;
				g3 = g3-(g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));

				g1 = data[m + ix + 2 + (itra-1 + m+iy*col+wrap_y*col_thread) * W] - data[ix + j + 2 + (itra-1 + i+iy*col+wrap_y*col_thread) * W];
				g1 = g1 * g1;
				g3 = g3-(g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));

				g1 = data[m + ix + 4 + (itra-1 + m+iy*col+wrap_y*col_thread) * W] - data[ix + j + 4 + (itra-1 + i+iy*col+wrap_y*col_thread) * W];
				g1 = g1 * g1;
				g3 = g3-(g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));

				g1 = data[m + ix + 6 + (itra-1 + m+iy*col+wrap_y*col_thread) * W] - data[ix + j + 6 + (itra-1 + i+iy*col+wrap_y*col_thread) * W];
				g1 = g1 * g1;
				g3 = g3-(g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));

				g1 = data[m + ix + 8 + (itra-1 + m+iy*col+wrap_y*col_thread) * W] - data[ix + j + 8 + (itra-1 + i+iy*col+wrap_y*col_thread) * W];
				g1 = g1 * g1;
				g3 = g3-(g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));


				g1 = data[m + ix + 10 + (itra-1 + m+iy*col+wrap_y*col_thread) * W] - data[ix + j + 10 + (itra-1 + i+iy*col+wrap_y*col_thread) * W];
				g1 = g1 * g1;
				g3 = g3-g1;






				g1 = data[m + ix + (itra + 10 + m+iy*col+wrap_y*col_thread) * W] - data[ix + j + (itra + 10 + i+iy*col+wrap_y*col_thread) * W];
				g1 = g1 * g1;
				g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);

				g1 = data[m + ix + 2 + (itra + 10 + m+iy*col+wrap_y*col_thread) * W] - data[ix + j + 2 + (itra + 10 + i+iy*col+wrap_y*col_thread) * W];
				g1 = g1 * g1;
				g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
				
				g1 = data[m + ix + 4 + (itra + 10 + m+iy*col+wrap_y*col_thread) * W] - data[ix + j + 4 + (itra + 10 + i+iy*col+wrap_y*col_thread) * W];
				g1 = g1 * g1;
				g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);

				g1 = data[m + ix + 6 + (itra + 10 + m+iy*col+wrap_y*col_thread) * W] - data[ix + j + 6 + (itra + 10 + i+iy*col+wrap_y*col_thread) * W];
				g1 = g1 * g1;
				g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);

				g1 = data[m + ix + 8 + (itra + 10 + m+iy*col+wrap_y*col_thread) * W] - data[ix + j + 8 + (itra + 10 + i+iy*col+wrap_y*col_thread) * W];
				g1 = g1 * g1;
				g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);

				g1 = data[m + ix + 10 + (itra + 10 + m+iy*col+wrap_y*col_thread) * W] - data[ix + j + 10 + (itra + 10 + i+iy*col+wrap_y*col_thread) * W];
				g1 = g1 * g1;
				g3 += g1;
				
				weight[ix_GM + j * (int)gridDim.x*32 + i * sr_size * (int)gridDim.x*32+(iy*BLOCK_Y+wrap_y)*(int)gridDim.x*32*sr_size*sr_size] = g3;			////#
				g3 = g3 / (float)(nb_size * nb_size);
				g3 = exp(-g3 / (PARAMETER * PARAMETER));
				gmid += g3;
				result += g3 * data[ix + n+j + (i +n+ itra+iy*col+wrap_y*col_thread)* W];
				g3 = 0;
			}
		}
		if(wrapLane<31 && ix<IMAGE_SIZE_X)out_image[ix+(itra+iy*col+wrap_y*col_thread)* IMAGE_SIZE_Y] = result / gmid;
		result = 0;
		gmid = 0;
	}

//*/	
}

__global__ void NLmeansOnGPU_shift_3(unsigned char *data,float *out_image,float *weight,int W, int H, int sr_size, int nb_size, int IMAGE_SIZE_X, int IMAGE_SIZE_Y)
{	
	unsigned int col = IMAGE_SIZE_Y/GRID_Y;	
	unsigned int col_thread = col/BLOCK_Y;																																//
	unsigned int iy = blockIdx.y;																											
	unsigned int wrapID = (threadIdx.x +blockIdx.x*32)/32, wrapLane=(threadIdx.x +blockIdx.x*32) % 32, wrap_y = threadIdx.y;											//
	unsigned int ix = wrapID*29+wrapLane, ix_GM = wrapID*32+wrapLane;																									////#
	unsigned int m=(sr_size - 1) / 2,  n=(nb_size - 1) / 2;									////

	float g1 = 0;
	float g3 = 0;
	float gmid = 0;
	float result = 0;
//out_image

	for (int i = 0; i < sr_size; i++)
	{
		for (int j = 0; j < sr_size; j++)
		{
			for(int k = 0; k < nb_size; k++)
			{
				g1 = data[m + ix + ( k + m +iy*col+wrap_y*col_thread) * W] - data[ix + j + ( k + i +iy*col+wrap_y*col_thread) * W];
				g1 = g1 * g1;
				g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
				g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
				g3 += __shfl_down_sync(0xffffffff, g1, 3, 32);
				
				g1 = data[m + ix + 4 + ( k + m +iy*col+wrap_y*col_thread) * W] - data[ix + j + 4 + ( k + i +iy*col+wrap_y*col_thread) * W];
				g1 = g1 * g1;
				g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
				g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
				g3 += __shfl_down_sync(0xffffffff, g1, 3, 32);				
				
				
				g1 = data[m + ix + 8 + ( k + m +iy*col+wrap_y*col_thread) * W] - data[ix + j + 8 + ( k + i +iy*col+wrap_y*col_thread) * W];
				g1 = g1 * g1;
				g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
				g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);

				
			} 
			weight[ix_GM + j *(int)gridDim.x*32  + i * sr_size * (int)gridDim.x*32+(iy*BLOCK_Y+wrap_y)*(int)gridDim.x*32*sr_size*sr_size] = g3;			////#
			g3 = g3 / (float)(nb_size * nb_size);
			g3 = exp(-g3 / (PARAMETER * PARAMETER));
			gmid += g3;
			result += g3 * data[ix +n+ j + (i +n+iy*col+wrap_y*col_thread)* W];
			g3 = 0;
       	}
	}
	if(wrapLane<29 && ix<IMAGE_SIZE_X)out_image[ix+(iy*col+wrap_y*col_thread)*IMAGE_SIZE_Y] = result / gmid;
	result = 0;
	gmid = 0;

///*
	for (int itra = 1; itra < col_thread; itra++)
	{
		for (int i = 0; i < sr_size; i++)
		{
			for (int j = 0; j < sr_size; j++)
			{
				g3 = weight[ix_GM + j * (int)gridDim.x*32 + i * sr_size * (int)gridDim.x*32 +(iy*BLOCK_Y+wrap_y)*(int)gridDim.x*32 *sr_size*sr_size];				////#
				
				g1 = data[m + ix + (itra-1 +m+iy*col+wrap_y*col_thread) * W] - data[ix + j + (itra-1 + i+iy*col+wrap_y*col_thread) * W];
				g1 = g1 * g1;
				g3 = g3-(g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
				g3 = g3-__shfl_down_sync(0xffffffff, g1, 2, 32);
				g3 = g3-__shfl_down_sync(0xffffffff, g1, 3, 32);
				
				g1 = data[m + ix + 4 + (itra-1 + m+iy*col+wrap_y*col_thread) * W] - data[ix + j + 4 + (itra-1 + i+iy*col+wrap_y*col_thread) * W];
				g1 = g1 * g1;
				g3 = g3-(g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
				g3 = g3-__shfl_down_sync(0xffffffff, g1, 2, 32);
				g3 = g3-__shfl_down_sync(0xffffffff, g1, 3, 32);				
				
				
				
				g1 = data[m + ix + 8 + (itra-1 + m+iy*col+wrap_y*col_thread) * W] - data[ix + j + 8 + (itra-1 + i+iy*col+wrap_y*col_thread) * W];
				g1 = g1 * g1;
				g3 = g3-(g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
				g3 = g3-__shfl_down_sync(0xffffffff, g1, 2, 32);




				g1 = data[m + ix + (itra + 10 + m+iy*col+wrap_y*col_thread) * W] - data[ix + j + (itra + 10 + i+iy*col+wrap_y*col_thread) * W];
				g1 = g1 * g1;
				g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
				g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
				g3 += __shfl_down_sync(0xffffffff, g1, 3, 32);
				
				g1 = data[m + ix + 4 + (itra + 10 + m+iy*col+wrap_y*col_thread) * W] - data[ix + j + 4 + (itra + 10 + i+iy*col+wrap_y*col_thread) * W];
				g1 = g1 * g1;
				g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
				g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
				g3 += __shfl_down_sync(0xffffffff, g1, 3, 32);				
				
				g1 = data[m + ix + 8 + (itra + 10 + m+iy*col+wrap_y*col_thread) * W] - data[ix + j + 8 + (itra + 10 + i+iy*col+wrap_y*col_thread) * W];
				g1 = g1 * g1;
				g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
				g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
				
				weight[ix_GM + j * (int)gridDim.x*32 + i * sr_size * (int)gridDim.x*32+(iy*BLOCK_Y+wrap_y)*(int)gridDim.x*32*sr_size*sr_size] = g3;			////#
				g3 = g3 / (float)(nb_size * nb_size);
				g3 = exp(-g3 / (PARAMETER * PARAMETER));
				gmid += g3;
				result += g3 * data[ix +n+ j + (i +n+ itra+iy*col+wrap_y*col_thread)* W];
				g3 = 0;
			}
		}
		if(wrapLane<29 && ix<IMAGE_SIZE_X)out_image[ix+(itra+iy*col+wrap_y*col_thread)* IMAGE_SIZE_Y] = result / gmid;
		result = 0;
		gmid = 0;
	}

//*/	
}


__global__ void shuffle_without_optimization(unsigned char *data,float *out_image,float *weight,int W, int H, int sr_size, int nb_size, int IMAGE_SIZE_X, int IMAGE_SIZE_Y)
{	
	unsigned int col = IMAGE_SIZE_Y/GRID_Y;	
	unsigned int col_thread = col/BLOCK_Y;																																//
	unsigned int iy = blockIdx.y;																											
	unsigned int wrapID = (threadIdx.x +blockIdx.x*32)/32, wrapLane=(threadIdx.x +blockIdx.x*32) % 32, wrap_y = threadIdx.y;											//
	unsigned int ix = wrapID*(33-nb_size)+wrapLane, ix_GM = wrapID*32+wrapLane;																									////#
	unsigned int m=(sr_size - 1) / 2,  n=(nb_size - 1) / 2;									////

	float g1 = 0;
	float g3 = 0;
	float gmid = 0;
	float result = 0;
//out_image

	for (int i = 0; i < sr_size; i++)
	{
		for (int j = 0; j < sr_size; j++)
		{
			for(int k = 0; k < nb_size; k++)
			{
				g1 = data[m + ix + ( k + m +iy*col+wrap_y*col_thread) * W] - data[ix + j + ( k + i +iy*col+wrap_y*col_thread) * W];
				g1 = g1 * g1;
				g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
				for(int ii=2; ii<nb_size; ii++)
				{
					g3 += __shfl_down_sync(0xffffffff, g1, ii, 32);
				}

				
			} 
			weight[ix_GM + j *(int)gridDim.x*32  + i * sr_size * (int)gridDim.x*32+(iy*BLOCK_Y+wrap_y)*(int)gridDim.x*32*sr_size*sr_size] = g3;			////#
			g3 = g3 / (float)(nb_size * nb_size);
			g3 = exp(-g3 / (PARAMETER * PARAMETER));
			gmid += g3;
			result += g3 * data[ix +n+ j + (i +n+iy*col+wrap_y*col_thread)* W];
			g3 = 0;
       	}
	}
	if(wrapLane<(33-nb_size) && ix<IMAGE_SIZE_X)out_image[ix+(iy*col+wrap_y*col_thread)*IMAGE_SIZE_Y] = result / gmid;
	result = 0;
	gmid = 0;

///*
	for (int itra = 1; itra < col_thread; itra++)
	{
		for (int i = 0; i < sr_size; i++)
		{
			for (int j = 0; j < sr_size; j++)
			{
				g3 = weight[ix_GM + j * (int)gridDim.x*32 + i * sr_size * (int)gridDim.x*32 +(iy*BLOCK_Y+wrap_y)*(int)gridDim.x*32 *sr_size*sr_size];				////#
				
				g1 = data[m + ix + (itra-1 + m+iy*col+wrap_y*col_thread) * W] - data[ix + j + (itra-1 + i+iy*col+wrap_y*col_thread) * W];
				g1 = g1 * g1;
				g3 = g3-(g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
				for(int ii=2; ii<nb_size; ii++)
				{
					g3 -= __shfl_down_sync(0xffffffff, g1, ii, 32);
				}




				g1 = data[m + ix + (itra + 10 + m+iy*col+wrap_y*col_thread) * W] - data[ix + j + (itra + 10 + i+iy*col+wrap_y*col_thread) * W];
				g1 = g1 * g1;
				g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
				for(int ii=2; ii<nb_size; ii++)
				{
					g3 += __shfl_down_sync(0xffffffff, g1, ii, 32);
				}
				
				weight[ix_GM + j * (int)gridDim.x*32 + i * sr_size * (int)gridDim.x*32+(iy*BLOCK_Y+wrap_y)*(int)gridDim.x*32*sr_size*sr_size] = g3;			////#
				g3 = g3 / (float)(nb_size * nb_size);
				g3 = exp(-g3 / (PARAMETER * PARAMETER));
				gmid += g3;
				result += g3 * data[ix +n+ j + (i +n+ itra+iy*col+wrap_y*col_thread)* W];
				g3 = 0;
			}
		}
		if(wrapLane<(33-nb_size) && ix<IMAGE_SIZE_X)out_image[ix+(itra+iy*col+wrap_y*col_thread)* IMAGE_SIZE_Y] = result / gmid;
		result = 0;
		gmid = 0;
	}

//*/	
}



void NLMeansProcessor::NL_Means(unsigned char *GPU_input, float *GPU_result, int W, int H, int sr_size, int nb_size, int IMAGE_SIZE_X, int IMAGE_SIZE_Y)
{
	unsigned char *data = nullptr;
	float *out_image = nullptr;
	float* weight = nullptr;

	cudaMallocCheck((void**)&data, W * H * sizeof(unsigned char));
	cudaMallocCheck((void**)&out_image, IMAGE_SIZE_X * IMAGE_SIZE_Y * sizeof(float));
	cudaMallocCheck((void**)&weight, (IMAGE_SIZE_X+32-nb_size)/(33-nb_size)*32*(GRID_Y*BLOCK_Y)*sr_size * sr_size * sizeof(float)); 						////#	

	cudaMemcpyCheck(data, GPU_input, W * H * sizeof(unsigned char), cudaMemcpyHostToDevice);
	
	unsigned char RUNS = 1;
	
	dim3 block(BLOCK_X, BLOCK_Y);
///*////////////////////shift_without_optimization
	dim3 grid22;
	grid22.x=(IMAGE_SIZE_X+32-nb_size)/(33-nb_size);
	grid22.y=GRID_Y;	
	for (int i = 0; i < RUNS; i++)
	{
		shuffle_without_optimization<< <grid22, block>> > (data, out_image,weight, W, H, sr_size, nb_size, IMAGE_SIZE_X, IMAGE_SIZE_Y);
	}	
///////////////////*/	

///*//////////////////NLmeansOnGPU_shift_1
	dim3 grid_1;
	grid_1.x=(IMAGE_SIZE_X+30)/31;
	grid_1.y=GRID_Y;
	for (int i = 0; i < RUNS; i++)
	{
		NLmeansOnGPU_shift_1<< <grid_1, block>> > (data, out_image,weight, W, H, sr_size, nb_size, IMAGE_SIZE_X, IMAGE_SIZE_Y);
	}
///////////////////*/

///*//////////////////NLmeansOnGPU_shift_2
	dim3 grid_2;
	grid_2.x=(IMAGE_SIZE_X+29)/30;
	grid_2.y=GRID_Y;
	for (int i = 0; i < RUNS; i++)
	{
		NLmeansOnGPU_shift_2<< <grid_2, block>> > (data, out_image,weight, W, H, sr_size, nb_size, IMAGE_SIZE_X, IMAGE_SIZE_Y);
	}
///////////////////*/


///*////////////////////NLmeansOnGPU_shift_3
	dim3 grid_3;
	grid_3.x=(IMAGE_SIZE_X+28)/29;
	grid_3.y=GRID_Y;
	for (int i = 0; i < RUNS; i++)
	{
		NLmeansOnGPU_shift_3<< <grid_3, block>> > (data, out_image,weight, W, H, sr_size, nb_size, IMAGE_SIZE_X, IMAGE_SIZE_Y);
	}
///////////////////*/




	cudaMemcpyCheck(GPU_result, out_image, IMAGE_SIZE_X * IMAGE_SIZE_Y * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFreeCheck(data);
	cudaFreeCheck(out_image);
	cudaFreeCheck(weight);

	cudaDeviceSynchronize();
	cudaProfilerStop();
}



