#include<iostream>
#include<vector>
#include<cuda_runtime.h>
#include<cuda_profiler_api.h>
#include<cmath>
#include<sys/time.h>
#include"cuda_utility.h"
#include"main.h"
#include "config.h"

#define WRAP 32

#define PARAMETER 10

#define BLOCK_X 32	
#define BLOCK_Y 4
#define BLOCK_Z 1
//

#define GRID_Y 16
#define GRID_Z 4

#define PI 3.1415926




__global__ void shuffle_without_optimization(unsigned char* data, float* out_image, float* weight)
{
	unsigned int wrapID = (threadIdx.x + blockIdx.x * 32) / 32, wrapLane = (threadIdx.x + blockIdx.x * 32) % 32, wrap_y = threadIdx.y;
	unsigned int ix = wrapID * (33 - NS) + wrapLane, ix_GM = wrapID * 32 + wrapLane;
	unsigned int iy = blockIdx.y;
	unsigned int iz = blockIdx.z;
	unsigned int col_per_blocky = IMAGE_SIZE_Y / GRID_Y;
	unsigned int col_per_thready = col_per_blocky / BLOCK_Y;
	unsigned int col_per_blockz = depth / (GRID_Z);                                   				////####////
	unsigned int m = (SS - 1) / 2, n = (NS - 1) / 2;
    unsigned int W = (IMAGE_SIZE_X+SS+NS-2);
	unsigned int H = (IMAGE_SIZE_Y+SS+NS-2);


	float g1 = 0;
	float g3 = 0;
	float g5 = 0;
	float gmid = 0;
	float gmid3 = 0;
	float result = 0;
	float result3 = 0; 

	for (int itra_z = 0; itra_z < col_per_blockz; itra_z+=2) {
		//for (int itra_y = 0; itra_y < col_per_thready; itra_y++) {
		for (int i = 0; i < SS; i++) {
			for (int j = 0; j < SS; j++) {
				for (int k = 0; k < SS; k++) {



						for (int jj = 0; jj < NS; jj++) {
							g1 = data[m + ix + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
	
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							
							for (int jkl = 2; jkl < NS; jkl++)
							{
								g3 += __shfl_down_sync(0xffffffff, g1, jkl, 32);
							}
						}




					for (int ii = 1; ii < NS; ii++) {
						
						for (int jj = 0; jj < NS; jj++) {
							g1 = data[m + ix + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
	
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32); 
							
							for (int jkl = 2; jkl < NS; jkl++)
							{
								g3 += __shfl_down_sync(0xffffffff, g1, jkl, 32);
								g5 += __shfl_down_sync(0xffffffff, g1, jkl, 32);
							}
						}

							
					}


						for (int jj = 0; jj < NS; jj++) {
							g1 = data[m + ix + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
	
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							
							for (int jkl = 2; jkl < NS; jkl++)
							{
								g5 += __shfl_down_sync(0xffffffff, g1, jkl, 32);
							}
						}
						
							
					
					weight[ix_GM + k * (int)gridDim.x * 32 + j * SS * (int)gridDim.x * 32 + i * SS * SS * (int)gridDim.x * 32 + (iy * BLOCK_Y + wrap_y) * (int)gridDim.x * 32 * SS * SS * SS + iz * (int)gridDim.x * 32 * (GRID_Y * BLOCK_Y) * SS * SS * SS] = g3;			////#
					g3 = g3 / (float)(NS * NS * NS);
					g3 = exp(-g3 / (PARAMETER * PARAMETER));
					gmid += g3;
					result += g3 * data[ix + n + k + (j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + n + itra_z + iz * col_per_blockz) * W * H];
					g3 = 0;


					g5 = g5 / (float)(NS * NS * NS);
					g5 = exp(-g5 / (PARAMETER * PARAMETER));
					gmid3 += g5;
					result3 += g5 * data[ix + n + k + (j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + 1 + n + itra_z + iz * col_per_blockz) * W * H];
					g5 = 0;					
					
				}
			}
		}
		//}
		if (wrapLane < (33 - NS) && ix < IMAGE_SIZE_X)out_image[ix + (iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result / gmid;
		if (wrapLane < (33 - NS) && ix < IMAGE_SIZE_X)out_image[ix + (iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + 1 + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result3 / gmid3;
		result = 0;
		gmid = 0;
		result3 = 0;
		gmid3 = 0;		

		for (int itra_y = 1; itra_y < col_per_thready; itra_y++)
		{
			for (int i = 0; i < SS; i++) {
				for (int j = 0; j < SS; j++) {
					for (int k = 0; k < SS; k++) {


							
						g3 = weight[ix_GM + k * (int)gridDim.x * 32 + j * SS * (int)gridDim.x * 32 + i * SS * SS * (int)gridDim.x * 32 + (iy * BLOCK_Y + wrap_y) * (int)gridDim.x * 32 * SS * SS * SS + iz * (int)gridDim.x * 32 * (GRID_Y * BLOCK_Y) * SS * SS * SS];
						g5 = g3;
							
							g1 = data[m + ix + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							for (int jkl = 2; jkl < NS; jkl++)
							{
								g3 -= __shfl_down_sync(0xffffffff, g1, jkl, 32);
							}


							g1 = data[m + ix + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] -
								data[k + ix + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							for (int jkl = 2; jkl < NS; jkl++)
							{
								g3 += __shfl_down_sync(0xffffffff, g1, jkl, 32);
							}						
			
							
						for (int ii = 1; ii < NS; ii++) {

							g1 = data[m + ix + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							for (int jkl = 2; jkl < NS; jkl++)
							{
								g3 -= __shfl_down_sync(0xffffffff, g1, jkl, 32);
								g5 -= __shfl_down_sync(0xffffffff, g1, jkl, 32);
							}


							g1 = data[m + ix + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							for (int jkl = 2; jkl < NS; jkl++)
							{
								g3 += __shfl_down_sync(0xffffffff, g1, jkl, 32);
								g5 += __shfl_down_sync(0xffffffff, g1, jkl, 32);
							}
	

						}
						
							g1 = data[m + ix + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							for (int jkl = 2; jkl < NS; jkl++)
							{
								g5 -= __shfl_down_sync(0xffffffff, g1, jkl, 32);
							}


							g1 = data[m + ix + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							for (int jkl = 2; jkl < NS; jkl++)
							{
								g5 += __shfl_down_sync(0xffffffff, g1, jkl, 32);
							}						
						
						
												
						
							  
							weight[ix_GM + k * (int)gridDim.x * 32 + j * SS * (int)gridDim.x * 32 + i * SS * SS * (int)gridDim.x * 32 + (iy * BLOCK_Y + wrap_y) * (int)gridDim.x * 32 * SS * SS * SS + iz * (int)gridDim.x * 32 * (GRID_Y * BLOCK_Y) * SS * SS * SS] = g3;			////#
							
							g3 = g3 / (float)(NS * NS * NS);
							g3 = exp(-g3 / (PARAMETER * PARAMETER));
							gmid += g3;
							result += g3 * data[ix + n + k + (itra_y + j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + n + itra_z + iz * col_per_blockz) * W * H];
							g3 = 0;

							g5 = g5 / (float)(NS * NS * NS);
							g5 = exp(-g5 / (PARAMETER * PARAMETER));
							gmid3 += g5;
							result3 += g5 * data[ix + n + k + (itra_y + j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + 1 + n + itra_z + iz * col_per_blockz) * W * H];
							g5 = 0;						
						
	
					}
				}
			}
			if (wrapLane < (33 - NS) && ix < IMAGE_SIZE_X)out_image[ix + (itra_y + iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result / gmid;
			if (wrapLane < (33 - NS) && ix < IMAGE_SIZE_X)out_image[ix + (itra_y + iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + 1 + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result3 / gmid3;
			result = 0;
			gmid = 0;
			result3 = 0;
			gmid3 = 0;
		}
	


	}
}












////////////////////////-----------------------------Template window = 5-----------------------------////////////////////////
#if NS == 5

__global__ void NLmeansOnGPU_shift_1(unsigned char* data, float* out_image, float* weight)
{
	unsigned int wrapID = (threadIdx.x + blockIdx.x * 32) / 32, wrapLane = (threadIdx.x + blockIdx.x * 32) % 32, wrap_y = threadIdx.y;
	unsigned int ix = wrapID * 31 + wrapLane, ix_GM = wrapID * 32 + wrapLane;
	unsigned int iy = blockIdx.y;
	unsigned int iz = blockIdx.z;
	unsigned int col_per_blocky = IMAGE_SIZE_Y / GRID_Y;
	unsigned int col_per_thready = col_per_blocky / BLOCK_Y;
	unsigned int col_per_blockz = depth / (GRID_Z);                                   				////####////
	unsigned int m = (SS - 1) / 2, n = (NS - 1) / 2;
    unsigned int W = (IMAGE_SIZE_X+SS+NS-2);
	unsigned int H = (IMAGE_SIZE_Y+SS+NS-2);


	float g1 = 0;
	float g3 = 0;
	float g5 = 0;
	float gmid = 0;
	float gmid3 = 0;
	float result = 0;
	float result3 = 0; 

	for (int itra_z = 0; itra_z < col_per_blockz; itra_z+=2) {
		//for (int itra_y = 0; itra_y < col_per_thready; itra_y++) {
		for (int i = 0; i < SS; i++) {
			for (int j = 0; j < SS; j++) {
				for (int k = 0; k < SS; k++) {



						for (int jj = 0; jj < NS; jj++) {
							g1 = data[m + ix + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							
							g1 = data[m + ix + 2 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 2 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);							
			
							g1 = data[m + ix + 4 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1;
						}




					for (int ii = 1; ii < NS; ii++) {
						for (int jj = 0; jj < NS; jj++) {
						
							g1 = data[m + ix + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							
							g1 = data[m + ix + 2 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 2 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);							
			
							g1 = data[m + ix + 4 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1;
							g5 += g1;
																					
						}		
					}


						for (int jj = 0; jj < NS; jj++) {
						
							g1 = data[m + ix + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							
							g1 = data[m + ix + 2 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 2 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);							
			
							g1 = data[m + ix + 4 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1;
																					
						}

						
							
					
					weight[ix_GM + k * (int)gridDim.x * 32 + j * SS * (int)gridDim.x * 32 + i * SS * SS * (int)gridDim.x * 32 + (iy * BLOCK_Y + wrap_y) * (int)gridDim.x * 32 * SS * SS * SS + iz * (int)gridDim.x * 32 * (GRID_Y * BLOCK_Y) * SS * SS * SS] = g3;			////#
					g3 = g3 / (float)(NS * NS * NS);
					g3 = exp(-g3 / (PARAMETER * PARAMETER));
					gmid += g3;
					result += g3 * data[ix + n + k + (j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + n + itra_z + iz * col_per_blockz) * W * H];
					g3 = 0;


					g5 = g5 / (float)(NS * NS * NS);
					g5 = exp(-g5 / (PARAMETER * PARAMETER));
					gmid3 += g5;
					result3 += g5 * data[ix + n + k + (j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + 1 + n + itra_z + iz * col_per_blockz) * W * H];
					g5 = 0;					
					
				}
			}
		}
		//}
		if (wrapLane < 31 && ix < IMAGE_SIZE_X)out_image[ix + (iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result / gmid;
		if (wrapLane < 31 && ix < IMAGE_SIZE_X)out_image[ix + (iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + 1 + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result3 / gmid3;
		result = 0;
		gmid = 0;
		result3 = 0;
		gmid3 = 0;		

		for (int itra_y = 1; itra_y < col_per_thready; itra_y++)
		{
			for (int i = 0; i < SS; i++) {
				for (int j = 0; j < SS; j++) {
					for (int k = 0; k < SS; k++) {


							
						g3 = weight[ix_GM + k * (int)gridDim.x * 32 + j * SS * (int)gridDim.x * 32 + i * SS * SS * (int)gridDim.x * 32 + (iy * BLOCK_Y + wrap_y) * (int)gridDim.x * 32 * SS * SS * SS + iz * (int)gridDim.x * 32 * (GRID_Y * BLOCK_Y) * SS * SS * SS];
						g5 = g3;


							
							g1 = data[m + ix + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							
							g1 = data[m + ix + 2 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 2 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3-(g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));	
							
							g1 = data[m + ix + 4 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3-g1;							




							g1 = data[m + ix + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							
							g1 = data[m + ix + 2 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 2 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));							
						
							g1 = data[m + ix + 4 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + g1;
							
								
							
						for (int ii = 1; ii < NS; ii++) {

							g1 = data[m + ix + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));

							g1 = data[m + ix + 2 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 2 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));

							g1 = data[m + ix + 4 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - g1;
							g5 = g5 - g1;



							g1 = data[m + ix + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							
							g1 = data[m + ix + 2 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 2 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));		
							
							g1 = data[m + ix + 4 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + g1;
							g5 = g5 + g1;					

						}
						
							g1 = data[m + ix + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));

							g1 = data[m + ix + 2 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 2 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));

							g1 = data[m + ix + 4 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - g1;





							g1 = data[m + ix + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
						
							g1 = data[m + ix + 2 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 2 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));						
						
							g1 = data[m + ix + 4 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + g1;												
						
							  
							weight[ix_GM + k * (int)gridDim.x * 32 + j * SS * (int)gridDim.x * 32 + i * SS * SS * (int)gridDim.x * 32 + (iy * BLOCK_Y + wrap_y) * (int)gridDim.x * 32 * SS * SS * SS + iz * (int)gridDim.x * 32 * (GRID_Y * BLOCK_Y) * SS * SS * SS] = g3;			////#
							
							g3 = g3 / (float)(NS * NS * NS);
							g3 = exp(-g3 / (PARAMETER * PARAMETER));
							gmid += g3;
							result += g3 * data[ix + n + k + (itra_y + j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + n + itra_z + iz * col_per_blockz) * W * H];
							g3 = 0;

							g5 = g5 / (float)(NS * NS * NS);
							g5 = exp(-g5 / (PARAMETER * PARAMETER));
							gmid3 += g5;
							result3 += g5 * data[ix + n + k + (itra_y + j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + 1 + n + itra_z + iz * col_per_blockz) * W * H];
							g5 = 0;						
						
	
					}
				}
			}
			if (wrapLane < 31 && ix < IMAGE_SIZE_X)out_image[ix + (itra_y + iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result / gmid;
			if (wrapLane < 31 && ix < IMAGE_SIZE_X)out_image[ix + (itra_y + iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + 1 + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result3 / gmid3;
			result = 0;
			gmid = 0;
			result3 = 0;
			gmid3 = 0;
		}
	


	}
}


__global__ void NLmeansOnGPU_shift_2(unsigned char* data, float* out_image, float* weight)
{
	unsigned int wrapID = (threadIdx.x + blockIdx.x * 32) / 32, wrapLane = (threadIdx.x + blockIdx.x * 32) % 32, wrap_y = threadIdx.y;
	unsigned int ix = wrapID * 30 + wrapLane, ix_GM = wrapID * 32 + wrapLane;
	unsigned int iy = blockIdx.y;
	unsigned int iz = blockIdx.z;
	unsigned int col_per_blocky = IMAGE_SIZE_Y / GRID_Y;
	unsigned int col_per_thready = col_per_blocky / BLOCK_Y;
	unsigned int col_per_blockz = depth / (GRID_Z);                                   				////####////
	unsigned int m = (SS - 1) / 2, n = (NS - 1) / 2;
    unsigned int W = (IMAGE_SIZE_X+SS+NS-2);
	unsigned int H = (IMAGE_SIZE_Y+SS+NS-2);


	float g1 = 0;
	float g3 = 0;
	float g5 = 0;
	float gmid = 0;
	float gmid3 = 0;
	float result = 0;
	float result3 = 0; 

	for (int itra_z = 0; itra_z < col_per_blockz; itra_z+=2) {
		//for (int itra_y = 0; itra_y < col_per_thready; itra_y++) {
		for (int i = 0; i < SS; i++) {
			for (int j = 0; j < SS; j++) {
				for (int k = 0; k < SS; k++) {



						for (int jj = 0; jj < NS; jj++) {
							g1 = data[m + ix + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							g1 = data[m + ix + 3 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 3 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);										
						}




					for (int ii = 1; ii < NS; ii++) {
						for (int jj = 0; jj < NS; jj++) {
						
							g1 = data[m + ix + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							g1 = data[m + ix + 3 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 3 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);							
															
						}		
					}


						for (int jj = 0; jj < NS; jj++) {
						
							g1 = data[m + ix + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							g1 = data[m + ix + 3 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 3 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);							
			
														
						}

						
							
					
					weight[ix_GM + k * (int)gridDim.x * 32 + j * SS * (int)gridDim.x * 32 + i * SS * SS * (int)gridDim.x * 32 + (iy * BLOCK_Y + wrap_y) * (int)gridDim.x * 32 * SS * SS * SS + iz * (int)gridDim.x * 32 * (GRID_Y * BLOCK_Y) * SS * SS * SS] = g3;			////#
					g3 = g3 / (float)(NS * NS * NS);
					g3 = exp(-g3 / (PARAMETER * PARAMETER));
					gmid += g3;
					result += g3 * data[ix + n + k + (j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + n + itra_z + iz * col_per_blockz) * W * H];
					g3 = 0;


					g5 = g5 / (float)(NS * NS * NS);
					g5 = exp(-g5 / (PARAMETER * PARAMETER));
					gmid3 += g5;
					result3 += g5 * data[ix + n + k + (j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + 1 + n + itra_z + iz * col_per_blockz) * W * H];
					g5 = 0;					
					
				}
			}
		}
		//}
		if (wrapLane < 30 && ix < IMAGE_SIZE_X)out_image[ix + (iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result / gmid;
		if (wrapLane < 30 && ix < IMAGE_SIZE_X)out_image[ix + (iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + 1 + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result3 / gmid3;
		result = 0;
		gmid = 0;
		result3 = 0;
		gmid3 = 0;		

		for (int itra_y = 1; itra_y < col_per_thready; itra_y++)
		{
			for (int i = 0; i < SS; i++) {
				for (int j = 0; j < SS; j++) {
					for (int k = 0; k < SS; k++) {


							
						g3 = weight[ix_GM + k * (int)gridDim.x * 32 + j * SS * (int)gridDim.x * 32 + i * SS * SS * (int)gridDim.x * 32 + (iy * BLOCK_Y + wrap_y) * (int)gridDim.x * 32 * SS * SS * SS + iz * (int)gridDim.x * 32 * (GRID_Y * BLOCK_Y) * SS * SS * SS];
						g5 = g3;


							
							g1 = data[m + ix + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							
							g1 = data[m + ix + 3 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 3 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3-(g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));	
							
						




							g1 = data[m + ix + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							
							g1 = data[m + ix + 3 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 3 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));							
						

			
			
							
								
							
						for (int ii = 1; ii < NS; ii++) {

							g1 = data[m + ix + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);

							g1 = data[m + ix + 3 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 3 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));




							g1 = data[m + ix + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							
							g1 = data[m + ix + 3 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 3 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));		
							
						}
		
						
							g1 = data[m + ix + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							

							g1 = data[m + ix + 3 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 3 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							


							g1 = data[m + ix + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							
						
							g1 = data[m + ix + 3 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 3 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));		
											

							  
							weight[ix_GM + k * (int)gridDim.x * 32 + j * SS * (int)gridDim.x * 32 + i * SS * SS * (int)gridDim.x * 32 + (iy * BLOCK_Y + wrap_y) * (int)gridDim.x * 32 * SS * SS * SS + iz * (int)gridDim.x * 32 * (GRID_Y * BLOCK_Y) * SS * SS * SS] = g3;			////#
							
							g3 = g3 / (float)(NS * NS * NS);
							g3 = exp(-g3 / (PARAMETER * PARAMETER));
							gmid += g3;
							result += g3 * data[ix + n + k + (itra_y + j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + n + itra_z + iz * col_per_blockz) * W * H];
							g3 = 0;

							g5 = g5 / (float)(NS * NS * NS);
							g5 = exp(-g5 / (PARAMETER * PARAMETER));
							gmid3 += g5;
							result3 += g5 * data[ix + n + k + (itra_y + j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + 1 + n + itra_z + iz * col_per_blockz) * W * H];
							g5 = 0;						
						
	
					}
				}
			}
			if (wrapLane < 30 && ix < IMAGE_SIZE_X)out_image[ix + (itra_y + iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result / gmid;
			if (wrapLane < 30 && ix < IMAGE_SIZE_X)out_image[ix + (itra_y + iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + 1 + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result3 / gmid3;
			result = 0;
			gmid = 0;
			result3 = 0;
			gmid3 = 0;
		}
	


	}
}

__global__ void NLmeansOnGPU_shift_3(unsigned char* data, float* out_image, float* weight)
{
	unsigned int wrapID = (threadIdx.x + blockIdx.x * 32) / 32, wrapLane = (threadIdx.x + blockIdx.x * 32) % 32, wrap_y = threadIdx.y;
	unsigned int ix = wrapID * 29 + wrapLane, ix_GM = wrapID * 32 + wrapLane;
	unsigned int iy = blockIdx.y;
	unsigned int iz = blockIdx.z;
	unsigned int col_per_blocky = IMAGE_SIZE_Y / GRID_Y;
	unsigned int col_per_thready = col_per_blocky / BLOCK_Y;
	unsigned int col_per_blockz = depth / (GRID_Z);                                   				////####////
	unsigned int m = (SS - 1) / 2, n = (NS - 1) / 2;
    unsigned int W = (IMAGE_SIZE_X+SS+NS-2);
	unsigned int H = (IMAGE_SIZE_Y+SS+NS-2);


	float g1 = 0;
	float g3 = 0;
	float g5 = 0;
	float gmid = 0;
	float gmid3 = 0;
	float result = 0;
	float result3 = 0; 

	for (int itra_z = 0; itra_z < col_per_blockz; itra_z+=2) {
		//for (int itra_y = 0; itra_y < col_per_thready; itra_y++) {
		for (int i = 0; i < SS; i++) {
			for (int j = 0; j < SS; j++) {
				for (int k = 0; k < SS; k++) {



						for (int jj = 0; jj < NS; jj++) {
							g1 = data[m + ix + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 3, 32);
							
							g1 = data[m + ix + 4 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1;										
						}




					for (int ii = 1; ii < NS; ii++) {
						for (int jj = 0; jj < NS; jj++) {
						
							g1 = data[m + ix + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 3, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 3, 32);
							
							g1 = data[m + ix + 4 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1;
							g5 += g1;							
															
						}		
					}


						for (int jj = 0; jj < NS; jj++) {
						
							g1 = data[m + ix + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 3, 32);
							
							g1 = data[m + ix + 4 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1;							
											
						}

						
							
					
					weight[ix_GM + k * (int)gridDim.x * 32 + j * SS * (int)gridDim.x * 32 + i * SS * SS * (int)gridDim.x * 32 + (iy * BLOCK_Y + wrap_y) * (int)gridDim.x * 32 * SS * SS * SS + iz * (int)gridDim.x * 32 * (GRID_Y * BLOCK_Y) * SS * SS * SS] = g3;			////#
					g3 = g3 / (float)(NS * NS * NS);
					g3 = exp(-g3 / (PARAMETER * PARAMETER));
					gmid += g3;
					result += g3 * data[ix + n + k + (j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + n + itra_z + iz * col_per_blockz) * W * H];
					g3 = 0;


					g5 = g5 / (float)(NS * NS * NS);
					g5 = exp(-g5 / (PARAMETER * PARAMETER));
					gmid3 += g5;
					result3 += g5 * data[ix + n + k + (j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + 1 + n + itra_z + iz * col_per_blockz) * W * H];
					g5 = 0;					
					
				}
			}
		}
		//}
		if (wrapLane < 29 && ix < IMAGE_SIZE_X)out_image[ix + (iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result / gmid;
		if (wrapLane < 29 && ix < IMAGE_SIZE_X)out_image[ix + (iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + 1 + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result3 / gmid3;
		result = 0;
		gmid = 0;
		result3 = 0;
		gmid3 = 0;		

		for (int itra_y = 1; itra_y < col_per_thready; itra_y++)
		{
			for (int i = 0; i < SS; i++) {
				for (int j = 0; j < SS; j++) {
					for (int k = 0; k < SS; k++) {


							
						g3 = weight[ix_GM + k * (int)gridDim.x * 32 + j * SS * (int)gridDim.x * 32 + i * SS * SS * (int)gridDim.x * 32 + (iy * BLOCK_Y + wrap_y) * (int)gridDim.x * 32 * SS * SS * SS + iz * (int)gridDim.x * 32 * (GRID_Y * BLOCK_Y) * SS * SS * SS];
						g5 = g3;


							
							g1 = data[m + ix + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 3, 32);
							
							
							g1 = data[m + ix + 4 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - g1;	
							
						




							g1 = data[m + ix + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 3, 32);
							
							
							g1 = data[m + ix + 4 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + g1;							
						

			
			
							
								
							
						for (int ii = 1; ii < NS; ii++) {

							g1 = data[m + ix + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 3, 32);
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 3, 32);

							g1 = data[m + ix + 4 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - g1;
							g5 = g5 - g1;




							g1 = data[m + ix + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 3, 32);
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 3, 32);
							
							
							g1 = data[m + ix + 4 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + g1;
							g5 = g5 + g1;		
							
						}
		
						
							g1 = data[m + ix + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 3, 32);
							

							g1 = data[m + ix + 4 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - g1;
							


							g1 = data[m + ix + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 3, 32);
							
						
							g1 = data[m + ix + 4 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + g1;		
											

							  
							weight[ix_GM + k * (int)gridDim.x * 32 + j * SS * (int)gridDim.x * 32 + i * SS * SS * (int)gridDim.x * 32 + (iy * BLOCK_Y + wrap_y) * (int)gridDim.x * 32 * SS * SS * SS + iz * (int)gridDim.x * 32 * (GRID_Y * BLOCK_Y) * SS * SS * SS] = g3;			////#
							
							g3 = g3 / (float)(NS * NS * NS);
							g3 = exp(-g3 / (PARAMETER * PARAMETER));
							gmid += g3;
							result += g3 * data[ix + n + k + (itra_y + j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + n + itra_z + iz * col_per_blockz) * W * H];
							g3 = 0;

							g5 = g5 / (float)(NS * NS * NS);
							g5 = exp(-g5 / (PARAMETER * PARAMETER));
							gmid3 += g5;
							result3 += g5 * data[ix + n + k + (itra_y + j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + 1 + n + itra_z + iz * col_per_blockz) * W * H];
							g5 = 0;						
						
	
					}
				}
			}
			if (wrapLane < 29 && ix < IMAGE_SIZE_X)out_image[ix + (itra_y + iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result / gmid;
			if (wrapLane < 29 && ix < IMAGE_SIZE_X)out_image[ix + (itra_y + iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + 1 + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result3 / gmid3;
			result = 0;
			gmid = 0;
			result3 = 0;
			gmid3 = 0;
		}
	


	}
}














////////////////////////-----------------------------Template window = 7-----------------------------////////////////////////
#elif NS == 7
__global__ void NLmeansOnGPU_shift_1(unsigned char* data, float* out_image, float* weight)
{
	unsigned int wrapID = (threadIdx.x + blockIdx.x * 32) / 32, wrapLane = (threadIdx.x + blockIdx.x * 32) % 32, wrap_y = threadIdx.y;
	unsigned int ix = wrapID * 31 + wrapLane, ix_GM = wrapID * 32 + wrapLane;
	unsigned int iy = blockIdx.y;
	unsigned int iz = blockIdx.z;
	unsigned int col_per_blocky = IMAGE_SIZE_Y / GRID_Y;
	unsigned int col_per_thready = col_per_blocky / BLOCK_Y;
	unsigned int col_per_blockz = depth / (GRID_Z);                                   				////####////
	unsigned int m = (SS - 1) / 2, n = (NS - 1) / 2;
    unsigned int W = (IMAGE_SIZE_X+SS+NS-2);
	unsigned int H = (IMAGE_SIZE_Y+SS+NS-2);


	float g1 = 0;
	float g3 = 0;
	float g5 = 0;
	float gmid = 0;
	float gmid3 = 0;
	float result = 0;
	float result3 = 0; 

	for (int itra_z = 0; itra_z < col_per_blockz; itra_z+=2) {
		//for (int itra_y = 0; itra_y < col_per_thready; itra_y++) {
		for (int i = 0; i < SS; i++) {
			for (int j = 0; j < SS; j++) {
				for (int k = 0; k < SS; k++) {



						for (int jj = 0; jj < NS; jj++) {
							g1 = data[m + ix + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							
							g1 = data[m + ix + 2 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 2 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);							
			
							g1 = data[m + ix + 4 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							
							g1 = data[m + ix + 6 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1;
						}




					for (int ii = 1; ii < NS; ii++) {
						for (int jj = 0; jj < NS; jj++) {
						
							g1 = data[m + ix + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							
							
							g1 = data[m + ix + 2 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 2 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);							
			
							g1 = data[m + ix + 4 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);	
							
							g1 = data[m + ix + 6 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1;
							g5 += g1;
																					
						}		
					}


						for (int jj = 0; jj < NS; jj++) {
						
							g1 = data[m + ix + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							
							g1 = data[m + ix + 2 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 2 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);							
			
							g1 = data[m + ix + 4 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);

							g1 = data[m + ix + 6 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1;
																					
						}

						
							
					
					weight[ix_GM + k * (int)gridDim.x * 32 + j * SS * (int)gridDim.x * 32 + i * SS * SS * (int)gridDim.x * 32 + (iy * BLOCK_Y + wrap_y) * (int)gridDim.x * 32 * SS * SS * SS + iz * (int)gridDim.x * 32 * (GRID_Y * BLOCK_Y) * SS * SS * SS] = g3;			////#
					g3 = g3 / (float)(NS * NS * NS);
					g3 = exp(-g3 / (PARAMETER * PARAMETER));
					gmid += g3;
					result += g3 * data[ix + n + k + (j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + n + itra_z + iz * col_per_blockz) * W * H];
					g3 = 0;


					g5 = g5 / (float)(NS * NS * NS);
					g5 = exp(-g5 / (PARAMETER * PARAMETER));
					gmid3 += g5;
					result3 += g5 * data[ix + n + k + (j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + 1 + n + itra_z + iz * col_per_blockz) * W * H];
					g5 = 0;					
					
				}
			}
		}
		//}
		if (wrapLane < 31 && ix < IMAGE_SIZE_X)out_image[ix + (iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result / gmid;
		if (wrapLane < 31 && ix < IMAGE_SIZE_X)out_image[ix + (iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + 1 + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result3 / gmid3;
		result = 0;
		gmid = 0;
		result3 = 0;
		gmid3 = 0;		

		for (int itra_y = 1; itra_y < col_per_thready; itra_y++)
		{
			for (int i = 0; i < SS; i++) {
				for (int j = 0; j < SS; j++) {
					for (int k = 0; k < SS; k++) {


							
						g3 = weight[ix_GM + k * (int)gridDim.x * 32 + j * SS * (int)gridDim.x * 32 + i * SS * SS * (int)gridDim.x * 32 + (iy * BLOCK_Y + wrap_y) * (int)gridDim.x * 32 * SS * SS * SS + iz * (int)gridDim.x * 32 * (GRID_Y * BLOCK_Y) * SS * SS * SS];
						g5 = g3;


							
							g1 = data[m + ix + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							
							g1 = data[m + ix + 2 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 2 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3-(g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));	
							
							g1 = data[m + ix + 4 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3-(g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));	
							
							g1 = data[m + ix + 6 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3-g1;														




							g1 = data[m + ix + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							
							g1 = data[m + ix + 2 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 2 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));							
						
							g1 = data[m + ix + 4 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));

							g1 = data[m + ix + 6 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + g1;							
								
							
						for (int ii = 1; ii < NS; ii++) {

							g1 = data[m + ix + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));

							g1 = data[m + ix + 2 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 2 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));

							g1 = data[m + ix + 4 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));

							g1 = data[m + ix + 6 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - g1;
							g5 = g5 - g1;



							g1 = data[m + ix + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							
							g1 = data[m + ix + 2 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 2 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));		
							
							g1 = data[m + ix + 4 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
												
							g1 = data[m + ix + 6 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + g1;
							g5 = g5 + g1;
						}
						
							g1 = data[m + ix + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));

							g1 = data[m + ix + 2 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 2 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));

							g1 = data[m + ix + 4 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));

							g1 = data[m + ix + 6 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - g1;



							g1 = data[m + ix + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
						
							g1 = data[m + ix + 2 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 2 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));						
						
							g1 = data[m + ix + 4 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));												

							g1 = data[m + ix + 6 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + g1;							
							  
							weight[ix_GM + k * (int)gridDim.x * 32 + j * SS * (int)gridDim.x * 32 + i * SS * SS * (int)gridDim.x * 32 + (iy * BLOCK_Y + wrap_y) * (int)gridDim.x * 32 * SS * SS * SS + iz * (int)gridDim.x * 32 * (GRID_Y * BLOCK_Y) * SS * SS * SS] = g3;			////#
							
							g3 = g3 / (float)(NS * NS * NS);
							g3 = exp(-g3 / (PARAMETER * PARAMETER));
							gmid += g3;
							result += g3 * data[ix + n + k + (itra_y + j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + n + itra_z + iz * col_per_blockz) * W * H];
							g3 = 0;

							g5 = g5 / (float)(NS * NS * NS);
							g5 = exp(-g5 / (PARAMETER * PARAMETER));
							gmid3 += g5;
							result3 += g5 * data[ix + n + k + (itra_y + j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + 1 + n + itra_z + iz * col_per_blockz) * W * H];
							g5 = 0;						
						
	
					}
				}
			}
			if (wrapLane < 31 && ix < IMAGE_SIZE_X)out_image[ix + (itra_y + iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result / gmid;
			if (wrapLane < 31 && ix < IMAGE_SIZE_X)out_image[ix + (itra_y + iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + 1 + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result3 / gmid3;
			result = 0;
			gmid = 0;
			result3 = 0;
			gmid3 = 0;
		}
	


	}
}


__global__ void NLmeansOnGPU_shift_2(unsigned char* data, float* out_image, float* weight)
{
	unsigned int wrapID = (threadIdx.x + blockIdx.x * 32) / 32, wrapLane = (threadIdx.x + blockIdx.x * 32) % 32, wrap_y = threadIdx.y;
	unsigned int ix = wrapID * 30 + wrapLane, ix_GM = wrapID * 32 + wrapLane;
	unsigned int iy = blockIdx.y;
	unsigned int iz = blockIdx.z;
	unsigned int col_per_blocky = IMAGE_SIZE_Y / GRID_Y;
	unsigned int col_per_thready = col_per_blocky / BLOCK_Y;
	unsigned int col_per_blockz = depth / (GRID_Z);                                   				////####////
	unsigned int m = (SS - 1) / 2, n = (NS - 1) / 2;
    unsigned int W = (IMAGE_SIZE_X+SS+NS-2);
	unsigned int H = (IMAGE_SIZE_Y+SS+NS-2);


	float g1 = 0;
	float g3 = 0;
	float g5 = 0;
	float gmid = 0;
	float gmid3 = 0;
	float result = 0;
	float result3 = 0; 

	for (int itra_z = 0; itra_z < col_per_blockz; itra_z+=2) {
		//for (int itra_y = 0; itra_y < col_per_thready; itra_y++) {
		for (int i = 0; i < SS; i++) {
			for (int j = 0; j < SS; j++) {
				for (int k = 0; k < SS; k++) {



						for (int jj = 0; jj < NS; jj++) {
							g1 = data[m + ix + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							g1 = data[m + ix + 3 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 3 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);			
							
							g1 = data[m + ix + 6 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1;
						
						}




					for (int ii = 1; ii < NS; ii++) {
						for (int jj = 0; jj < NS; jj++) {
						
							g1 = data[m + ix + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							g1 = data[m + ix + 3 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 3 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);	
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);	
							
							g1 = data[m + ix + 6 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1;
							g5 += g1;						
															
						}		
					}


						for (int jj = 0; jj < NS; jj++) {
						
							g1 = data[m + ix + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;							
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							g1 = data[m + ix + 3 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 3 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);	
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);						

							g1 = data[m + ix + 6 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1;				
														
						}

						
							
					
					weight[ix_GM + k * (int)gridDim.x * 32 + j * SS * (int)gridDim.x * 32 + i * SS * SS * (int)gridDim.x * 32 + (iy * BLOCK_Y + wrap_y) * (int)gridDim.x * 32 * SS * SS * SS + iz * (int)gridDim.x * 32 * (GRID_Y * BLOCK_Y) * SS * SS * SS] = g3;			////#
					g3 = g3 / (float)(NS * NS * NS);
					g3 = exp(-g3 / (PARAMETER * PARAMETER));
					gmid += g3;
					result += g3 * data[ix + n + k + (j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + n + itra_z + iz * col_per_blockz) * W * H];
					g3 = 0;


					g5 = g5 / (float)(NS * NS * NS);
					g5 = exp(-g5 / (PARAMETER * PARAMETER));
					gmid3 += g5;
					result3 += g5 * data[ix + n + k + (j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + 1 + n + itra_z + iz * col_per_blockz) * W * H];
					g5 = 0;					
					
				}
			}
		}
		//}
		if (wrapLane < 30 && ix < IMAGE_SIZE_X)out_image[ix + (iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result / gmid;
		if (wrapLane < 30 && ix < IMAGE_SIZE_X)out_image[ix + (iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + 1 + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result3 / gmid3;
		result = 0;
		gmid = 0;
		result3 = 0;
		gmid3 = 0;		

		for (int itra_y = 1; itra_y < col_per_thready; itra_y++)
		{
			for (int i = 0; i < SS; i++) {
				for (int j = 0; j < SS; j++) {
					for (int k = 0; k < SS; k++) {


							
						g3 = weight[ix_GM + k * (int)gridDim.x * 32 + j * SS * (int)gridDim.x * 32 + i * SS * SS * (int)gridDim.x * 32 + (iy * BLOCK_Y + wrap_y) * (int)gridDim.x * 32 * SS * SS * SS + iz * (int)gridDim.x * 32 * (GRID_Y * BLOCK_Y) * SS * SS * SS];
						g5 = g3;


							
							g1 = data[m + ix + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							
							g1 = data[m + ix + 3 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 3 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3-(g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));	
							g3 = g3-__shfl_down_sync(0xffffffff, g1, 2, 32);
							
							g1 = data[m + ix + 6 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3-g1;	
						




							g1 = data[m + ix + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							
							g1 = data[m + ix + 3 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 3 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));		
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);					

							g1 = data[m + ix + 6 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + g1;							

			
			
							
								
							
						for (int ii = 1; ii < NS; ii++) {

							g1 = data[m + ix + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);

							g1 = data[m + ix + 3 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 3 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							
							g1 = data[m + ix + 6 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - g1;
							g5 = g5 - g1;



							g1 = data[m + ix + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							
							g1 = data[m + ix + 3 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 3 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);	

							g1 = data[m + ix + 6 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + g1;
							g5 = g5 + g1;
		
							
						}
		
						
							g1 = data[m + ix + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							

							g1 = data[m + ix + 3 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 3 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							g1 = data[m + ix + 6 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - g1;							
							


							g1 = data[m + ix + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							
						
							g1 = data[m + ix + 3 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 3 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);		
							
							g1 = data[m + ix + 6 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + g1;
											

							  
							weight[ix_GM + k * (int)gridDim.x * 32 + j * SS * (int)gridDim.x * 32 + i * SS * SS * (int)gridDim.x * 32 + (iy * BLOCK_Y + wrap_y) * (int)gridDim.x * 32 * SS * SS * SS + iz * (int)gridDim.x * 32 * (GRID_Y * BLOCK_Y) * SS * SS * SS] = g3;			////#
							
							g3 = g3 / (float)(NS * NS * NS);
							g3 = exp(-g3 / (PARAMETER * PARAMETER));
							gmid += g3;
							result += g3 * data[ix + n + k + (itra_y + j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + n + itra_z + iz * col_per_blockz) * W * H];
							g3 = 0;

							g5 = g5 / (float)(NS * NS * NS);
							g5 = exp(-g5 / (PARAMETER * PARAMETER));
							gmid3 += g5;
							result3 += g5 * data[ix + n + k + (itra_y + j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + 1 + n + itra_z + iz * col_per_blockz) * W * H];
							g5 = 0;						
						
	
					}
				}
			}
			if (wrapLane < 30 && ix < IMAGE_SIZE_X)out_image[ix + (itra_y + iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result / gmid;
			if (wrapLane < 30 && ix < IMAGE_SIZE_X)out_image[ix + (itra_y + iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + 1 + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result3 / gmid3;
			result = 0;
			gmid = 0;
			result3 = 0;
			gmid3 = 0;
		}
	


	}
}

__global__ void NLmeansOnGPU_shift_3(unsigned char* data, float* out_image, float* weight)
{
	unsigned int wrapID = (threadIdx.x + blockIdx.x * 32) / 32, wrapLane = (threadIdx.x + blockIdx.x * 32) % 32, wrap_y = threadIdx.y;
	unsigned int ix = wrapID * 29 + wrapLane, ix_GM = wrapID * 32 + wrapLane;
	unsigned int iy = blockIdx.y;
	unsigned int iz = blockIdx.z;
	unsigned int col_per_blocky = IMAGE_SIZE_Y / GRID_Y;
	unsigned int col_per_thready = col_per_blocky / BLOCK_Y;
	unsigned int col_per_blockz = depth / (GRID_Z);                                   				////####////
	unsigned int m = (SS - 1) / 2, n = (NS - 1) / 2;
    unsigned int W = (IMAGE_SIZE_X+SS+NS-2);
	unsigned int H = (IMAGE_SIZE_Y+SS+NS-2);


	float g1 = 0;
	float g3 = 0;
	float g5 = 0;
	float gmid = 0;
	float gmid3 = 0;
	float result = 0;
	float result3 = 0; 

	for (int itra_z = 0; itra_z < col_per_blockz; itra_z+=2) {
		//for (int itra_y = 0; itra_y < col_per_thready; itra_y++) {
		for (int i = 0; i < SS; i++) {
			for (int j = 0; j < SS; j++) {
				for (int k = 0; k < SS; k++) {



						for (int jj = 0; jj < NS; jj++) {
							g1 = data[m + ix + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 3, 32);
							
							g1 = data[m + ix + 4 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);										
						}




					for (int ii = 1; ii < NS; ii++) {
						for (int jj = 0; jj < NS; jj++) {
						
							g1 = data[m + ix + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 3, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 3, 32);
							
							g1 = data[m + ix + 4 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);							
															
						}		
					}


						for (int jj = 0; jj < NS; jj++) {
						
							g1 = data[m + ix + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 3, 32);
							
							g1 = data[m + ix + 4 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);							
			
														
						}

						
							
					
					weight[ix_GM + k * (int)gridDim.x * 32 + j * SS * (int)gridDim.x * 32 + i * SS * SS * (int)gridDim.x * 32 + (iy * BLOCK_Y + wrap_y) * (int)gridDim.x * 32 * SS * SS * SS + iz * (int)gridDim.x * 32 * (GRID_Y * BLOCK_Y) * SS * SS * SS] = g3;			////#
					g3 = g3 / (float)(NS * NS * NS);
					g3 = exp(-g3 / (PARAMETER * PARAMETER));
					gmid += g3;
					result += g3 * data[ix + n + k + (j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + n + itra_z + iz * col_per_blockz) * W * H];
					g3 = 0;


					g5 = g5 / (float)(NS * NS * NS);
					g5 = exp(-g5 / (PARAMETER * PARAMETER));
					gmid3 += g5;
					result3 += g5 * data[ix + n + k + (j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + 1 + n + itra_z + iz * col_per_blockz) * W * H];
					g5 = 0;					
					
				}
			}
		}
		//}
		if (wrapLane < 29 && ix < IMAGE_SIZE_X)out_image[ix + (iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result / gmid;
		if (wrapLane < 29 && ix < IMAGE_SIZE_X)out_image[ix + (iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + 1 + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result3 / gmid3;
		result = 0;
		gmid = 0;
		result3 = 0;
		gmid3 = 0;		

		for (int itra_y = 1; itra_y < col_per_thready; itra_y++)
		{
			for (int i = 0; i < SS; i++) {
				for (int j = 0; j < SS; j++) {
					for (int k = 0; k < SS; k++) {


							
						g3 = weight[ix_GM + k * (int)gridDim.x * 32 + j * SS * (int)gridDim.x * 32 + i * SS * SS * (int)gridDim.x * 32 + (iy * BLOCK_Y + wrap_y) * (int)gridDim.x * 32 * SS * SS * SS + iz * (int)gridDim.x * 32 * (GRID_Y * BLOCK_Y) * SS * SS * SS];
						g5 = g3;


							
							g1 = data[m + ix + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 3, 32);
							
							
							g1 = data[m + ix + 4 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 2, 32);	
							
						




							g1 = data[m + ix + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 3, 32);
							
							
							g1 = data[m + ix + 4 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);							
						

			
			
							
								
							
						for (int ii = 1; ii < NS; ii++) {

							g1 = data[m + ix + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 3, 32);
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 3, 32);

							g1 = data[m + ix + 4 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);




							g1 = data[m + ix + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 3, 32);
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 3, 32);
							
							
							g1 = data[m + ix + 4 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);		
							
						}
		
						
							g1 = data[m + ix + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 3, 32);
							

							g1 = data[m + ix + 4 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							


							g1 = data[m + ix + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 3, 32);
							
						
							g1 = data[m + ix + 4 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);		
											

							  
							weight[ix_GM + k * (int)gridDim.x * 32 + j * SS * (int)gridDim.x * 32 + i * SS * SS * (int)gridDim.x * 32 + (iy * BLOCK_Y + wrap_y) * (int)gridDim.x * 32 * SS * SS * SS + iz * (int)gridDim.x * 32 * (GRID_Y * BLOCK_Y) * SS * SS * SS] = g3;			////#
							
							g3 = g3 / (float)(NS * NS * NS);
							g3 = exp(-g3 / (PARAMETER * PARAMETER));
							gmid += g3;
							result += g3 * data[ix + n + k + (itra_y + j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + n + itra_z + iz * col_per_blockz) * W * H];
							g3 = 0;

							g5 = g5 / (float)(NS * NS * NS);
							g5 = exp(-g5 / (PARAMETER * PARAMETER));
							gmid3 += g5;
							result3 += g5 * data[ix + n + k + (itra_y + j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + 1 + n + itra_z + iz * col_per_blockz) * W * H];
							g5 = 0;						
						
	
					}
				}
			}
			if (wrapLane < 29 && ix < IMAGE_SIZE_X)out_image[ix + (itra_y + iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result / gmid;
			if (wrapLane < 29 && ix < IMAGE_SIZE_X)out_image[ix + (itra_y + iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + 1 + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result3 / gmid3;
			result = 0;
			gmid = 0;
			result3 = 0;
			gmid3 = 0;
		}
	


	}
}















////////////////////////-----------------------------Template window = 9-----------------------------////////////////////////
#elif NS == 9
__global__ void NLmeansOnGPU_shift_1(unsigned char* data, float* out_image, float* weight)
{
	unsigned int wrapID = (threadIdx.x + blockIdx.x * 32) / 32, wrapLane = (threadIdx.x + blockIdx.x * 32) % 32, wrap_y = threadIdx.y;
	unsigned int ix = wrapID * 31 + wrapLane, ix_GM = wrapID * 32 + wrapLane;
	unsigned int iy = blockIdx.y;
	unsigned int iz = blockIdx.z;
	unsigned int col_per_blocky = IMAGE_SIZE_Y / GRID_Y;
	unsigned int col_per_thready = col_per_blocky / BLOCK_Y;
	unsigned int col_per_blockz = depth / (GRID_Z);                                   				////####////
	unsigned int m = (SS - 1) / 2, n = (NS - 1) / 2;
    unsigned int W = (IMAGE_SIZE_X+SS+NS-2);
	unsigned int H = (IMAGE_SIZE_Y+SS+NS-2);


	float g1 = 0;
	float g3 = 0;
	float g5 = 0;
	float gmid = 0;
	float gmid3 = 0;
	float result = 0;
	float result3 = 0; 

	for (int itra_z = 0; itra_z < col_per_blockz; itra_z+=2) {
		//for (int itra_y = 0; itra_y < col_per_thready; itra_y++) {
		for (int i = 0; i < SS; i++) {
			for (int j = 0; j < SS; j++) {
				for (int k = 0; k < SS; k++) {



						for (int jj = 0; jj < NS; jj++) {
							g1 = data[m + ix + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							
							g1 = data[m + ix + 2 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 2 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);							
			
							g1 = data[m + ix + 4 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							
							g1 = data[m + ix + 6 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
						
							g1 = data[m + ix + 8 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1;
						}




					for (int ii = 1; ii < NS; ii++) {
						for (int jj = 0; jj < NS; jj++) {
						
							g1 = data[m + ix + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							
							
							g1 = data[m + ix + 2 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 2 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);							
			
							g1 = data[m + ix + 4 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);	
							
							g1 = data[m + ix + 6 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);	
							
							g1 = data[m + ix + 8 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1;
							g5 += g1;
																					
						}		
					}


						for (int jj = 0; jj < NS; jj++) {
						
							g1 = data[m + ix + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							
							g1 = data[m + ix + 2 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 2 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);							
			
							g1 = data[m + ix + 4 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);

							g1 = data[m + ix + 6 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
					
							g1 = data[m + ix + 8 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1;
																					
						}

						
							
					
					weight[ix_GM + k * (int)gridDim.x * 32 + j * SS * (int)gridDim.x * 32 + i * SS * SS * (int)gridDim.x * 32 + (iy * BLOCK_Y + wrap_y) * (int)gridDim.x * 32 * SS * SS * SS + iz * (int)gridDim.x * 32 * (GRID_Y * BLOCK_Y) * SS * SS * SS] = g3;			////#
					g3 = g3 / (float)(NS * NS * NS);
					g3 = exp(-g3 / (PARAMETER * PARAMETER));
					gmid += g3;
					result += g3 * data[ix + n + k + (j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + n + itra_z + iz * col_per_blockz) * W * H];
					g3 = 0;


					g5 = g5 / (float)(NS * NS * NS);
					g5 = exp(-g5 / (PARAMETER * PARAMETER));
					gmid3 += g5;
					result3 += g5 * data[ix + n + k + (j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + 1 + n + itra_z + iz * col_per_blockz) * W * H];
					g5 = 0;					
					
				}
			}
		}
		//}
		if (wrapLane < 31 && ix < IMAGE_SIZE_X)out_image[ix + (iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result / gmid;
		if (wrapLane < 31 && ix < IMAGE_SIZE_X)out_image[ix + (iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + 1 + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result3 / gmid3;
		result = 0;
		gmid = 0;
		result3 = 0;
		gmid3 = 0;		

		for (int itra_y = 1; itra_y < col_per_thready; itra_y++)
		{
			for (int i = 0; i < SS; i++) {
				for (int j = 0; j < SS; j++) {
					for (int k = 0; k < SS; k++) {


							
						g3 = weight[ix_GM + k * (int)gridDim.x * 32 + j * SS * (int)gridDim.x * 32 + i * SS * SS * (int)gridDim.x * 32 + (iy * BLOCK_Y + wrap_y) * (int)gridDim.x * 32 * SS * SS * SS + iz * (int)gridDim.x * 32 * (GRID_Y * BLOCK_Y) * SS * SS * SS];
						g5 = g3;


							
							g1 = data[m + ix + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							
							g1 = data[m + ix + 2 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 2 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3-(g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));	
							
							g1 = data[m + ix + 4 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3-(g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));	
							
							g1 = data[m + ix + 6 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3-(g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));		
					
							g1 = data[m + ix + 8 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3-g1;													




							g1 = data[m + ix + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							
							g1 = data[m + ix + 2 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 2 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));							
						
							g1 = data[m + ix + 4 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));

							g1 = data[m + ix + 6 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));	
							
							g1 = data[m + ix + 8 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + g1;							
								
							
						for (int ii = 1; ii < NS; ii++) {

							g1 = data[m + ix + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));

							g1 = data[m + ix + 2 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 2 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));

							g1 = data[m + ix + 4 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));

							g1 = data[m + ix + 6 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							
							g1 = data[m + ix + 8 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - g1;
							g5 = g5 - g1;							



							g1 = data[m + ix + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							
							g1 = data[m + ix + 2 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 2 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));		
							
							g1 = data[m + ix + 4 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
												
							g1 = data[m + ix + 6 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							
							g1 = data[m + ix + 8 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + g1;
							g5 = g5 + g1;
						}
						
							g1 = data[m + ix + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));

							g1 = data[m + ix + 2 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 2 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));

							g1 = data[m + ix + 4 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));

							g1 = data[m + ix + 6 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
						
							g1 = data[m + ix + 8 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - g1;



							g1 = data[m + ix + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
						
							g1 = data[m + ix + 2 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 2 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));						
						
							g1 = data[m + ix + 4 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));												

							g1 = data[m + ix + 6 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));							
							  
							g1 = data[m + ix + 8 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + g1;							
							
							weight[ix_GM + k * (int)gridDim.x * 32 + j * SS * (int)gridDim.x * 32 + i * SS * SS * (int)gridDim.x * 32 + (iy * BLOCK_Y + wrap_y) * (int)gridDim.x * 32 * SS * SS * SS + iz * (int)gridDim.x * 32 * (GRID_Y * BLOCK_Y) * SS * SS * SS] = g3;			////#
							
							g3 = g3 / (float)(NS * NS * NS);
							g3 = exp(-g3 / (PARAMETER * PARAMETER));
							gmid += g3;
							result += g3 * data[ix + n + k + (itra_y + j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + n + itra_z + iz * col_per_blockz) * W * H];
							g3 = 0;

							g5 = g5 / (float)(NS * NS * NS);
							g5 = exp(-g5 / (PARAMETER * PARAMETER));
							gmid3 += g5;
							result3 += g5 * data[ix + n + k + (itra_y + j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + 1 + n + itra_z + iz * col_per_blockz) * W * H];
							g5 = 0;						
						
	
					}
				}
			}
			if (wrapLane < 31 && ix < IMAGE_SIZE_X)out_image[ix + (itra_y + iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result / gmid;
			if (wrapLane < 31 && ix < IMAGE_SIZE_X)out_image[ix + (itra_y + iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + 1 + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result3 / gmid3;
			result = 0;
			gmid = 0;
			result3 = 0;
			gmid3 = 0;
		}
	


	}
}


__global__ void NLmeansOnGPU_shift_2(unsigned char* data, float* out_image, float* weight)
{
	unsigned int wrapID = (threadIdx.x + blockIdx.x * 32) / 32, wrapLane = (threadIdx.x + blockIdx.x * 32) % 32, wrap_y = threadIdx.y;
	unsigned int ix = wrapID * 30 + wrapLane, ix_GM = wrapID * 32 + wrapLane;
	unsigned int iy = blockIdx.y;
	unsigned int iz = blockIdx.z;
	unsigned int col_per_blocky = IMAGE_SIZE_Y / GRID_Y;
	unsigned int col_per_thready = col_per_blocky / BLOCK_Y;
	unsigned int col_per_blockz = depth / (GRID_Z);                                   				////####////
	unsigned int m = (SS - 1) / 2, n = (NS - 1) / 2;
    unsigned int W = (IMAGE_SIZE_X+SS+NS-2);
	unsigned int H = (IMAGE_SIZE_Y+SS+NS-2);


	float g1 = 0;
	float g3 = 0;
	float g5 = 0;
	float gmid = 0;
	float gmid3 = 0;
	float result = 0;
	float result3 = 0; 

	for (int itra_z = 0; itra_z < col_per_blockz; itra_z+=2) {
		//for (int itra_y = 0; itra_y < col_per_thready; itra_y++) {
		for (int i = 0; i < SS; i++) {
			for (int j = 0; j < SS; j++) {
				for (int k = 0; k < SS; k++) {



						for (int jj = 0; jj < NS; jj++) {
							g1 = data[m + ix + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							g1 = data[m + ix + 3 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 3 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);			
							
							g1 = data[m + ix + 6 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);	
						
						}




					for (int ii = 1; ii < NS; ii++) {
						for (int jj = 0; jj < NS; jj++) {
						
							g1 = data[m + ix + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							g1 = data[m + ix + 3 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 3 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);	
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);	
							
							g1 = data[m + ix + 6 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);	
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);							
															
						}		
					}


						for (int jj = 0; jj < NS; jj++) {
						
							g1 = data[m + ix + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;							
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							g1 = data[m + ix + 3 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 3 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);	
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);						

							g1 = data[m + ix + 6 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);	
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);					
														
						}

						
							
					
					weight[ix_GM + k * (int)gridDim.x * 32 + j * SS * (int)gridDim.x * 32 + i * SS * SS * (int)gridDim.x * 32 + (iy * BLOCK_Y + wrap_y) * (int)gridDim.x * 32 * SS * SS * SS + iz * (int)gridDim.x * 32 * (GRID_Y * BLOCK_Y) * SS * SS * SS] = g3;			////#
					g3 = g3 / (float)(NS * NS * NS);
					g3 = exp(-g3 / (PARAMETER * PARAMETER));
					gmid += g3;
					result += g3 * data[ix + n + k + (j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + n + itra_z + iz * col_per_blockz) * W * H];
					g3 = 0;


					g5 = g5 / (float)(NS * NS * NS);
					g5 = exp(-g5 / (PARAMETER * PARAMETER));
					gmid3 += g5;
					result3 += g5 * data[ix + n + k + (j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + 1 + n + itra_z + iz * col_per_blockz) * W * H];
					g5 = 0;					
					
				}
			}
		}
		//}
		if (wrapLane < 30 && ix < IMAGE_SIZE_X)out_image[ix + (iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result / gmid;
		if (wrapLane < 30 && ix < IMAGE_SIZE_X)out_image[ix + (iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + 1 + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result3 / gmid3;
		result = 0;
		gmid = 0;
		result3 = 0;
		gmid3 = 0;		

		for (int itra_y = 1; itra_y < col_per_thready; itra_y++)
		{
			for (int i = 0; i < SS; i++) {
				for (int j = 0; j < SS; j++) {
					for (int k = 0; k < SS; k++) {


							
						g3 = weight[ix_GM + k * (int)gridDim.x * 32 + j * SS * (int)gridDim.x * 32 + i * SS * SS * (int)gridDim.x * 32 + (iy * BLOCK_Y + wrap_y) * (int)gridDim.x * 32 * SS * SS * SS + iz * (int)gridDim.x * 32 * (GRID_Y * BLOCK_Y) * SS * SS * SS];
						g5 = g3;


							
							g1 = data[m + ix + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							
							g1 = data[m + ix + 3 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 3 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3-(g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));	
							g3 = g3-__shfl_down_sync(0xffffffff, g1, 2, 32);
							
							g1 = data[m + ix + 6 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3-(g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));	
							g3 = g3-__shfl_down_sync(0xffffffff, g1, 2, 32);
						




							g1 = data[m + ix + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							
							g1 = data[m + ix + 3 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 3 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));		
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);					

							g1 = data[m + ix + 6 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));		
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);							

			
			
							
								
							
						for (int ii = 1; ii < NS; ii++) {

							g1 = data[m + ix + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);

							g1 = data[m + ix + 3 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 3 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							
							g1 = data[m + ix + 6 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);



							g1 = data[m + ix + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							
							g1 = data[m + ix + 3 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 3 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);	

							g1 = data[m + ix + 6 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);
		
							
						}
		
						
							g1 = data[m + ix + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							

							g1 = data[m + ix + 3 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 3 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							g1 = data[m + ix + 6 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);							
							


							g1 = data[m + ix + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							
						
							g1 = data[m + ix + 3 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 3 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);		
							
							g1 = data[m + ix + 6 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);
											

							  
							weight[ix_GM + k * (int)gridDim.x * 32 + j * SS * (int)gridDim.x * 32 + i * SS * SS * (int)gridDim.x * 32 + (iy * BLOCK_Y + wrap_y) * (int)gridDim.x * 32 * SS * SS * SS + iz * (int)gridDim.x * 32 * (GRID_Y * BLOCK_Y) * SS * SS * SS] = g3;			////#
							
							g3 = g3 / (float)(NS * NS * NS);
							g3 = exp(-g3 / (PARAMETER * PARAMETER));
							gmid += g3;
							result += g3 * data[ix + n + k + (itra_y + j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + n + itra_z + iz * col_per_blockz) * W * H];
							g3 = 0;

							g5 = g5 / (float)(NS * NS * NS);
							g5 = exp(-g5 / (PARAMETER * PARAMETER));
							gmid3 += g5;
							result3 += g5 * data[ix + n + k + (itra_y + j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + 1 + n + itra_z + iz * col_per_blockz) * W * H];
							g5 = 0;						
						
	
					}
				}
			}
			if (wrapLane < 30 && ix < IMAGE_SIZE_X)out_image[ix + (itra_y + iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result / gmid;
			if (wrapLane < 30 && ix < IMAGE_SIZE_X)out_image[ix + (itra_y + iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + 1 + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result3 / gmid3;
			result = 0;
			gmid = 0;
			result3 = 0;
			gmid3 = 0;
		}
	


	}
}

__global__ void NLmeansOnGPU_shift_3(unsigned char* data, float* out_image, float* weight)
{
	unsigned int wrapID = (threadIdx.x + blockIdx.x * 32) / 32, wrapLane = (threadIdx.x + blockIdx.x * 32) % 32, wrap_y = threadIdx.y;
	unsigned int ix = wrapID * 29 + wrapLane, ix_GM = wrapID * 32 + wrapLane;
	unsigned int iy = blockIdx.y;
	unsigned int iz = blockIdx.z;
	unsigned int col_per_blocky = IMAGE_SIZE_Y / GRID_Y;
	unsigned int col_per_thready = col_per_blocky / BLOCK_Y;
	unsigned int col_per_blockz = depth / (GRID_Z);                                   				////####////
	unsigned int m = (SS - 1) / 2, n = (NS - 1) / 2;
    unsigned int W = (IMAGE_SIZE_X+SS+NS-2);
	unsigned int H = (IMAGE_SIZE_Y+SS+NS-2);


	float g1 = 0;
	float g3 = 0;
	float g5 = 0;
	float gmid = 0;
	float gmid3 = 0;
	float result = 0;
	float result3 = 0; 

	for (int itra_z = 0; itra_z < col_per_blockz; itra_z+=2) {
		//for (int itra_y = 0; itra_y < col_per_thready; itra_y++) {
		for (int i = 0; i < SS; i++) {
			for (int j = 0; j < SS; j++) {
				for (int k = 0; k < SS; k++) {



						for (int jj = 0; jj < NS; jj++) {
							g1 = data[m + ix + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 3, 32);
							
							g1 = data[m + ix + 4 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);	
							g3 += __shfl_down_sync(0xffffffff, g1, 3, 32);	
							
							g1 = data[m + ix + 8 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1;
								
						}




					for (int ii = 1; ii < NS; ii++) {
						for (int jj = 0; jj < NS; jj++) {
						
							g1 = data[m + ix + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 3, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 3, 32);
							
							g1 = data[m + ix + 4 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 3, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);	
							g5 += __shfl_down_sync(0xffffffff, g1, 3, 32);	
							
							g1 = data[m + ix + 8 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1;
							g5 += g1;				
				
										
															
						}		
					}


						for (int jj = 0; jj < NS; jj++) {
						
							g1 = data[m + ix + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 3, 32);
							
							g1 = data[m + ix + 4 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);	
							g5 += __shfl_down_sync(0xffffffff, g1, 3, 32);							
			
							g1 = data[m + ix + 8 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1;
															
						}

						
							
					
					weight[ix_GM + k * (int)gridDim.x * 32 + j * SS * (int)gridDim.x * 32 + i * SS * SS * (int)gridDim.x * 32 + (iy * BLOCK_Y + wrap_y) * (int)gridDim.x * 32 * SS * SS * SS + iz * (int)gridDim.x * 32 * (GRID_Y * BLOCK_Y) * SS * SS * SS] = g3;			////#
					g3 = g3 / (float)(NS * NS * NS);
					g3 = exp(-g3 / (PARAMETER * PARAMETER));
					gmid += g3;
					result += g3 * data[ix + n + k + (j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + n + itra_z + iz * col_per_blockz) * W * H];
					g3 = 0;


					g5 = g5 / (float)(NS * NS * NS);
					g5 = exp(-g5 / (PARAMETER * PARAMETER));
					gmid3 += g5;
					result3 += g5 * data[ix + n + k + (j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + 1 + n + itra_z + iz * col_per_blockz) * W * H];
					g5 = 0;					
					
				}
			}
		}
		//}
		if (wrapLane < 29 && ix < IMAGE_SIZE_X)out_image[ix + (iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result / gmid;
		if (wrapLane < 29 && ix < IMAGE_SIZE_X)out_image[ix + (iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + 1 + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result3 / gmid3;
		result = 0;
		gmid = 0;
		result3 = 0;
		gmid3 = 0;		

		for (int itra_y = 1; itra_y < col_per_thready; itra_y++)
		{
			for (int i = 0; i < SS; i++) {
				for (int j = 0; j < SS; j++) {
					for (int k = 0; k < SS; k++) {


							
						g3 = weight[ix_GM + k * (int)gridDim.x * 32 + j * SS * (int)gridDim.x * 32 + i * SS * SS * (int)gridDim.x * 32 + (iy * BLOCK_Y + wrap_y) * (int)gridDim.x * 32 * SS * SS * SS + iz * (int)gridDim.x * 32 * (GRID_Y * BLOCK_Y) * SS * SS * SS];
						g5 = g3;


							
							g1 = data[m + ix + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 3, 32);
							
							
							g1 = data[m + ix + 4 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 2, 32);	
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 3, 32);
							
							g1 = data[m + ix + 8 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - g1;
						




							g1 = data[m + ix + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 3, 32);
							
							
							g1 = data[m + ix + 4 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);	
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 3, 32);						
						
							g1 = data[m + ix + 8 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + g1;
			
			
							
								
							
						for (int ii = 1; ii < NS; ii++) {

							g1 = data[m + ix + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 3, 32);
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 3, 32);

							g1 = data[m + ix + 4 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 3, 32);
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 3, 32);

							g1 = data[m + ix + 8 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - g1;
							g5 = g5 - g1;

							


							g1 = data[m + ix + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 3, 32);
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 3, 32);
							
							
							g1 = data[m + ix + 4 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 3, 32);
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 3, 32);
				
							g1 = data[m + ix + 8 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + g1;
							g5 = g5 + g1;	
							
						}
		
						
							g1 = data[m + ix + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 3, 32);
							

							g1 = data[m + ix + 4 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 3, 32);
							
							g1 = data[m + ix + 8 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - g1;
							


							g1 = data[m + ix + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 3, 32);
							
						
							g1 = data[m + ix + 4 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);		
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 3, 32);				

							g1 = data[m + ix + 8 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + g1;

							  
							weight[ix_GM + k * (int)gridDim.x * 32 + j * SS * (int)gridDim.x * 32 + i * SS * SS * (int)gridDim.x * 32 + (iy * BLOCK_Y + wrap_y) * (int)gridDim.x * 32 * SS * SS * SS + iz * (int)gridDim.x * 32 * (GRID_Y * BLOCK_Y) * SS * SS * SS] = g3;			////#
							
							g3 = g3 / (float)(NS * NS * NS);
							g3 = exp(-g3 / (PARAMETER * PARAMETER));
							gmid += g3;
							result += g3 * data[ix + n + k + (itra_y + j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + n + itra_z + iz * col_per_blockz) * W * H];
							g3 = 0;

							g5 = g5 / (float)(NS * NS * NS);
							g5 = exp(-g5 / (PARAMETER * PARAMETER));
							gmid3 += g5;
							result3 += g5 * data[ix + n + k + (itra_y + j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + 1 + n + itra_z + iz * col_per_blockz) * W * H];
							g5 = 0;						
						
	
					}
				}
			}
			if (wrapLane < 29 && ix < IMAGE_SIZE_X)out_image[ix + (itra_y + iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result / gmid;
			if (wrapLane < 29 && ix < IMAGE_SIZE_X)out_image[ix + (itra_y + iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + 1 + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result3 / gmid3;
			result = 0;
			gmid = 0;
			result3 = 0;
			gmid3 = 0;
		}
	


	}
}













////////////////////////-----------------------------Template window = 11-----------------------------////////////////////////
#elif NS == 11
__global__ void NLmeansOnGPU_shift_1(unsigned char* data, float* out_image, float* weight)
{
	unsigned int wrapID = (threadIdx.x + blockIdx.x * 32) / 32, wrapLane = (threadIdx.x + blockIdx.x * 32) % 32, wrap_y = threadIdx.y;
	unsigned int ix = wrapID * 31 + wrapLane, ix_GM = wrapID * 32 + wrapLane;
	unsigned int iy = blockIdx.y;
	unsigned int iz = blockIdx.z;
	unsigned int col_per_blocky = IMAGE_SIZE_Y / GRID_Y;
	unsigned int col_per_thready = col_per_blocky / BLOCK_Y;
	unsigned int col_per_blockz = depth / (GRID_Z);                                   				////####////
	unsigned int m = (SS - 1) / 2, n = (NS - 1) / 2;
    unsigned int W = (IMAGE_SIZE_X+SS+NS-2);
	unsigned int H = (IMAGE_SIZE_Y+SS+NS-2);


	float g1 = 0;
	float g3 = 0;
	float g5 = 0;
	float gmid = 0;
	float gmid3 = 0;
	float result = 0;
	float result3 = 0; 

	for (int itra_z = 0; itra_z < col_per_blockz; itra_z+=2) {
		//for (int itra_y = 0; itra_y < col_per_thready; itra_y++) {
		for (int i = 0; i < SS; i++) {
			for (int j = 0; j < SS; j++) {
				for (int k = 0; k < SS; k++) {



						for (int jj = 0; jj < NS; jj++) {
							g1 = data[m + ix + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							
							g1 = data[m + ix + 2 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 2 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);							
			
							g1 = data[m + ix + 4 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							
							g1 = data[m + ix + 6 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
						
							g1 = data[m + ix + 8 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							
							g1 = data[m + ix + 10 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 10 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1;
						}




					for (int ii = 1; ii < NS; ii++) {
						for (int jj = 0; jj < NS; jj++) {
						
							g1 = data[m + ix + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							
							
							g1 = data[m + ix + 2 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 2 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);							
			
							g1 = data[m + ix + 4 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);	
							
							g1 = data[m + ix + 6 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);	
							
							g1 = data[m + ix + 8 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);	
					
							g1 = data[m + ix + 10 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 10 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1;
							g5 += g1;
																					
						}		
					}


						for (int jj = 0; jj < NS; jj++) {
						
							g1 = data[m + ix + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							
							g1 = data[m + ix + 2 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 2 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);							
			
							g1 = data[m + ix + 4 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);

							g1 = data[m + ix + 6 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
					
							g1 = data[m + ix + 8 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							
							g1 = data[m + ix + 10 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 10 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1;
																					
						}

						
							
					
					weight[ix_GM + k * (int)gridDim.x * 32 + j * SS * (int)gridDim.x * 32 + i * SS * SS * (int)gridDim.x * 32 + (iy * BLOCK_Y + wrap_y) * (int)gridDim.x * 32 * SS * SS * SS + iz * (int)gridDim.x * 32 * (GRID_Y * BLOCK_Y) * SS * SS * SS] = g3;			////#
					g3 = g3 / (float)(NS * NS * NS);
					g3 = exp(-g3 / (PARAMETER * PARAMETER));
					gmid += g3;
					result += g3 * data[ix + n + k + (j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + n + itra_z + iz * col_per_blockz) * W * H];
					g3 = 0;


					g5 = g5 / (float)(NS * NS * NS);
					g5 = exp(-g5 / (PARAMETER * PARAMETER));
					gmid3 += g5;
					result3 += g5 * data[ix + n + k + (j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + 1 + n + itra_z + iz * col_per_blockz) * W * H];
					g5 = 0;					
					
				}
			}
		}
		//}
		if (wrapLane < 31 && ix < IMAGE_SIZE_X)out_image[ix + (iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result / gmid;
		if (wrapLane < 31 && ix < IMAGE_SIZE_X)out_image[ix + (iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + 1 + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result3 / gmid3;
		result = 0;
		gmid = 0;
		result3 = 0;
		gmid3 = 0;		

		for (int itra_y = 1; itra_y < col_per_thready; itra_y++)
		{
			for (int i = 0; i < SS; i++) {
				for (int j = 0; j < SS; j++) {
					for (int k = 0; k < SS; k++) {


							
						g3 = weight[ix_GM + k * (int)gridDim.x * 32 + j * SS * (int)gridDim.x * 32 + i * SS * SS * (int)gridDim.x * 32 + (iy * BLOCK_Y + wrap_y) * (int)gridDim.x * 32 * SS * SS * SS + iz * (int)gridDim.x * 32 * (GRID_Y * BLOCK_Y) * SS * SS * SS];
						g5 = g3;


							
							g1 = data[m + ix + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							
							g1 = data[m + ix + 2 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 2 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3-(g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));	
							
							g1 = data[m + ix + 4 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3-(g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));	
							
							g1 = data[m + ix + 6 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3-(g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));		
					
							g1 = data[m + ix + 8 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3-(g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));													

							g1 = data[m + ix + 10 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 10 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3-g1;	



							g1 = data[m + ix + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							
							g1 = data[m + ix + 2 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 2 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));							
						
							g1 = data[m + ix + 4 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));

							g1 = data[m + ix + 6 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));	
							
							g1 = data[m + ix + 8 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));	
					
							g1 = data[m + ix + 10 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 10 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + g1;						
								
							
						for (int ii = 1; ii < NS; ii++) {

							g1 = data[m + ix + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));

							g1 = data[m + ix + 2 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 2 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));

							g1 = data[m + ix + 4 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));

							g1 = data[m + ix + 6 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							
							g1 = data[m + ix + 8 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
					
							g1 = data[m + ix + 10 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 10 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - g1;
							g5 = g5 - g1;							



							g1 = data[m + ix + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							
							g1 = data[m + ix + 2 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 2 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));		
							
							g1 = data[m + ix + 4 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
												
							g1 = data[m + ix + 6 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							
							g1 = data[m + ix + 8 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));

							g1 = data[m + ix + 10 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 10 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + g1;
							g5 = g5 + g1;
						}
						
							g1 = data[m + ix + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));

							g1 = data[m + ix + 2 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 2 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));

							g1 = data[m + ix + 4 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));

							g1 = data[m + ix + 6 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
						
							g1 = data[m + ix + 8 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							
							g1 = data[m + ix + 10 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 10 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - g1;



							g1 = data[m + ix + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
						
							g1 = data[m + ix + 2 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 2 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));						
						
							g1 = data[m + ix + 4 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));												

							g1 = data[m + ix + 6 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));							
							  
							g1 = data[m + ix + 8 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							
							g1 = data[m + ix + 10 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 10 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + g1;								
							
							weight[ix_GM + k * (int)gridDim.x * 32 + j * SS * (int)gridDim.x * 32 + i * SS * SS * (int)gridDim.x * 32 + (iy * BLOCK_Y + wrap_y) * (int)gridDim.x * 32 * SS * SS * SS + iz * (int)gridDim.x * 32 * (GRID_Y * BLOCK_Y) * SS * SS * SS] = g3;			////#
							
							g3 = g3 / (float)(NS * NS * NS);
							g3 = exp(-g3 / (PARAMETER * PARAMETER));
							gmid += g3;
							result += g3 * data[ix + n + k + (itra_y + j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + n + itra_z + iz * col_per_blockz) * W * H];
							g3 = 0;

							g5 = g5 / (float)(NS * NS * NS);
							g5 = exp(-g5 / (PARAMETER * PARAMETER));
							gmid3 += g5;
							result3 += g5 * data[ix + n + k + (itra_y + j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + 1 + n + itra_z + iz * col_per_blockz) * W * H];
							g5 = 0;						
						
	
					}
				}
			}
			if (wrapLane < 31 && ix < IMAGE_SIZE_X)out_image[ix + (itra_y + iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result / gmid;
			if (wrapLane < 31 && ix < IMAGE_SIZE_X)out_image[ix + (itra_y + iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + 1 + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result3 / gmid3;
			result = 0;
			gmid = 0;
			result3 = 0;
			gmid3 = 0;
		}
	


	}
}


__global__ void NLmeansOnGPU_shift_2(unsigned char* data, float* out_image, float* weight)
{
	unsigned int wrapID = (threadIdx.x + blockIdx.x * 32) / 32, wrapLane = (threadIdx.x + blockIdx.x * 32) % 32, wrap_y = threadIdx.y;
	unsigned int ix = wrapID * 30 + wrapLane, ix_GM = wrapID * 32 + wrapLane;
	unsigned int iy = blockIdx.y;
	unsigned int iz = blockIdx.z;
	unsigned int col_per_blocky = IMAGE_SIZE_Y / GRID_Y;
	unsigned int col_per_thready = col_per_blocky / BLOCK_Y;
	unsigned int col_per_blockz = depth / (GRID_Z);                                   				////####////
	unsigned int m = (SS - 1) / 2, n = (NS - 1) / 2;
    unsigned int W = (IMAGE_SIZE_X+SS+NS-2);
	unsigned int H = (IMAGE_SIZE_Y+SS+NS-2);


	float g1 = 0;
	float g3 = 0;
	float g5 = 0;
	float gmid = 0;
	float gmid3 = 0;
	float result = 0;
	float result3 = 0; 

	for (int itra_z = 0; itra_z < col_per_blockz; itra_z+=2) {
		//for (int itra_y = 0; itra_y < col_per_thready; itra_y++) {
		for (int i = 0; i < SS; i++) {
			for (int j = 0; j < SS; j++) {
				for (int k = 0; k < SS; k++) {



						for (int jj = 0; jj < NS; jj++) {
							g1 = data[m + ix + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							g1 = data[m + ix + 3 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 3 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);			
							
							g1 = data[m + ix + 6 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);	
							
							g1 = data[m + ix + 9 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 9 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
						
						}




					for (int ii = 1; ii < NS; ii++) {
						for (int jj = 0; jj < NS; jj++) {
						
							g1 = data[m + ix + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							g1 = data[m + ix + 3 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 3 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);	
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);	
							
							g1 = data[m + ix + 6 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);	
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);	
							
							g1 = data[m + ix + 9 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 9 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);							
															
						}		
					}


						for (int jj = 0; jj < NS; jj++) {
						
							g1 = data[m + ix + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;							
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							g1 = data[m + ix + 3 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 3 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);	
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);						

							g1 = data[m + ix + 6 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);	
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);					
														
							g1 = data[m + ix + 9 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 9 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);	
						}

						
							
					
					weight[ix_GM + k * (int)gridDim.x * 32 + j * SS * (int)gridDim.x * 32 + i * SS * SS * (int)gridDim.x * 32 + (iy * BLOCK_Y + wrap_y) * (int)gridDim.x * 32 * SS * SS * SS + iz * (int)gridDim.x * 32 * (GRID_Y * BLOCK_Y) * SS * SS * SS] = g3;			////#
					g3 = g3 / (float)(NS * NS * NS);
					g3 = exp(-g3 / (PARAMETER * PARAMETER));
					gmid += g3;
					result += g3 * data[ix + n + k + (j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + n + itra_z + iz * col_per_blockz) * W * H];
					g3 = 0;


					g5 = g5 / (float)(NS * NS * NS);
					g5 = exp(-g5 / (PARAMETER * PARAMETER));
					gmid3 += g5;
					result3 += g5 * data[ix + n + k + (j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + 1 + n + itra_z + iz * col_per_blockz) * W * H];
					g5 = 0;					
					
				}
			}
		}
		//}
		if (wrapLane < 30 && ix < IMAGE_SIZE_X)out_image[ix + (iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result / gmid;
		if (wrapLane < 30 && ix < IMAGE_SIZE_X)out_image[ix + (iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + 1 + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result3 / gmid3;
		result = 0;
		gmid = 0;
		result3 = 0;
		gmid3 = 0;		

		for (int itra_y = 1; itra_y < col_per_thready; itra_y++)
		{
			for (int i = 0; i < SS; i++) {
				for (int j = 0; j < SS; j++) {
					for (int k = 0; k < SS; k++) {


							
						g3 = weight[ix_GM + k * (int)gridDim.x * 32 + j * SS * (int)gridDim.x * 32 + i * SS * SS * (int)gridDim.x * 32 + (iy * BLOCK_Y + wrap_y) * (int)gridDim.x * 32 * SS * SS * SS + iz * (int)gridDim.x * 32 * (GRID_Y * BLOCK_Y) * SS * SS * SS];
						g5 = g3;


							
							g1 = data[m + ix + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							
							g1 = data[m + ix + 3 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 3 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3-(g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));	
							g3 = g3- __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							g1 = data[m + ix + 6 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3-(g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));	
							g3 = g3-__shfl_down_sync(0xffffffff, g1, 2, 32);
						
							g1 = data[m + ix + 9 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 9 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3-(g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));	



							g1 = data[m + ix + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							
							g1 = data[m + ix + 3 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 3 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));		
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);					

							g1 = data[m + ix + 6 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));		
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);							

							g1 = data[m + ix + 9 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 9 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));					
			
							
								
							
						for (int ii = 1; ii < NS; ii++) {

							g1 = data[m + ix + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);

							g1 = data[m + ix + 3 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 3 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);
														
							g1 = data[m + ix + 6 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);

							g1 = data[m + ix + 9 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 9 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));




							g1 = data[m + ix + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							
							g1 = data[m + ix + 3 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 3 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);	

							g1 = data[m + ix + 6 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);
								
							g1 = data[m + ix + 9 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 9 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));

		
							
						}
		
						
							g1 = data[m + ix + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							

							g1 = data[m + ix + 3 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 3 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							g1 = data[m + ix + 6 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);							
							
							g1 = data[m + ix + 9 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 9 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));



							g1 = data[m + ix + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							
						
							g1 = data[m + ix + 3 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 3 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);		
							
							g1 = data[m + ix + 6 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							g1 = data[m + ix + 9 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 9 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));

											
							  
							weight[ix_GM + k * (int)gridDim.x * 32 + j * SS * (int)gridDim.x * 32 + i * SS * SS * (int)gridDim.x * 32 + (iy * BLOCK_Y + wrap_y) * (int)gridDim.x * 32 * SS * SS * SS + iz * (int)gridDim.x * 32 * (GRID_Y * BLOCK_Y) * SS * SS * SS] = g3;			////#
							
							g3 = g3 / (float)(NS * NS * NS);
							g3 = exp(-g3 / (PARAMETER * PARAMETER));
							gmid += g3;
							result += g3 * data[ix + n + k + (itra_y + j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + n + itra_z + iz * col_per_blockz) * W * H];
							g3 = 0;

							g5 = g5 / (float)(NS * NS * NS);
							g5 = exp(-g5 / (PARAMETER * PARAMETER));
							gmid3 += g5;
							result3 += g5 * data[ix + n + k + (itra_y + j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + 1 + n + itra_z + iz * col_per_blockz) * W * H];
							g5 = 0;						
						
	
					}
				}
			}
			if (wrapLane < 30 && ix < IMAGE_SIZE_X)out_image[ix + (itra_y + iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result / gmid;
			if (wrapLane < 30 && ix < IMAGE_SIZE_X)out_image[ix + (itra_y + iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + 1 + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result3 / gmid3;
			result = 0;
			gmid = 0;
			result3 = 0;
			gmid3 = 0;
		}
	


	}
}

__global__ void NLmeansOnGPU_shift_3(unsigned char* data, float* out_image, float* weight)
{
	unsigned int wrapID = (threadIdx.x + blockIdx.x * 32) / 32, wrapLane = (threadIdx.x + blockIdx.x * 32) % 32, wrap_y = threadIdx.y;
	unsigned int ix = wrapID * 29 + wrapLane, ix_GM = wrapID * 32 + wrapLane;
	unsigned int iy = blockIdx.y;
	unsigned int iz = blockIdx.z;
	unsigned int col_per_blocky = IMAGE_SIZE_Y / GRID_Y;
	unsigned int col_per_thready = col_per_blocky / BLOCK_Y;
	unsigned int col_per_blockz = depth / (GRID_Z);                                   				////####////
	unsigned int m = (SS - 1) / 2, n = (NS - 1) / 2;
    unsigned int W = (IMAGE_SIZE_X+SS+NS-2);
	unsigned int H = (IMAGE_SIZE_Y+SS+NS-2);


	float g1 = 0;
	float g3 = 0;
	float g5 = 0;
	float gmid = 0;
	float gmid3 = 0;
	float result = 0;
	float result3 = 0; 

	for (int itra_z = 0; itra_z < col_per_blockz; itra_z+=2) {
		//for (int itra_y = 0; itra_y < col_per_thready; itra_y++) {
		for (int i = 0; i < SS; i++) {
			for (int j = 0; j < SS; j++) {
				for (int k = 0; k < SS; k++) {



						for (int jj = 0; jj < NS; jj++) {
							g1 = data[m + ix + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 3, 32);
							
							g1 = data[m + ix + 4 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);	
							g3 += __shfl_down_sync(0xffffffff, g1, 3, 32);	
							
							g1 = data[m + ix + 8 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
								
						}




					for (int ii = 1; ii < NS; ii++) {
						for (int jj = 0; jj < NS; jj++) {
						
							g1 = data[m + ix + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 3, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 3, 32);
							
							g1 = data[m + ix + 4 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 3, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);	
							g5 += __shfl_down_sync(0xffffffff, g1, 3, 32);	
							
							g1 = data[m + ix + 8 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);				
				
										
															
						}		
					}


						for (int jj = 0; jj < NS; jj++) {
						
							g1 = data[m + ix + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 3, 32);
							
							g1 = data[m + ix + 4 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);	
							g5 += __shfl_down_sync(0xffffffff, g1, 3, 32);							
			
							g1 = data[m + ix + 8 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);
															
						}

						
							
					
					weight[ix_GM + k * (int)gridDim.x * 32 + j * SS * (int)gridDim.x * 32 + i * SS * SS * (int)gridDim.x * 32 + (iy * BLOCK_Y + wrap_y) * (int)gridDim.x * 32 * SS * SS * SS + iz * (int)gridDim.x * 32 * (GRID_Y * BLOCK_Y) * SS * SS * SS] = g3;			////#
					g3 = g3 / (float)(NS * NS * NS);
					g3 = exp(-g3 / (PARAMETER * PARAMETER));
					gmid += g3;
					result += g3 * data[ix + n + k + (j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + n + itra_z + iz * col_per_blockz) * W * H];
					g3 = 0;


					g5 = g5 / (float)(NS * NS * NS);
					g5 = exp(-g5 / (PARAMETER * PARAMETER));
					gmid3 += g5;
					result3 += g5 * data[ix + n + k + (j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + 1 + n + itra_z + iz * col_per_blockz) * W * H];
					g5 = 0;					
					
				}
			}
		}
		//}
		if (wrapLane < 29 && ix < IMAGE_SIZE_X)out_image[ix + (iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result / gmid;
		if (wrapLane < 29 && ix < IMAGE_SIZE_X)out_image[ix + (iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + 1 + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result3 / gmid3;
		result = 0;
		gmid = 0;
		result3 = 0;
		gmid3 = 0;		

		for (int itra_y = 1; itra_y < col_per_thready; itra_y++)
		{
			for (int i = 0; i < SS; i++) {
				for (int j = 0; j < SS; j++) {
					for (int k = 0; k < SS; k++) {


							
						g3 = weight[ix_GM + k * (int)gridDim.x * 32 + j * SS * (int)gridDim.x * 32 + i * SS * SS * (int)gridDim.x * 32 + (iy * BLOCK_Y + wrap_y) * (int)gridDim.x * 32 * SS * SS * SS + iz * (int)gridDim.x * 32 * (GRID_Y * BLOCK_Y) * SS * SS * SS];
						g5 = g3;


							
							g1 = data[m + ix + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 3, 32);
							
							
							g1 = data[m + ix + 4 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 2, 32);	
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 3, 32);
							
							g1 = data[m + ix + 8 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 2, 32);
						




							g1 = data[m + ix + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 3, 32);
							
							
							g1 = data[m + ix + 4 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);	
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 3, 32);						
						
							g1 = data[m + ix + 8 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);
			
			
							
								
							
						for (int ii = 1; ii < NS; ii++) {

							g1 = data[m + ix + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 3, 32);
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 3, 32);

							g1 = data[m + ix + 4 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 3, 32);
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 3, 32);

							g1 = data[m + ix + 8 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);

							


							g1 = data[m + ix + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 3, 32);
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 3, 32);
							
							
							g1 = data[m + ix + 4 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 3, 32);
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 3, 32);
				
							g1 = data[m + ix + 8 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);	
							
						}
		
						
							g1 = data[m + ix + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 3, 32);
							

							g1 = data[m + ix + 4 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 3, 32);
							
							g1 = data[m + ix + 8 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							


							g1 = data[m + ix + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 3, 32);
							
						
							g1 = data[m + ix + 4 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);		
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 3, 32);				

							g1 = data[m + ix + 8 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);

							  
							weight[ix_GM + k * (int)gridDim.x * 32 + j * SS * (int)gridDim.x * 32 + i * SS * SS * (int)gridDim.x * 32 + (iy * BLOCK_Y + wrap_y) * (int)gridDim.x * 32 * SS * SS * SS + iz * (int)gridDim.x * 32 * (GRID_Y * BLOCK_Y) * SS * SS * SS] = g3;			////#
							
							g3 = g3 / (float)(NS * NS * NS);
							g3 = exp(-g3 / (PARAMETER * PARAMETER));
							gmid += g3;
							result += g3 * data[ix + n + k + (itra_y + j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + n + itra_z + iz * col_per_blockz) * W * H];
							g3 = 0;

							g5 = g5 / (float)(NS * NS * NS);
							g5 = exp(-g5 / (PARAMETER * PARAMETER));
							gmid3 += g5;
							result3 += g5 * data[ix + n + k + (itra_y + j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + 1 + n + itra_z + iz * col_per_blockz) * W * H];
							g5 = 0;						
						
	
					}
				}
			}
			if (wrapLane < 29 && ix < IMAGE_SIZE_X)out_image[ix + (itra_y + iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result / gmid;
			if (wrapLane < 29 && ix < IMAGE_SIZE_X)out_image[ix + (itra_y + iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + 1 + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result3 / gmid3;
			result = 0;
			gmid = 0;
			result3 = 0;
			gmid3 = 0;
		}
	


	}
}














////////////////////////-----------------------------Template window = 13-----------------------------////////////////////////
#elif NS == 13
__global__ void NLmeansOnGPU_shift_1(unsigned char* data, float* out_image, float* weight)
{
	unsigned int wrapID = (threadIdx.x + blockIdx.x * 32) / 32, wrapLane = (threadIdx.x + blockIdx.x * 32) % 32, wrap_y = threadIdx.y;
	unsigned int ix = wrapID * 31 + wrapLane, ix_GM = wrapID * 32 + wrapLane;
	unsigned int iy = blockIdx.y;
	unsigned int iz = blockIdx.z;
	unsigned int col_per_blocky = IMAGE_SIZE_Y / GRID_Y;
	unsigned int col_per_thready = col_per_blocky / BLOCK_Y;
	unsigned int col_per_blockz = depth / (GRID_Z);                                   				////####////
	unsigned int m = (SS - 1) / 2, n = (NS - 1) / 2;
    unsigned int W = (IMAGE_SIZE_X+SS+NS-2);
	unsigned int H = (IMAGE_SIZE_Y+SS+NS-2);


	float g1 = 0;
	float g3 = 0;
	float g5 = 0;
	float gmid = 0;
	float gmid3 = 0;
	float result = 0;
	float result3 = 0; 

	for (int itra_z = 0; itra_z < col_per_blockz; itra_z+=2) {
		//for (int itra_y = 0; itra_y < col_per_thready; itra_y++) {
		for (int i = 0; i < SS; i++) {
			for (int j = 0; j < SS; j++) {
				for (int k = 0; k < SS; k++) {



						for (int jj = 0; jj < NS; jj++) {
							g1 = data[m + ix + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							
							g1 = data[m + ix + 2 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 2 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);							
			
							g1 = data[m + ix + 4 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							
							g1 = data[m + ix + 6 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
						
							g1 = data[m + ix + 8 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							
							g1 = data[m + ix + 10 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 10 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							
							g1 = data[m + ix + 12 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1;
						}




					for (int ii = 1; ii < NS; ii++) {
						for (int jj = 0; jj < NS; jj++) {
						
							g1 = data[m + ix + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							
							
							g1 = data[m + ix + 2 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 2 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);							
			
							g1 = data[m + ix + 4 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);	
							
							g1 = data[m + ix + 6 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);	
							
							g1 = data[m + ix + 8 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);	
					
							g1 = data[m + ix + 10 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 10 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);	
										
							g1 = data[m + ix + 12 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1;
							g5 += g1;											
						}		
					}


						for (int jj = 0; jj < NS; jj++) {
						
							g1 = data[m + ix + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							
							g1 = data[m + ix + 2 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 2 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);							
			
							g1 = data[m + ix + 4 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);

							g1 = data[m + ix + 6 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
					
							g1 = data[m + ix + 8 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							
							g1 = data[m + ix + 10 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 10 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
											
							g1 = data[m + ix + 12 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1;										
						}

						
							
					
					weight[ix_GM + k * (int)gridDim.x * 32 + j * SS * (int)gridDim.x * 32 + i * SS * SS * (int)gridDim.x * 32 + (iy * BLOCK_Y + wrap_y) * (int)gridDim.x * 32 * SS * SS * SS + iz * (int)gridDim.x * 32 * (GRID_Y * BLOCK_Y) * SS * SS * SS] = g3;			////#
					g3 = g3 / (float)(NS * NS * NS);
					g3 = exp(-g3 / (PARAMETER * PARAMETER));
					gmid += g3;
					result += g3 * data[ix + n + k + (j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + n + itra_z + iz * col_per_blockz) * W * H];
					g3 = 0;


					g5 = g5 / (float)(NS * NS * NS);
					g5 = exp(-g5 / (PARAMETER * PARAMETER));
					gmid3 += g5;
					result3 += g5 * data[ix + n + k + (j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + 1 + n + itra_z + iz * col_per_blockz) * W * H];
					g5 = 0;					
					
				}
			}
		}
		//}
		if (wrapLane < 31 && ix < IMAGE_SIZE_X)out_image[ix + (iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result / gmid;
		if (wrapLane < 31 && ix < IMAGE_SIZE_X)out_image[ix + (iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + 1 + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result3 / gmid3;
		result = 0;
		gmid = 0;
		result3 = 0;
		gmid3 = 0;		

		for (int itra_y = 1; itra_y < col_per_thready; itra_y++)
		{
			for (int i = 0; i < SS; i++) {
				for (int j = 0; j < SS; j++) {
					for (int k = 0; k < SS; k++) {


							
						g3 = weight[ix_GM + k * (int)gridDim.x * 32 + j * SS * (int)gridDim.x * 32 + i * SS * SS * (int)gridDim.x * 32 + (iy * BLOCK_Y + wrap_y) * (int)gridDim.x * 32 * SS * SS * SS + iz * (int)gridDim.x * 32 * (GRID_Y * BLOCK_Y) * SS * SS * SS];
						g5 = g3;


							
							g1 = data[m + ix + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							
							g1 = data[m + ix + 2 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 2 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3-(g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));	
							
							g1 = data[m + ix + 4 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3-(g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));	
							
							g1 = data[m + ix + 6 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3-(g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));		
					
							g1 = data[m + ix + 8 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3-(g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));													

							g1 = data[m + ix + 10 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 10 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3-(g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));	

							g1 = data[m + ix + 12 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3-g1;	



							g1 = data[m + ix + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							
							g1 = data[m + ix + 2 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 2 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));							
						
							g1 = data[m + ix + 4 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));

							g1 = data[m + ix + 6 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));	
							
							g1 = data[m + ix + 8 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));	
					
							g1 = data[m + ix + 10 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 10 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));						
	
							g1 = data[m + ix + 12 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + g1;									
							
							
						for (int ii = 1; ii < NS; ii++) {

							g1 = data[m + ix + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));

							g1 = data[m + ix + 2 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 2 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));

							g1 = data[m + ix + 4 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));

							g1 = data[m + ix + 6 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							
							g1 = data[m + ix + 8 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
					
							g1 = data[m + ix + 10 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 10 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));	
							
							g1 = data[m + ix + 12 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - g1;
							g5 = g5 - g1;						



							g1 = data[m + ix + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							
							g1 = data[m + ix + 2 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 2 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));		
							
							g1 = data[m + ix + 4 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
												
							g1 = data[m + ix + 6 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							
							g1 = data[m + ix + 8 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));

							g1 = data[m + ix + 10 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 10 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));

							g1 = data[m + ix + 12 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + g1;
							g5 = g5 + g1;
						

						}
						
							g1 = data[m + ix + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));

							g1 = data[m + ix + 2 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 2 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));

							g1 = data[m + ix + 4 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));

							g1 = data[m + ix + 6 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
						
							g1 = data[m + ix + 8 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							
							g1 = data[m + ix + 10 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 10 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							
							g1 = data[m + ix + 12 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - g1;



							g1 = data[m + ix + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
						
							g1 = data[m + ix + 2 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 2 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));						
						
							g1 = data[m + ix + 4 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));												

							g1 = data[m + ix + 6 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));							
							  
							g1 = data[m + ix + 8 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							
							g1 = data[m + ix + 10 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 10 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							
							g1 = data[m + ix + 12 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + g1;															
							
							weight[ix_GM + k * (int)gridDim.x * 32 + j * SS * (int)gridDim.x * 32 + i * SS * SS * (int)gridDim.x * 32 + (iy * BLOCK_Y + wrap_y) * (int)gridDim.x * 32 * SS * SS * SS + iz * (int)gridDim.x * 32 * (GRID_Y * BLOCK_Y) * SS * SS * SS] = g3;			////#
							
							g3 = g3 / (float)(NS * NS * NS);
							g3 = exp(-g3 / (PARAMETER * PARAMETER));
							gmid += g3;
							result += g3 * data[ix + n + k + (itra_y + j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + n + itra_z + iz * col_per_blockz) * W * H];
							g3 = 0;

							g5 = g5 / (float)(NS * NS * NS);
							g5 = exp(-g5 / (PARAMETER * PARAMETER));
							gmid3 += g5;
							result3 += g5 * data[ix + n + k + (itra_y + j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + 1 + n + itra_z + iz * col_per_blockz) * W * H];
							g5 = 0;						
						
	
					}
				}
			}
			if (wrapLane < 31 && ix < IMAGE_SIZE_X)out_image[ix + (itra_y + iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result / gmid;
			if (wrapLane < 31 && ix < IMAGE_SIZE_X)out_image[ix + (itra_y + iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + 1 + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result3 / gmid3;
			result = 0;
			gmid = 0;
			result3 = 0;
			gmid3 = 0;
		}
	


	}
}


__global__ void NLmeansOnGPU_shift_2(unsigned char* data, float* out_image, float* weight)
{
	unsigned int wrapID = (threadIdx.x + blockIdx.x * 32) / 32, wrapLane = (threadIdx.x + blockIdx.x * 32) % 32, wrap_y = threadIdx.y;
	unsigned int ix = wrapID * 30 + wrapLane, ix_GM = wrapID * 32 + wrapLane;
	unsigned int iy = blockIdx.y;
	unsigned int iz = blockIdx.z;
	unsigned int col_per_blocky = IMAGE_SIZE_Y / GRID_Y;
	unsigned int col_per_thready = col_per_blocky / BLOCK_Y;
	unsigned int col_per_blockz = depth / (GRID_Z);                                   				////####////
	unsigned int m = (SS - 1) / 2, n = (NS - 1) / 2;
    unsigned int W = (IMAGE_SIZE_X+SS+NS-2);
	unsigned int H = (IMAGE_SIZE_Y+SS+NS-2);


	float g1 = 0;
	float g3 = 0;
	float g5 = 0;
	float gmid = 0;
	float gmid3 = 0;
	float result = 0;
	float result3 = 0; 

	for (int itra_z = 0; itra_z < col_per_blockz; itra_z+=2) {
		//for (int itra_y = 0; itra_y < col_per_thready; itra_y++) {
		for (int i = 0; i < SS; i++) {
			for (int j = 0; j < SS; j++) {
				for (int k = 0; k < SS; k++) {



						for (int jj = 0; jj < NS; jj++) {
							g1 = data[m + ix + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							g1 = data[m + ix + 3 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 3 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);			
							
							g1 = data[m + ix + 6 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);	
							
							g1 = data[m + ix + 9 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 9 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							g1 = data[m + ix + 12 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1;
						
						}




					for (int ii = 1; ii < NS; ii++) {
						for (int jj = 0; jj < NS; jj++) {
						
							g1 = data[m + ix + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							g1 = data[m + ix + 3 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 3 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);	
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);	
							
							g1 = data[m + ix + 6 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);	
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);	
							
							g1 = data[m + ix + 9 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 9 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);	
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);						
															
							g1 = data[m + ix + 12 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1;
							g5 += g1;	
						}		
					}


						for (int jj = 0; jj < NS; jj++) {
						
							g1 = data[m + ix + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;							
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							g1 = data[m + ix + 3 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 3 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);	
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);						

							g1 = data[m + ix + 6 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);	
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);					
														
							g1 = data[m + ix + 9 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 9 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);

							g1 = data[m + ix + 12 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1;		
						}

						
							
					
					weight[ix_GM + k * (int)gridDim.x * 32 + j * SS * (int)gridDim.x * 32 + i * SS * SS * (int)gridDim.x * 32 + (iy * BLOCK_Y + wrap_y) * (int)gridDim.x * 32 * SS * SS * SS + iz * (int)gridDim.x * 32 * (GRID_Y * BLOCK_Y) * SS * SS * SS] = g3;			////#
					g3 = g3 / (float)(NS * NS * NS);
					g3 = exp(-g3 / (PARAMETER * PARAMETER));
					gmid += g3;
					result += g3 * data[ix + n + k + (j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + n + itra_z + iz * col_per_blockz) * W * H];
					g3 = 0;


					g5 = g5 / (float)(NS * NS * NS);
					g5 = exp(-g5 / (PARAMETER * PARAMETER));
					gmid3 += g5;
					result3 += g5 * data[ix + n + k + (j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + 1 + n + itra_z + iz * col_per_blockz) * W * H];
					g5 = 0;					
					
				}
			}
		}
		//}
		if (wrapLane < 30 && ix < IMAGE_SIZE_X)out_image[ix + (iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result / gmid;
		if (wrapLane < 30 && ix < IMAGE_SIZE_X)out_image[ix + (iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + 1 + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result3 / gmid3;
		result = 0;
		gmid = 0;
		result3 = 0;
		gmid3 = 0;		

		for (int itra_y = 1; itra_y < col_per_thready; itra_y++)
		{
			for (int i = 0; i < SS; i++) {
				for (int j = 0; j < SS; j++) {
					for (int k = 0; k < SS; k++) {


							
						g3 = weight[ix_GM + k * (int)gridDim.x * 32 + j * SS * (int)gridDim.x * 32 + i * SS * SS * (int)gridDim.x * 32 + (iy * BLOCK_Y + wrap_y) * (int)gridDim.x * 32 * SS * SS * SS + iz * (int)gridDim.x * 32 * (GRID_Y * BLOCK_Y) * SS * SS * SS];
						g5 = g3;


							
							g1 = data[m + ix + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							
							g1 = data[m + ix + 3 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 3 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3-(g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));	
							g3 = g3- __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							g1 = data[m + ix + 6 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3-(g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));	
							g3 = g3-__shfl_down_sync(0xffffffff, g1, 2, 32);
						
							g1 = data[m + ix + 9 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 9 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3-(g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));	
							g3 = g3-__shfl_down_sync(0xffffffff, g1, 2, 32);
							
							g1 = data[m + ix + 12 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3-g1;	


							g1 = data[m + ix + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							
							g1 = data[m + ix + 3 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 3 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));		
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);					

							g1 = data[m + ix + 6 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));		
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);							

							g1 = data[m + ix + 9 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 9 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));	
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);				

							g1 = data[m + ix + 12 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + g1;			
							
								
							
						for (int ii = 1; ii < NS; ii++) {

							g1 = data[m + ix + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);

							g1 = data[m + ix + 3 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 3 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);
														
							g1 = data[m + ix + 6 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);

							g1 = data[m + ix + 9 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 9 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);


							g1 = data[m + ix + 12 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - g1;
							g5 = g5 - g1;



							g1 = data[m + ix + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							
							g1 = data[m + ix + 3 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 3 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);	

							g1 = data[m + ix + 6 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);
								
							g1 = data[m + ix + 9 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 9 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);

							g1 = data[m + ix + 12 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + g1;
							g5 = g5 + g1;
		
							
						}
		
						
							g1 = data[m + ix + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							

							g1 = data[m + ix + 3 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 3 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							g1 = data[m + ix + 6 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);		
					
							g1 = data[m + ix + 9 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 9 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);				
							
							g1 = data[m + ix + 12 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - g1;
		


							g1 = data[m + ix + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							
						
							g1 = data[m + ix + 3 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 3 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);		
							
							g1 = data[m + ix + 6 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							g1 = data[m + ix + 9 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 9 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							g1 = data[m + ix + 12 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + g1;

											
							  
							weight[ix_GM + k * (int)gridDim.x * 32 + j * SS * (int)gridDim.x * 32 + i * SS * SS * (int)gridDim.x * 32 + (iy * BLOCK_Y + wrap_y) * (int)gridDim.x * 32 * SS * SS * SS + iz * (int)gridDim.x * 32 * (GRID_Y * BLOCK_Y) * SS * SS * SS] = g3;			////#
							
							g3 = g3 / (float)(NS * NS * NS);
							g3 = exp(-g3 / (PARAMETER * PARAMETER));
							gmid += g3;
							result += g3 * data[ix + n + k + (itra_y + j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + n + itra_z + iz * col_per_blockz) * W * H];
							g3 = 0;

							g5 = g5 / (float)(NS * NS * NS);
							g5 = exp(-g5 / (PARAMETER * PARAMETER));
							gmid3 += g5;
							result3 += g5 * data[ix + n + k + (itra_y + j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + 1 + n + itra_z + iz * col_per_blockz) * W * H];
							g5 = 0;						
						
	
					}
				}
			}
			if (wrapLane < 30 && ix < IMAGE_SIZE_X)out_image[ix + (itra_y + iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result / gmid;
			if (wrapLane < 30 && ix < IMAGE_SIZE_X)out_image[ix + (itra_y + iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + 1 + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result3 / gmid3;
			result = 0;
			gmid = 0;
			result3 = 0;
			gmid3 = 0;
		}
	


	}
}

__global__ void NLmeansOnGPU_shift_3(unsigned char* data, float* out_image, float* weight)
{
	unsigned int wrapID = (threadIdx.x + blockIdx.x * 32) / 32, wrapLane = (threadIdx.x + blockIdx.x * 32) % 32, wrap_y = threadIdx.y;
	unsigned int ix = wrapID * 29 + wrapLane, ix_GM = wrapID * 32 + wrapLane;
	unsigned int iy = blockIdx.y;
	unsigned int iz = blockIdx.z;
	unsigned int col_per_blocky = IMAGE_SIZE_Y / GRID_Y;
	unsigned int col_per_thready = col_per_blocky / BLOCK_Y;
	unsigned int col_per_blockz = depth / (GRID_Z);                                   				////####////
	unsigned int m = (SS - 1) / 2, n = (NS - 1) / 2;
    unsigned int W = (IMAGE_SIZE_X+SS+NS-2);
	unsigned int H = (IMAGE_SIZE_Y+SS+NS-2);


	float g1 = 0;
	float g3 = 0;
	float g5 = 0;
	float gmid = 0;
	float gmid3 = 0;
	float result = 0;
	float result3 = 0; 

	for (int itra_z = 0; itra_z < col_per_blockz; itra_z+=2) {
		//for (int itra_y = 0; itra_y < col_per_thready; itra_y++) {
		for (int i = 0; i < SS; i++) {
			for (int j = 0; j < SS; j++) {
				for (int k = 0; k < SS; k++) {



						for (int jj = 0; jj < NS; jj++) {
							g1 = data[m + ix + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 3, 32);
							
							g1 = data[m + ix + 4 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);	
							g3 += __shfl_down_sync(0xffffffff, g1, 3, 32);	
							
							g1 = data[m + ix + 8 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 3, 32);

							g1 = data[m + ix + 12 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1;								
						}




					for (int ii = 1; ii < NS; ii++) {
						for (int jj = 0; jj < NS; jj++) {
						
							g1 = data[m + ix + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 3, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 3, 32);
							
							g1 = data[m + ix + 4 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 3, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);	
							g5 += __shfl_down_sync(0xffffffff, g1, 3, 32);	
							
							g1 = data[m + ix + 8 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 3, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);	
							g5 += __shfl_down_sync(0xffffffff, g1, 3, 32);			
				
							g1 = data[m + ix + 12 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1;
							g5 += g1;

										
															
						}		
					}


						for (int jj = 0; jj < NS; jj++) {
						
							g1 = data[m + ix + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 3, 32);
							
							g1 = data[m + ix + 4 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);	
							g5 += __shfl_down_sync(0xffffffff, g1, 3, 32);							
			
							g1 = data[m + ix + 8 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 3, 32);
															
							g1 = data[m + ix + 12 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1;

						}

						
							
					
					weight[ix_GM + k * (int)gridDim.x * 32 + j * SS * (int)gridDim.x * 32 + i * SS * SS * (int)gridDim.x * 32 + (iy * BLOCK_Y + wrap_y) * (int)gridDim.x * 32 * SS * SS * SS + iz * (int)gridDim.x * 32 * (GRID_Y * BLOCK_Y) * SS * SS * SS] = g3;			////#
					g3 = g3 / (float)(NS * NS * NS);
					g3 = exp(-g3 / (PARAMETER * PARAMETER));
					gmid += g3;
					result += g3 * data[ix + n + k + (j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + n + itra_z + iz * col_per_blockz) * W * H];
					g3 = 0;


					g5 = g5 / (float)(NS * NS * NS);
					g5 = exp(-g5 / (PARAMETER * PARAMETER));
					gmid3 += g5;
					result3 += g5 * data[ix + n + k + (j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + 1 + n + itra_z + iz * col_per_blockz) * W * H];
					g5 = 0;					
					
				}
			}
		}
		//}
		if (wrapLane < 29 && ix < IMAGE_SIZE_X)out_image[ix + (iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result / gmid;
		if (wrapLane < 29 && ix < IMAGE_SIZE_X)out_image[ix + (iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + 1 + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result3 / gmid3;
		result = 0;
		gmid = 0;
		result3 = 0;
		gmid3 = 0;		

		for (int itra_y = 1; itra_y < col_per_thready; itra_y++)
		{
			for (int i = 0; i < SS; i++) {
				for (int j = 0; j < SS; j++) {
					for (int k = 0; k < SS; k++) {


							
						g3 = weight[ix_GM + k * (int)gridDim.x * 32 + j * SS * (int)gridDim.x * 32 + i * SS * SS * (int)gridDim.x * 32 + (iy * BLOCK_Y + wrap_y) * (int)gridDim.x * 32 * SS * SS * SS + iz * (int)gridDim.x * 32 * (GRID_Y * BLOCK_Y) * SS * SS * SS];
						g5 = g3;


							
							g1 = data[m + ix + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 3, 32);
							
							
							g1 = data[m + ix + 4 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 2, 32);	
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 3, 32);
							
							g1 = data[m + ix + 8 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 3, 32);

							g1 = data[m + ix + 12 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - g1;
						




							g1 = data[m + ix + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 3, 32);
							
							
							g1 = data[m + ix + 4 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);	
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 3, 32);						
						
							g1 = data[m + ix + 8 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 3, 32);
							
							g1 = data[m + ix + 12 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + g1;
	
			
			
							
								
							
						for (int ii = 1; ii < NS; ii++) {

							g1 = data[m + ix + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 3, 32);
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 3, 32);

							g1 = data[m + ix + 4 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 3, 32);
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 3, 32);

							g1 = data[m + ix + 8 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 3, 32);
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 3, 32);
							
							g1 = data[m + ix + 12 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - g1;
							g5 = g5 - g1;
							

							


							g1 = data[m + ix + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 3, 32);
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 3, 32);
							
							
							g1 = data[m + ix + 4 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 3, 32);
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 3, 32);
				
							g1 = data[m + ix + 8 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 3, 32);
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);	
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 3, 32);
						
							g1 = data[m + ix + 12 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + g1;
							g5 = g5 + g1;

							
						}
		
						
							g1 = data[m + ix + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 3, 32);
							

							g1 = data[m + ix + 4 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 3, 32);
							
							g1 = data[m + ix + 8 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 3, 32);
							
							g1 = data[m + ix + 12 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - g1;
							


							g1 = data[m + ix + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 3, 32);
							
						
							g1 = data[m + ix + 4 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);		
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 3, 32);				

							g1 = data[m + ix + 8 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 3, 32);	
							
							g1 = data[m + ix + 12 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + g1;

							  
							weight[ix_GM + k * (int)gridDim.x * 32 + j * SS * (int)gridDim.x * 32 + i * SS * SS * (int)gridDim.x * 32 + (iy * BLOCK_Y + wrap_y) * (int)gridDim.x * 32 * SS * SS * SS + iz * (int)gridDim.x * 32 * (GRID_Y * BLOCK_Y) * SS * SS * SS] = g3;			////#
							
							g3 = g3 / (float)(NS * NS * NS);
							g3 = exp(-g3 / (PARAMETER * PARAMETER));
							gmid += g3;
							result += g3 * data[ix + n + k + (itra_y + j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + n + itra_z + iz * col_per_blockz) * W * H];
							g3 = 0;

							g5 = g5 / (float)(NS * NS * NS);
							g5 = exp(-g5 / (PARAMETER * PARAMETER));
							gmid3 += g5;
							result3 += g5 * data[ix + n + k + (itra_y + j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + 1 + n + itra_z + iz * col_per_blockz) * W * H];
							g5 = 0;						
						
	
					}
				}
			}
			if (wrapLane < 29 && ix < IMAGE_SIZE_X)out_image[ix + (itra_y + iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result / gmid;
			if (wrapLane < 29 && ix < IMAGE_SIZE_X)out_image[ix + (itra_y + iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + 1 + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result3 / gmid3;
			result = 0;
			gmid = 0;
			result3 = 0;
			gmid3 = 0;
		}
	


	}
}










////////////////////////-----------------------------Template window = 15-----------------------------////////////////////////
#elif NS == 15
__global__ void NLmeansOnGPU_shift_1(unsigned char* data, float* out_image, float* weight)
{
	unsigned int wrapID = (threadIdx.x + blockIdx.x * 32) / 32, wrapLane = (threadIdx.x + blockIdx.x * 32) % 32, wrap_y = threadIdx.y;
	unsigned int ix = wrapID * 31 + wrapLane, ix_GM = wrapID * 32 + wrapLane;
	unsigned int iy = blockIdx.y;
	unsigned int iz = blockIdx.z;
	unsigned int col_per_blocky = IMAGE_SIZE_Y / GRID_Y;
	unsigned int col_per_thready = col_per_blocky / BLOCK_Y;
	unsigned int col_per_blockz = depth / (GRID_Z);                                   				////####////
	unsigned int m = (SS - 1) / 2, n = (NS - 1) / 2;
    unsigned int W = (IMAGE_SIZE_X+SS+NS-2);
	unsigned int H = (IMAGE_SIZE_Y+SS+NS-2);


	float g1 = 0;
	float g3 = 0;
	float g5 = 0;
	float gmid = 0;
	float gmid3 = 0;
	float result = 0;
	float result3 = 0; 

	for (int itra_z = 0; itra_z < col_per_blockz; itra_z+=2) {
		//for (int itra_y = 0; itra_y < col_per_thready; itra_y++) {
		for (int i = 0; i < SS; i++) {
			for (int j = 0; j < SS; j++) {
				for (int k = 0; k < SS; k++) {



						for (int jj = 0; jj < NS; jj++) {
							g1 = data[m + ix + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							
							g1 = data[m + ix + 2 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 2 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);							
			
							g1 = data[m + ix + 4 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							
							g1 = data[m + ix + 6 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
						
							g1 = data[m + ix + 8 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							
							g1 = data[m + ix + 10 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 10 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							
							g1 = data[m + ix + 12 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
				
							g1 = data[m + ix + 14 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 14 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1;
						}




					for (int ii = 1; ii < NS; ii++) {
						for (int jj = 0; jj < NS; jj++) {
						
							g1 = data[m + ix + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							
							
							g1 = data[m + ix + 2 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 2 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);							
			
							g1 = data[m + ix + 4 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);	
							
							g1 = data[m + ix + 6 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);	
							
							g1 = data[m + ix + 8 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);	
					
							g1 = data[m + ix + 10 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 10 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);	
										
							g1 = data[m + ix + 12 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);				
							
							g1 = data[m + ix + 14 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 14 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1;
							g5 += g1;								
						}		
					}


						for (int jj = 0; jj < NS; jj++) {
						
							g1 = data[m + ix + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							
							g1 = data[m + ix + 2 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 2 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);							
			
							g1 = data[m + ix + 4 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);

							g1 = data[m + ix + 6 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
					
							g1 = data[m + ix + 8 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							
							g1 = data[m + ix + 10 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 10 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
											
							g1 = data[m + ix + 12 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);		
							
							g1 = data[m + ix + 14 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 14 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1;									
						}

						
							
					
					weight[ix_GM + k * (int)gridDim.x * 32 + j * SS * (int)gridDim.x * 32 + i * SS * SS * (int)gridDim.x * 32 + (iy * BLOCK_Y + wrap_y) * (int)gridDim.x * 32 * SS * SS * SS + iz * (int)gridDim.x * 32 * (GRID_Y * BLOCK_Y) * SS * SS * SS] = g3;			////#
					g3 = g3 / (float)(NS * NS * NS);
					g3 = exp(-g3 / (PARAMETER * PARAMETER));
					gmid += g3;
					result += g3 * data[ix + n + k + (j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + n + itra_z + iz * col_per_blockz) * W * H];
					g3 = 0;


					g5 = g5 / (float)(NS * NS * NS);
					g5 = exp(-g5 / (PARAMETER * PARAMETER));
					gmid3 += g5;
					result3 += g5 * data[ix + n + k + (j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + 1 + n + itra_z + iz * col_per_blockz) * W * H];
					g5 = 0;					
					
				}
			}
		}
		//}
		if (wrapLane < 31 && ix < IMAGE_SIZE_X)out_image[ix + (iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result / gmid;
		if (wrapLane < 31 && ix < IMAGE_SIZE_X)out_image[ix + (iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + 1 + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result3 / gmid3;
		result = 0;
		gmid = 0;
		result3 = 0;
		gmid3 = 0;		

		for (int itra_y = 1; itra_y < col_per_thready; itra_y++)
		{
			for (int i = 0; i < SS; i++) {
				for (int j = 0; j < SS; j++) {
					for (int k = 0; k < SS; k++) {


							
						g3 = weight[ix_GM + k * (int)gridDim.x * 32 + j * SS * (int)gridDim.x * 32 + i * SS * SS * (int)gridDim.x * 32 + (iy * BLOCK_Y + wrap_y) * (int)gridDim.x * 32 * SS * SS * SS + iz * (int)gridDim.x * 32 * (GRID_Y * BLOCK_Y) * SS * SS * SS];
						g5 = g3;


							
							g1 = data[m + ix + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							
							g1 = data[m + ix + 2 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 2 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3-(g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));	
							
							g1 = data[m + ix + 4 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3-(g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));	
							
							g1 = data[m + ix + 6 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3-(g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));		
					
							g1 = data[m + ix + 8 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3-(g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));													

							g1 = data[m + ix + 10 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 10 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3-(g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));	

							g1 = data[m + ix + 12 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3-(g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							
							g1 = data[m + ix + 14 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 14 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3-g1;



							g1 = data[m + ix + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							
							g1 = data[m + ix + 2 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 2 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));							
						
							g1 = data[m + ix + 4 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));

							g1 = data[m + ix + 6 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));	
							
							g1 = data[m + ix + 8 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));	
					
							g1 = data[m + ix + 10 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 10 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));						
	
							g1 = data[m + ix + 12 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));	

							g1 = data[m + ix + 14 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 14 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + g1;								
							
							
						for (int ii = 1; ii < NS; ii++) {

							g1 = data[m + ix + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));

							g1 = data[m + ix + 2 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 2 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));

							g1 = data[m + ix + 4 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));

							g1 = data[m + ix + 6 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							
							g1 = data[m + ix + 8 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
					
							g1 = data[m + ix + 10 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 10 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));	
							
							g1 = data[m + ix + 12 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));						

							g1 = data[m + ix + 14 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 14 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - g1;
							g5 = g5 - g1;	


							g1 = data[m + ix + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							
							g1 = data[m + ix + 2 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 2 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));		
							
							g1 = data[m + ix + 4 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
												
							g1 = data[m + ix + 6 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							
							g1 = data[m + ix + 8 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));

							g1 = data[m + ix + 10 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 10 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));

							g1 = data[m + ix + 12 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							
							g1 = data[m + ix + 14 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 14 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + g1;
							g5 = g5 + g1;
						

						}
						
							g1 = data[m + ix + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));

							g1 = data[m + ix + 2 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 2 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));

							g1 = data[m + ix + 4 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));

							g1 = data[m + ix + 6 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
						
							g1 = data[m + ix + 8 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							
							g1 = data[m + ix + 10 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 10 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							
							g1 = data[m + ix + 12 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
					
							g1 = data[m + ix + 14 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 14 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - g1;



							g1 = data[m + ix + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
						
							g1 = data[m + ix + 2 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 2 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));						
						
							g1 = data[m + ix + 4 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));												

							g1 = data[m + ix + 6 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));							
							  
							g1 = data[m + ix + 8 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							
							g1 = data[m + ix + 10 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 10 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							
							g1 = data[m + ix + 12 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));			
					
							g1 = data[m + ix + 14 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 14 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + g1;												
							
							weight[ix_GM + k * (int)gridDim.x * 32 + j * SS * (int)gridDim.x * 32 + i * SS * SS * (int)gridDim.x * 32 + (iy * BLOCK_Y + wrap_y) * (int)gridDim.x * 32 * SS * SS * SS + iz * (int)gridDim.x * 32 * (GRID_Y * BLOCK_Y) * SS * SS * SS] = g3;			////#
							
							g3 = g3 / (float)(NS * NS * NS);
							g3 = exp(-g3 / (PARAMETER * PARAMETER));
							gmid += g3;
							result += g3 * data[ix + n + k + (itra_y + j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + n + itra_z + iz * col_per_blockz) * W * H];
							g3 = 0;

							g5 = g5 / (float)(NS * NS * NS);
							g5 = exp(-g5 / (PARAMETER * PARAMETER));
							gmid3 += g5;
							result3 += g5 * data[ix + n + k + (itra_y + j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + 1 + n + itra_z + iz * col_per_blockz) * W * H];
							g5 = 0;						
						
	
					}
				}
			}
			if (wrapLane < 31 && ix < IMAGE_SIZE_X)out_image[ix + (itra_y + iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result / gmid;
			if (wrapLane < 31 && ix < IMAGE_SIZE_X)out_image[ix + (itra_y + iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + 1 + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result3 / gmid3;
			result = 0;
			gmid = 0;
			result3 = 0;
			gmid3 = 0;
		}
	


	}
}


__global__ void NLmeansOnGPU_shift_2(unsigned char* data, float* out_image, float* weight)
{
	unsigned int wrapID = (threadIdx.x + blockIdx.x * 32) / 32, wrapLane = (threadIdx.x + blockIdx.x * 32) % 32, wrap_y = threadIdx.y;
	unsigned int ix = wrapID * 30 + wrapLane, ix_GM = wrapID * 32 + wrapLane;
	unsigned int iy = blockIdx.y;
	unsigned int iz = blockIdx.z;
	unsigned int col_per_blocky = IMAGE_SIZE_Y / GRID_Y;
	unsigned int col_per_thready = col_per_blocky / BLOCK_Y;
	unsigned int col_per_blockz = depth / (GRID_Z);                                   				////####////
	unsigned int m = (SS - 1) / 2, n = (NS - 1) / 2;
    unsigned int W = (IMAGE_SIZE_X+SS+NS-2);
	unsigned int H = (IMAGE_SIZE_Y+SS+NS-2);


	float g1 = 0;
	float g3 = 0;
	float g5 = 0;
	float gmid = 0;
	float gmid3 = 0;
	float result = 0;
	float result3 = 0; 

	for (int itra_z = 0; itra_z < col_per_blockz; itra_z+=2) {
		//for (int itra_y = 0; itra_y < col_per_thready; itra_y++) {
		for (int i = 0; i < SS; i++) {
			for (int j = 0; j < SS; j++) {
				for (int k = 0; k < SS; k++) {



						for (int jj = 0; jj < NS; jj++) {
							g1 = data[m + ix + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							g1 = data[m + ix + 3 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 3 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);			
							
							g1 = data[m + ix + 6 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);	
							
							g1 = data[m + ix + 9 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 9 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							g1 = data[m + ix + 12 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							

						
						}




					for (int ii = 1; ii < NS; ii++) {
						for (int jj = 0; jj < NS; jj++) {
						
							g1 = data[m + ix + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							g1 = data[m + ix + 3 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 3 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);	
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);	
							
							g1 = data[m + ix + 6 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);	
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);	
							
							g1 = data[m + ix + 9 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 9 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);	
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);						
															
							g1 = data[m + ix + 12 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);	
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);	
						}		
					}


						for (int jj = 0; jj < NS; jj++) {
						
							g1 = data[m + ix + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;							
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							g1 = data[m + ix + 3 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 3 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);	
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);						

							g1 = data[m + ix + 6 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);	
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);					
														
							g1 = data[m + ix + 9 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 9 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);

							g1 = data[m + ix + 12 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);		
						}

						
							
					
					weight[ix_GM + k * (int)gridDim.x * 32 + j * SS * (int)gridDim.x * 32 + i * SS * SS * (int)gridDim.x * 32 + (iy * BLOCK_Y + wrap_y) * (int)gridDim.x * 32 * SS * SS * SS + iz * (int)gridDim.x * 32 * (GRID_Y * BLOCK_Y) * SS * SS * SS] = g3;			////#
					g3 = g3 / (float)(NS * NS * NS);
					g3 = exp(-g3 / (PARAMETER * PARAMETER));
					gmid += g3;
					result += g3 * data[ix + n + k + (j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + n + itra_z + iz * col_per_blockz) * W * H];
					g3 = 0;


					g5 = g5 / (float)(NS * NS * NS);
					g5 = exp(-g5 / (PARAMETER * PARAMETER));
					gmid3 += g5;
					result3 += g5 * data[ix + n + k + (j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + 1 + n + itra_z + iz * col_per_blockz) * W * H];
					g5 = 0;					
					
				}
			}
		}
		//}
		if (wrapLane < 30 && ix < IMAGE_SIZE_X)out_image[ix + (iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result / gmid;
		if (wrapLane < 30 && ix < IMAGE_SIZE_X)out_image[ix + (iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + 1 + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result3 / gmid3;
		result = 0;
		gmid = 0;
		result3 = 0;
		gmid3 = 0;		

		for (int itra_y = 1; itra_y < col_per_thready; itra_y++)
		{
			for (int i = 0; i < SS; i++) {
				for (int j = 0; j < SS; j++) {
					for (int k = 0; k < SS; k++) {


							
						g3 = weight[ix_GM + k * (int)gridDim.x * 32 + j * SS * (int)gridDim.x * 32 + i * SS * SS * (int)gridDim.x * 32 + (iy * BLOCK_Y + wrap_y) * (int)gridDim.x * 32 * SS * SS * SS + iz * (int)gridDim.x * 32 * (GRID_Y * BLOCK_Y) * SS * SS * SS];
						g5 = g3;


							
							g1 = data[m + ix + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							
							g1 = data[m + ix + 3 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 3 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3-(g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));	
							g3 = g3- __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							g1 = data[m + ix + 6 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3-(g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));	
							g3 = g3-__shfl_down_sync(0xffffffff, g1, 2, 32);
						
							g1 = data[m + ix + 9 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 9 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3-(g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));	
							g3 = g3-__shfl_down_sync(0xffffffff, g1, 2, 32);
							
							g1 = data[m + ix + 12 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3-(g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));	
							g3 = g3-__shfl_down_sync(0xffffffff, g1, 2, 32);		



							g1 = data[m + ix + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							
							g1 = data[m + ix + 3 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 3 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));		
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);					

							g1 = data[m + ix + 6 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));		
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);							

							g1 = data[m + ix + 9 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 9 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));	
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);				

							g1 = data[m + ix + 12 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));	
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);				
							
								
							
						for (int ii = 1; ii < NS; ii++) {

							g1 = data[m + ix + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);

							g1 = data[m + ix + 3 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 3 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);
														
							g1 = data[m + ix + 6 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);

							g1 = data[m + ix + 9 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 9 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);


							g1 = data[m + ix + 12 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);



							g1 = data[m + ix + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							
							g1 = data[m + ix + 3 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 3 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);	

							g1 = data[m + ix + 6 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);
								
							g1 = data[m + ix + 9 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 9 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);

							g1 = data[m + ix + 12 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);
		
							
						}
		
						
							g1 = data[m + ix + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							

							g1 = data[m + ix + 3 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 3 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							g1 = data[m + ix + 6 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);		
							
							g1 = data[m + ix + 9 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 9 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);	
						
							g1 = data[m + ix + 12 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);											
							


							g1 = data[m + ix + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							
						
							g1 = data[m + ix + 3 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 3 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);		
							
							g1 = data[m + ix + 6 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							g1 = data[m + ix + 9 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 9 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							g1 = data[m + ix + 12 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);

											
							  
							weight[ix_GM + k * (int)gridDim.x * 32 + j * SS * (int)gridDim.x * 32 + i * SS * SS * (int)gridDim.x * 32 + (iy * BLOCK_Y + wrap_y) * (int)gridDim.x * 32 * SS * SS * SS + iz * (int)gridDim.x * 32 * (GRID_Y * BLOCK_Y) * SS * SS * SS] = g3;			////#
							
							g3 = g3 / (float)(NS * NS * NS);
							g3 = exp(-g3 / (PARAMETER * PARAMETER));
							gmid += g3;
							result += g3 * data[ix + n + k + (itra_y + j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + n + itra_z + iz * col_per_blockz) * W * H];
							g3 = 0;

							g5 = g5 / (float)(NS * NS * NS);
							g5 = exp(-g5 / (PARAMETER * PARAMETER));
							gmid3 += g5;
							result3 += g5 * data[ix + n + k + (itra_y + j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + 1 + n + itra_z + iz * col_per_blockz) * W * H];
							g5 = 0;						
						
	
					}
				}
			}
			if (wrapLane < 30 && ix < IMAGE_SIZE_X)out_image[ix + (itra_y + iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result / gmid;
			if (wrapLane < 30 && ix < IMAGE_SIZE_X)out_image[ix + (itra_y + iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + 1 + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result3 / gmid3;
			result = 0;
			gmid = 0;
			result3 = 0;
			gmid3 = 0;
		}
	


	}
}

__global__ void NLmeansOnGPU_shift_3(unsigned char* data, float* out_image, float* weight)
{
	unsigned int wrapID = (threadIdx.x + blockIdx.x * 32) / 32, wrapLane = (threadIdx.x + blockIdx.x * 32) % 32, wrap_y = threadIdx.y;
	unsigned int ix = wrapID * 29 + wrapLane, ix_GM = wrapID * 32 + wrapLane;
	unsigned int iy = blockIdx.y;
	unsigned int iz = blockIdx.z;
	unsigned int col_per_blocky = IMAGE_SIZE_Y / GRID_Y;
	unsigned int col_per_thready = col_per_blocky / BLOCK_Y;
	unsigned int col_per_blockz = depth / (GRID_Z);                                   				////####////
	unsigned int m = (SS - 1) / 2, n = (NS - 1) / 2;
    unsigned int W = (IMAGE_SIZE_X+SS+NS-2);
	unsigned int H = (IMAGE_SIZE_Y+SS+NS-2);


	float g1 = 0;
	float g3 = 0;
	float g5 = 0;
	float gmid = 0;
	float gmid3 = 0;
	float result = 0;
	float result3 = 0; 

	for (int itra_z = 0; itra_z < col_per_blockz; itra_z+=2) {
		//for (int itra_y = 0; itra_y < col_per_thready; itra_y++) {
		for (int i = 0; i < SS; i++) {
			for (int j = 0; j < SS; j++) {
				for (int k = 0; k < SS; k++) {



						for (int jj = 0; jj < NS; jj++) {
							g1 = data[m + ix + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 3, 32);
							
							g1 = data[m + ix + 4 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);	
							g3 += __shfl_down_sync(0xffffffff, g1, 3, 32);	
							
							g1 = data[m + ix + 8 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 3, 32);

							g1 = data[m + ix + 12 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);								
						}




					for (int ii = 1; ii < NS; ii++) {
						for (int jj = 0; jj < NS; jj++) {
						
							g1 = data[m + ix + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 3, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 3, 32);
							
							g1 = data[m + ix + 4 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 3, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);	
							g5 += __shfl_down_sync(0xffffffff, g1, 3, 32);	
							
							g1 = data[m + ix + 8 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 3, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);	
							g5 += __shfl_down_sync(0xffffffff, g1, 3, 32);			
				
							g1 = data[m + ix + 12 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);																										
						}		
					}


						for (int jj = 0; jj < NS; jj++) {
						
							g1 = data[m + ix + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 3, 32);
							
							g1 = data[m + ix + 4 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);	
							g5 += __shfl_down_sync(0xffffffff, g1, 3, 32);							
			
							g1 = data[m + ix + 8 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 3, 32);
															
							g1 = data[m + ix + 12 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);

						}

						
							
					
					weight[ix_GM + k * (int)gridDim.x * 32 + j * SS * (int)gridDim.x * 32 + i * SS * SS * (int)gridDim.x * 32 + (iy * BLOCK_Y + wrap_y) * (int)gridDim.x * 32 * SS * SS * SS + iz * (int)gridDim.x * 32 * (GRID_Y * BLOCK_Y) * SS * SS * SS] = g3;			////#
					g3 = g3 / (float)(NS * NS * NS);
					g3 = exp(-g3 / (PARAMETER * PARAMETER));
					gmid += g3;
					result += g3 * data[ix + n + k + (j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + n + itra_z + iz * col_per_blockz) * W * H];
					g3 = 0;


					g5 = g5 / (float)(NS * NS * NS);
					g5 = exp(-g5 / (PARAMETER * PARAMETER));
					gmid3 += g5;
					result3 += g5 * data[ix + n + k + (j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + 1 + n + itra_z + iz * col_per_blockz) * W * H];
					g5 = 0;					
					
				}
			}
		}
		//}
		if (wrapLane < 29 && ix < IMAGE_SIZE_X)out_image[ix + (iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result / gmid;
		if (wrapLane < 29 && ix < IMAGE_SIZE_X)out_image[ix + (iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + 1 + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result3 / gmid3;
		result = 0;
		gmid = 0;
		result3 = 0;
		gmid3 = 0;		

		for (int itra_y = 1; itra_y < col_per_thready; itra_y++)
		{
			for (int i = 0; i < SS; i++) {
				for (int j = 0; j < SS; j++) {
					for (int k = 0; k < SS; k++) {


							
						g3 = weight[ix_GM + k * (int)gridDim.x * 32 + j * SS * (int)gridDim.x * 32 + i * SS * SS * (int)gridDim.x * 32 + (iy * BLOCK_Y + wrap_y) * (int)gridDim.x * 32 * SS * SS * SS + iz * (int)gridDim.x * 32 * (GRID_Y * BLOCK_Y) * SS * SS * SS];
						g5 = g3;


							
							g1 = data[m + ix + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 3, 32);
							
							
							g1 = data[m + ix + 4 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 2, 32);	
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 3, 32);
							
							g1 = data[m + ix + 8 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 3, 32);

							g1 = data[m + ix + 12 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 2, 32);
						




							g1 = data[m + ix + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 3, 32);
							
							
							g1 = data[m + ix + 4 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);	
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 3, 32);						
						
							g1 = data[m + ix + 8 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 3, 32);
							
							g1 = data[m + ix + 12 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);
	
			
			
							
								
							
						for (int ii = 1; ii < NS; ii++) {

							g1 = data[m + ix + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 3, 32);
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 3, 32);

							g1 = data[m + ix + 4 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 3, 32);
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 3, 32);

							g1 = data[m + ix + 8 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 3, 32);
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 3, 32);
							
							g1 = data[m + ix + 12 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							

							


							g1 = data[m + ix + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 3, 32);
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 3, 32);
							
							
							g1 = data[m + ix + 4 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 3, 32);
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 3, 32);
				
							g1 = data[m + ix + 8 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 3, 32);
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);	
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 3, 32);
						
							g1 = data[m + ix + 12 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);	

							
						}
		
						
							g1 = data[m + ix + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 3, 32);
							

							g1 = data[m + ix + 4 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 3, 32);
							
							g1 = data[m + ix + 8 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 3, 32);
							
							g1 = data[m + ix + 12 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							


							g1 = data[m + ix + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 3, 32);
							
						
							g1 = data[m + ix + 4 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);		
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 3, 32);				

							g1 = data[m + ix + 8 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 3, 32);	
							
							g1 = data[m + ix + 12 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);

							  
							weight[ix_GM + k * (int)gridDim.x * 32 + j * SS * (int)gridDim.x * 32 + i * SS * SS * (int)gridDim.x * 32 + (iy * BLOCK_Y + wrap_y) * (int)gridDim.x * 32 * SS * SS * SS + iz * (int)gridDim.x * 32 * (GRID_Y * BLOCK_Y) * SS * SS * SS] = g3;			////#
							
							g3 = g3 / (float)(NS * NS * NS);
							g3 = exp(-g3 / (PARAMETER * PARAMETER));
							gmid += g3;
							result += g3 * data[ix + n + k + (itra_y + j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + n + itra_z + iz * col_per_blockz) * W * H];
							g3 = 0;

							g5 = g5 / (float)(NS * NS * NS);
							g5 = exp(-g5 / (PARAMETER * PARAMETER));
							gmid3 += g5;
							result3 += g5 * data[ix + n + k + (itra_y + j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + 1 + n + itra_z + iz * col_per_blockz) * W * H];
							g5 = 0;						
						
	
					}
				}
			}
			if (wrapLane < 29 && ix < IMAGE_SIZE_X)out_image[ix + (itra_y + iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result / gmid;
			if (wrapLane < 29 && ix < IMAGE_SIZE_X)out_image[ix + (itra_y + iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + 1 + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result3 / gmid3;
			result = 0;
			gmid = 0;
			result3 = 0;
			gmid3 = 0;
		}
	


	}
}





















////////////////////////-----------------------------Template window = 17-----------------------------////////////////////////
#elif NS == 17
__global__ void NLmeansOnGPU_shift_1(unsigned char* data, float* out_image, float* weight)
{
	unsigned int wrapID = (threadIdx.x + blockIdx.x * 32) / 32, wrapLane = (threadIdx.x + blockIdx.x * 32) % 32, wrap_y = threadIdx.y;
	unsigned int ix = wrapID * 31 + wrapLane, ix_GM = wrapID * 32 + wrapLane;
	unsigned int iy = blockIdx.y;
	unsigned int iz = blockIdx.z;
	unsigned int col_per_blocky = IMAGE_SIZE_Y / GRID_Y;
	unsigned int col_per_thready = col_per_blocky / BLOCK_Y;
	unsigned int col_per_blockz = depth / (GRID_Z);                                   				////####////
	unsigned int m = (SS - 1) / 2, n = (NS - 1) / 2;
    unsigned int W = (IMAGE_SIZE_X+SS+NS-2);
	unsigned int H = (IMAGE_SIZE_Y+SS+NS-2);


	float g1 = 0;
	float g3 = 0;
	float g5 = 0;
	float gmid = 0;
	float gmid3 = 0;
	float result = 0;
	float result3 = 0; 

	for (int itra_z = 0; itra_z < col_per_blockz; itra_z+=2) {
		//for (int itra_y = 0; itra_y < col_per_thready; itra_y++) {
		for (int i = 0; i < SS; i++) {
			for (int j = 0; j < SS; j++) {
				for (int k = 0; k < SS; k++) {



						for (int jj = 0; jj < NS; jj++) {
							g1 = data[m + ix + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							
							g1 = data[m + ix + 2 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 2 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);							
			
							g1 = data[m + ix + 4 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							
							g1 = data[m + ix + 6 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
						
							g1 = data[m + ix + 8 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							
							g1 = data[m + ix + 10 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 10 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							
							g1 = data[m + ix + 12 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
				
							g1 = data[m + ix + 14 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 14 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							
							g1 = data[m + ix + 16 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 16 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1;
						}




					for (int ii = 1; ii < NS; ii++) {
						for (int jj = 0; jj < NS; jj++) {
						
							g1 = data[m + ix + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							
							
							g1 = data[m + ix + 2 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 2 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);							
			
							g1 = data[m + ix + 4 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);	
							
							g1 = data[m + ix + 6 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);	
							
							g1 = data[m + ix + 8 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);	
					
							g1 = data[m + ix + 10 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 10 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);	
										
							g1 = data[m + ix + 12 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);				
							
							g1 = data[m + ix + 14 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 14 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);		
							
							g1 = data[m + ix + 16 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 16 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1;
							g5 += g1;						
						}		
					}


						for (int jj = 0; jj < NS; jj++) {
						
							g1 = data[m + ix + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							
							g1 = data[m + ix + 2 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 2 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);							
			
							g1 = data[m + ix + 4 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);

							g1 = data[m + ix + 6 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
					
							g1 = data[m + ix + 8 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							
							g1 = data[m + ix + 10 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 10 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
											
							g1 = data[m + ix + 12 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);		
							
							g1 = data[m + ix + 14 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 14 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);		
							
							g1 = data[m + ix + 16 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 16 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1;							
						}

						
							
					
					weight[ix_GM + k * (int)gridDim.x * 32 + j * SS * (int)gridDim.x * 32 + i * SS * SS * (int)gridDim.x * 32 + (iy * BLOCK_Y + wrap_y) * (int)gridDim.x * 32 * SS * SS * SS + iz * (int)gridDim.x * 32 * (GRID_Y * BLOCK_Y) * SS * SS * SS] = g3;			////#
					g3 = g3 / (float)(NS * NS * NS);
					g3 = exp(-g3 / (PARAMETER * PARAMETER));
					gmid += g3;
					result += g3 * data[ix + n + k + (j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + n + itra_z + iz * col_per_blockz) * W * H];
					g3 = 0;


					g5 = g5 / (float)(NS * NS * NS);
					g5 = exp(-g5 / (PARAMETER * PARAMETER));
					gmid3 += g5;
					result3 += g5 * data[ix + n + k + (j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + 1 + n + itra_z + iz * col_per_blockz) * W * H];
					g5 = 0;					
					
				}
			}
		}
		//}
		if (wrapLane < 31 && ix < IMAGE_SIZE_X)out_image[ix + (iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result / gmid;
		if (wrapLane < 31 && ix < IMAGE_SIZE_X)out_image[ix + (iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + 1 + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result3 / gmid3;
		result = 0;
		gmid = 0;
		result3 = 0;
		gmid3 = 0;		

		for (int itra_y = 1; itra_y < col_per_thready; itra_y++)
		{
			for (int i = 0; i < SS; i++) {
				for (int j = 0; j < SS; j++) {
					for (int k = 0; k < SS; k++) {


							
						g3 = weight[ix_GM + k * (int)gridDim.x * 32 + j * SS * (int)gridDim.x * 32 + i * SS * SS * (int)gridDim.x * 32 + (iy * BLOCK_Y + wrap_y) * (int)gridDim.x * 32 * SS * SS * SS + iz * (int)gridDim.x * 32 * (GRID_Y * BLOCK_Y) * SS * SS * SS];
						g5 = g3;


							
							g1 = data[m + ix + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							
							g1 = data[m + ix + 2 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 2 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3-(g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));	
							
							g1 = data[m + ix + 4 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3-(g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));	
							
							g1 = data[m + ix + 6 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3-(g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));		
					
							g1 = data[m + ix + 8 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3-(g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));													

							g1 = data[m + ix + 10 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 10 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3-(g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));	

							g1 = data[m + ix + 12 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3-(g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							
							g1 = data[m + ix + 14 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 14 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3-(g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
					
							g1 = data[m + ix + 16 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 16 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3-g1;



							g1 = data[m + ix + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							
							g1 = data[m + ix + 2 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 2 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));							
						
							g1 = data[m + ix + 4 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));

							g1 = data[m + ix + 6 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));	
							
							g1 = data[m + ix + 8 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));	
					
							g1 = data[m + ix + 10 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 10 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));						
	
							g1 = data[m + ix + 12 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));	

							g1 = data[m + ix + 14 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 14 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));	
							
							g1 = data[m + ix + 16 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 16 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + g1;							
							
							
						for (int ii = 1; ii < NS; ii++) {

							g1 = data[m + ix + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));

							g1 = data[m + ix + 2 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 2 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));

							g1 = data[m + ix + 4 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));

							g1 = data[m + ix + 6 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							
							g1 = data[m + ix + 8 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
					
							g1 = data[m + ix + 10 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 10 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));	
							
							g1 = data[m + ix + 12 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));						

							g1 = data[m + ix + 14 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 14 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));		
						
							g1 = data[m + ix + 16 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 16 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - g1;
							g5 = g5 - g1;


							g1 = data[m + ix + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							
							g1 = data[m + ix + 2 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 2 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));		
							
							g1 = data[m + ix + 4 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
												
							g1 = data[m + ix + 6 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							
							g1 = data[m + ix + 8 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));

							g1 = data[m + ix + 10 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 10 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));

							g1 = data[m + ix + 12 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							
							g1 = data[m + ix + 14 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 14 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							
							g1 = data[m + ix + 16 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 16 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + g1;
							g5 = g5 + g1;
						

						}
						
							g1 = data[m + ix + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));

							g1 = data[m + ix + 2 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 2 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));

							g1 = data[m + ix + 4 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));

							g1 = data[m + ix + 6 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
						
							g1 = data[m + ix + 8 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							
							g1 = data[m + ix + 10 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 10 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							
							g1 = data[m + ix + 12 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
					
							g1 = data[m + ix + 14 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 14 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));

							g1 = data[m + ix + 16 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 16 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - g1;



							g1 = data[m + ix + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
						
							g1 = data[m + ix + 2 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 2 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));						
						
							g1 = data[m + ix + 4 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));												

							g1 = data[m + ix + 6 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));							
							  
							g1 = data[m + ix + 8 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							
							g1 = data[m + ix + 10 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 10 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							
							g1 = data[m + ix + 12 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));			
					
							g1 = data[m + ix + 14 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 14 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));	
							
							g1 = data[m + ix + 16 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 16 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + g1;												
							
							weight[ix_GM + k * (int)gridDim.x * 32 + j * SS * (int)gridDim.x * 32 + i * SS * SS * (int)gridDim.x * 32 + (iy * BLOCK_Y + wrap_y) * (int)gridDim.x * 32 * SS * SS * SS + iz * (int)gridDim.x * 32 * (GRID_Y * BLOCK_Y) * SS * SS * SS] = g3;			////#
							
							g3 = g3 / (float)(NS * NS * NS);
							g3 = exp(-g3 / (PARAMETER * PARAMETER));
							gmid += g3;
							result += g3 * data[ix + n + k + (itra_y + j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + n + itra_z + iz * col_per_blockz) * W * H];
							g3 = 0;

							g5 = g5 / (float)(NS * NS * NS);
							g5 = exp(-g5 / (PARAMETER * PARAMETER));
							gmid3 += g5;
							result3 += g5 * data[ix + n + k + (itra_y + j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + 1 + n + itra_z + iz * col_per_blockz) * W * H];
							g5 = 0;						
						
	
					}
				}
			}
			if (wrapLane < 31 && ix < IMAGE_SIZE_X)out_image[ix + (itra_y + iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result / gmid;
			if (wrapLane < 31 && ix < IMAGE_SIZE_X)out_image[ix + (itra_y + iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + 1 + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result3 / gmid3;
			result = 0;
			gmid = 0;
			result3 = 0;
			gmid3 = 0;
		}
	


	}
}


__global__ void NLmeansOnGPU_shift_2(unsigned char* data, float* out_image, float* weight)
{
	unsigned int wrapID = (threadIdx.x + blockIdx.x * 32) / 32, wrapLane = (threadIdx.x + blockIdx.x * 32) % 32, wrap_y = threadIdx.y;
	unsigned int ix = wrapID * 30 + wrapLane, ix_GM = wrapID * 32 + wrapLane;
	unsigned int iy = blockIdx.y;
	unsigned int iz = blockIdx.z;
	unsigned int col_per_blocky = IMAGE_SIZE_Y / GRID_Y;
	unsigned int col_per_thready = col_per_blocky / BLOCK_Y;
	unsigned int col_per_blockz = depth / (GRID_Z);                                   				////####////
	unsigned int m = (SS - 1) / 2, n = (NS - 1) / 2;
    unsigned int W = (IMAGE_SIZE_X+SS+NS-2);
	unsigned int H = (IMAGE_SIZE_Y+SS+NS-2);


	float g1 = 0;
	float g3 = 0;
	float g5 = 0;
	float gmid = 0;
	float gmid3 = 0;
	float result = 0;
	float result3 = 0; 

	for (int itra_z = 0; itra_z < col_per_blockz; itra_z+=2) {
		//for (int itra_y = 0; itra_y < col_per_thready; itra_y++) {
		for (int i = 0; i < SS; i++) {
			for (int j = 0; j < SS; j++) {
				for (int k = 0; k < SS; k++) {



						for (int jj = 0; jj < NS; jj++) {
							g1 = data[m + ix + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							g1 = data[m + ix + 3 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 3 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);			
							
							g1 = data[m + ix + 6 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);	
							
							g1 = data[m + ix + 9 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 9 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							g1 = data[m + ix + 12 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							g1 = data[m + ix + 15 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 15 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
						}




					for (int ii = 1; ii < NS; ii++) {
						for (int jj = 0; jj < NS; jj++) {
						
							g1 = data[m + ix + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							g1 = data[m + ix + 3 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 3 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);	
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);	
							
							g1 = data[m + ix + 6 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);	
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);	
							
							g1 = data[m + ix + 9 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 9 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);	
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);						
															
							g1 = data[m + ix + 12 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);	
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							g1 = data[m + ix + 15 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 15 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);		
						}		
					}


						for (int jj = 0; jj < NS; jj++) {
						
							g1 = data[m + ix + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;							
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							g1 = data[m + ix + 3 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 3 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);	
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);						

							g1 = data[m + ix + 6 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);	
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);					
														
							g1 = data[m + ix + 9 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 9 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);

							g1 = data[m + ix + 12 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							g1 = data[m + ix + 15 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 15 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);		
						}

						
							
					
					weight[ix_GM + k * (int)gridDim.x * 32 + j * SS * (int)gridDim.x * 32 + i * SS * SS * (int)gridDim.x * 32 + (iy * BLOCK_Y + wrap_y) * (int)gridDim.x * 32 * SS * SS * SS + iz * (int)gridDim.x * 32 * (GRID_Y * BLOCK_Y) * SS * SS * SS] = g3;			////#
					g3 = g3 / (float)(NS * NS * NS);
					g3 = exp(-g3 / (PARAMETER * PARAMETER));
					gmid += g3;
					result += g3 * data[ix + n + k + (j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + n + itra_z + iz * col_per_blockz) * W * H];
					g3 = 0;


					g5 = g5 / (float)(NS * NS * NS);
					g5 = exp(-g5 / (PARAMETER * PARAMETER));
					gmid3 += g5;
					result3 += g5 * data[ix + n + k + (j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + 1 + n + itra_z + iz * col_per_blockz) * W * H];
					g5 = 0;					
					
				}
			}
		}
		//}
		if (wrapLane < 30 && ix < IMAGE_SIZE_X)out_image[ix + (iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result / gmid;
		if (wrapLane < 30 && ix < IMAGE_SIZE_X)out_image[ix + (iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + 1 + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result3 / gmid3;
		result = 0;
		gmid = 0;
		result3 = 0;
		gmid3 = 0;		

		for (int itra_y = 1; itra_y < col_per_thready; itra_y++)
		{
			for (int i = 0; i < SS; i++) {
				for (int j = 0; j < SS; j++) {
					for (int k = 0; k < SS; k++) {


							
						g3 = weight[ix_GM + k * (int)gridDim.x * 32 + j * SS * (int)gridDim.x * 32 + i * SS * SS * (int)gridDim.x * 32 + (iy * BLOCK_Y + wrap_y) * (int)gridDim.x * 32 * SS * SS * SS + iz * (int)gridDim.x * 32 * (GRID_Y * BLOCK_Y) * SS * SS * SS];
						g5 = g3;


							
							g1 = data[m + ix + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							
							g1 = data[m + ix + 3 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 3 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3-(g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));	
							g3 = g3- __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							g1 = data[m + ix + 6 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3-(g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));	
							g3 = g3-__shfl_down_sync(0xffffffff, g1, 2, 32);
						
							g1 = data[m + ix + 9 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 9 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3-(g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));	
							g3 = g3-__shfl_down_sync(0xffffffff, g1, 2, 32);
							
							g1 = data[m + ix + 12 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3-(g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));	
							g3 = g3-__shfl_down_sync(0xffffffff, g1, 2, 32);
							
							g1 = data[m + ix + 15 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 15 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3-(g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));			



							g1 = data[m + ix + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							
							g1 = data[m + ix + 3 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 3 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));		
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);					

							g1 = data[m + ix + 6 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));		
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);							

							g1 = data[m + ix + 9 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 9 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));	
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);				

							g1 = data[m + ix + 12 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));	
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							g1 = data[m + ix + 15 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 15 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));				
							
								
							
						for (int ii = 1; ii < NS; ii++) {

							g1 = data[m + ix + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);

							g1 = data[m + ix + 3 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 3 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);
														
							g1 = data[m + ix + 6 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);

							g1 = data[m + ix + 9 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 9 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);


							g1 = data[m + ix + 12 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);

							g1 = data[m + ix + 15 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 15 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							
							
							
							


							g1 = data[m + ix + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							
							g1 = data[m + ix + 3 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 3 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);	

							g1 = data[m + ix + 6 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);
								
							g1 = data[m + ix + 9 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 9 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);

							g1 = data[m + ix + 12 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							g1 = data[m + ix + 15 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 15 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
						}
		
						
							g1 = data[m + ix + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							

							g1 = data[m + ix + 3 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 3 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							g1 = data[m + ix + 6 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);		
							
							g1 = data[m + ix + 9 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 9 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);	
						
							g1 = data[m + ix + 12 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);	
							
							g1 = data[m + ix + 15 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 15 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));										
							


							g1 = data[m + ix + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							
						
							g1 = data[m + ix + 3 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 3 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);		
							
							g1 = data[m + ix + 6 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 6 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							g1 = data[m + ix + 9 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 9 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							g1 = data[m + ix + 12 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							
							g1 = data[m + ix + 15 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 15 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));

											
							  
							weight[ix_GM + k * (int)gridDim.x * 32 + j * SS * (int)gridDim.x * 32 + i * SS * SS * (int)gridDim.x * 32 + (iy * BLOCK_Y + wrap_y) * (int)gridDim.x * 32 * SS * SS * SS + iz * (int)gridDim.x * 32 * (GRID_Y * BLOCK_Y) * SS * SS * SS] = g3;			////#
							
							g3 = g3 / (float)(NS * NS * NS);
							g3 = exp(-g3 / (PARAMETER * PARAMETER));
							gmid += g3;
							result += g3 * data[ix + n + k + (itra_y + j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + n + itra_z + iz * col_per_blockz) * W * H];
							g3 = 0;

							g5 = g5 / (float)(NS * NS * NS);
							g5 = exp(-g5 / (PARAMETER * PARAMETER));
							gmid3 += g5;
							result3 += g5 * data[ix + n + k + (itra_y + j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + 1 + n + itra_z + iz * col_per_blockz) * W * H];
							g5 = 0;						
						
	
					}
				}
			}
			if (wrapLane < 30 && ix < IMAGE_SIZE_X)out_image[ix + (itra_y + iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result / gmid;
			if (wrapLane < 30 && ix < IMAGE_SIZE_X)out_image[ix + (itra_y + iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + 1 + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result3 / gmid3;
			result = 0;
			gmid = 0;
			result3 = 0;
			gmid3 = 0;
		}
	


	}
}

__global__ void NLmeansOnGPU_shift_3(unsigned char* data, float* out_image, float* weight)
{
	unsigned int wrapID = (threadIdx.x + blockIdx.x * 32) / 32, wrapLane = (threadIdx.x + blockIdx.x * 32) % 32, wrap_y = threadIdx.y;
	unsigned int ix = wrapID * 29 + wrapLane, ix_GM = wrapID * 32 + wrapLane;
	unsigned int iy = blockIdx.y;
	unsigned int iz = blockIdx.z;
	unsigned int col_per_blocky = IMAGE_SIZE_Y / GRID_Y;
	unsigned int col_per_thready = col_per_blocky / BLOCK_Y;
	unsigned int col_per_blockz = depth / (GRID_Z);                                   				////####////
	unsigned int m = (SS - 1) / 2, n = (NS - 1) / 2;
    unsigned int W = (IMAGE_SIZE_X+SS+NS-2);
	unsigned int H = (IMAGE_SIZE_Y+SS+NS-2);


	float g1 = 0;
	float g3 = 0;
	float g5 = 0;
	float gmid = 0;
	float gmid3 = 0;
	float result = 0;
	float result3 = 0; 

	for (int itra_z = 0; itra_z < col_per_blockz; itra_z+=2) {
		//for (int itra_y = 0; itra_y < col_per_thready; itra_y++) {
		for (int i = 0; i < SS; i++) {
			for (int j = 0; j < SS; j++) {
				for (int k = 0; k < SS; k++) {



						for (int jj = 0; jj < NS; jj++) {
							g1 = data[m + ix + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 3, 32);
							
							g1 = data[m + ix + 4 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);	
							g3 += __shfl_down_sync(0xffffffff, g1, 3, 32);	
							
							g1 = data[m + ix + 8 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 3, 32);

							g1 = data[m + ix + 12 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 3, 32);	
							
							g1 = data[m + ix + 16 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 16 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1;							
						}




					for (int ii = 1; ii < NS; ii++) {
						for (int jj = 0; jj < NS; jj++) {
						
							g1 = data[m + ix + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 3, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 3, 32);
							
							g1 = data[m + ix + 4 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 3, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);	
							g5 += __shfl_down_sync(0xffffffff, g1, 3, 32);	
							
							g1 = data[m + ix + 8 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 3, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);	
							g5 += __shfl_down_sync(0xffffffff, g1, 3, 32);			
				
							g1 = data[m + ix + 12 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 += __shfl_down_sync(0xffffffff, g1, 3, 32);
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);	
							g5 += __shfl_down_sync(0xffffffff, g1, 3, 32);		
							
							g1 = data[m + ix + 16 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 16 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 += g1;
							g5 += g1;																							
						}		
					}


						for (int jj = 0; jj < NS; jj++) {
						
							g1 = data[m + ix + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 3, 32);
							
							g1 = data[m + ix + 4 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);	
							g5 += __shfl_down_sync(0xffffffff, g1, 3, 32);							
			
							g1 = data[m + ix + 8 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 3, 32);
															
							g1 = data[m + ix + 12 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 += __shfl_down_sync(0xffffffff, g1, 3, 32);
							
							g1 = data[m + ix + 16 + (jj + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 16 + (jj + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 += g1;

						}

						
							
					
					weight[ix_GM + k * (int)gridDim.x * 32 + j * SS * (int)gridDim.x * 32 + i * SS * SS * (int)gridDim.x * 32 + (iy * BLOCK_Y + wrap_y) * (int)gridDim.x * 32 * SS * SS * SS + iz * (int)gridDim.x * 32 * (GRID_Y * BLOCK_Y) * SS * SS * SS] = g3;			////#
					g3 = g3 / (float)(NS * NS * NS);
					g3 = exp(-g3 / (PARAMETER * PARAMETER));
					gmid += g3;
					result += g3 * data[ix + n + k + (j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + n + itra_z + iz * col_per_blockz) * W * H];
					g3 = 0;


					g5 = g5 / (float)(NS * NS * NS);
					g5 = exp(-g5 / (PARAMETER * PARAMETER));
					gmid3 += g5;
					result3 += g5 * data[ix + n + k + (j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + 1 + n + itra_z + iz * col_per_blockz) * W * H];
					g5 = 0;					
					
				}
			}
		}
		//}
		if (wrapLane < 29 && ix < IMAGE_SIZE_X)out_image[ix + (iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result / gmid;
		if (wrapLane < 29 && ix < IMAGE_SIZE_X)out_image[ix + (iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + 1 + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result3 / gmid3;
		result = 0;
		gmid = 0;
		result3 = 0;
		gmid3 = 0;		

		for (int itra_y = 1; itra_y < col_per_thready; itra_y++)
		{
			for (int i = 0; i < SS; i++) {
				for (int j = 0; j < SS; j++) {
					for (int k = 0; k < SS; k++) {


							
						g3 = weight[ix_GM + k * (int)gridDim.x * 32 + j * SS * (int)gridDim.x * 32 + i * SS * SS * (int)gridDim.x * 32 + (iy * BLOCK_Y + wrap_y) * (int)gridDim.x * 32 * SS * SS * SS + iz * (int)gridDim.x * 32 * (GRID_Y * BLOCK_Y) * SS * SS * SS];
						g5 = g3;


							
							g1 = data[m + ix + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 3, 32);
							
							
							g1 = data[m + ix + 4 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 2, 32);	
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 3, 32);
							
							g1 = data[m + ix + 8 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 3, 32);

							g1 = data[m + ix + 12 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 3, 32);
							
							g1 = data[m + ix + 16 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 16 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - g1;




							g1 = data[m + ix + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 3, 32);
							
							
							g1 = data[m + ix + 4 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);	
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 3, 32);						
						
							g1 = data[m + ix + 8 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 3, 32);
							
							g1 = data[m + ix + 12 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + g1 + __shfl_down_sync(0xffffffff, g1, 1, 32);
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 3, 32);
						
							g1 = data[m + ix + 16 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 16 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + g1;
			
			
							
								
							
						for (int ii = 1; ii < NS; ii++) {

							g1 = data[m + ix + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 3, 32);
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 3, 32);

							g1 = data[m + ix + 4 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 3, 32);
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 3, 32);

							g1 = data[m + ix + 8 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 3, 32);
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 3, 32);
							
							g1 = data[m + ix + 12 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 = g3 - __shfl_down_sync(0xffffffff, g1, 3, 32);
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 3, 32);
							
							g1 = data[m + ix + 16 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 16 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 - g1;
							g5 = g5 - g1;

							


							g1 = data[m + ix + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 3, 32);
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 3, 32);
							
							
							g1 = data[m + ix + 4 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 3, 32);
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 3, 32);
				
							g1 = data[m + ix + 8 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 3, 32);
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);	
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 3, 32);
						
							g1 = data[m + ix + 12 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g3 = g3 + __shfl_down_sync(0xffffffff, g1, 3, 32);
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);	
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 3, 32);
						
							g1 = data[m + ix + 16 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 16 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (ii + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g3 = g3 + g1;
							g5 = g5 + g1;

							
						}
		
						
							g1 = data[m + ix + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 3, 32);
							

							g1 = data[m + ix + 4 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 3, 32);
							
							g1 = data[m + ix + 8 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 3, 32);
							
							g1 = data[m + ix + 12 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 - __shfl_down_sync(0xffffffff, g1, 3, 32);

							g1 = data[m + ix + 16 + (itra_y - 1 + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 16 + (itra_y - 1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 - g1;						
							


							g1 = data[m + ix + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 3, 32);
							
						
							g1 = data[m + ix + 4 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 4 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);		
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 3, 32);				

							g1 = data[m + ix + 8 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 8 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 3, 32);	
							
							g1 = data[m + ix + 12 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 12 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + (g1 + __shfl_down_sync(0xffffffff, g1, 1, 32));
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 2, 32);
							g5 = g5 + __shfl_down_sync(0xffffffff, g1, 3, 32);
							
							g1 = data[m + ix + 16 + (itra_y + NS-1  + m + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + m + itra_z + iz * col_per_blockz) * W * H] - data[k + ix + 16 + (itra_y + NS-1 + j + iy * col_per_blocky + wrap_y * col_per_thready) * W + (NS + i + itra_z + iz * col_per_blockz) * W * H];
							g1 = g1 * g1;
							g5 = g5 + g1;


							  
							weight[ix_GM + k * (int)gridDim.x * 32 + j * SS * (int)gridDim.x * 32 + i * SS * SS * (int)gridDim.x * 32 + (iy * BLOCK_Y + wrap_y) * (int)gridDim.x * 32 * SS * SS * SS + iz * (int)gridDim.x * 32 * (GRID_Y * BLOCK_Y) * SS * SS * SS] = g3;			////#
							
							g3 = g3 / (float)(NS * NS * NS);
							g3 = exp(-g3 / (PARAMETER * PARAMETER));
							gmid += g3;
							result += g3 * data[ix + n + k + (itra_y + j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + n + itra_z + iz * col_per_blockz) * W * H];
							g3 = 0;

							g5 = g5 / (float)(NS * NS * NS);
							g5 = exp(-g5 / (PARAMETER * PARAMETER));
							gmid3 += g5;
							result3 += g5 * data[ix + n + k + (itra_y + j + n + iy * col_per_blocky + wrap_y * col_per_thready) * W + (i + 1 + n + itra_z + iz * col_per_blockz) * W * H];
							g5 = 0;						
						
	
					}
				}
			}
			if (wrapLane < 29 && ix < IMAGE_SIZE_X)out_image[ix + (itra_y + iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result / gmid;
			if (wrapLane < 29 && ix < IMAGE_SIZE_X)out_image[ix + (itra_y + iy * col_per_blocky + wrap_y * col_per_thready) * IMAGE_SIZE_X + (itra_z + 1 + iz * col_per_blockz) * IMAGE_SIZE_X * IMAGE_SIZE_Y] = result3 / gmid3;
			result = 0;
			gmid = 0;
			result3 = 0;
			gmid3 = 0;
		}
	


	}
}
#endif







void NLMeansProcessor::NL_Means(unsigned char* GPU_input, float* GPU_result)
{
	unsigned char* data = nullptr;
	float* out_image = nullptr;
	float* weight = nullptr;
    unsigned int W = (IMAGE_SIZE_X+SS+NS-2);
	unsigned int H = (IMAGE_SIZE_Y+SS+NS-2);
	unsigned int dim_after_paddig = (depth+SS+NS-2);
		
	cudaMallocCheck((void**)&data, W * H * dim_after_paddig * sizeof(unsigned char));	
	cudaMallocCheck((void**)&out_image, IMAGE_SIZE_X * IMAGE_SIZE_Y * depth * sizeof(float));
	cudaMallocCheck((void**)&weight, (IMAGE_SIZE_X + 32 - NS) / (33 - NS) * 32 * (GRID_Y * BLOCK_Y) * (GRID_Z) * SS * SS * SS * sizeof(float)); ////#
	
	cudaMemcpyCheck(data, GPU_input, W * H * dim_after_paddig * sizeof(unsigned char), cudaMemcpyHostToDevice);

	unsigned char RUNS = 10;
	
	dim3 block(BLOCK_X, BLOCK_Y, BLOCK_Z);

///*////////////////////shift_without_optimization
	dim3 grid22;
	grid22.x = (IMAGE_SIZE_X + 32 - NS) / (33 - NS);
	grid22.y = GRID_Y;
	grid22.z = GRID_Z;
	for (int i = 0; i < RUNS; i++)
	{
		shuffle_without_optimization << <grid22, block >> > (data, out_image, weight);
	}
///////////////////*/

///*//////////////////NLmeansOnGPU_shift_1
	dim3 grid_1;
	grid_1.x = (IMAGE_SIZE_X + 30) / 31;
	grid_1.y = GRID_Y;
	grid_1.z = GRID_Z;
	for (int i = 0; i < RUNS; i++)
	{
		NLmeansOnGPU_shift_1 << <grid_1, block >> > (data, out_image, weight);
	}
///////////////////*/

///*//////////////////NLmeansOnGPU_shift_2
	dim3 grid_2;
	grid_2.x = (IMAGE_SIZE_X + 29) / 30;
	grid_2.y = GRID_Y;
	grid_2.z = GRID_Z;
	for (int i = 0; i < RUNS; i++)
	{
		NLmeansOnGPU_shift_2 << <grid_2, block >> > (data, out_image, weight);
	}
///////////////////*/

///*////////////////////NLmeansOnGPU_shift_3
	dim3 grid_3;
	grid_3.x = (IMAGE_SIZE_X + 28) / 29;
	grid_3.y = GRID_Y;
	grid_3.z = GRID_Z;
	for (int i = 0; i < RUNS; i++)
	{
		NLmeansOnGPU_shift_3 << <grid_3, block >> > (data, out_image, weight);
	}
/////////////////*/



	cudaMemcpyCheck(GPU_result, out_image, IMAGE_SIZE_X * IMAGE_SIZE_Y * depth * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFreeCheck(data);
	cudaFreeCheck(out_image);
	cudaFreeCheck(weight);

	cudaDeviceSynchronize();
	cudaProfilerStop();
}









