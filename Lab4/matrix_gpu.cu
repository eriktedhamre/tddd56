// Matrix addition, CPU version
// gcc matrix_cpu.c -o matrix_cpu -std=c99

#include <stdio.h>
#include "milli.h"

__global__
void add_matrix(float *a, float *b, float *c, int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	//printf("%d \n",idx * N + idy);
	c[idx * N + idy] = a[idx * N + idy] + b[idx * N + idy];
	//c[idy * N + idx] = a[idy * N + idx] + b[idy * N + idx];
/*
// Bad pattern to access after each other since each thread has to read memory for each access and threads can not use one read for multiple data points
	for (int i = 0; i < N; i++) {
			c[idx * N + i] = a[idx * N + i] + b[idx * N + i];
			//c[idx + N * i] = a[idx + N * i] + b[idx + N * i];
	}
*/
}
/*
cudaMalloc( (void**)&cd, size );
dim3 dimBlock( blocksize, 1 );
dim3 dimGrid( 1, 1 );
simple<<<dimGrid, dimBlock>>>(cd);
cudaThreadSynchronize();
cudaMemcpy( c, cd, size, cudaMemcpyDeviceToHost );
cudaFree( cd );
*/

void add_matrix_cpu(float *a, float *b, float *c, int N)
{
	int index;

	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
		{
			index = i + j*N;
			c[index] = a[index] + b[index];
		}
}

int main()
{
	const int N = 6144 ;
	const int BLOCKSIZE = 32;
	const int GRIDSIZE = 192;
	long size = (long) N*N*sizeof(float);
	float * a_h,* a_d,* b_h,* b_d,* c_d,* c_h;
	a_h = (float *)malloc(size);
	cudaMalloc((void **) &a_d, size);
	b_h = (float *)malloc(size);
	cudaMalloc((void **) &b_d, size);
	c_h = (float *)malloc(size);
	cudaMalloc((void **) &c_d, size);

	for (int i = 0; i < N; i++){
		for (int j = 0; j < N; j++)
		{
			a_h[i+j*N] = 10 + i;
			b_h[i+j*N] = (float)j / N;
			c_h[i+j*N] = 0;
		}
	}
	float GPUTime;
		cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
		cudaMemcpy(b_d, b_h, size, cudaMemcpyHostToDevice);
		cudaMemcpy(c_d, c_h, size, cudaMemcpyHostToDevice);

  cudaEvent_t start;
		cudaEvent_t stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		dim3 dimBlock( BLOCKSIZE, BLOCKSIZE );
		dim3 dimGrid( GRIDSIZE, GRIDSIZE );
		cudaEventRecord(start, 0);
		add_matrix<<<dimGrid, dimBlock>>>(a_d, b_d, c_d, N);
		cudaEventRecord(stop, 0);
		cudaThreadSynchronize();
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);

		cudaEventElapsedTime(&GPUTime, start, stop);

		cudaError_t err = cudaGetLastError();
	  	if (err != cudaSuccess)
	    	printf("Error: %s\n", cudaGetErrorString(err));
		cudaMemcpy(c_h, c_d, size, cudaMemcpyDeviceToHost);

		float * a, *b, *c;
		a = (float *)malloc(size);
		b = (float *)malloc(size);
		c = (float *)malloc(size);

		for (int i = 0; i < N; i++){
			for (int j = 0; j < N; j++)
			{
				a[i+j*N] = 10 + i;
				b[i+j*N] = (float)j /N;
				c[i+j*N] = 0;
			}
		}
		int CPUTime;
		GetMilliseconds();
		add_matrix_cpu(a, b, c, N);
		CPUTime = GetMilliseconds();



	for (int i = 0; i < N; i+=1000)
	{
		for (int j = 0; j < N; j+=1000)
		{
			printf("%0.2f ", c_h[i+j*N]);
			printf("%0.2f ", c[i+j*N]);
		}
		printf("\n");
	}

	printf("GPU time = %f ms\n", GPUTime);
	printf("CPU time = %d ms\n", CPUTime);

}
