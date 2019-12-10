// Lab 5, image filters with CUDA.

// Compile with a command-line similar to Lab 4:
// nvcc filter.cu -c -arch=sm_30 -o filter.o
// g++ filter.o milli.c readppm.c -lGL -lm -lcuda -lcudart -L/usr/local/cuda/lib -lglut -o filter
// or (multicore lab)
// nvcc filter.cu -c -arch=sm_20 -o filter.o
// g++ filter.o milli.c readppm.c -lGL -lm -lcuda -L/usr/local/cuda/lib64 -lcudart -lglut -o filter

// 2017-11-27: Early pre-release, dubbed "beta".
// 2017-12-03: First official version! Brand new lab 5 based on the old lab 6.
// Better variable names, better prepared for some lab tasks. More changes may come
// but I call this version 1.0b2.
// 2017-12-04: Two fixes: Added command-lines (above), fixed a bug in computeImages
// that allocated too much memory. b3
// 2017-12-04: More fixes: Tightened up the kernel with edge clamping.
// Less code, nicer result (no borders). Cleaned up some messed up X and Y. b4

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#ifdef __APPLE__
  #include <GLUT/glut.h>
  #include <OpenGL/gl.h>
#else
  #include <GL/glut.h>
#endif
#include "readppm.h"
#include "milli.h"

// Use these for setting shared memory size.
#define maxKernelSizeX 10
#define maxKernelSizeY 10


__global__ void filter(unsigned char *image, unsigned char *out, const unsigned int imagesizex, const unsigned int imagesizey, const int kernelsizex, const int kernelsizey)
{
  // map from blockIdx to pixel position
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

  int dy, dx;
  unsigned int sumx, sumy, sumz;

  int divby = (2*kernelsizex+1)*(2*kernelsizey+1); // Works for box filters only!

	if (x < imagesizex && y < imagesizey) // If inside image
	{
// Filter kernel (simple box filter)
	sumx=0;sumy=0;sumz=0;
	for(dy=-kernelsizey;dy<=kernelsizey;dy++)
		for(dx=-kernelsizex;dx<=kernelsizex;dx++)
		{
			// Use max and min to avoid branching!
			int yy = min(max(y+dy, 0), imagesizey-1);
			int xx = min(max(x+dx, 0), imagesizex-1);

			sumx += image[((yy)*imagesizex+(xx))*3+0];
			sumy += image[((yy)*imagesizex+(xx))*3+1];
			sumz += image[((yy)*imagesizex+(xx))*3+2];
		}
	out[(y*imagesizex+x)*3+0] = sumx/divby;
	out[(y*imagesizex+x)*3+1] = sumy/divby;
	out[(y*imagesizex+x)*3+2] = sumz/divby;
	}
}

__global__ void filterOptimized(unsigned char *image, unsigned char *out, const unsigned int imagesizex, const unsigned int imagesizey, const int kernelsizex, const int kernelsizey)
{
  // map from blockIdx to pixel position
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

  int dy, dx;
  unsigned int sumx, sumy, sumz;

  int divby = (2*kernelsizex+1)*(2*kernelsizey+1); // Works for box filters only!
  extern __shared__ unsigned char sData[]; //size = (2*kernelsizex+1)*(2*kernelsizey+1)*3*sizeof(unsigned char)
// use threadIdx.x and .y insetad of x and y since we are in a block that is executed on a MP?
/*
  sData[(y*imagesizex+x)*3+0] = image[(y*imagesizex+x)*3+0];
  sData[(y*imagesizex+x)*3+1] = image[(y*imagesizex+x)*3+1];
  sData[(y*imagesizex+x)*3+2] = image[(y*imagesizex+x)*3+2];

  sData[((2*kernelsizex+1)*kernelsizey+1)*(y*imagesizex+x)*3+0] = image[(y*imagesizex+x)*3+0];
  sData[((2*kernelsizex+1)*kernelsizey+1)*(y*imagesizex+x)*3+1] = image[(y*imagesizex+x)*3+1];
  sData[((2*kernelsizex+1)*kernelsizey+1)*(y*imagesizex+x)*3+2] = image[(y*imagesizex+x)*3+2];
*/

// want to get x-threadidx and y-threadidy to get the values outside of the used grid
  int imy = min(max(y-2*threadIdx.y,0), imagesizey-1);
  int imx = min(max(x-2*threadIdx.x,0), imagesizex-1);
  sData[(threadIdx.y*kernelsizex+threadIdx.x)*3+0] = image[(imy*imagesizex+x-2*imx)*3+0];
  sData[(threadIdx.y*kernelsizex+threadIdx.x)*3+1] = image[(imy*imagesizex+x-2*imx)*3+1];
  sData[(threadIdx.y*kernelsizex+threadIdx.x)*3+2] = image[(imy*imagesizex+x-2*imx)*3+2];

  imy = min(max(y*imagesizex, 0), imagesizey-1);
  imx = min(max(x,0), imagesizex-1);
  sData[((2*kernelsizex+1)*kernelsizey+1)+(threadIdx.y*kernelsizex+threadIdx.x)*3+0] = image[(imy*imagesizex+imx)*3+0];
  sData[((2*kernelsizex+1)*kernelsizey+1)+(threadIdx.y*kernelsizex+threadIdx.x)*3+1] = image[(imy*imagesizex+imx)*3+1];
  sData[((2*kernelsizex+1)*kernelsizey+1)+(threadIdx.y*kernelsizex+threadIdx.x)*3+2] = image[(imy*imagesizex+imx)*3+2];

  __syncthreads();
  if (x < imagesizex && y < imagesizey) // If inside image
	{
// Filter kernel (simple box filter)
	sumx=0;sumy=0;sumz=0;
	for(dy=0;dy<=2*kernelsizey+1;dy++)
		for(dx=0;dx<=2*kernelsizex+1;dx++)
		{
			// Use max and min to avoid branching!
			int yy = min(max(threadIdx.y+dy, 0), 2*kernelsizey);
			int xx = min(max(threadIdx.x+dx, 0), 2*kernelsizex);

			sumx += sData[((yy)*imagesizex+(xx))*3+0];
			sumy += sData[((yy)*imagesizex+(xx))*3+1];
			sumz += sData[((yy)*imagesizex+(xx))*3+2];
		}
	out[(y*imagesizex+x)*3+0] = sumx/divby;
	out[(y*imagesizex+x)*3+1] = sumy/divby;
	out[(y*imagesizex+x)*3+2] = sumz/divby;
    __syncthreads();
	}
}


// Global variables for image data

unsigned char *image, *pixels, *dev_bitmap, *dev_input;
unsigned int imagesizey, imagesizex; // Image size

////////////////////////////////////////////////////////////////////////////////
// main computation function
////////////////////////////////////////////////////////////////////////////////
void computeImages(int kernelsizex, int kernelsizey)
{
	if (kernelsizex > maxKernelSizeX || kernelsizey > maxKernelSizeY)
	{
		printf("Kernel size out of bounds!\n");
		return;
	}

	pixels = (unsigned char *) malloc(imagesizex*imagesizey*3);
	cudaMalloc( (void**)&dev_input, imagesizex*imagesizey*3);
	cudaMemcpy( dev_input, image, imagesizey*imagesizex*3, cudaMemcpyHostToDevice );
	cudaMalloc( (void**)&dev_bitmap, imagesizex*imagesizey*3);
	dim3 grid(imagesizex,imagesizey);
  filter<<<grid,1>>>(dev_input, dev_bitmap, imagesizex, imagesizey, kernelsizex, kernelsizey); // Awful load balance
	cudaThreadSynchronize();
//	Check for errors!
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));
	cudaMemcpy( pixels, dev_bitmap, imagesizey*imagesizex*3, cudaMemcpyDeviceToHost );
  unsigned char* sharedData;
  cudaMalloc((void**)&sharedData, (kernelsizex*2+1)*(kernelsizey*2+1)*3*sizeof(unsigned char));
  filterOptimized<<<grid, 1, (kernelsizex*2+1)*(kernelsizey*2+1)*3*sizeof(unsigned char)>>>(dev_input, dev_bitmap, imagesizex, imagesizey, kernelsizex, kernelsizey);
  cudaThreadSynchronize();
  //	Check for errors!
    err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Optimized version Error: %s\n", cudaGetErrorString(err));
	cudaMemcpy( pixels, dev_bitmap, imagesizey*imagesizex*3, cudaMemcpyDeviceToHost );
  cudaFree( sharedData );
  cudaFree( dev_bitmap );
	cudaFree( dev_input );
}

// Display images
void Draw()
{
// Dump the whole picture onto the screen.
	glClearColor( 0.0, 0.0, 0.0, 1.0 );
	glClear( GL_COLOR_BUFFER_BIT );

	if (imagesizey >= imagesizex)
	{ // Not wide - probably square. Original left, result right.
		glRasterPos2f(-1, -1);
		glDrawPixels( imagesizex, imagesizey, GL_RGB, GL_UNSIGNED_BYTE, image );
		glRasterPos2i(0, -1);
		glDrawPixels( imagesizex, imagesizey, GL_RGB, GL_UNSIGNED_BYTE,  pixels);
	}
	else
	{ // Wide image! Original on top, result below.
		glRasterPos2f(-1, -1);
		glDrawPixels( imagesizex, imagesizey, GL_RGB, GL_UNSIGNED_BYTE, pixels );
		glRasterPos2i(-1, 0);
		glDrawPixels( imagesizex, imagesizey, GL_RGB, GL_UNSIGNED_BYTE, image );
	}
	glFlush();
}

// Main program, inits
int main( int argc, char** argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode( GLUT_SINGLE | GLUT_RGBA );

	if (argc > 1)
		image = readppm(argv[1], (int *)&imagesizex, (int *)&imagesizey);
	else
		image = readppm((char *)"maskros512.ppm", (int *)&imagesizex, (int *)&imagesizey);

	if (imagesizey >= imagesizex)
		glutInitWindowSize( imagesizex*2, imagesizey );
	else
		glutInitWindowSize( imagesizex, imagesizey*2 );
	glutCreateWindow("Lab 5");
	glutDisplayFunc(Draw);

	ResetMilli();

	computeImages(2, 2);

// You can save the result to a file like this:
//	writeppm("out.ppm", imagesizey, imagesizex, pixels);

	glutMainLoop();
	return 0;
}
