#ifndef SKEPU_PRECOMPILED
#define SKEPU_PRECOMPILED
#endif
#ifndef SKEPU_OPENMP
#define SKEPU_OPENMP
#endif
#ifndef SKEPU_OPENCL
#define SKEPU_OPENCL
#endif
/**************************
** TDDD56 Lab 3
***************************
** Author:
** August Ernstsson
**************************/

#include <stdio.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <time.h>
#include <iterator>

#include <skepu2.hpp>

#include "support.h"


unsigned char median_kernel(int ox, int oy, size_t stride, const unsigned char *image, size_t elemPerPx)
{
	//float scaling = 1.0 / ((ox/elemPerPx*2+1)*(oy*2+1));
	float res = 0;
	int size = (2*ox/elemPerPx+1)*(2*oy+1);
	unsigned char element[size];
	int i = 0;
	for (int y = -oy; y <= oy; ++y)
		for (int x = -ox; x <= ox; x += elemPerPx){
			element[i] = image[y*(int)stride*x];
			i++;
			}

	unsigned char element2[(2*ox/elemPerPx+1)*(2*oy+1)];

	unsigned char qselect(unsigned char *v, int len, int k)
	{
	#	define SWAP(a, b) { tmp = v[a]; v[a] = v[b]; v[b] = tmp; }
		int i, st, tmp;

		for (st = i = 0; i < len - 1; i++) {
			if (v[i] > v[len-1]) continue;
			SWAP(i, st);
			st++;
		}

		SWAP(len-1, st);

		return k == st	?v[st]
				:st > k	? qselect(v, st, k)
					: qselect(v + st, len - st, k - st);
	}

	res = qselect(element, size, (2*ox/elemPerPx+1)*oy+1);
	//return res * scaling;
	return res;
}
struct skepu2_userfunction_calculateMedian_median_kernel
{
constexpr static size_t totalArity = 5;
constexpr static bool indexed = 0;
using ElwiseArgs = std::tuple<int, int, size_t, const unsigned char *>;
using ContainerArgs = std::tuple<>;
using UniformArgs = std::tuple<size_t>;
constexpr static skepu2::AccessMode anyAccessMode[] = {
};

using Ret = unsigned char;

constexpr static bool prefersMatrix = 0;

#define SKEPU_USING_BACKEND_OMP 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block)
#define VARIANT_OPENMP(block) block
#define VARIANT_CUDA(block)
static inline SKEPU_ATTRIBUTE_FORCE_INLINE unsigned char OMP(int ox, int oy, size_t stride, const unsigned char *image, size_t elemPerPx)
{
	//float scaling = 1.0 / ((ox/elemPerPx*2+1)*(oy*2+1));
	float res = 0;
	int size = (2*ox/elemPerPx+1)*(2*oy+1);
	unsigned char element[size];
	int i = 0;
	for (int y = -oy; y <= oy; ++y)
		for (int x = -ox; x <= ox; x += elemPerPx){
			element[i] = image[y*(int)stride*x];
			i++;
			}

	unsigned char element2[(2*ox/elemPerPx+1)*(2*oy+1)];

	unsigned char qselect(unsigned char *v, int len, int k)
	{
	#	define SWAP(a, b) { tmp = v[a]; v[a] = v[b]; v[b] = tmp; }
		int i, st, tmp;

		for (st = i = 0; i < len - 1; i++) {
			if (v[i] > v[len-1]) continue;
			SWAP(i, st);
			st++;
		}

		SWAP(len-1, st);

		return k == st	?v[st]
				:st > k	? qselect(v, st, k)
					: qselect(v + st, len - st, k - st);
	}

	res = qselect(element, size, (2*ox/elemPerPx+1)*oy+1);
	//return res * scaling;
	return res;
}
#undef SKEPU_USING_BACKEND_OMP

#define SKEPU_USING_BACKEND_CPU 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block) block
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block) block
static inline SKEPU_ATTRIBUTE_FORCE_INLINE unsigned char CPU(int ox, int oy, size_t stride, const unsigned char *image, size_t elemPerPx)
{
	//float scaling = 1.0 / ((ox/elemPerPx*2+1)*(oy*2+1));
	float res = 0;
	int size = (2*ox/elemPerPx+1)*(2*oy+1);
	unsigned char element[size];
	int i = 0;
	for (int y = -oy; y <= oy; ++y)
		for (int x = -ox; x <= ox; x += elemPerPx){
			element[i] = image[y*(int)stride*x];
			i++;
			}

	unsigned char element2[(2*ox/elemPerPx+1)*(2*oy+1)];

	unsigned char qselect(unsigned char *v, int len, int k)
	{
	#	define SWAP(a, b) { tmp = v[a]; v[a] = v[b]; v[b] = tmp; }
		int i, st, tmp;

		for (st = i = 0; i < len - 1; i++) {
			if (v[i] > v[len-1]) continue;
			SWAP(i, st);
			st++;
		}

		SWAP(len-1, st);

		return k == st	?v[st]
				:st > k	? qselect(v, st, k)
					: qselect(v + st, len - st, k - st);
	}

	res = qselect(element, size, (2*ox/elemPerPx+1)*oy+1);
	//return res * scaling;
	return res;
}
#undef SKEPU_USING_BACKEND_CPU
};

constexpr skepu2::AccessMode skepu2_userfunction_calculateMedian_median_kernel::anyAccessMode[];






#include "median_precompiled_Overlap2DKernel_median_kernel_cl_source.inl"
int main(int argc, char* argv[])
{
	LodePNGColorType colorType = LCT_RGB;

	if (argc < 5)
	{
		std::cout << "Usage: " << argv[0] << "input output radius [backend]\n";
		exit(1);
	}

	std::string inputFileName = argv[1];
	std::string outputFileName = argv[2];
	const int radius = atoi(argv[3]);
	auto spec = skepu2::BackendSpec{skepu2::Backend::typeFromString(argv[4])};

	// Create the full path for writing the image.
	std::stringstream ss;
	ss << (2 * radius + 1) << "x" << (2 * radius + 1);
	std::string outputFileNamePad = outputFileName + ss.str() + "-median.png";

	// Read the padded image into a matrix. Create the output matrix without padding.
	ImageInfo imageInfo;
	skepu2::Matrix<unsigned char> inputMatrix = ReadAndPadPngFileToMatrix(inputFileName, radius, colorType, imageInfo);
	skepu2::Matrix<unsigned char> outputMatrix(imageInfo.height, imageInfo.width * imageInfo.elementsPerPixel, 120);

	// Skeleton instance
	skepu2::backend::MapOverlap2D<skepu2_userfunction_calculateMedian_median_kernel, bool, CLWrapperClass_median_precompiled_Overlap2DKernel_median_kernel> calculateMedian(false);
	calculateMedian.setBackend(spec);
	calculateMedian.setOverlap(radius, radius  * imageInfo.elementsPerPixel);

	auto timeTaken = skepu2::benchmark::measureExecTime([&]
	{
		calculateMedian(outputMatrix, inputMatrix, imageInfo.elementsPerPixel);
	});

	WritePngFileMatrix(outputMatrix, outputFileNamePad, colorType, imageInfo);

	std::cout << "Time: " << (timeTaken.count() / 10E6) << "\n";

	return 0;
}
