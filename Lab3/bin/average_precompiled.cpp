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

unsigned char average_kernel(int ox, int oy, size_t stride, const unsigned char *m, size_t elemPerPx)
{
	float scaling = 1.0 / ((ox/elemPerPx*2+1)*(oy*2+1));
	float res = 0;
	for (int y = -oy; y <= oy; ++y)
		for (int x = -ox; x <= ox; x += elemPerPx){
			//printf("%d\n", y*stride+x);
			res += m[y*(int)stride+x];
		}
	return res * scaling;
}
struct skepu2_userfunction_conv_average_kernel
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
static inline SKEPU_ATTRIBUTE_FORCE_INLINE unsigned char OMP(int ox, int oy, size_t stride, const unsigned char *m, size_t elemPerPx)
{
	float scaling = 1.0 / ((ox/elemPerPx*2+1)*(oy*2+1));
	float res = 0;
	for (int y = -oy; y <= oy; ++y)
		for (int x = -ox; x <= ox; x += elemPerPx){
			//printf("%d\n", y*stride+x);
			res += m[y*(int)stride+x];
		}
	return res * scaling;
}
#undef SKEPU_USING_BACKEND_OMP

#define SKEPU_USING_BACKEND_CPU 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block) block
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block) block
static inline SKEPU_ATTRIBUTE_FORCE_INLINE unsigned char CPU(int ox, int oy, size_t stride, const unsigned char *m, size_t elemPerPx)
{
	float scaling = 1.0 / ((ox/elemPerPx*2+1)*(oy*2+1));
	float res = 0;
	for (int y = -oy; y <= oy; ++y)
		for (int x = -ox; x <= ox; x += elemPerPx){
			//printf("%d\n", y*stride+x);
			res += m[y*(int)stride+x];
		}
	return res * scaling;
}
#undef SKEPU_USING_BACKEND_CPU
};

constexpr skepu2::AccessMode skepu2_userfunction_conv_average_kernel::anyAccessMode[];


unsigned char average_kernel_1d(int o, size_t stride, const unsigned char *m, size_t elemPerPx)
{
	float res = 0;
	int addvalue;
	if(stride == 1){
		addvalue = elemPerPx;
	} else {
		addvalue = 1;
	}
	float scaling = 1.0 /(o/addvalue*2+1);
	for (int i = -o; i <= o; i += addvalue) {
		res += m[i*stride];
	}
	return res * scaling;
}
struct skepu2_userfunction_conv_average_kernel_1d
{
constexpr static size_t totalArity = 4;
constexpr static bool indexed = 0;
using ElwiseArgs = std::tuple<int, size_t, const unsigned char *>;
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
static inline SKEPU_ATTRIBUTE_FORCE_INLINE unsigned char OMP(int o, size_t stride, const unsigned char *m, size_t elemPerPx)
{
	float res = 0;
	int addvalue;
	if(stride == 1){
		addvalue = elemPerPx;
	} else {
		addvalue = 1;
	}
	float scaling = 1.0 /(o/addvalue*2+1);
	for (int i = -o; i <= o; i += addvalue) {
		res += m[i*stride];
	}
	return res * scaling;
}
#undef SKEPU_USING_BACKEND_OMP

#define SKEPU_USING_BACKEND_CPU 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block) block
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block) block
static inline SKEPU_ATTRIBUTE_FORCE_INLINE unsigned char CPU(int o, size_t stride, const unsigned char *m, size_t elemPerPx)
{
	float res = 0;
	int addvalue;
	if(stride == 1){
		addvalue = elemPerPx;
	} else {
		addvalue = 1;
	}
	float scaling = 1.0 /(o/addvalue*2+1);
	for (int i = -o; i <= o; i += addvalue) {
		res += m[i*stride];
	}
	return res * scaling;
}
#undef SKEPU_USING_BACKEND_CPU
};

constexpr skepu2::AccessMode skepu2_userfunction_conv_average_kernel_1d::anyAccessMode[];




unsigned char gaussian_kernel(int o, size_t stride, const unsigned char *m, const skepu2::Vec<float> stencil, size_t elemPerPx)
{
	float res = 0;
	int addvalue;
	if(stride == 1){
		addvalue = elemPerPx;
	} else {
		addvalue = 1;
	}
	float scaling = 1.0 /(o/addvalue*2+1);
	for (int i = -o, j=0; i <= o; i += addvalue,j++) {
		res += m[i*stride] *stencil[j];
	}
	return res;
}
struct skepu2_userfunction_conv_gaussian_kernel
{
constexpr static size_t totalArity = 5;
constexpr static bool indexed = 0;
using ElwiseArgs = std::tuple<int, size_t, const unsigned char *>;
using ContainerArgs = std::tuple<const skepu2::Vec<float>>;
using UniformArgs = std::tuple<size_t>;
constexpr static skepu2::AccessMode anyAccessMode[] = {
skepu2::AccessMode::Read, };

using Ret = unsigned char;

constexpr static bool prefersMatrix = 0;

#define SKEPU_USING_BACKEND_OMP 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block)
#define VARIANT_OPENMP(block) block
#define VARIANT_CUDA(block)
static inline SKEPU_ATTRIBUTE_FORCE_INLINE unsigned char OMP(int o, size_t stride, const unsigned char *m, const skepu2::Vec<float> stencil, size_t elemPerPx)
{
	float res = 0;
	int addvalue;
	if(stride == 1){
		addvalue = elemPerPx;
	} else {
		addvalue = 1;
	}
	float scaling = 1.0 /(o/addvalue*2+1);
	for (int i = -o, j=0; i <= o; i += addvalue,j++) {
		res += m[i*stride] *stencil.data[j];
	}
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
static inline SKEPU_ATTRIBUTE_FORCE_INLINE unsigned char CPU(int o, size_t stride, const unsigned char *m, const skepu2::Vec<float> stencil, size_t elemPerPx)
{
	float res = 0;
	int addvalue;
	if(stride == 1){
		addvalue = elemPerPx;
	} else {
		addvalue = 1;
	}
	float scaling = 1.0 /(o/addvalue*2+1);
	for (int i = -o, j=0; i <= o; i += addvalue,j++) {
		res += m[i*stride] *stencil.data[j];
	}
	return res;
}
#undef SKEPU_USING_BACKEND_CPU
};

constexpr skepu2::AccessMode skepu2_userfunction_conv_gaussian_kernel::anyAccessMode[];






#include "average_precompiled_OverlapKernel_gaussian_kernel_cl_source.inl"

#include "average_precompiled_OverlapKernel_average_kernel_1d_cl_source.inl"

#include "average_precompiled_Overlap2DKernel_average_kernel_cl_source.inl"
int main(int argc, char* argv[])
{
	if (argc < 5)
	{
		std::cout << "Usage: " << argv[0] << " input output radius [backend]\n";
		exit(1);
	}

	LodePNGColorType colorType = LCT_RGB;
	std::string inputFileName = argv[1];
	std::string outputFileName = argv[2];
	const int radius = atoi(argv[3]);
	auto spec = skepu2::BackendSpec{skepu2::Backend::typeFromString(argv[4])};

	// Create the full path for writing the image.
	std::stringstream ss;
	ss << (2 * radius + 1) << "x" << (2 * radius + 1);
	std::string outputFile = outputFileName + ss.str();

	// Read the padded image into a matrix. Create the output matrix without padding.
	// Padded version for 2D MapOverlap, non-padded for 1D MapOverlap
	ImageInfo imageInfo;
	skepu2::Matrix<unsigned char> inputMatrixPad = ReadAndPadPngFileToMatrix(inputFileName, radius, colorType, imageInfo);
	skepu2::Matrix<unsigned char> inputMatrix = ReadPngFileToMatrix(inputFileName, colorType, imageInfo);
	skepu2::Matrix<unsigned char> outputMatrix(imageInfo.height, imageInfo.width * imageInfo.elementsPerPixel, 120);
	skepu2::Matrix<unsigned char> intermediateMatrix(imageInfo.height, imageInfo.width * imageInfo.elementsPerPixel, 120);
	// more containers...?

	// Original version
	{
		skepu2::backend::MapOverlap2D<skepu2_userfunction_conv_average_kernel, bool, CLWrapperClass_average_precompiled_Overlap2DKernel_average_kernel> conv(false);
		conv.setOverlap(radius, radius  * imageInfo.elementsPerPixel);
		conv.setBackend(spec);

		auto timeTaken = skepu2::benchmark::measureExecTime([&]
		{
			conv(outputMatrix, inputMatrixPad, imageInfo.elementsPerPixel);
		});

		WritePngFileMatrix(outputMatrix, outputFile + "-average.png", colorType, imageInfo);
		std::cout << "Time for combined: " << (timeTaken.count() / 10E6) << "\n";
	}


	// Separable version
	// use conv.setOverlapMode(skepu2::Overlap::[ColWise RowWise]);
	// and conv.setOverlap(<integer>)
	{
		skepu2::backend::MapOverlap1D<skepu2_userfunction_conv_average_kernel_1d, bool, bool, bool, bool, CLWrapperClass_average_precompiled_OverlapKernel_average_kernel_1d> conv(false, false, false, false);
		conv.setOverlapMode(skepu2::Overlap::RowWise);
		conv.setOverlap(radius * imageInfo.elementsPerPixel);
		conv.setBackend(spec);


		auto timeTaken = skepu2::benchmark::measureExecTime([&]
		{
			conv(intermediateMatrix, inputMatrix, imageInfo.elementsPerPixel);
			//conv(outputMatrix, inputMatrix, imageInfo.elementsPerPixel);

			conv.setOverlapMode(skepu2::Overlap::ColWise);
			conv.setOverlap(radius);
			//conv(outputMatrix, inputMatrix, imageInfo.elementsPerPixel);
			conv(outputMatrix, intermediateMatrix, imageInfo.elementsPerPixel);
		});

		WritePngFileMatrix(outputMatrix, outputFile + "-separable.png", colorType, imageInfo);
		std::cout << "Time for separable: " << (timeTaken.count() / 10E6) << "\n";

	}


	// Separable gaussian
	{
		skepu2::backend::MapOverlap1D<skepu2_userfunction_conv_gaussian_kernel, bool, bool, bool, bool, CLWrapperClass_average_precompiled_OverlapKernel_gaussian_kernel> conv(false, false, false, false);
		skepu2::Vector<float> stencil = sampleGaussian(radius);
		conv.setOverlapMode(skepu2::Overlap::RowWise);
		conv.setOverlap(radius * imageInfo.elementsPerPixel);
		conv.setBackend(spec);
		// skeleton instance, etc here (remember to set backend)
		std::cout << stencil << '\n';
		auto timeTaken = skepu2::benchmark::measureExecTime([&]
		{
			conv(intermediateMatrix, inputMatrix, stencil, imageInfo.elementsPerPixel);
			//conv(outputMatrix, inputMatrix, imageInfo.elementsPerPixel);

			conv.setOverlapMode(skepu2::Overlap::ColWise);
			conv.setOverlap(radius);
			//conv(outputMatrix, inputMatrix, imageInfo.elementsPerPixel);
			conv(outputMatrix, intermediateMatrix, stencil, imageInfo.elementsPerPixel);
		});

		WritePngFileMatrix(outputMatrix, outputFile + "-gaussian.png", colorType, imageInfo);
		std::cout << "Time for gaussian: " << (timeTaken.count() / 10E6) << "\n";
	}



	return 0;
}
