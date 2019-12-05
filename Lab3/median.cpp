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



	res = qselect(element, size, (2*ox/elemPerPx+1)*oy+1);
	//return res * scaling;
	return res;
}




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
	auto calculateMedian = skepu2::MapOverlap(median_kernel);
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
