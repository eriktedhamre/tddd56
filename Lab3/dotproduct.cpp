/**************************
** TDDD56 Lab 3
***************************
** Author:
** August Ernstsson
**************************/

#include <iostream>

#include <skepu2.hpp>

/* SkePU user functions */

/*
float userfunction(...)
{
	// your code here
}

// more user functions...

*/
float multiply(float a, float b)
{
	return a * b;
}

float add(float a, float b)
{
	return a + b;
}


int main(int argc, const char* argv[])
{
	if (argc < 2)
	{
		std::cout << "Usage: " << argv[0] << " <input size> <backend>\n";
		exit(1);
	}

	const size_t size = std::stoul(argv[1]);
	auto spec = skepu2::BackendSpec{skepu2::Backend::typeFromString(argv[2])};
//	spec.setCPUThreads(<integer value>);


	/* Skeleton instances */
	auto mapInstance = skepu2::Map<2>(multiply);
	auto reduceInstance = skepu2::Reduce(add);
	auto mapReduceInstance = skepu2::MapReduce<2>(multiply, add);

// ...

	/* Set backend (important, do for all instances!) */
	mapInstance.setBackend(spec);
	reduceInstance.setBackend(spec);
	mapReduceInstance.setBackend(spec);


	/* SkePU containers */
	skepu2::Vector<float> v1(size, 1.0f), v2(size, 2.0f), v3(size, 3.0f);
	v1.randomize(0, 9);
	v2.randomize(0, 9);

	/* Compute and measure time */
	float resComb, resSep;

	auto timeComb = skepu2::benchmark::measureExecTime([&]
	{
		resComb = mapReduceInstance(v1, v2);
	});

	auto timeSep = skepu2::benchmark::measureExecTime([&]
	{
		mapInstance(v3, v1, v2);
		resSep = reduceInstance(v3);
	});

	std::cout << "Time Combined: " << (timeComb.count() / 10E6) << " seconds.\n";
	std::cout << "Time Separate: " << ( timeSep.count() / 10E6) << " seconds.\n";


	std::cout << "Result Combined: " << resComb << "\n";
	std::cout << "Result Separate: " << resSep  << "\n";

	return 0;
}
