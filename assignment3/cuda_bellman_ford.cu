/**
 * Name:
 * Student id:
 * ITSC email:
 */
/*
 * This is a CUDA version of bellman_ford algorithm
 * Compile: nvcc -std=c++11 -arch=sm_52 -o cuda_bellman_ford cuda_bellman_ford.cu
 * Run: ./cuda_bellman_ford <input file> <number of blocks per grid> <number of threads per block>, you will find the output file 'output.txt'
 * */

#include <string>
#include <cassert>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <iomanip>
#include <cstring>
#include <sys/time.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using std::string;
using std::cout;
using std::endl;

#define INF 1000000

/*
 * This is a CHECK function to check CUDA calls
 */
#define CHECK(call)                                                            \
		{                                                                              \
	const cudaError_t error = call;                                            \
	if (error != cudaSuccess)                                                  \
	{                                                                          \
		fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
		fprintf(stderr, "code: %d, reason: %s\n", error,                       \
				cudaGetErrorString(error));                                    \
				exit(1);                                                               \
	}                                                                          \
		}


/**
 * utils is a namespace for utility functions
 * including I/O (read input file and print results) and matrix dimension convert(2D->1D) function
 */
namespace utils {
int N; //number of vertices
int *mat; // the adjacency matrix

void abort_with_error_message(string msg) {
	std::cerr << msg << endl;
	abort();
}

//translate 2-dimension coordinate to 1-dimension
int convert_dimension_2D_1D(int x, int y, int n) {
	return x * n + y;
}

int read_file(string filename) {
	std::ifstream inputf(filename, std::ifstream::in);
	if (!inputf.good()) {
		abort_with_error_message("ERROR OCCURRED WHILE READING INPUT FILE");
	}
	inputf >> N;
	//input matrix should be smaller than 20MB * 20MB (400MB, we don't have too much memory for multi-processors)
	assert(N < (1024 * 1024 * 20));
	mat = (int *) malloc(N * N * sizeof(int));
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++) {
			inputf >> mat[convert_dimension_2D_1D(i, j, N)];
		}
	return 0;
}

int print_result(bool has_negative_cycle, int *dist) {
	std::ofstream outputf("output.txt", std::ofstream::out);
	if (!has_negative_cycle) {
		for (int i = 0; i < N; i++) {
			if (dist[i] > INF)
				dist[i] = INF;
			outputf << dist[i] << '\n';
		}
		outputf.flush();
	} else {
		outputf << "FOUND NEGATIVE CYCLE!" << endl;
	}
	outputf.close();
	return 0;
}
}//namespace utils

// you may add some helper/kernel functions here.

/**
 * Bellman-Ford algorithm. Find the shortest path from vertex 0 to other vertices.
 * @param blockPerGrid number of blocks per grid
 * @param threadsPerBlock number of threads per block
 * @param n input size
 * @param *mat input adjacency matrix
 * @param *dist distance array
 * @param *has_negative_cycle a bool variable to recode if there are negative cycles
 */
void bellman_ford(int blocksPerGrid, int threadsPerBlock, int n, int *mat, int *dist, bool *has_negative_cycle) {
	//------your code starts from here------

	//------end of your code------
}

int main(int argc, char **argv) {
	if (argc <= 1) {
		utils::abort_with_error_message("INPUT FILE WAS NOT FOUND!");
	}
	if (argc <= 3) {
		utils::abort_with_error_message("blocksPerGrid or threadsPerBlock WAS NOT FOUND!");
	}

	string filename = argv[1];
	int blockPerGrid = atoi(argv[2]);
	int threadsPerBlock = atoi(argv[3]);

	int *dist;
	bool has_negative_cycle = false;


	assert(utils::read_file(filename) == 0);
	dist = (int *) calloc(sizeof(int), utils::N);


	//time counter
	timeval start_wall_time_t, end_wall_time_t;
	float ms_wall;
	cudaDeviceReset();
	//start timer
	gettimeofday(&start_wall_time_t, nullptr);
	//bellman-ford algorithm
	bellman_ford(blockPerGrid, threadsPerBlock, utils::N, utils::mat, dist, &has_negative_cycle);
	CHECK(cudaDeviceSynchronize());
	//end timer
	gettimeofday(&end_wall_time_t, nullptr);
	ms_wall = ((end_wall_time_t.tv_sec - start_wall_time_t.tv_sec) * 1000 * 1000
			+ end_wall_time_t.tv_usec - start_wall_time_t.tv_usec) / 1000.0;

	std::cerr.setf(std::ios::fixed);
	std::cerr << std::setprecision(6) << "Time(s): " << (ms_wall/1000.0) << endl;
	utils::print_result(has_negative_cycle, dist);
	free(dist);
	free(utils::mat);

	return 0;
}
