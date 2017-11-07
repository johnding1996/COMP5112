/**
 * Name: DING, Mu Cong
 * Student id: 20323458
 * ITSC email: mcding@connect.ust.hk
 */
/*
 * This is a openmp version of bellman_ford algorithm
 * Compile: g++ -std=c++11 -fopenmp -o openmp_bellman_ford openmp_bellman_ford.cpp
 * Run: ./openmp_bellman_ford <input file> <number of threads>, you will find the output file 'output.txt'
 * */

#include <string>
#include <cassert>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <iomanip>
#include <cstring>
#include <sys/time.h>

#include "omp.h"

using std::string;
using std::cout;
using std::endl;

#define INF 1000000

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
        //input matrix should be smaller than 20MB * 20MB (400MB, we don't have too much memory for multi-threads)
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

// you may add some helper functions here.

/**
 * Bellman-Ford algorithm. Find the shortest path from vertex 0 to other vertices.
 * @param p number of threads
 * @param n input size
 * @param *mat input adjacency matrix
 * @param *dist distance array
 * @param *has_negative_cycle a bool variable to recode if there are negative cycles
*/
void bellman_ford(int p, int n, int *mat, int *dist, bool *has_negative_cycle) {
    //initialize results
    *has_negative_cycle = false;
    int i;
#pragma omp parallel for num_threads(p) \
        default(none) shared(dist, n) private(i) schedule(guided, 1)
    for (i = 0; i < n; i++) {
        dist[i] = INF;
    }
    //root vertex always has distance 0
    dist[0] = 0;

    //a flag to record if there is any distance change in this iteration
    bool has_change;
    int u, v, weight;
    //bellman-ford edge relaxation
    for (i = 0; i < n - 1; i++) {// n - 1 iteration
        has_change = false;
#pragma omp parallel for num_threads(p) \
        default(none) shared(mat, dist, n, has_change) private(u, v, weight) schedule(guided, 1) collapse(2)
        for (u = 0; u < n; u++) {
            for (v = 0; v < n; v++) {
                weight = mat[utils::convert_dimension_2D_1D(u, v, n)];
                if (weight < INF) {//test if u--v has an edge
                    if (dist[u] + weight < dist[v]) {
                        has_change = true;
                        dist[v] = dist[u] + weight;
                    }
                }
            }
        }
        //if there is no change in this iteration, then we have finished
        if (!has_change) {
            return;
        }
    }

    //do one more iteration to check negative cycles
    for (u = 0; u < n; u++) {
        for (v = 0; v < n; v++) {
            weight = mat[utils::convert_dimension_2D_1D(u, v, n)];
            if (weight < INF) {
                if (dist[u] + weight < dist[v]) { // if we can relax one more step, then we find a negative cycle
                    *has_negative_cycle = true;
                    return;
                }
            }
        }
    }
}

int main(int argc, char **argv) {
    if (argc <= 1) {
        utils::abort_with_error_message("INPUT FILE WAS NOT FOUND!");
    }
    if (argc <= 2) {
        utils::abort_with_error_message("NUMBER OF THREADS WAS NOT FOUND!");
    }
    string filename = argv[1];
    int p = atoi(argv[2]);

    int *dist;
    bool has_negative_cycle = false;


    assert(utils::read_file(filename) == 0);
    dist = (int *) malloc(sizeof(int) * utils::N);

    //time counter
    timeval start_wall_time_t, end_wall_time_t;
    float ms_wall;

    //start timer
    gettimeofday(&start_wall_time_t, nullptr);

    //bellman-ford algorithm
    bellman_ford(p, utils::N, utils::mat, dist, &has_negative_cycle);

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
