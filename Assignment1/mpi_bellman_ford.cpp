/**
 * Name:
 * Student id:
 * ITSC email:
 */
/*
 * This is a mpi version of bellman_ford algorithm
 * Compile: mpic++ -std=c++11 -o mpi_bellman_ford mpi_bellman_ford.cpp
 * Run: mpiexec -n <number of processes> ./mpi_bellman_ford <input file>, you will find the output file 'output.txt'
 * */

#include <string>
#include <cassert>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <iomanip>
#include <cstring>
//Debug
#include <unistd.h>

#include "mpi.h"

using std::string;
using std::cout;
using std::endl;

#define INF 1000000
#define DEBUG

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

// you may add some helper functions here.

/**
 * Bellman-Ford algorithm. Find the shortest path from vertex 0 to other vertices.
 * @param my_rank the rank of current process
 * @param p number of processes
 * @param comm the MPI communicator
 * @param n input size
 * @param *mat input adjacency matrix
 * @param *dist distance array
 * @param *has_negative_cycle a bool variable to recode if there are negative cycles
*/
void bellman_ford(int my_rank, int p, MPI_Comm comm, int n, int *mat, int *dist, bool *has_negative_cycle) {
    int local_N;
    int local_low;
    int local_up;
    int local_n;
    int *local_mat;
    int *local_dist;
    int *global_dist;
    int *local_has_negative_cycle;
    int *global_has_negative_cycle;

    //step 1: broadcast N
    if (my_rank == 0) {
        local_N = n;
    }
    MPI_Bcast(&local_N, 1, MPI_INT, 0, comm);

#ifdef DEBUG
    std::cout << "MPI process: " << my_rank << "  local_N: " << local_N << std::endl;
#endif

    //step 2: find local task range
    local_low = local_N * my_rank / p;
    local_up = local_N * (my_rank + 1) / p - 1;
    local_n = local_up - local_low + 1;

#ifdef DEBUG
    std::cout << "MPI process: " << my_rank << " local_low " << local_low <<" local_up " << local_up << std::endl;
#endif

    //step 3: allocate local memory
    local_mat = (int *) malloc(local_n * local_N * sizeof(int));
    local_dist = (int *) malloc(local_N * sizeof(int));
    global_dist = (int *) malloc(local_N * sizeof(int));
    local_has_negative_cycle = (int *) malloc(sizeof(int));
    global_has_negative_cycle = (int *) malloc(sizeof(int));

    //step 4: broadcast matrix mat
    MPI_Scatter(mat, local_n * local_N, MPI_INT,
                local_mat, local_n * local_N, MPI_INT, 0, comm);

    //step 5: bellman-ford algorithm
    //initialize results
    *local_has_negative_cycle = 0;
    *global_has_negative_cycle = 0;
    for (int i = 0; i < local_N; i++) {
        global_dist[i] = INF;
    }
    //root vertex always has distance 0
    global_dist[0] = 0;

    bool has_change;
    for (int i = 0; i < local_N - 1; i++) {// n - 1 iteration
        has_change = false;
        memcpy(local_dist, global_dist, local_N * sizeof(int));
        for (int u = 0; u < local_n; u++) {
            for (int v = 0; v < local_N; v++) {
                int weight = local_mat[utils::convert_dimension_2D_1D(u, v, n)];
                if (weight < INF) {//test if u--v has an edge
                    if (local_dist[local_low + u] + weight < local_dist[v]) {
#ifdef DEBUG
                        std::cout << " MPI process: " << my_rank << " i " << i << " u " << local_low + u << " v " << v << std::endl;
#endif
                        local_dist[v] = local_dist[local_low + u] + weight;
                        has_change = true;
#ifdef DEBUG
                        std::cout << " local_dist ";
                        for (int j = 0; j < local_N; j++)
                            cout << " " << local_dist[j] << " ";
                        std::cout << std::endl;
#endif
                    }
                }
            }
        }
        //MPI_Allreduce(local_dist, global_dist, local_N, MPI_INT, MPI_MIN, comm);
        //if there is no change in this iteration, then we have finished
        if(!has_change) {
            break;
        }
    }
    //do one more iteration to check negative cycles
    for (int u = 0; u < local_n; u++) {
        for (int v = 0; v < n; v++) {
            int weight = local_mat[utils::convert_dimension_2D_1D(u, v, n)];
            if (weight < INF) {
                if (global_dist[local_low + u] + weight < global_dist[v]) { // if we can relax one more step, then we find a negative cycle
                    *local_has_negative_cycle = 1;
                    break;
                }
            }
        }
    }
    //MPI_Reduce(local_has_negative_cycle, global_has_negative_cycle, 1, MPI_INT, MPI_LOR, 0, comm);

#ifdef DEBUG
    std::cout << " final_global_dist ";
    for (int j = 0; j < local_N; j++)
        cout << " " << global_dist[j] << " ";
    std::cout << std::endl;
#endif

    //step 6: retrieve results back
    if (my_rank == 0) {
        memcpy(dist, global_dist, local_N * sizeof(int));
    }
    if (my_rank == 0) {
        if (*global_has_negative_cycle == 1) {
            *has_negative_cycle = true;
        }
    }

    //step 7: remember to free memory
    free(local_mat);
    free(local_dist);
    free(global_dist);
    free(local_has_negative_cycle);
    free(global_has_negative_cycle);

    return;
}

int main(int argc, char **argv) {

    if (argc <= 1) {
        utils::abort_with_error_message("INPUT FILE WAS NOT FOUND!");
    }
    string filename = argv[1];

    int *dist;
    bool has_negative_cycle = false;

    //MPI initialization
    MPI_Init(&argc, &argv);
    MPI_Comm comm;

    int p;//number of processors
    int my_rank;//my global rank
    comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &my_rank);

    //Debug
    printf("Rank %d of %d with PID %d\n", my_rank, p, getpid());
    fflush(stdout);

    //only rank 0 process do the I/O
    if (my_rank == 0) {
        assert(utils::read_file(filename) == 0);
        dist = (int *) malloc(sizeof(int) * utils::N);
    }

    //time counter
    double t1, t2;
    MPI_Barrier(comm);
    t1 = MPI_Wtime();

    //bellman-ford algorithm
    bellman_ford(my_rank, p, comm, utils::N, utils::mat, dist, &has_negative_cycle);
    MPI_Barrier(comm);

    //end timer
    t2 = MPI_Wtime();

    if (my_rank == 0) {
        std::cerr.setf(std::ios::fixed);
        std::cerr << std::setprecision(6) << "Time(s): " << (t2 - t1) << endl;
        utils::print_result(has_negative_cycle, dist);
        free(dist);
        free(utils::mat);
    }
    MPI_Finalize();
    return 0;
}
