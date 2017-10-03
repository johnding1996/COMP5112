## MPI environment setup

Add the OpenMPI installation path to your shell environment:

```
$ echo 'setenv PATH "${PATH}:/usr/local/software/openmpi/bin"' >> ~/.cshrc_user
```
After that, re-login (`logout` and `login`) to active the new environment.

You can use command `which mpicc` to check your configuration.

## Compile and run the examples

### mpi_hello.c

compile and run:

```sh
$ mpicc -std=c99 -o mpi_hello mpi_hello.c
$ mpiexec -n <number of processes> ./mpi_hello
```
e.g. `mpiexec -n 4 ./mpi_hello`

### mpi_output.c

compile and run:

```sh
$ mpicc -std=c99 -o mpi_output mpi_output.c
$ mpiexec -n <number of processes> ./mpi_output
```

### mpi_trap1.c

compile and run:

```sh
$ mpicc -std=c99 -o mpi_trap1 mpi_trap1.c
$ mpiexec -n <number of processes> ./mpi_trap1
```
