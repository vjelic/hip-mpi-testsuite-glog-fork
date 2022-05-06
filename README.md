# hip-mpi-testsuite

The hip-mpi-testsuite contains a collection of MPI tests capable of utilizing arbitray combinations of different memory types.
The code requires a rocm enabled implementation of Open MPI, i.e. compiled with --with-rocm available in Open MPI starting 05/06/2022,
and UCX compiled with rocm support.

All tests take the same set of optional arguments:

```
Usage: executable_name -s <sendBufType> -r <recvBufType> -n <elements> -t <sleepTime>
       with sendBufType and recvBufType being :
                  D      Device memory (i.e. hipMalloc) - default if not specified
                  H      Host memory (i.e. malloc)
                  M      Unified memory (i.e hipMallocManaged)
                  O      Device accessible page locked host memory (i.e. hipHostMalloc)
                  R      Registered host memory (i.e. hipHostRegister)
            elements:  number of elements to send/recv
            sleepTime: time in seconds to sleep
```

To run all tests in the testsuite 

```
make
./run_all.sh
```
