/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
** Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
*/


/*
** Template to create a new test in the testuite
**
*/

#include <stdio.h>
#include "mpi.h"

#include <hip/hip_runtime.h>
#include <chrono>

#include "hip_mpitest_utils.h"
#include "hip_mpitest_buffer.h"

int elements=100;                  //Adjust
hip_mpitest_buffer *sendbuf=NULL;
hip_mpitest_buffer *recvbuf=NULL;

static void init_sendbuf (double *sendbuf, int count, int mynode)
{
    //Implement function
}

static void init_recvbuf (double *recvbuf, int count)
{
    //Implement function
}

static bool check_recvbuf(double *recvbuf, int nprocs, int rank, int count)
{
    //Implement function
}

int main (int argc, char *argv[])
{
    int res;
    int rank, size;

    
    MPI_Init      (&argc, &argv);
    MPI_Comm_size (MPI_COMM_WORLD, &size);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    
    bind_device();
    parse_args(argc, argv, MPI_COMM_WORLD);

    //Replace type and extent_of_type in the code
    type *tmp_sendbuf=NULL, *tmp_recvbuf=NULL;
    
    // Initialise send buffer
    ALLOCATE_SENDBUFFER(sendbuf, tmp_sendbuf, type, elements, extent_of_type, rank, MPI_COMM_WORLD, init_sendbuf);
    
    // Initialize recv buffer
    ALLOCATE_RECVBUFFER(recvbuf, tmp_recvbuf, type, elements, extent_of_type, rank, MPI_COMM_WORLD, init_recvbuf);
    
    //Warmup
    //execute warmup function if necessary/desired
    //provide actual test_name
    res = execute_test;                 
    
    // execute the allreduce test
    MPI_Barrier(MPI_COMM_WORLD);
    auto t1s = std::chrono::high_resolution_clock::now();
    // provide actual test_name
    res = execute_test;                  
    if (MPI_SUCCESS != res) {
        fprintf(stderr, "Error in #func_name test. Aborting\n");
        MPI_Abort (MPI_COMM_WORLD, 1);
        return 1;
    }
    auto t1e = std::chrono::high_resolution_clock::now();    
    double t1 = std::chrono::duration<double>(t1e-t1s).count();

    // verify results 
    bool ret = true;
    if (recvbuf->NeedsStagingBuffer()) {
        HIP_CHECK(recvbuf->CopyFrom(tmp_recvbuf, elements*extent_of_type));
        ret = check_recvbuf((type*)tmp_recvbuf, size, rank, elements);
    }
    else {
        ret = check_recvbuf((type*) recvbuf->get_buffer(), size, rank, elements);
    }

    bool fret = report_testresult(argv[0], MPI_COMM_WORLD, sendbuf->get_memchar(), recvbuf->get_memchar(), ret);
    report_performance (argv[0], MPI_COMM_WORLD, sendbuf->get_memchar(), recvbuf->get_memchar(),
                        elements, (size_t)(elements * extent_of_type), NITER, t1);

    //Free buffers
    FREE_BUFFER(sendbuf, tmp_sendbuf);
    FREE_BUFFER(recvbuf, tmp_recvbuf);

    delete (sendbuf);
    delete (recvbuf);
    
    MPI_Finalize ();    
    return fret ? 0 : 1;
}
