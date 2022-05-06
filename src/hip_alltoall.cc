/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
** Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
*/

#include <stdio.h>
#include "mpi.h"

#include <hip/hip_runtime.h>
#include <chrono>

#include "hip_mpitest_utils.h"
#include "hip_mpitest_buffer.h"

#define NITER 25
int elements=100;
hip_mpitest_buffer *sendbuf=NULL;
hip_mpitest_buffer *recvbuf=NULL;

static void init_sendbuf (double *sendbuf, int count, int mynode)
{
    for (int i = 0; i < count; i++) {
        sendbuf[i] = (double)mynode;
    }
}

static void init_recvbuf (double *recvbuf, int count)
{
    for (int i = 0; i < count; i++) {
        recvbuf[i] = 0.0;
    }
}

static bool check_recvbuf(double *recvbuf, int nprocs, int rank, int count)
{
    bool res=true;
    int l=0;
    
    for (int j=0; j<nprocs; j++) {
        double result = (double)j;
        for (int i=0; i<count; i++, l++) {
            if (recvbuf[l] != result) {
                res = false;
#ifdef VERBOSE
                printf("recvbuf[%d] = %d\n", i, recvbuf[l]);
#endif
            }
        }
    }

    return res;
}

int alltoall_test (void *sendbuf, void *recvbuf, int count,
                   MPI_Datatype datatype, MPI_Comm comm,
                   int niterations);

int main (int argc, char *argv[])
{
    int res;
    int rank, size;

    MPI_Init      (&argc, &argv);
    MPI_Comm_size (MPI_COMM_WORLD, &size);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    
    bind_device();
    parse_args(argc, argv, MPI_COMM_WORLD);

    double *tmp_sendbuf=NULL, *tmp_recvbuf=NULL;
    
    // Initialise send buffer
    ALLOCATE_SENDBUFFER(sendbuf, tmp_sendbuf, double, size*elements, sizeof(double),
                        rank, MPI_COMM_WORLD, init_sendbuf);
    
    // Initialize recv buffer
    ALLOCATE_RECVBUFFER(recvbuf, tmp_recvbuf, double, size*elements, sizeof(double),
                        rank, MPI_COMM_WORLD, init_recvbuf);
    
    //Warmup
    res = alltoall_test (sendbuf->get_buffer(), recvbuf->get_buffer(), elements,
                         MPI_DOUBLE, MPI_COMM_WORLD, 1);
    if (MPI_SUCCESS != res ) {
        fprintf(stderr, "Error in alltoall_test. Aborting\n");
        MPI_Abort (MPI_COMM_WORLD, 1);
        return 1;
    }
    
    // execute the allreduce test
    MPI_Barrier(MPI_COMM_WORLD);
    auto t1s = std::chrono::high_resolution_clock::now();
    res = alltoall_test (sendbuf->get_buffer(), recvbuf->get_buffer(), elements,
                         MPI_DOUBLE, MPI_COMM_WORLD, NITER);
    if (MPI_SUCCESS != res) {
        fprintf(stderr, "Error in alltoall_test. Aborting\n");
        MPI_Abort (MPI_COMM_WORLD, 1);
        return 1;
    }
    auto t1e = std::chrono::high_resolution_clock::now();    
    double t1 = std::chrono::duration<double>(t1e-t1s).count();
    
    // verify results 
    bool ret = true;
    if (recvbuf->NeedsStagingBuffer()) {
        HIP_CHECK(recvbuf->CopyFrom(tmp_recvbuf, elements*size*sizeof(double)));
        ret = check_recvbuf(tmp_recvbuf, size, rank, elements);
    }
    else {
        ret = check_recvbuf((double*) recvbuf->get_buffer(), size, rank, elements);
    }
    
    bool fret = report_testresult(argv[0], MPI_COMM_WORLD, sendbuf->get_memchar(), recvbuf->get_memchar(), ret);
    report_performance (argv[0], MPI_COMM_WORLD, sendbuf->get_memchar(), recvbuf->get_memchar(),
                        elements, (size_t)(elements * sizeof(double)), NITER, t1);
    
    //Free buffers
    FREE_BUFFER(sendbuf, tmp_sendbuf);
    FREE_BUFFER(recvbuf, tmp_recvbuf);
    
    delete (sendbuf);
    delete (recvbuf);
    
    MPI_Finalize ();    
    return fret ? 0 : 1;
}


int alltoall_test ( void *sendbuf, void *recvbuf, int count,
                    MPI_Datatype datatype, MPI_Comm comm,
                    int niterations)
{
    int ret;
#ifdef HIP_MPITEST_ALLTOALLV
    int *scounts, *rcounts;
    int *sdispls, *rdispls;
    int size;

    MPI_Comm_size (comm, &size);
    
    scounts = (int*)malloc(size *sizeof(int));
    rcounts = (int*)malloc(size *sizeof(int));
    sdispls = (int*)malloc(size *sizeof(int));
    rdispls = (int*)malloc(size *sizeof(int));
    if (NULL==scounts || NULL==rcounts || NULL==sdispls || NULL==rdispls) {
        printf("Alltoallv test: Could not allocate memory\n");
        return MPI_ERR_OTHER;
    }

    for (int i=0; i<size; i++) {
        scounts[i]=count;
        rcounts[i]=count;
        sdispls[i]=i*count;
        rdispls[i]=i*count;
    }
#endif
    
    for (int i=0; i<niterations; i++) {
#ifdef HIP_MPITEST_ALLTOALLV
        ret = MPI_Alltoallv(sendbuf, scounts, sdispls, datatype,
                            recvbuf, rcounts, rdispls, datatype, comm);
#else
        ret = MPI_Alltoall(sendbuf, count, datatype, recvbuf, count, datatype, comm);
#endif
        if (MPI_SUCCESS != ret) {
            return ret;
        }
    }

#ifdef HIP_MPITEST_ALLTOALLV
    free (scounts);
    free (rcounts);
    free (sdispls);
    free (rdispls);
#endif

    return MPI_SUCCESS;
}
