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
#include "hip_mpitest_bench.h"

#define NITER 1000
int elements=100;
hip_mpitest_buffer *sendbuf=NULL;
hip_mpitest_buffer *recvbuf=NULL;

static void init_sendbuf (double *sendbuf, int count, int mynode)
{
    for (int i = 0; i < count; i++) {
        sendbuf[i] = (double)mynode+1;
    }
}

static void init_recvbuf (double *recvbuf, int count)
{
    for (int i = 0; i < count; i++) {
        recvbuf[i] = 0.0;
    }
}

static bool check_recvbuf(double *recvbuf, int nprocs, int root, int count)
{
    bool res=true;
    double result = (double) root+1;

    for (int i=0; i<count; i++) {
        if (recvbuf[i] != result) {
            res = false;
#ifdef VERBOSE
            printf("recvbuf[%d] = %d\n", i, recvbuf[i]);
#endif
        }
    }

    return res;
}

#define ROOT 0
int bcast_test (void *sendbuf, int count, MPI_Datatype datatype, MPI_Comm comm,
                int niterations);

int main (int argc, char *argv[])
{
    int res;
    int rank, size;
    int root = 0;

    bind_device();

    MPI_Init      (&argc, &argv);
    MPI_Comm_size (MPI_COMM_WORLD, &size);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);

    parse_args(argc, argv, MPI_COMM_WORLD);

    int max_elements = elements;

    if (rank == 0 ) {
        printf("Benchmark: %s %c %c - %d processes\n\n", argv[0],  sendbuf->get_memchar(), recvbuf->get_memchar(), size);
        printf("No. of elems \t msg. length \t time\n");
        printf("================================================================\n");
    }

    for (elements=1; elements<=max_elements; elements *=2 ) {
        double *tmp_sendbuf=NULL, *tmp_recvbuf=NULL;

        // Initialise send buffer
        ALLOCATE_SENDBUFFER(sendbuf, tmp_sendbuf, double, elements, sizeof(double),
                            rank, MPI_COMM_WORLD, init_sendbuf);
        
        //Warmup
        res = bcast_test (sendbuf->get_buffer(), elements, MPI_DOUBLE, MPI_COMM_WORLD, 1);
        if (MPI_SUCCESS != res ) {
            fprintf(stderr, "Error in bcast_test. Aborting\n");
            MPI_Abort (MPI_COMM_WORLD, 1);
            return 1;
        }
        
        // execute the allreduce test
        MPI_Barrier(MPI_COMM_WORLD);
        auto t1s = std::chrono::high_resolution_clock::now();
        res = bcast_test (sendbuf->get_buffer(), elements, MPI_DOUBLE, MPI_COMM_WORLD, NITER);
        if (MPI_SUCCESS != res) {
            fprintf(stderr, "Error in bcast_test. Aborting\n");
            MPI_Abort (MPI_COMM_WORLD, 1);
            return 1;
        }
        auto t1e = std::chrono::high_resolution_clock::now();
        double t1 = std::chrono::duration<double>(t1e-t1s).count();
        
#if 0
        // verify results 
        bool ret = true;
        if (sendbuf->NeedsStagingBuffer()) {
            HIP_CHECK(sendbuf->CopyFrom(tmp_sendbuf, elements*sizeof(double)));
            ret = check_recvbuf(tmp_sendbuf, size, ROOT, elements);
        }
        else {
            ret = check_recvbuf((double*) sendbuf->get_buffer(), size, ROOT, elements);
        }
        
        bool fret = report_testresult(argv[0], MPI_COMM_WORLD, sendbuf->get_memchar(), recvbuf->get_memchar(), ret);
#endif
        bench_performance (argv[0], MPI_COMM_WORLD, sendbuf->get_memchar(), recvbuf->get_memchar(),
                           elements, (size_t)(elements * sizeof(double)), NITER, t1);
        
        //Free buffers
        FREE_BUFFER(sendbuf, tmp_sendbuf);
    }

    delete (sendbuf);

    MPI_Finalize ();    
    return 0;
}


int bcast_test ( void *sendbuf, int count, MPI_Datatype datatype, MPI_Comm comm,
                 int niterations)
{
    int ret;

    for (int i=0; i<niterations; i++) {
        ret = MPI_Bcast (sendbuf, count, datatype, ROOT, comm);
        if (MPI_SUCCESS != ret) {
            return ret;
        }
    }

    return MPI_SUCCESS;
}
