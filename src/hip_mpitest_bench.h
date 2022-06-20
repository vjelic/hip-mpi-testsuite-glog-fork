/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
** Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
*/

#ifndef __HIP_MPITEST_BENCH__
#define __HIP_MPITEST_BENCH__

#include <unistd.h>
#include <string.h>
#include <stdlib.h>

#include "mpi.h"


static void bench_performance (char *exec, MPI_Comm comm, char sendtype, char recvtype,
                               int elements, long nBytes, int niter, double time)
{
    int rank, size;
    double t1_sum=0.0;
    double t1_avg=0.0;

    MPI_Comm_rank (comm, &rank);
    MPI_Comm_size (comm, &size);

    MPI_Reduce(&time, &t1_sum, 1, MPI_DOUBLE, MPI_SUM, 0, comm);    
    
    if (rank == 0) {
        t1_avg = t1_sum/(size*niter);
        printf("%10d \t %10lu \t %lf\n", elements, (size_t)nBytes, t1_avg);
    }
}

#endif
