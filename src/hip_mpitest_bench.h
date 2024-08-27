/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/******************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

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
