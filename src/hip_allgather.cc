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
                break;
            }
        }
    }

    return res;
}

int allgather_test (void *sendbuf, void *recvbuf, int count,
                    MPI_Datatype datatype, MPI_Comm comm,
                    int niterations);

int main (int argc, char *argv[])
{
    int ret;
    int rank, size;
    int root = 0;
    double t1;
    std::chrono::high_resolution_clock::time_point t1s, t1e;

    bind_device();

    MPI_Init      (&argc, &argv);
    MPI_Comm_size (MPI_COMM_WORLD, &size);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);

    parse_args(argc, argv, MPI_COMM_WORLD);

    double *tmp_sendbuf=NULL, *tmp_recvbuf=NULL;

    // Initialise send buffer
    ALLOCATE_SENDBUFFER(sendbuf, tmp_sendbuf, double, size*elements, sizeof(double),
                        rank, MPI_COMM_WORLD, init_sendbuf, out);

    // Initialize recv buffer
    ALLOCATE_RECVBUFFER(recvbuf, tmp_recvbuf, double, size*elements, sizeof(double),
                        rank, MPI_COMM_WORLD, init_recvbuf, out);

    //Warmup
    ret = allgather_test (sendbuf->get_buffer(), recvbuf->get_buffer(), elements,
                          MPI_DOUBLE, MPI_COMM_WORLD, 1);
    if (MPI_SUCCESS != ret) {
        fprintf(stderr, "Error in allgather_test. Aborting\n");
        goto out;
    }

    // execute the allreduce test
    MPI_Barrier(MPI_COMM_WORLD);
    t1s = std::chrono::high_resolution_clock::now();
    ret = allgather_test (sendbuf->get_buffer(), recvbuf->get_buffer(), elements,
                          MPI_DOUBLE, MPI_COMM_WORLD, NITER);
    if (MPI_SUCCESS != ret) {
        fprintf(stderr, "Error in allgather_test. Aborting\n");
        goto out;
    }
    t1e = std::chrono::high_resolution_clock::now();
    t1 = std::chrono::duration<double>(t1e-t1s).count();

    // verify results
    bool res, fret;
    res = true;
    if (recvbuf->NeedsStagingBuffer()) {
        HIP_CHECK(recvbuf->CopyFrom(tmp_recvbuf, elements*size*sizeof(double)));
#if defined HIP_MPITEST_GATHER || defined HIP_MPITEST_GATHERV
        if (rank == 0)
#endif
        res = check_recvbuf(tmp_recvbuf, size, rank, elements);
    }
    else {
#if defined HIP_MPITEST_GATHER || defined HIP_MPITEST_GATHERV
        if (rank == 0)
#endif
        res = check_recvbuf((double*) recvbuf->get_buffer(), size, rank, elements);
    }

    fret = report_testresult(argv[0], MPI_COMM_WORLD, sendbuf->get_memchar(), recvbuf->get_memchar(), res);
    report_performance (argv[0], MPI_COMM_WORLD, sendbuf->get_memchar(), recvbuf->get_memchar(),
                        elements, (size_t)(elements * sizeof(double)), NITER, t1);

 out:
    //Free buffers
    FREE_BUFFER(sendbuf, tmp_sendbuf);
    FREE_BUFFER(recvbuf, tmp_recvbuf);
    delete (sendbuf);
    delete (recvbuf);

    if (MPI_SUCCESS != ret) {
        MPI_Abort (MPI_COMM_WORLD, 1);
        return 1;
    }

    MPI_Finalize ();
    return fret ? 0 : 1;
}


int allgather_test (void *sendbuf, void *recvbuf, int count,
                    MPI_Datatype datatype, MPI_Comm comm,
                    int niterations)
{
    int ret;
#if defined HIP_MPITEST_ALLGATHERV || defined HIP_MPITEST_GATHERV
    int *rcounts = NULL, *rdispls = NULL;
    int size;

    MPI_Comm_size (comm, &size);

    rcounts = (int*)malloc(size *sizeof(int));
    if (NULL == rcounts) {
        printf("(All)gatherv test: Could not allocate memory\n");
        return MPI_ERR_OTHER;
    }
    rdispls = (int*)malloc(size *sizeof(int));
    if (NULL == rdispls) {
        printf("(All)gatherv test: Could not allocate memory\n");
        ret = MPI_ERR_OTHER;
        goto out;
    }

    for (int i=0; i<size; i++) {
        rcounts[i]=count;
        rdispls[i]=i*count;
    }
#endif

    for (int i=0; i<niterations; i++) {
#if defined HIP_MPITEST_GATHER
        ret = MPI_Gather (sendbuf, count, datatype, recvbuf, count, datatype, 0, comm);
#elif defined HIP_MPITEST_GATHERV
        ret = MPI_Gatherv (sendbuf, count, datatype, recvbuf, rcounts, rdispls, datatype, 0, comm);
#elif defined HIP_MPITEST_ALLGATHERV
        ret = MPI_Allgatherv (sendbuf, count, datatype, recvbuf, rcounts, rdispls, datatype, comm);
#else
        ret = MPI_Allgather (sendbuf, count, datatype, recvbuf, count, datatype, comm);
#endif
        if (MPI_SUCCESS != ret) {
            goto out;
        }
    }

 out:
#if defined HIP_MPITEST_ALLGATHERV || defined HIP_MPITEST_GATHERV
    free (rcounts);
    free (rdispls);
#endif
    return ret;
}
