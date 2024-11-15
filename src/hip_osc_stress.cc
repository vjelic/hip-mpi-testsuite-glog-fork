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

#include "hip_mpitest_utils.h"
#include "hip_mpitest_buffer.h"
#define NUM_NB_ITERATIONS 29
int elements=1024;
hip_mpitest_buffer *sendbuf=NULL;
hip_mpitest_buffer *recvbuf=NULL;

static void init_buf (int *sendbuf, int count, int mynode)
{
    int realcount = count / 2;
    int scount = realcount / NUM_NB_ITERATIONS;
    int nProcs = scount / elements;

    /* first half of the buffer used as result/receive buffer */
    for (int i = 0; i < realcount; i++) {
        sendbuf[i] = 0;
    }

    /* second half contains the actual data that will be fetched/provided */
    int l=0;
    for (int iteration=0; iteration < NUM_NB_ITERATIONS; iteration++) {
        for (int i = 0; i < scount; i++, l++) {
            sendbuf[realcount+l] = mynode + 1 + iteration * nProcs;
        }
    }
}

static bool check_recvbuf (int *recvbuf, int nProcs, int rank, int count)
{
    bool res=true;
    int  l=0;
    for (int iteration=0; iteration < NUM_NB_ITERATIONS; iteration++) {
        for (int recvrank=0; recvrank < nProcs; recvrank++) {
            for (int i=0; i < count; i++, l++) {
                if (recvbuf[l] != recvrank + 1 + iteration * nProcs) {
                    res = false;
#ifdef VERBOSE
                    printf("[%d] recvbuf[%d] = %d expected %d\n", rank, l, recvbuf[l],
                           (recvrank+1 + iteration * nProcs));
#endif
                    break;
                }
            }
        }
    }
    return res;
}

int type_osc_stress_test (int *buf, int count,  MPI_Comm comm, MPI_Win win);

int main (int argc, char *argv[])
{
    int rank, nProcs, ret;
    int root = 0;
    MPI_Win win = MPI_WIN_NULL;

    bind_device();

    MPI_Init      (&argc, &argv);
    MPI_Comm_size (MPI_COMM_WORLD, &nProcs);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);

    parse_args(argc, argv, MPI_COMM_WORLD);

    int *tmpbuf=NULL;
    // Initialise global buffer
    ALLOCATE_SENDBUFFER(sendbuf, tmpbuf, int, 2*nProcs*elements*NUM_NB_ITERATIONS, sizeof(int),
                        rank, MPI_COMM_WORLD, init_buf, out);

    //Create window
    ret = MPI_Win_create (sendbuf->get_buffer(), 2*nProcs*elements*sizeof(int), sizeof(int), MPI_INFO_NULL,
                          MPI_COMM_WORLD, &win);
    if (MPI_SUCCESS != ret) {
        goto out;
    }

    //execute osc stress test
    ret = type_osc_stress_test ((int *)sendbuf->get_buffer(), elements, MPI_COMM_WORLD, win);
    if (MPI_SUCCESS != ret) {
        printf("Error in type_osc_stress_test. Aborting\n");
        goto out;
    }

    // verify results
    bool res, fret;
    res = true;
    if (sendbuf->NeedsStagingBuffer()) {
        HIP_CHECK(sendbuf->CopyFrom(tmpbuf, 2*nProcs*elements*NUM_NB_ITERATIONS*sizeof(int)));
        res = check_recvbuf(tmpbuf, nProcs, rank, elements);
    }
    else {
        res = check_recvbuf((int*) sendbuf->get_buffer(), nProcs, rank, elements);
    }
    fret = report_testresult(argv[0], MPI_COMM_WORLD, sendbuf->get_memchar(), '-', res);
    report_performance (argv[0], MPI_COMM_WORLD, sendbuf->get_memchar(), '-', elements,
                        (size_t)(elements *sizeof(int)), 0, 0.0);

 out:
    //Cleanup dynamic buffers
    if (MPI_WIN_NULL != win) {
        MPI_Win_free (&win);
    }
    FREE_BUFFER(sendbuf, tmpbuf);
    delete (sendbuf);

    if (MPI_SUCCESS != ret ) {
        MPI_Abort (MPI_COMM_WORLD, 1);
        return 1;
    }    
    MPI_Finalize ();
    return fret ? 0 : 1;
}


int type_osc_stress_test (int *sbuf, int count, MPI_Comm comm, MPI_Win win)
{
    int size, rank, ret;
    MPI_Request *reqs;
    int *tbuf;
    MPI_Aint rdisp;

    MPI_Comm_size (comm, &size);
    MPI_Comm_rank (comm, &rank);

    reqs = (MPI_Request*)malloc (size*NUM_NB_ITERATIONS*sizeof(MPI_Request));
    if (NULL == reqs) {
        printf("4. Could not allocate memory. Aborting\n");
        return MPI_ERR_OTHER;
    }

    int datadisp = count * size * NUM_NB_ITERATIONS;

    ret = MPI_Win_lock_all(MPI_MODE_NOCHECK, win);
    if (MPI_SUCCESS != ret) {
        goto out;
    }
    for (int j=0; j<NUM_NB_ITERATIONS; j++) {
        for (int i=0; i<size; i++) {
#ifdef HIP_MPITEST_OSC_RGET
            tbuf = &sbuf[i*count+j*count*size];
            rdisp = datadisp + rank*count + j*count*size;
#ifdef VERBOSE
            printf("[%d] about to Rget from proc %d local_elem %d disp %lu\n", rank, i,
                   (i*count+j*count*size), rdisp);
#endif
            ret = MPI_Rget (tbuf, count, MPI_INT, i, rdisp, count, MPI_INT, win, &reqs[size*j+i]);
#elif defined HIP_MPITEST_OSC_RPUT
            tbuf = &sbuf[datadisp+i*count+j*count*size];
            rdisp = rank*count + j*count*size;
#ifdef VERBOSE
            printf("[%d] about to Rput to proc %d local_elemt %d [value %d] disp %lu\n", rank, i,
                   (datadisp+i*count+j*count*size), *tbuf, rdisp);
#endif
            ret = MPI_Rput (tbuf, count, MPI_INT, i, rdisp, count, MPI_INT, win, &reqs[size*j+i]);
#endif
            if (MPI_SUCCESS != ret) {
                goto out;
            }
        }
    }
    ret = MPI_Waitall (size*NUM_NB_ITERATIONS, reqs, MPI_STATUSES_IGNORE);
    if (MPI_SUCCESS != ret) {
        goto out;
    }

    ret = MPI_Win_unlock_all(win);
    if (MPI_SUCCESS != ret) {
        goto out;
    }
 out:
    free (reqs);
    return ret;
}

