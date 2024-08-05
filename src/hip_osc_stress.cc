/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
** Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
*/

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
    int rank, nProcs, status;
    int root = 0;
    MPI_Win win;

    bind_device();

    MPI_Init      (&argc, &argv);
    MPI_Comm_size (MPI_COMM_WORLD, &nProcs);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);

    parse_args(argc, argv, MPI_COMM_WORLD);

    int *tmpbuf=NULL;
    // Initialise global buffer
    ALLOCATE_SENDBUFFER(sendbuf, tmpbuf, int, 2*nProcs*elements*NUM_NB_ITERATIONS, sizeof(int),
                        rank, MPI_COMM_WORLD, init_buf);

    //Create window
    status = MPI_Win_create (sendbuf->get_buffer(), 2*nProcs*elements*sizeof(int), sizeof(int), MPI_INFO_NULL,
                             MPI_COMM_WORLD, &win);
    if (MPI_SUCCESS != status) {
        return status;
    }

    //execute osc stress test
    int res = type_osc_stress_test ((int *)sendbuf->get_buffer(), elements, MPI_COMM_WORLD, win);
    if (MPI_SUCCESS != res) {
        printf("Error in type_osc_stress_test. Aborting\n");
        MPI_Abort (MPI_COMM_WORLD, 1);
        return 1;
    }

    // verify results
    bool ret;
    if (sendbuf->NeedsStagingBuffer()) {
        HIP_CHECK(sendbuf->CopyFrom(tmpbuf, 2*nProcs*elements*NUM_NB_ITERATIONS*sizeof(int)));
        ret = check_recvbuf(tmpbuf, nProcs, rank, elements);
    }
    else {
        ret = check_recvbuf((int*) sendbuf->get_buffer(), nProcs, rank, elements);
    }
    bool fret = report_testresult(argv[0], MPI_COMM_WORLD, sendbuf->get_memchar(), '-', ret);
    report_performance (argv[0], MPI_COMM_WORLD, sendbuf->get_memchar(), '-', elements,
                        (size_t)(elements *sizeof(int)), 0, 0.0);

    //Cleanup dynamic buffers
    FREE_BUFFER(sendbuf, tmpbuf);
    delete (sendbuf);

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
        MPI_Abort(comm, 1);
    }

    int datadisp = count * size * NUM_NB_ITERATIONS;

    ret = MPI_Win_lock_all(MPI_MODE_NOCHECK, win);
    if (MPI_SUCCESS != ret) {
        return ret;
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
                return ret;
            }
        }
    }
    ret = MPI_Waitall (size*NUM_NB_ITERATIONS, reqs, MPI_STATUSES_IGNORE);
    if (MPI_SUCCESS != ret) {
        return ret;
    }

    ret = MPI_Win_unlock_all(win);
    if (MPI_SUCCESS != ret) {
        return ret;
    }
    free (reqs);

    return MPI_SUCCESS;
}

