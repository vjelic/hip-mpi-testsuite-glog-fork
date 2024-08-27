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

int elements=1024;
hip_mpitest_buffer *sendbuf=NULL;
hip_mpitest_buffer *recvbuf=NULL;

static void init_sendbuf (int *sendbuf, int count, int mynode)
{
    for (int i = 0; i < count; i++) {
        sendbuf[i] = mynode;
    }
}

static void init_recvbuf (int *recvbuf, int count )
{
    for (int i = 0; i < count; i++) {
        recvbuf[i] = 0;
    }
}

static bool check_recvbuf (int *recvbuf, int nProcs, int rank, int count)
{
    bool res = true;
    int result = (nProcs * (nProcs - 1)) / 2 ;
        for (int j=0; j < count; j++) {
            if (recvbuf[j] != result) {
                res = false;
#ifdef VERBOSE
                printf("recvbuf[%d] = %d\n", k, recvbuf[k]);
#endif
                break;
            }
        }
    return res;
}

static int type_osc_accumulate_test ( void *sendbuf, void *recvbuf, int count,
                           MPI_Datatype datatype, MPI_Comm comm, MPI_Win win);

int main (int argc, char *argv[])
{
    int rank, nProcs, status;
    MPI_Win win;

    bind_device();

    MPI_Init      (&argc, &argv);
    MPI_Comm_size (MPI_COMM_WORLD, &nProcs);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);

    parse_args(argc, argv, MPI_COMM_WORLD);

    int *tmp_sendbuf=NULL, *tmp_recvbuf=NULL;
    // Initialise send buffer
    ALLOCATE_SENDBUFFER(sendbuf, tmp_sendbuf, int, elements, sizeof(int),
                        rank, MPI_COMM_WORLD, init_sendbuf);

    // Initialize recv buffer
    ALLOCATE_RECVBUFFER(recvbuf, tmp_recvbuf, int, nProcs*elements, sizeof(int),
                        rank, MPI_COMM_WORLD, init_recvbuf);

    //Create window
        status = MPI_Win_create (recvbuf->get_buffer(), elements*sizeof(int), sizeof(int), MPI_INFO_NULL,
                                 MPI_COMM_WORLD, &win);
    if (MPI_SUCCESS != status) {
        return status;
    }

    //execute one-sided operations
    int res = type_osc_accumulate_test (sendbuf->get_buffer(), recvbuf->get_buffer(),
                             elements, MPI_INT, MPI_COMM_WORLD, win);
    if (MPI_SUCCESS != res) {
        printf("Error in type_osc_accumulate_test. Aborting\n");
        MPI_Abort (MPI_COMM_WORLD, 1);
        return 1;
    }

    // verify results
    bool ret=true;
        if (recvbuf->NeedsStagingBuffer()) {
            HIP_CHECK(recvbuf->CopyFrom(tmp_recvbuf, nProcs*elements*sizeof(int)));
            ret = check_recvbuf(tmp_recvbuf, nProcs, rank, elements);
        }
        else {
            ret = check_recvbuf((int*) recvbuf->get_buffer(), nProcs, rank, elements);
        }
    bool fret = report_testresult(argv[0], MPI_COMM_WORLD, sendbuf->get_memchar(), recvbuf->get_memchar(), ret);

    //Cleanup dynamic buffers
    MPI_Win_free (&win);
    FREE_BUFFER(sendbuf, tmp_sendbuf);
    FREE_BUFFER(recvbuf, tmp_recvbuf);

    delete (sendbuf);
    delete (recvbuf);

    MPI_Finalize ();
    return fret ? 0 : 1;
}

int type_osc_accumulate_test (void *sbuf, void *rbuf, int count,
                   MPI_Datatype datatype, MPI_Comm comm, MPI_Win win)
{
    int size, rank, ret;
    int tsize;

    MPI_Comm_size (comm, &size);
    MPI_Comm_rank (comm, &rank);
    MPI_Type_size (datatype, &tsize);

#ifdef HIP_MPITEST_OSC_ACCUMULATE_FENCE
    ret = MPI_Win_fence (0, win);
    if (MPI_SUCCESS != ret) {
        return ret;
    }
#endif

    for (int i = 0; i < size; i ++) {
#ifdef HIP_MPITEST_OSC_ACCUMULATE_LOCK
        ret = MPI_Win_lock(MPI_LOCK_EXCLUSIVE, i, 0, win);
        if (MPI_SUCCESS != ret) {
            return ret;
        }
#endif
        ret = MPI_Accumulate(sbuf, count, datatype, i, 0, count, datatype, MPI_SUM, win);
        if (MPI_SUCCESS != ret) {
            return ret;
        }
#ifdef HIP_MPITEST_OSC_ACCUMULATE_LOCK
        ret = MPI_Win_unlock(i, win);
        if (MPI_SUCCESS != ret) {
            return ret;
        }
#endif
    }

#ifdef HIP_MPITEST_OSC_ACCUMULATE_FENCE
    ret = MPI_Win_fence (0, win);
    if (MPI_SUCCESS != ret) {
        return ret;
    }
#endif

#ifdef HIP_MPITEST_OSC_ACCUMULATE_LOCK
    ret = MPI_Barrier(comm);
    if (MPI_SUCCESS != ret) {
        return ret;
    }
#endif
    return MPI_SUCCESS;
}
