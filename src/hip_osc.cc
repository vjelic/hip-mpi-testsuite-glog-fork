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
    int k=0;
    for (int i=0; i<nProcs; i++) {
        for (int j=0; j < count; j++, k++) {
            if (recvbuf[k] != i) {
                res = false;
#ifdef VERBOSE
                printf("recvbuf[%d] = %d\n", k, recvbuf[k]);
#endif
                break;
            }
        }
    }
    return res;
}

static int type_osc_test ( void *sendbuf, void *recvbuf, int count,
                           MPI_Datatype datatype, int root, MPI_Comm comm, MPI_Win win);

int main (int argc, char *argv[])
{
    int rank, nProcs;
    int root = 0; //checkbuff will not work for any other root value right now
    MPI_Win win = MPI_WIN_NULL;
    int ret;

    bind_device();

    MPI_Init      (&argc, &argv);
    MPI_Comm_size (MPI_COMM_WORLD, &nProcs);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);

    parse_args(argc, argv, MPI_COMM_WORLD);

    int *tmp_sendbuf=NULL, *tmp_recvbuf=NULL;
    // Initialise send buffer
    ALLOCATE_SENDBUFFER(sendbuf, tmp_sendbuf, int, elements, sizeof(int),
                        rank, MPI_COMM_WORLD, init_sendbuf, out);

    // Initialize recv buffer
    ALLOCATE_RECVBUFFER(recvbuf, tmp_recvbuf, int, nProcs*elements, sizeof(int),
                        rank, MPI_COMM_WORLD, init_recvbuf, out);

    //Create window
    if (rank == root) {
        ret = MPI_Win_create (recvbuf->get_buffer(), nProcs*elements*sizeof(int), sizeof(int), MPI_INFO_NULL,
                              MPI_COMM_WORLD, &win);
    }
    else {
        ret = MPI_Win_create (sendbuf->get_buffer(), elements*sizeof(int), sizeof(int), MPI_INFO_NULL,
                              MPI_COMM_WORLD, &win);
    }
    if (MPI_SUCCESS != ret) {
        goto out;
    }

    //execute one-sided operations
    ret = type_osc_test (sendbuf->get_buffer(), recvbuf->get_buffer(),
                         elements, MPI_INT, root, MPI_COMM_WORLD, win);
    if (MPI_SUCCESS != ret) {
        printf("Error in type_osc_test. Aborting\n");
        goto out;
    }

    // verify results
    bool res, fret;
    res=true;
    if (rank == 0) {
        if (recvbuf->NeedsStagingBuffer()) {
            HIP_CHECK(recvbuf->CopyFrom(tmp_recvbuf, nProcs*elements*sizeof(int)));
            res = check_recvbuf(tmp_recvbuf, nProcs, rank, elements);
        }
        else {
            res = check_recvbuf((int*) recvbuf->get_buffer(), nProcs, rank, elements);
        }
    }
    fret = report_testresult(argv[0], MPI_COMM_WORLD, sendbuf->get_memchar(), recvbuf->get_memchar(), res);
    report_performance (argv[0], MPI_COMM_WORLD, sendbuf->get_memchar(), recvbuf->get_memchar(), elements,
                        (size_t)(elements *sizeof(int)), 0, 0.0);

 out:
    //Cleanup dynamic buffers
    if (MPI_WIN_NULL != win) {
        MPI_Win_free (&win);
    }
    FREE_BUFFER(sendbuf, tmp_sendbuf);
    FREE_BUFFER(recvbuf, tmp_recvbuf);

    delete (sendbuf);
    delete (recvbuf);

    if (MPI_SUCCESS != ret ) {
        MPI_Abort (MPI_COMM_WORLD, 1);
        return 1;
    }    
    MPI_Finalize ();
    return fret ? 0 : 1;
}


int type_osc_test (void *sbuf, void *rbuf, int count,
                   MPI_Datatype datatype, int root, MPI_Comm comm, MPI_Win win)
{
    int size, rank, ret;
    int tsize;

    MPI_Comm_size (comm, &size);
    MPI_Comm_rank (comm, &rank);
    MPI_Type_size (datatype, &tsize);

#ifdef HIP_MPITEST_OSC_FENCE
    ret = MPI_Win_fence (0, win);
    if (MPI_SUCCESS != ret) {
        return ret;
    }
#endif

#if defined HIP_MPITEST_OSC_GET || defined HIP_MPITEST_OSC_RGET
    if (rank == root ) {
        char *r = (char *)rbuf;
        for (int i = 0; i < size; i ++) {
            if (i != root) {
#ifdef HIP_MPITEST_OSC_LOCK
                ret = MPI_Win_lock(MPI_LOCK_EXCLUSIVE, i, 0, win);
                if (MPI_SUCCESS != ret) {
                    return ret;
                }
#endif
#if defined HIP_MPITEST_OSC_GET
                ret = MPI_Get (r, count, datatype, i, 0, count, datatype, win);
                if (MPI_SUCCESS != ret) {
                    return ret;
                }
#elif defined HIP_MPITEST_OSC_RGET
                MPI_Request req;
                ret = MPI_Rget (r, count, datatype, i, 0, count, datatype, win, &req);
                if (MPI_SUCCESS != ret) {
                    return ret;
                }
                ret = MPI_Wait (&req, MPI_STATUS_IGNORE);
                if (MPI_SUCCESS != ret) {
                    return ret;
                }
#endif
#ifdef HIP_MPITEST_OSC_LOCK
                ret = MPI_Win_unlock(i, win);
                if (MPI_SUCCESS != ret) {
                    return ret;
                }
#endif
            }
            r +=count*tsize;
        }
    }

#elif defined HIP_MPITEST_OSC_PUT || defined HIP_MPITEST_OSC_RPUT
    if (rank != root) {
        MPI_Aint disp = rank * count;
#ifdef HIP_MPITEST_OSC_LOCK
        ret = MPI_Win_lock(MPI_LOCK_EXCLUSIVE, root, 0, win);
        if (MPI_SUCCESS != ret) {
            return ret;
        }
#endif
#if defined HIP_MPITEST_OSC_PUT
        ret = MPI_Put(sbuf, count, datatype, root, disp, count, datatype, win);
        if (MPI_SUCCESS != ret) {
            return ret;
        }
#elif defined HIP_MPITEST_OSC_RPUT
        MPI_Request req;
        ret = MPI_Rput(sbuf, count, datatype, root, disp, count, datatype, win, &req);
        if (MPI_SUCCESS != ret) {
            return ret;
        }
        ret = MPI_Wait (&req, MPI_STATUS_IGNORE);
        if (MPI_SUCCESS != ret) {
            return ret;
        }
#endif
#ifdef HIP_MPITEST_OSC_LOCK
        ret = MPI_Win_unlock(root, win);
        if (MPI_SUCCESS != ret) {
            return ret;
        }
#endif
    }
#endif


#ifdef HIP_MPITEST_OSC_FENCE
    ret = MPI_Win_fence (0, win);
    if (MPI_SUCCESS != ret) {
        return ret;
    }
#endif

#ifdef HIP_MPITEST_OSC_LOCK
    ret = MPI_Barrier(comm);
    if (MPI_SUCCESS != ret) {
        return ret;
    }
#endif

    return MPI_SUCCESS;
}
