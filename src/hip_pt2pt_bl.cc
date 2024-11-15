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

int elements = 1024;
hip_mpitest_buffer *sendbuf = NULL;
hip_mpitest_buffer *recvbuf = NULL;

static void init_sendbuf(int *sendbuf, int count, int mynode)
{
    // Rank 0 sends "1" and Rank 1 sends "2"
    for (int i = 0; i < count; i++) {
        sendbuf[i] = mynode + 1;
    }
}

static void init_recvbuf(int *recvbuf, int count)
{
    for (int i = 0; i < count; i++) {
        recvbuf[i] = 0;
    }
}

static bool check_recvbuf(int *recvbuf, int nProcs, int rank, int count)
{
    bool res = true;
    int result = 0;
    // Rank 0 receives "2" and Rank 1 receives "1"
    if (rank == 0) {
        result = 2;
    }
    else {
        result = 1;
    }

    for (int i = 0; i < count; i++) {
        if (recvbuf[i] != result) {
            res = false;
#ifdef VERBOSE
            printf("recvbuf[%d] = %d expected %d\n", i, recvbuf[i], result);
#endif
            break;
        }
    }
    return res;
}

int type_p2p_bl_test(int *sendbuf, int *recvbuf, int count, MPI_Comm comm);
int type_p2p_bsend_test(int *sendbuf, int *recvbuf, int count, MPI_Comm comm);
int type_p2p_ssend_test(int *sendbuf, int *recvbuf, int count, MPI_Comm comm);

int main(int argc, char *argv[])
{
    int rank, nProcs;
    int root = 0;
    int ret;

    bind_device();

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (nProcs != 2) {
        printf("This test requires exactly two processes!\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    parse_args(argc, argv, MPI_COMM_WORLD);

    int *tmp_sendbuf = NULL, *tmp_recvbuf = NULL;
    // Initialise send buffer
    ALLOCATE_SENDBUFFER(sendbuf, tmp_sendbuf, int, nProcs *elements, sizeof(int),
                        rank, MPI_COMM_WORLD, init_sendbuf, out);

    // Initialize recv buffer
    ALLOCATE_RECVBUFFER(recvbuf, tmp_recvbuf, int, nProcs *elements, sizeof(int),
                        rank, MPI_COMM_WORLD, init_recvbuf, out);

    // execute point-to-point operations
#if defined HIP_MPITEST_BSEND
    ret = type_p2p_bsend_test((int *)sendbuf->get_buffer(), (int *)recvbuf->get_buffer(),
                              elements, MPI_COMM_WORLD);
    if (MPI_SUCCESS != ret) {
        printf("Error in type_p2p_bsend_test. Aborting\n");
        goto out;
    }
#elif defined HIP_MPITEST_SSEND
    ret = type_p2p_ssend_test((int *)sendbuf->get_buffer(), (int *)recvbuf->get_buffer(),
                              elements, MPI_COMM_WORLD);
    if (MPI_SUCCESS != ret) {
        printf("Error in type_p2p_ssend_test. Aborting\n");
        goto out;
    }
#else
    ret = type_p2p_bl_test((int *)sendbuf->get_buffer(), (int *)recvbuf->get_buffer(),
                            elements, MPI_COMM_WORLD);
    if (MPI_SUCCESS != ret) {
        printf("Error in type_p2p_bl_test. Aborting\n");
        goto out;
    }
#endif

    // verify results
    bool res, fret;
    res = true;
    if (recvbuf->NeedsStagingBuffer()) {
        HIP_CHECK(recvbuf->CopyFrom(tmp_recvbuf, nProcs * elements * sizeof(int)));
        res = check_recvbuf(tmp_recvbuf, nProcs, rank, elements);
    }
    else {
        res = check_recvbuf((int *)recvbuf->get_buffer(), nProcs, rank, elements);
    }
    fret = report_testresult(argv[0], MPI_COMM_WORLD, sendbuf->get_memchar(), recvbuf->get_memchar(), res);

out:
    // Cleanup dynamic buffers
    FREE_BUFFER(sendbuf, tmp_sendbuf);
    FREE_BUFFER(recvbuf, tmp_recvbuf);
    delete (sendbuf);
    delete (recvbuf);

    if (MPI_SUCCESS != ret) {
        MPI_Abort(MPI_COMM_WORLD, 1);
        return ret;
    }

    MPI_Finalize();
    return fret ? 0 : 1;
}

int type_p2p_bl_test(int *sbuf, int *rbuf, int count, MPI_Comm comm)
{
    int size, rank, ret;
    int tag = 251;
    MPI_Status status;

    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    if (rank == 0) {
        ret = MPI_Send(sbuf, count, MPI_INT, 1, tag, comm);
        if (MPI_SUCCESS != ret) {
            return ret;
        }
        ret = MPI_Recv(rbuf, count, MPI_INT, 1, tag, comm, &status);
        if (MPI_SUCCESS != ret) {
            return ret;
        }
    }
    if (rank == 1) {
        ret = MPI_Recv(rbuf, count, MPI_INT, 0, tag, comm, &status);
        if (MPI_SUCCESS != ret) {
            return ret;
        }
        ret = MPI_Send(sbuf, count, MPI_INT, 0, tag, comm);
        if (MPI_SUCCESS != ret) {
            return ret;
        }
    }

    return MPI_SUCCESS;
}

int type_p2p_bsend_test(int *sbuf, int *rbuf, int count, MPI_Comm comm)
{
    int size, rank, ret = MPI_SUCCESS;
    int tag = 251;
    MPI_Status status;
    int *buffer;
    int msg_size, buffersize;

    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    MPI_Pack_size(count, MPI_INT, comm, &msg_size);
    buffersize = MPI_BSEND_OVERHEAD + msg_size;
    buffer = (int *) malloc(buffersize);
    if (NULL == buffer) {
        return MPI_ERR_OTHER;
    }

    ret = MPI_Buffer_attach(buffer, buffersize);
    if (MPI_SUCCESS != ret) {
        free (buffer);
        return ret;
    }

    if (rank == 0) {
        ret = MPI_Bsend(sbuf, count, MPI_INT, 1, tag, comm);
        if (MPI_SUCCESS != ret) {
            goto out;
        }
        ret = MPI_Recv(rbuf, count, MPI_INT, 1, tag, comm, &status);
        if (MPI_SUCCESS != ret) {
            goto out;
        }
    }
    if (rank == 1) {
        ret = MPI_Recv(rbuf, count, MPI_INT, 0, tag, comm, &status);
        if (MPI_SUCCESS != ret) {
            goto out;
        }
        ret = MPI_Bsend(sbuf, count, MPI_INT, 0, tag, comm);
        if (MPI_SUCCESS != ret) {
            goto out;
        }
    }

 out:
    if (NULL != buffer) {
        MPI_Buffer_detach(&buffer, &buffersize);
        free (buffer);
    }

    return ret;
}

int type_p2p_ssend_test(int *sbuf, int *rbuf, int count, MPI_Comm comm) {
    int size, rank, ret;
    int tag = 251;

    MPI_Status status;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (rank == 0) {
        ret = MPI_Ssend(sbuf, count, MPI_INT, 1, tag, comm);
        if (MPI_SUCCESS != ret) {
            return ret;
        }
        ret = MPI_Recv(rbuf, count, MPI_INT, 1, tag, comm, &status);
        if (MPI_SUCCESS != ret) {
            return ret;
        }
    } else if (rank == 1) {
        ret = MPI_Recv(rbuf, count, MPI_INT, 0, tag, comm, &status);
        if (MPI_SUCCESS != ret) {
            return ret;
        }
        ret = MPI_Ssend(sbuf, count, MPI_INT, 0, tag, comm);
        if (MPI_SUCCESS != ret) {
            return ret;
        }
    }

    return MPI_SUCCESS;
}
