/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
** Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
*/

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

int type_osc_test ( void *sendbuf, void *recvbuf, int count,
                    MPI_Datatype datatype, int root, MPI_Comm comm);

int main (int argc, char *argv[])
{
    int rank, nProcs;
    int root = 0; //checkbuff will not work for any other root value right now
    
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

    //execute one-sided operations
    int res = type_osc_test (sendbuf->get_buffer(), recvbuf->get_buffer(),
                             elements, MPI_INT, root, MPI_COMM_WORLD);
    if (MPI_SUCCESS != res) {
        printf("Error in type_osc_test. Aborting\n");
        MPI_Abort (MPI_COMM_WORLD, 1);
        return 1;
    }

    // verify results 
    bool ret=true;
    if (rank == 0) {
        if (recvbuf->NeedsStagingBuffer()) {
            HIP_CHECK(recvbuf->CopyFrom(tmp_recvbuf, nProcs*elements*sizeof(int)));
            ret = check_recvbuf(tmp_recvbuf, nProcs, rank, elements);
        }
        else {
            ret = check_recvbuf((int*) recvbuf->get_buffer(), nProcs, rank, elements);
        }
    }
    bool fret = report_testresult(argv[0], MPI_COMM_WORLD, sendbuf->get_memchar(), recvbuf->get_memchar(), ret);
    report_performance (argv[0], MPI_COMM_WORLD, sendbuf->get_memchar(), recvbuf->get_memchar(), elements,
                        (size_t)(elements *sizeof(int)), 0, 0.0);

    //Cleanup dynamic buffers
    FREE_BUFFER(sendbuf, tmp_sendbuf);
    FREE_BUFFER(recvbuf, tmp_recvbuf);

    delete (sendbuf);
    delete (recvbuf);

    MPI_Finalize ();
    return fret ? 0 : 1;
}


int type_osc_test (void *sbuf, void *rbuf, int count,
                   MPI_Datatype datatype, int root, MPI_Comm comm)
{
    int size, rank, ret;
    MPI_Win win;
    int tsize;

    MPI_Comm_size (comm, &size);
    MPI_Comm_rank (comm, &rank);
    MPI_Type_size (datatype, &tsize);

    if (rank == root) {
        ret = MPI_Win_create (rbuf, count*tsize*size, tsize, MPI_INFO_NULL, comm, &win);
    }
    else {
        ret = MPI_Win_create (sbuf, tsize*size, tsize, MPI_INFO_NULL, comm, &win);
    }
    if (MPI_SUCCESS != ret) {
        return ret;
    }

    ret = MPI_Win_fence (0, win);
    if (MPI_SUCCESS != ret) {
        return ret;
    }

#ifdef HIP_MPITEST_OSC_GET
    if (rank == root ) {
        char *r = (char *)rbuf;
        for (int i = 0; i < size; i ++) {
            if (i != root) {
                ret = MPI_Get (r, count, datatype, i, 0, count, datatype, win);
                if (MPI_SUCCESS != ret) {
                    return ret;
                }
            }
            r +=count*tsize;
        }
    }
#elif defined HIP_MPITEST_OSC_PUT || 1
    if (rank != root) {
        MPI_Aint disp = rank * count;
        ret = MPI_Put(sbuf, count, datatype, root, disp, count, datatype, win);
        if (MPI_SUCCESS != ret) {
            return ret;
        }
    }
#endif
    MPI_Barrier(comm);

    ret = MPI_Win_fence (0, win);
    if (MPI_SUCCESS != ret) {
        return ret;
    }

    ret = MPI_Win_free (&win);

    return MPI_SUCCESS;
}
