/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
** Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
*/

#include <stdio.h>
#include "mpi.h"

#include <hip/hip_runtime.h>
#include <chrono>

#include "hip_mpitest_utils.h"
#include "hip_mpitest_datatype.h"
#include "hip_mpitest_buffer.h"


#define NITER 10
int  elements=3;

hip_mpitest_buffer *sendbuf=NULL;
hip_mpitest_buffer *recvbuf=NULL;

int type_p2p_nb_test (void *sendbuf, void *recvbuf, int count,
                      MPI_Datatype datatype, MPI_Comm comm, int niterations);

int main (int argc, char *argv[])
{
    int res, rank, size;
    MPI_Comm comm = MPI_COMM_WORLD;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    bind_device();
    parse_args(argc, argv, comm);

    hip_mpitest_datatype *dat = new (TEST_DATATYPE);
    char *tmp_sendbuf=NULL, *tmp_recvbuf=NULL;

    // Initialise send buffer
    ALLOCATE_SENDBUFFER(sendbuf, tmp_sendbuf, char, elements, dat->get_extent(),
                        rank, comm, dat->init_sendbuf);

    // Initialize recv buffer
    ALLOCATE_RECVBUFFER(recvbuf, tmp_recvbuf, char, elements, dat->get_extent(),
                        rank, comm, dat->init_recvbuf);

    //Warmup
    res = type_p2p_nb_test (sendbuf->get_buffer(), recvbuf->get_buffer(), elements,
                            dat->get_mpi_type(), comm, 1);
    if (MPI_SUCCESS != res ) {
        fprintf(stderr, "Error in type_p2p_test. Aborting\n");
        MPI_Abort (comm, 1);
        return 1;
    }

    // execute the point-to-point tests
    MPI_Barrier(comm);
    auto t1s = std::chrono::high_resolution_clock::now();
    res = type_p2p_nb_test(sendbuf->get_buffer(), recvbuf->get_buffer(), elements,
                           dat->get_mpi_type(), comm, NITER);
    if (MPI_SUCCESS != res) {
        fprintf(stderr, "Error in type_p2p_test. Aborting\n");
        MPI_Abort (comm, 1);
        return 1;
    }
    auto t1e = std::chrono::high_resolution_clock::now();    
    double t1 = std::chrono::duration<double>(t1e-t1s).count();

    // verify results 
    bool ret = true;
    if (recvbuf->NeedsStagingBuffer()) {
        HIP_CHECK(recvbuf->CopyFrom(tmp_recvbuf, elements*dat->get_extent()));
        ret = dat->check_recvbuf(tmp_recvbuf, size, rank, elements);
    }
    else {
        ret = dat->check_recvbuf(recvbuf->get_buffer(), size, rank, elements);
    }

    bool fret = report_testresult(argv[0], comm, sendbuf->get_memchar(), recvbuf->get_memchar(), ret);
    report_performance (argv[0], comm, sendbuf->get_memchar(), recvbuf->get_memchar(), elements,
                        (size_t)(elements * dat->get_size()), NITER, t1);

    //Free buffers
    FREE_BUFFER(sendbuf, tmp_sendbuf);
    FREE_BUFFER(recvbuf, tmp_recvbuf);

    delete (sendbuf);
    delete (recvbuf);
    delete (dat);

    MPI_Finalize ();
    return fret ? 0 : 1;
}


int type_p2p_nb_test ( void *sendbuf, void *recvbuf, int count,
                       MPI_Datatype datatype, MPI_Comm comm, int niterations)
{
    int size, rank, ret;
    int tag=251;
    MPI_Request reqs[2];

    MPI_Comm_size (comm, &size);
    MPI_Comm_rank (comm, &rank);

    // send buffer to right, receive from left
    int left = rank - 1;
    if (left < 0) left = size-1;
    int right = rank + 1;
    if (right == size) right = 0;

    for (int i=0; i<niterations; i++) {
      ret = MPI_Irecv (recvbuf, count, datatype, left, tag, comm, &reqs[1]);
      if (MPI_SUCCESS != ret) {
	return ret;
      }
      ret = MPI_Isend (sendbuf, count, datatype, right, tag, comm, &reqs[0]);
      if (MPI_SUCCESS != ret) {
	return ret;
      }
      ret = MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
      if (MPI_SUCCESS != ret) {
	return ret;
      }
    }

    return MPI_SUCCESS;
}
