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
#include "hip_mpitest_datatype.h"
#include "hip_mpitest_buffer.h"


#define NITER 10
int  elements=3;

hip_mpitest_buffer *sendbuf=NULL;
hip_mpitest_buffer *recvbuf=NULL;

static void init_contg_sendbuf (void *buf, int totalcount, int rank)
{
    int *sbuf = (int  *)buf;
    int l=0;
    int count = totalcount / (2*A_WIDTH);

    for (int i=0; i<count; i++) {
        for (int j=0; j<2*A_WIDTH; j++, l++) {
            sbuf[l] = rank*3+i;
        }
    }
}

static void init_contg_recvbuf (void *buf, int count)
{
    int *rbuf = (int*)buf;
    for (int i=0; i<count; i++) {
        rbuf[i]=-1;
    }
}

static bool check_contg_recvbuf(void *buf, int numprocs, int rank, int totalcount)
{
    int *recvbuf = (int*)buf;
    bool res = true;
    int l=0;
    int count = totalcount / (2*A_WIDTH);

    for (int i=0; i<count && res != false; i++) {
        for (int j=0; j<2*A_WIDTH; j++, l++) {
             if ( (recvbuf[l] != (rank*3)+i) ) {
                 res = false;
#ifdef VERBOSE
                 printf("recvbuf[%d] = %d\n", i, recvbuf[i]);
#endif
                 break;
             }
        }
    }
    return res;
}

static int packunpack_test (void *sendbuf, void *recvbuf, int count,
                            MPI_Datatype datatype, MPI_Comm comm, int niterations);

int main (int argc, char *argv[])
{
    int res, rank, size;
    MPI_Comm comm = MPI_COMM_WORLD;

    bind_device();

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    parse_args(argc, argv, comm);

    hip_mpitest_datatype *dat = new (TEST_DATATYPE);
    int *tmp_sendbuf=NULL, *tmp_recvbuf=NULL;

    // Initialise send buffer
#ifdef HIP_MPITEST_UNPACK
    ALLOCATE_SENDBUFFER(sendbuf, tmp_sendbuf, int, dat->get_num_elements()*elements, sizeof(int),
                        rank, comm, init_contg_sendbuf);
#else
    ALLOCATE_SENDBUFFER(sendbuf, tmp_sendbuf, int, elements, dat->get_extent(),
                        rank, comm, dat->init_sendbuf);
#endif

    // Initialize recv buffer
#ifdef HIP_MPITEST_UNPACK
    ALLOCATE_RECVBUFFER(recvbuf, tmp_recvbuf, int, elements, dat->get_extent(),
                        rank, comm, dat->init_recvbuf);
#else
    ALLOCATE_RECVBUFFER(recvbuf, tmp_recvbuf, int, dat->get_num_elements()*elements, sizeof(int),
                        rank, comm, init_contg_recvbuf);
#endif
    //Warmup
    res = packunpack_test (sendbuf->get_buffer(), recvbuf->get_buffer(), elements,
                           dat->get_mpi_type(), comm, 1);
    if (MPI_SUCCESS != res ) {
        fprintf(stderr, "Error in packunpack_test. Aborting\n");
        MPI_Abort (comm, 1);
        return 1;
    }

    // execute the point-to-point tests
    MPI_Barrier(comm);
    auto t1s = std::chrono::high_resolution_clock::now();
    res = packunpack_test(sendbuf->get_buffer(), recvbuf->get_buffer(), elements,
                          dat->get_mpi_type(), comm, NITER);
    if (MPI_SUCCESS != res) {
        fprintf(stderr, "Error in packunpack_test. Aborting\n");
        MPI_Abort (comm, 1);
        return 1;
    }
    auto t1e = std::chrono::high_resolution_clock::now();
    double t1 = std::chrono::duration<double>(t1e-t1s).count();

    // verify results
    bool ret = true;
    if (recvbuf->NeedsStagingBuffer()) {
#ifdef HIP_MPITEST_UNPACK
        HIP_CHECK(recvbuf->CopyFrom(tmp_recvbuf, elements*dat->get_extent()));
        ret = dat->check_recvbuf(tmp_recvbuf, size, rank+1, elements);
#else
        HIP_CHECK(recvbuf->CopyFrom(tmp_recvbuf, elements*dat->get_num_elements()*sizeof(int)));
        ret = check_contg_recvbuf(tmp_recvbuf, 1, rank, dat->get_num_elements()*elements);
#endif
    }
    else {
#ifdef HIP_MPITEST_UNPACK
        ret = dat->check_recvbuf(recvbuf->get_buffer(), size, rank+1, elements);
#else
        ret = check_contg_recvbuf(recvbuf->get_buffer(), 1, rank, dat->get_num_elements()*elements);
#endif
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


int packunpack_test ( void *sendbuf, void *recvbuf, int count,
                      MPI_Datatype datatype, MPI_Comm comm,
                      int niterations)
{
    int      ret, pos;
    int      type_size;
    MPI_Aint lb, type_extent;

    MPI_Type_size(datatype, &type_size);
    MPI_Type_get_extent(datatype, &lb, &type_extent);

    for (int i=0; i<niterations; i++) {
        pos = 0;
#ifdef HIP_MPITEST_UNPACK
        ret = MPI_Unpack(sendbuf, type_size*count, &pos, recvbuf, count, datatype, comm);
#else
        ret = MPI_Pack(sendbuf, count, datatype, recvbuf, type_extent*count, &pos, comm);
#endif
        if (MPI_SUCCESS != ret) {
            return ret;
        }
    }

    return MPI_SUCCESS;
}
