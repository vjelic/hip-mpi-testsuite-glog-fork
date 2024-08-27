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
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "mpi.h"

#include <hip/hip_runtime.h>
#include <chrono>

#include "hip_mpitest_utils.h"
#include "hip_mpitest_buffer.h"

int elements=64*1024*1024;
hip_mpitest_buffer *sendbuf=NULL;
hip_mpitest_buffer *recvbuf=NULL;

static int coord[2];
static int procs_per_dim;
static int nelem_per_dim;

static void SL_write (int hdl, void *buf, size_t num);


static void init_sendbuf (long *sendbuf, int count, int unused)
{
    for (long i = 0; i < count; i++) {
        sendbuf[i] = i+1;
    }
}

static void init_recvbuf (long *recvbuf, int count)
{
    for (long i = 0; i < count; i++) {
        recvbuf[i] = 0.0;
    }
}

static bool check_recvbuf(long *recvbuf, int nprocs_unused, int rank, int count)
{
    bool res=true;
    long c = 0;

    for (long i=0; i<nelem_per_dim; i++) {
        for (long j = 0; j < nelem_per_dim; j++) {
            long should_be = (coord[0] * procs_per_dim * nelem_per_dim * nelem_per_dim) +
                             (coord[1] * nelem_per_dim) + (i*procs_per_dim * nelem_per_dim) + j+1;
            if (recvbuf[c] != should_be) {
                res = false;
#ifdef VERBOSE
                printf("recvbuf[%d] = %ld should be %ld\n", i, recvbuf[i], should_be);
#endif
                break;
            }
            c++;
        }
    }

    return res;
}

int file_read_all_test (void *sendbuf, int count,
                        MPI_Datatype datatype, MPI_File fh);

int main (int argc, char *argv[])
{
    int res, fd, old_mask, perm;
    int rank, size;
    MPI_File fh;

    bind_device();

    MPI_Init      (&argc, &argv);
    MPI_Comm_size (MPI_COMM_WORLD, &size);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);

    parse_args(argc, argv, MPI_COMM_WORLD);

    // Verify that the number of processes is a perfect square
    procs_per_dim = sqrt(size);
    assert ((procs_per_dim *procs_per_dim) == size);

    // Verify that the number of elements is a perfect square
    nelem_per_dim = sqrt(elements);
    assert ((nelem_per_dim*nelem_per_dim) == elements);

    // Create 2D cartesian topology
    MPI_Comm gridComm;
    int dim[2] = {procs_per_dim, procs_per_dim};
    int period[2] = {0, 0};
    int reorder = 0;

    MPI_Cart_create(MPI_COMM_WORLD, 2, dim, period, reorder, &gridComm);
    MPI_Cart_coords(gridComm, rank, 2, coord);

    long *tmp_sendbuf=NULL, *tmp_recvbuf=NULL;
    // Initialise send buffer
    if (rank == 0) {
        // Forcing the temporary buffer used to write the input data
        // to be hostbuffer, independent of user input. Only recvbuffer
        // relevant for the read test.
        delete sendbuf;
        sendbuf = new hip_mpitest_buffer_host;
        ALLOCATE_SENDBUFFER(sendbuf, tmp_sendbuf, long, elements * size, sizeof(long),
                            rank, MPI_COMM_WORLD, init_sendbuf);
    }

    // Initialize recv buffer
    ALLOCATE_RECVBUFFER(recvbuf, tmp_recvbuf, long, elements, sizeof(long),
                        rank, MPI_COMM_WORLD, init_recvbuf);

    // Create input file
    if (rank == 0) {
        old_mask = umask(022);
        umask (old_mask);
        perm = old_mask^0666;

        fd = open ("testout.out", O_CREAT|O_WRONLY, perm );
        if (-1 == fd) {
            fprintf(stderr, "Error in creating input file. Aborting\n");
            MPI_Abort (MPI_COMM_WORLD, 1);
            return 1;
        }

        SL_write(fd, sendbuf->get_buffer(), elements*size*sizeof(long));
        close (fd);
        rename ("testout.out", "testin.in");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // open file and set file view
    MPI_Datatype fview;
    int startV[2] = {coord[0]*nelem_per_dim, coord[1]*nelem_per_dim};
    int arrsizeV[2] = {dim[0]*nelem_per_dim, dim[1]*nelem_per_dim};
    int gridsizeV[2] = {nelem_per_dim, nelem_per_dim};

#ifdef DEBUG
    printf("%d: coords[%d][%d] start[%d][%d] size[%d][%d] gridsize[%d][%d]\n", rank,
           coord[0], coord[1], coord[0]*nelem_per_dim, coord[1]*nelem_per_dim,
           dim[0]*nelem_per_dim, dim[1]*nelem_per_dim, nelem_per_dim, nelem_per_dim);
#endif
    MPI_Type_create_subarray(2, arrsizeV, gridsizeV, startV, MPI_ORDER_C, MPI_LONG, &fview);
    MPI_Type_commit (&fview);

    MPI_File_open(gridComm, "testin.in", MPI_MODE_RDONLY,
                  MPI_INFO_NULL, &fh);
    MPI_File_set_view (fh, 0, MPI_LONG, fview, "native", MPI_INFO_NULL);

    MPI_Barrier(MPI_COMM_WORLD);
    auto t1s = std::chrono::high_resolution_clock::now();
    res = file_read_all_test (recvbuf->get_buffer(), elements,
                              MPI_LONG, fh);
    if (MPI_SUCCESS != res) {
        fprintf(stderr, "Error in file_read_test. Aborting\n");
        MPI_Abort (MPI_COMM_WORLD, 1);
        return 1;
    }
    MPI_File_close (&fh);
    auto t1e = std::chrono::high_resolution_clock::now();
    double t1 = std::chrono::duration<double>(t1e-t1s).count();

    // verify results
    bool ret;
    if (recvbuf->NeedsStagingBuffer()) {
        HIP_CHECK(recvbuf->CopyFrom(tmp_recvbuf, elements*sizeof(long)));
        ret = check_recvbuf(tmp_recvbuf, size, rank, elements);
    }
    else {
        ret = check_recvbuf((long*) recvbuf->get_buffer(), size, rank, elements);
    }

    bool fret = report_testresult(argv[0], MPI_COMM_WORLD, '-', recvbuf->get_memchar(),
                                  ret);
    report_performance (argv[0], MPI_COMM_WORLD, '-', recvbuf->get_memchar(),
                        elements, (size_t)(elements * sizeof(long)), 1, t1);

    //Free buffers
    if (rank == 0) {
        FREE_BUFFER(sendbuf, tmp_sendbuf);
        delete (sendbuf);
        unlink("testin.in");
    }

    FREE_BUFFER(recvbuf, tmp_recvbuf);
    delete (recvbuf);

    MPI_Type_free(&fview);

    MPI_Finalize ();
    return fret ? 0 : 1;
}

int file_read_all_test (void *recvbuf, int count, MPI_Datatype datatype, MPI_File fh )
{
    int ret;

    ret = MPI_File_read_all (fh, recvbuf, count, datatype, MPI_STATUS_IGNORE);
    return ret;
}

void SL_write (int hdl, void *buf, size_t num )
{
    int lcount=0;
    int a;
    char *wbuf = ( char *)buf;

    do {
    a = write ( hdl, wbuf, num);
    if ( a == -1 ) {
        if ( errno == EINTR || errno == EAGAIN ||
             errno == EINPROGRESS || errno == EWOULDBLOCK) {
            continue;
        } else {
            printf("SL_write: error while writing to file %d %s\n", hdl, strerror(errno));
            return;
        }
        lcount++;
        a=0;
    }

    num -= a;
    wbuf += a;

    } while ( num > 0 );

    return;
}

