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

#ifndef __HIP_TYPE_RESIZED__
#define __HIP_TYPE_RESIZED__

#include "mpi.h"
#include "hip_mpitest_datatype.h"

#undef  TEST_DATATYPE
#define TEST_DATATYPE hip_type_resized

#ifndef A_WIDTH
#define A_WIDTH 1024
#endif
#define GAPSIZE 32

struct _s2 {
    int        a[A_WIDTH];
    int doNotUse[GAPSIZE];
};
typedef struct _s2 _s2;

class hip_type_resized: public hip_mpitest_datatype {
 public:
    hip_type_resized() {
        _s2 testarr[2];
        MPI_Aint a1, a2;

        MPI_Get_address(&(testarr[0].a[0]), &a1);
        MPI_Get_address(&(testarr[1].a[0]), &a2);

        MPI_Aint extent = a2-a1;
        MPI_Datatype dat1;

        MPI_Type_contiguous(A_WIDTH, MPI_INT, &dat1);
        MPI_Type_create_resized(dat1, 0, extent, &datatype);
        MPI_Type_commit(&datatype);
        MPI_Type_free (&dat1);
    }
    ~hip_type_resized() {
        MPI_Type_free (&datatype);
    }

    int get_num_elements() {
        return A_WIDTH;
    }

    void init_sendbuf (void *sbuf, int count, int mynode)
    {
        _s2 *sendbuf = (_s2 *) sbuf;

        for (int i=0; i<count; i++) {
            for (int j=0; j<A_WIDTH; j++) {
                sendbuf[i].a[j] = mynode * 3 + i;
            }
            for (int j=0; j<GAPSIZE; j++) {
                sendbuf[i].doNotUse[j] = mynode;
            }
        }
    }

    void init_recvbuf (void *rbuf, int count)
    {
        _s2 *recvbuf = (_s2*) rbuf;

        for (int i=0; i<count; i++) {
            for (int j=0; j<A_WIDTH; j++ ) {
                recvbuf[i].a[j] = -1;
            }
            for (int j=0; j<GAPSIZE; j++) {
                recvbuf[i].doNotUse[j] = -1;
            }
        }
    }

    bool check_recvbuf (void *rbuf, int numprocs, int rank, int count)
    {
        bool res = true;
        _s2 *recvbuf = (_s2 *) rbuf;
        int recvfrom = rank - 1;
        if (recvfrom < 0 ) recvfrom = numprocs -1;

        for (int i=0; i<count; i++) {
            for (int l=0; l<A_WIDTH; l++) {
                if ( (recvbuf[i].a[l] != (recvfrom*3)+i) ) {
                    res = false;
#ifdef VERBOSE
                    printf("recvbuf[%d].a[%d] = %d \n", i, l, recvbuf[i].a[l]);
#endif
                }
            }
            for (int l=0; l<GAPSIZE; l++) {
                if ( recvbuf[i].doNotUse[l] != -1 ) {
                    res = false;
#ifdef VERBOSE
                    printf("recvbuf[%d].doNotUse[%d] = %d \n", k, l, recvbuf[i].doNotUse[l]);
#endif
                }
            }
        }

        return res;
    }
};

#endif
