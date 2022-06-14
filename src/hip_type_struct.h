/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
** Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
*/

#ifndef __HIP_TYPE_STRUCT__
#define __HIP_TYPE_STRUCT__

#include "mpi.h"
#include "hip_mpitest_datatype.h"

#undef  TEST_DATATYPE
#define TEST_DATATYPE hip_type_struct

#ifndef A_WIDTH
#define A_WIDTH 512
#endif
#define GAPSIZE 32

struct _s2 {
    int        a[A_WIDTH];
    int doNotUse[GAPSIZE];
    int        b[A_WIDTH];
};
typedef struct _s2 _s2;

class hip_type_struct: public hip_mpitest_datatype {
 public:
    hip_type_struct() {
        _s2 testarr[2];
        MPI_Aint a1, a2, b1;

        MPI_Get_address(&(testarr[0].a[0]), &a1);
        MPI_Get_address(&(testarr[1].a[0]), &a2);
        MPI_Get_address(&(testarr[0].b[0]), &b1);

        MPI_Aint extent = a2-a1;
        MPI_Aint displs[2] = {0, b1-a1};
        MPI_Datatype dats[2] = {MPI_INT, MPI_INT};
        int blength[2] = {A_WIDTH, A_WIDTH};

        MPI_Type_create_struct(2, blength, displs, dats, &datatype);
        MPI_Type_commit(&datatype);
    }
    ~hip_type_struct() {
        MPI_Type_free (&datatype);
    }

    int get_num_elements() {
        return 2*A_WIDTH;
    }

    void init_sendbuf (void *sbuf, int count, int rank)
    {
        _s2 *sendbuf = (_s2 *) sbuf;

        for (int i = 0; i < count; i++) {
            for (int j=0; j<A_WIDTH; j++ ) {
                sendbuf[i].a[j] = rank * 3 + i;
            }
            for (int j = 0; j < GAPSIZE; j++) {
                sendbuf[i].doNotUse[j] = rank;
            }
            for (int j=0; j<A_WIDTH; j++ ) {
                sendbuf[i].b[j] = rank * 3 + i;
            }        
        }  
    }

    void init_recvbuf (void *rbuf, int count)
    {
        _s2 *recvbuf = (_s2*) rbuf;

        for (int i = 0; i < count; i++) {
            for ( int j=0; j<A_WIDTH; j++ ) {
                recvbuf[i].a[j] = -1;
            }
            for (int j = 0; j<GAPSIZE; j++) {
                recvbuf[i].doNotUse[j] = -1;
            }
            for ( int j=0; j<A_WIDTH; j++ ) {
                recvbuf[i].b[j] = -1;
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
                    printf("recvbuf[%d].doNotUse[%d] = %d \n", i, l, recvbuf[i].doNotUse[l]);
#endif
                }
            }
            for (int l=0; l<A_WIDTH; l++) {
                if ( (recvbuf[i].b[l] != (recvfrom*3)+i) ) {
                    res = false;
#ifdef VERBOSE
                    printf("recvbuf[%d].b[%d] = %d \n", i, l, recvbuf[i].b[l]);
#endif
                }
            }
        }

        return res;
    }
};

#endif
