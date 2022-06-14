/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
** Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
*/

#ifndef __HIP_MPITEST_DATATYPE__
#define __HIP_MPITEST_DATATYPE__

#include "mpi.h"

class hip_mpitest_datatype {
 protected:
    MPI_Datatype datatype;
 public:
    MPI_Datatype get_mpi_type() {
	return datatype;
    }
    MPI_Aint get_extent() {
	MPI_Aint lb, extent;
	MPI_Type_get_extent (datatype, &lb, &extent);
	return extent;
    }
    int get_size() {
	int type_size;
	MPI_Type_size (datatype, &type_size);
	return type_size;
    }
    virtual int get_num_elements()=0;
    virtual void init_sendbuf   (void *sendbuf, int count, int mynode)=0;
    virtual void init_recvbuf   (void *recvbuf, int count)=0;
    virtual bool check_recvbuf  (void *recvbuf, int numprocs, int rank, int count)=0;
};


#if defined (HIP_TYPE_RESIZED)
#include "hip_type_resized.h"
#elif defined (HIP_TYPE_STRUCT)
#include "hip_type_struct.h"
#endif

#endif // __HIP_MPITEST_DATATYPE__
