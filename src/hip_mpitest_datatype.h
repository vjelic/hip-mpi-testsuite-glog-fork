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
