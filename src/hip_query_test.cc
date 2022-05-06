/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
** Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
*/

#include <stdio.h>
#include "mpi.h"

#include "mpi-ext.h" /* Needed for ROCm-aware check */

int main(int argc, char *argv[])
{

    MPI_Init(&argc, &argv);

    if (MPIX_Query_rocm_support()) {
        printf("This MPI library has support for ROCm buffers.\n");
    } else {
        printf("This MPI library does not have support for ROCm buffers.\n");
    }
    MPI_Finalize();

    return 0;
}
