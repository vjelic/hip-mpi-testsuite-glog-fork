/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
** Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
*/
#ifndef __HIP_MPITEST_COMPUTE_KERNEL__
#define __HIP_MPITEST_COMPUTE_KERNEL__

#include <hip/hip_runtime_api.h>

typedef struct hip_mpitest_compute_params_s {
    int         N, K, Kthresh, Rthresh, niter;
    long       *Ahost, *Adevice;
    double     *Afhost, *Afdevice;
    double      est_runtime;
    hipStream_t stream;
} hip_mpitest_compute_params_t;

int  hip_mpitest_compute_init(hip_mpitest_compute_params_t &params);
void hip_mpitest_compute_set_params(hip_mpitest_compute_params_t &params, double runtime);
void hip_mpitest_compute_launch (hip_mpitest_compute_params_t &params);
void hip_mpitest_compute_fini(hip_mpitest_compute_params_t &params);

#endif
