/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
** Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
*/

#ifndef __HIP_MPITEST_BUFFER__
#define __HIP_MPITEST_BUFFER__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <hip/hip_runtime.h>


enum HIP_MPITEST_MEMTYPE {
      HIP_MPITEST_MEMTYPE_HOST=0,
      HIP_MPITEST_MEMTYPE_DEVICE,
      HIP_MPITEST_MEMTYPE_MANAGED,
      HIP_MPITEST_MEMTYPE_HOSTMALLOC,
      HIP_MPITEST_MEMTYPE_HOSTREGISTER,
      HIP_MPITEST_MEMTYPE_LAST
};

const char hip_mpitest_memtype_chars[HIP_MPITEST_MEMTYPE_LAST] = {'H','D','M','O','R'};

class hip_mpitest_buffer {
 protected:
    void                *buffer;
    HIP_MPITEST_MEMTYPE memtype;
    char                memchar;
    char            memname[32];

 public:
    void* get_buffer() {
	return buffer;
    }
    char get_memchar() {
	return memchar;
    }
    char *get_memname() {
	return memname;
    }

    virtual hipError_t  Allocate(size_t nBytes)=0;
    virtual hipError_t  CopyTo(void* src, size_t nBytes)=0;
    virtual hipError_t  CopyFrom(void* dst, size_t nBytes)=0;
    virtual hipError_t  Free ()=0;
    virtual bool        NeedsStagingBuffer()=0;
};


class hip_mpitest_buffer_host: public hip_mpitest_buffer {
 public:
    hip_mpitest_buffer_host () {
	memtype = HIP_MPITEST_MEMTYPE_HOST;
	memchar = 'H';
	strncpy (memname, "malloc", 32);
    }

    bool NeedsStagingBuffer() {
	return false;
    }

    hipError_t Allocate (size_t nBytes) {
	hipError_t err = hipErrorMemoryAllocation;
	char *tbuf = (char *) malloc (nBytes);
	if (NULL != tbuf) {
	    err = hipSuccess;
	    buffer = tbuf;
	}
	return err;
    }

    hipError_t Free () {
	free(buffer);
	buffer = NULL;
	return hipSuccess;
    }

    hipError_t CopyTo(void *src, size_t nBytes) {
	memcpy(buffer, src, nBytes);
	return hipSuccess;
    }

    hipError_t CopyFrom(void *dst, size_t nBytes) {
	memcpy(dst, buffer, nBytes);
	return hipSuccess;
    }
};

class hip_mpitest_buffer_device: public hip_mpitest_buffer {
 public:
    hip_mpitest_buffer_device () {
	memtype = HIP_MPITEST_MEMTYPE_DEVICE;
	memchar = 'D';
	strncpy (memname, "hipMalloc", 32);
    }

    bool NeedsStagingBuffer() {
	return true;
    }

    hipError_t Allocate (size_t nBytes) {
	return hipMalloc((void **)&buffer, nBytes);
    }

    hipError_t Free () {
	hipError_t err = hipFree(buffer);
	buffer = NULL;
	return err;
    }

    hipError_t CopyTo(void *src, size_t nBytes) {
	return hipMemcpy(buffer, src, nBytes, hipMemcpyDefault);
    }
    hipError_t CopyFrom(void *dst, size_t nBytes) {
	return hipMemcpy(dst, buffer, nBytes, hipMemcpyDefault);
    }

};


class hip_mpitest_buffer_managed: public hip_mpitest_buffer {
 public:
    hip_mpitest_buffer_managed () {
	memtype = HIP_MPITEST_MEMTYPE_MANAGED;
	memchar = 'M';
	strncpy (memname, "hipMallocManaged", 32);
    }

    bool NeedsStagingBuffer() {
	return false;
    }

    hipError_t Allocate(size_t nBytes) {
	return hipMallocManaged((void**) &buffer, nBytes);
    }

    hipError_t Free() {
	hipError_t err = hipFree(buffer);
	buffer = NULL;
	return err;
    }

    hipError_t CopyTo(void *src, size_t nBytes) {
	return hipMemcpy(buffer, src, nBytes, hipMemcpyDefault);
    }

    hipError_t CopyFrom(void *dst, size_t nBytes) {
	return hipMemcpy(dst, buffer, nBytes, hipMemcpyDefault);
    }
};

class hip_mpitest_buffer_hostmalloc: public hip_mpitest_buffer {
 public:
    hip_mpitest_buffer_hostmalloc () {
	memtype = HIP_MPITEST_MEMTYPE_HOSTMALLOC;
	memchar = 'O';
	strncpy (memname, "hipHostMalloc", 32);
    }

    bool NeedsStagingBuffer() {
	return false;
    }

    hipError_t Allocate(size_t nBytes) {
	return hipHostMalloc((void **)&buffer, nBytes);
    }

    hipError_t Free() {
	hipError_t err = hipFree(buffer);
	buffer = NULL;
	return err;
    }

    hipError_t CopyTo(void *src, size_t nBytes) {
	return hipMemcpy(buffer, src, nBytes, hipMemcpyDefault);
    }

    hipError_t CopyFrom(void *dst, size_t nBytes) {
	return hipMemcpy(dst, buffer, nBytes, hipMemcpyDefault);
    }
};

class hip_mpitest_buffer_hostregister: public hip_mpitest_buffer {
 public:
    hip_mpitest_buffer_hostregister () {
	memtype = HIP_MPITEST_MEMTYPE_HOSTREGISTER;
	memchar = 'R';
	strncpy (memname, "hipHostRegister", 32);
    }

    bool NeedsStagingBuffer() {
	return false;
    }

    hipError_t Allocate(size_t nBytes) {
	hipError_t err = hipErrorMemoryAllocation;
	char *tbuf = (char*) malloc (nBytes);
	if (NULL != tbuf) {
	    err = hipHostRegister(tbuf, nBytes, 0);
	    buffer = tbuf;
	}
	return err;
    }

    hipError_t Free() {
	hipError_t err = hipHostUnregister(buffer);
	free(buffer);
	buffer = NULL;
	return err;
    }

    hipError_t CopyTo(void *src, size_t nBytes) {
	memcpy(buffer, src, nBytes);
	return hipSuccess;
    }

    hipError_t CopyFrom(void *dst, size_t nBytes) {
	memcpy(dst, buffer, nBytes);
	return hipSuccess;
    }
};

// Some convinience macros
#define ALLOCATE_SENDBUFFER(_sendbuf, _tmp_sendbuf, _type, _elements, _extent, _rank, _comm, _init) { \
    if (_sendbuf->NeedsStagingBuffer() ) {                                                            \
        _tmp_sendbuf = (_type *) malloc (_elements * _extent);                                        \
        if (NULL == _tmp_sendbuf) {                                                                   \
            MPI_Abort(_comm,1);                                                                       \
        }                                                                                             \
        _init(_tmp_sendbuf, _elements, _rank);                                                        \
	HIP_CHECK(_sendbuf->Allocate(_elements * _extent));                                           \
        HIP_CHECK(_sendbuf->CopyTo(_tmp_sendbuf, _elements * _extent));                               \
    }                                                                                                 \
    else {                                                                                            \
        HIP_CHECK(_sendbuf->Allocate(_elements * _extent));                                           \
        _init((_type *)_sendbuf->get_buffer(), _elements, _rank);                                     \
    }                                                                                                 \
    report_buffertype(_comm, "Sendbuf", sendbuf);                                                     \
}

#define ALLOCATE_RECVBUFFER(_recvbuf, _tmp_recvbuf, _type, _elements, _extent, _rank, _comm, _init) { \
    if (_recvbuf->NeedsStagingBuffer() ) {                                                            \
        _tmp_recvbuf = (_type *) malloc (_elements * _extent);                                        \
        if (NULL == _tmp_recvbuf) {                                                                   \
            MPI_Abort(_comm,1);                                                                       \
        }                                                                                             \
        _init(_tmp_recvbuf, _elements);                                                               \
        HIP_CHECK(_recvbuf->Allocate(_elements * _extent));                                           \
        HIP_CHECK(_recvbuf->CopyTo(_tmp_recvbuf, _elements * _extent));                               \
    }                                                                                                 \
    else {                                                                                            \
        HIP_CHECK(_recvbuf->Allocate(_elements * _extent));                                           \
        _init((_type*)_recvbuf->get_buffer(), _elements);	                                      \
    }                                                                                                 \
    report_buffertype(_comm, "Recvbuf", _recvbuf);                                                    \
}

#define FREE_BUFFER(_buf, _tmp_buf) { \
    if (_buf->NeedsStagingBuffer() ){ \
       free (_tmp_buf);               \
    }                                 \
    HIP_CHECK(_buf->Free());          \
}

#endif // __HIP_MPITEST_BUFFER__
