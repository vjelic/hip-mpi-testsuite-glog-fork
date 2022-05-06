#
# Copyright (c) 2022      Advanced Micro Devices, Inc. All rights reserved.
#

ROCM_DIR         = /opt/rocm
ROCM_INCLUDE_DIR = $(ROCM_DIR)/include
ROCM_LIB_DIR     = $(ROCM_DIR)/lib
ROCM_LIBS        = amdhip64


CC      = mpiCC
CFLAGS  = -D__HIP_PLATFORM_AMD__ -I$(ROCM_INCLUDE_DIR)/hip -I$(ROCM_INCLUDE_DIR)
LDFLAGS = -L$(ROCM_LIB_DIR) -l$(ROCM_LIBS)

RM      = rm -f


HEADERS = hip_mpitest_utils.h hip_mpitest_buffer.h hip_mpitest_datatype.h


all:    hip_pt2pt_nb           \
	hip_sendtoself         \
	hip_type_struct_short  \
	hip_type_struct_long   \
	hip_type_resized_short \
	hip_type_resized_long  \
	hip_allreduce          \
	hip_alltoall           \
	hip_alltoallv          \
	hip_osc_put            \
	hip_osc_get            \
	hip_query_test


hip_pt2pt_nb: hip_pt2pt_nb.cc $(HEADERS)
	$(CC) $(CFLAGS) -o hip_pt2pt_nb hip_pt2pt_nb.cc $(LDFLAGS)

hip_sendtoself: hip_sendtoself.cc $(HEADERS)
	$(CC) $(CFLAGS) -o hip_sendtoself hip_sendtoself.cc $(LDFLAGS)

hip_allreduce: hip_allreduce.cc $(HEADERS)
	$(CC) $(CFLAGS) -o hip_allreduce hip_allreduce.cc $(LDFLAGS)

hip_alltoall: hip_alltoall.cc $(HEADERS)
	$(CC) $(CFLAGS) -o hip_alltoall hip_alltoall.cc $(LDFLAGS)

hip_alltoallv: hip_alltoall.cc $(HEADERS)
	$(CC) $(CFLAGS) -o hip_alltoallv hip_alltoall.cc -DHIP_MPITEST_ALLTOALLV $(LDFLAGS)

hip_osc_put: hip_osc.cc $(HEADERS)
	$(CC) $(CFLAGS) -o hip_osc_put hip_osc.cc -DHIP_MPITEST_OSC_PUT $(LDFLAGS)

hip_osc_get: hip_osc.cc $(HEADERS)
	$(CC) $(CFLAGS) -o hip_osc_get hip_osc.cc -DHIP_MPITEST_OSC_GET $(LDFLAGS)

hip_type_resized_short: hip_ddt.cc $(HEADERS)
	$(CC) $(CFLAGS) -o hip_type_resized_short hip_ddt.cc -DHIP_TYPE_RESIZED -DA_WIDTH=32 $(LDFLAGS)

hip_type_resized_long: hip_ddt.cc $(HEADERS)
	$(CC) $(CFLAGS) -o hip_type_resized_long hip_ddt.cc -DHIP_TYPE_RESIZED -DA_WIDTH=1024 $(LDFLAGS)

hip_type_struct_short: hip_ddt.cc $(HEADERS)
	$(CC) $(CFLAGS) -o hip_type_struct_short hip_ddt.cc -DHIP_TYPE_STRUCT -DA_WIDTH=32 $(LDFLAGS)

hip_type_struct_long: hip_ddt.cc $(HEADERS)
	$(CC) $(CFLAGS) -o hip_type_struct_long hip_ddt.cc -DHIP_TYPE_STRUCT -DA_WIDTH=1024 $(LDFLAGS)

hip_query_test: hip_query_test.cc
	$(CC) $(CFLAGS) -o hip_query_test hip_query_test.cc $(LDFLAGS)


clean:
	$(RM) *.o *~
	$(RM) hip_pt2pt_nb hip_sendtoself
	$(RM) hip_allreduce hip_alltoall hip_alltoallv
	$(RM) hip_type_resized_short hip_type_struct_short
	$(RM) hip_type_resized_long hip_type_struct_long
	$(RM) hip_osc_put hip_osc_get hip_query_test
