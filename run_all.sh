#!/bin/bash
#
# Copyright (c) 2022      Advanced Micro Devices, Inc. All rights reserved.
#

OPTIONS="--mca pml ucx --mca osc ucx --mca btl ^openib"

ExecTest() {

    for NUMELEMS in $3 ; do 
	for MEM1 in $4 ; do
	    for MEM2 in $4 ; do
		let COUNTER=COUNTER+1
		mpirun $OPTIONS -np $2 ./src/$1 $MEM1 $MEM2 $NUMELEMS
		if [ $? -eq 0 ]
		then
		    let SUCCESS=SUCCESS+1
		else
		    let FAILED=FAILED+1
		fi
	    done
	done
    done	    
}

let COUNTER=0
let SUCCESS=0
let FAILED=0
ExecTest "hip_query_test"         "1" "1"          "D"
ExecTest "hip_pt2pt_nb"           "2" "32 1048576" "D H M O R"
ExecTest "hip_sendtoself"         "1" "32 1048576" "D H M O R"
ExecTest "hip_type_resized_short" "2" "32"         "D H M O R"
ExecTest "hip_type_resized_long"  "2" "32"         "D H M O R"
ExecTest "hip_type_struct_short"  "2" "32"         "D H M O R"
ExecTest "hip_type_struct_long"   "2" "32"         "D H M O R"
ExecTest "hip_osc_put"            "2" "32 1048576" "D H"
ExecTest "hip_osc_get"            "2" "32 1048576" "D H"
ExecTest "hip_allreduce"          "4" "32 1048576" "D H"
ExecTest "hip_alltoall"           "4" "1024"       "D H"
ExecTest "hip_alltoallv"          "4" "1024"       "D H"
printf "\n Executed %d Tests (%d passed %d failed)\n" $COUNTER $SUCCESS $FAILED
