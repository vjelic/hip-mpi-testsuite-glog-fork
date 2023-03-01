#!/bin/bash
#
# Copyright (c) 2022      Advanced Micro Devices, Inc. All rights reserved.
#

OPTIONS="--mca pml ucx --mca osc ucx"

ExecTest() {

    for NUMELEMS in $3 ; do
	for MEM1 in $4 ; do
	    for MEM2 in $4 ; do
		let COUNTER=COUNTER+1
		mpirun $OPTIONS -x UCX_RNDV_SCHEME=$5 -np $2 ../src/$1 -s $MEM1 -r $MEM2 -n $NUMELEMS
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

echo "RNDV_SCHEME=am"
ExecTest "hip_pt2pt_nb_stress"    "2" "10485760" "D H M O R" "am"
ExecTest "hip_sendtoself_stress"  "1" "10485760" "D H M O R" "am"

echo "RNDV_SCHEME=rkey_ptr"
ExecTest "hip_pt2pt_nb_stress"    "2" "10485760" "D H M O R" "rkey_ptr"
ExecTest "hip_sendtoself_stress"  "1" "10485760" "D H M O R" "am"

echo "RNDV_SCHEME=put_zcopy"
ExecTest "hip_pt2pt_nb_stress"    "2" "10485760" "D H M O R" "put_zcopy"
ExecTest "hip_sendtoself_stress"  "1" "10485760" "D H M O R" "am"

echo "RNDV_SCHEME=get_zcopy"
ExecTest "hip_pt2pt_nb_stress"    "2" "10485760" "D H M O R" "get_zcopy"
ExecTest "hip_sendtoself_stress"  "1" "10485760" "D H M O R" "am"

echo "RNDV_SCHEME=put_ppln"
ExecTest "hip_pt2pt_nb_stress"    "2" "10485760" "D H M O R" "put_ppln"
ExecTest "hip_sendtoself_stress"  "1" "10485760" "D H M O R" "am"

echo "RNDV_SCHEME=get_ppln"
ExecTest "hip_pt2pt_nb_stress"    "2" "10485760" "D H M O R" "get_ppln"
ExecTest "hip_sendtoself_stress"  "1" "10485760" "D H M O R" "am"

printf "\n Executed %d Tests (%d passed %d failed)\n" $COUNTER $SUCCESS $FAILED
