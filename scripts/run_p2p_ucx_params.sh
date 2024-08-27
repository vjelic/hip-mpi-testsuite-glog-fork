#!/bin/bash
###############################################################################
# Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
###############################################################################

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
ExecTest "hip_pt2pt_nb"           "2" "1048576" "D H M O R" "am"
ExecTest "hip_type_struct_long"   "2" "32"      "D H M O R" "am"
ExecTest "hip_osc_put_fence"      "2" "1048576" "D H" "am"
ExecTest "hip_osc_get_fence"      "2" "1048576" "D H" "am"

echo "RNDV_SCHEME=rkey_ptr"
ExecTest "hip_pt2pt_nb"           "2" "1048576" "D H M O R" "rkey_ptr"
ExecTest "hip_type_struct_long"   "2" "32"      "D H M O R" "rkey_ptr"
ExecTest "hip_osc_put_fence"      "2" "1048576" "D H" "rkey_ptr"
ExecTest "hip_osc_get_fence"      "2" "1048576" "D H" "rkey_ptr"

echo "RNDV_SCHEME=put_zcopy"
ExecTest "hip_pt2pt_nb"           "2" "1048576" "D H M O R" "put_zcopy"
ExecTest "hip_type_struct_long"   "2" "32"      "D H M O R" "put_zcopy"
ExecTest "hip_osc_put_fence"      "2" "1048576" "D H" "put_zcopy"
ExecTest "hip_osc_get_fence"      "2" "1048576" "D H" "put_zcopy"

echo "RNDV_SCHEME=get_zcopy"
ExecTest "hip_pt2pt_nb"           "2" "1048576" "D H M O R" "get_zcopy"
ExecTest "hip_type_struct_long"   "2" "32"      "D H M O R" "get_zcopy"
ExecTest "hip_osc_put_fence"      "2" "1048576" "D H" "get_zcopy"
ExecTest "hip_osc_get_fence"      "2" "1048576" "D H" "get_zcopy"

echo "RNDV_SCHEME=put_ppln"
ExecTest "hip_pt2pt_nb"           "2" "1048576" "D H M O R" "put_ppln"
ExecTest "hip_type_struct_long"   "2" "32"      "D H M O R" "put_ppln"
ExecTest "hip_osc_put_fence"      "2" "1048576" "D H" "put_ppln"
ExecTest "hip_osc_get_fence"      "2" "1048576" "D H" "put_ppln"

echo "RNDV_SCHEME=get_ppln"
ExecTest "hip_pt2pt_nb"           "2" "1048576" "D H M O R" "get_ppln"
ExecTest "hip_type_struct_long"   "2" "32"      "D H M O R" "get_ppln"
ExecTest "hip_osc_put_fence"      "2" "1048576" "D H" "get_ppln"
ExecTest "hip_osc_get_fence"      "2" "1048576" "D H" "get_ppln"

printf "\n Executed %d Tests (%d passed %d failed)\n" $COUNTER $SUCCESS $FAILED
