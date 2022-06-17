#
# Copyright (c) 2022      Advanced Micro Devices, Inc. All rights reserved.
#

all:
	cd src ; make

bench:
	cd benchmarks ; make

clean:
	cd src ; make clean
	cd benchmarks ; make clean
	rm -f *~ 
