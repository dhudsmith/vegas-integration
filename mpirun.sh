#!/usr/bin/env bash
mpirun -np 8 python integrator_mpi.py 16 15 10000000 10 > mpioutput.txt