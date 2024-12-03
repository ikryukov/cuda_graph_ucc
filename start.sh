#!/bin/bash

mpirun --allow-run-as-root --tag-output --mca coll_ucc_enable 0 -x UCX_TLS=^cuda_ipc -x UCC_TLS=cuda -x UCC_COLL_TRACE=debug -x xUCC_LOG_LEVEL=debug -x LD_LIBRARY_PATH=/home/ikryukov/work/ucc/install/lib:$LD_LIBRARY_PATH -np 3 ./cuda_graph_ucc