#!/bin/bash

mpirun --allow-run-as-root --tag-output --mca coll_ucc_enable 0 -x UCC_TL_NCCL_LAZY_INIT=no -x NCCL_DEBUG=info -x NCCL_DEBUG_SUBSYS=coll  -x UCX_TLS=^cuda_ipc -x UCC_TLS=nccl -x UCC_COLL_TRACE=debug -x UCC_LOG_LEVEL=debug -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH -np 8 ./cuda_graph_ucc
