#include <mpi.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <iostream>

#include <ucc/api/ucc.h>
#include <cuda_runtime.h>

#include "kernels.h"

// Error handling macros
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define UCC_CHECK(call) \
    do { \
        ucc_status_t status = call; \
        if (status != UCC_OK) { \
            std::cerr << "UCC Error: " << ucc_status_string(status) << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

void run_allgather_with_cuda_graph(ucc_team_h team, ucc_context_h ctx, int rank, int size, int N) {
    float *d_sendbuf, *d_recvbuf;
    size_t bytes = N * sizeof(float);

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void **)&d_sendbuf, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_recvbuf, bytes * size));

    // Initialize data
    initData<<<(N + 255) / 256, 256>>>(d_sendbuf, rank, N);

    // CUDA graph setup
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    // UCC allgather collective
    ucc_coll_req_h req;
    ucc_coll_args_t args;
    args.mask = 0;
    args.coll_type = UCC_COLL_TYPE_ALLGATHER;
    args.src.info.buffer = d_sendbuf;
    args.src.info.count = N;
    args.src.info.datatype = UCC_DT_FLOAT32;
    args.src.info.mem_type = UCC_MEMORY_TYPE_CUDA;
    args.dst.info.buffer = d_recvbuf;
    args.dst.info.count = N * size;
    args.dst.info.datatype = UCC_DT_FLOAT32;
    args.dst.info.mem_type = UCC_MEMORY_TYPE_CUDA;
    UCC_CHECK(ucc_collective_init(&args, &req, team));

    ucc_ev_t comp_ev, *post_ev;
    comp_ev.ev_type = UCC_EVENT_COMPUTE_COMPLETE;
    comp_ev.ev_context = nullptr;
    comp_ev.ev_context_size = 0;
    comp_ev.req = req;

    ucc_ee_h ee = nullptr;

    ucc_ee_params_t ee_params;
    ee_params.ee_type         = UCC_EE_CUDA_STREAM;
    ee_params.ee_context_size = sizeof(cudaStream_t);
    ee_params.ee_context      = stream;
    UCC_CHECK(ucc_ee_create(team, &ee_params, &ee));

    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;

    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

    UCC_CHECK(ucc_collective_triggered_post(ee, &comp_ev));
    ucc_status_t st = ucc_ee_get_event(ee, &post_ev);

    ucc_ee_ack_event(ee, post_ev);

    // Capture collective in CUDA graph
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

    // Execute the graph
    CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Cleanup graph
    CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
    CUDA_CHECK(cudaGraphDestroy(graph));

    // Cleanup
    UCC_CHECK(ucc_collective_finalize(req));
    CUDA_CHECK(cudaFree(d_sendbuf));
    CUDA_CHECK(cudaFree(d_recvbuf));

    CUDA_CHECK(cudaStreamDestroy(stream));
}

static ucc_status_t oob_allgather(void *sbuf, void *rbuf, size_t msglen,
                                  void *coll_info, void **req)
{
    MPI_Comm    comm = (MPI_Comm)coll_info;
    MPI_Request request;

    MPI_Iallgather(sbuf, msglen, MPI_BYTE, rbuf, msglen, MPI_BYTE, comm,
                   &request);
    *req = (void *)request;
    return UCC_OK;
}

static ucc_status_t oob_allgather_test(void *req)
{
    MPI_Request request = (MPI_Request)req;
    int         completed;

    MPI_Test(&request, &completed, MPI_STATUS_IGNORE);
    return completed ? UCC_OK : UCC_INPROGRESS;
}

static ucc_status_t oob_allgather_free(void *req)
{
    return UCC_OK;
}

/* Creates UCC team for a group of processes represented by MPI
   communicator. UCC API provides different ways to create a team,
   one of them is to use out-of-band (OOB) allgather provided by
   the calling runtime. */
static ucc_team_h create_ucc_team(MPI_Comm comm, ucc_context_h ctx)
{
    int               rank, size;
    ucc_team_h        team;
    ucc_team_params_t team_params;
    ucc_status_t      status;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    team_params.mask          = UCC_TEAM_PARAM_FIELD_OOB;
    team_params.oob.allgather = oob_allgather;
    team_params.oob.req_test  = oob_allgather_test;
    team_params.oob.req_free  = oob_allgather_free;
    team_params.oob.coll_info = (void*)comm;
    team_params.oob.n_oob_eps = size;
    team_params.oob.oob_ep    = rank;

    UCC_CHECK(ucc_team_create_post(&ctx, 1, &team_params, &team));
    while (UCC_INPROGRESS == (status = ucc_team_create_test(team))) {
        UCC_CHECK(ucc_context_progress(ctx));
    };
    if (UCC_OK != status) {
        fprintf(stderr, "failed to create ucc team\n");
        MPI_Abort(MPI_COMM_WORLD, status);
    }
    return team;
}

int main(int argc, char **argv) {
    ucc_lib_config_h     lib_config;
    ucc_context_config_h ctx_config;
    int                  rank, size;
    ucc_team_h           team;
    ucc_context_h        ctx;
    ucc_lib_h            lib;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char *v = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    int local_rank = atoi(v);
    CUDA_CHECK(cudaSetDevice(local_rank));

    /* Init ucc library */
    ucc_lib_params_t lib_params = {
        .mask        = UCC_LIB_PARAM_FIELD_THREAD_MODE,
        .thread_mode = UCC_THREAD_SINGLE
    };
    UCC_CHECK(ucc_lib_config_read(NULL, NULL, &lib_config));
    UCC_CHECK(ucc_init(&lib_params, lib_config, &lib));
    ucc_lib_config_release(lib_config);

    /* Init ucc context for a specified UCC_TEST_TLS */
    ucc_context_params_t ctx_params = {};
    ctx_params.mask = UCC_CONTEXT_PARAM_FIELD_OOB;
    ctx_params.oob.allgather = oob_allgather;
    ctx_params.oob.req_test = oob_allgather_test;
    ctx_params.oob.req_free = oob_allgather_free;
    ctx_params.oob.coll_info = (void *)MPI_COMM_WORLD;
    ctx_params.oob.n_oob_eps = size;
    ctx_params.oob.oob_ep = rank;

    UCC_CHECK(ucc_context_config_read(lib, NULL, &ctx_config));

    UCC_CHECK(ucc_context_create(lib, &ctx_params, ctx_config, &ctx));
    ucc_context_config_release(ctx_config);

    team = create_ucc_team(MPI_COMM_WORLD, ctx);

    run_allgather_with_cuda_graph(team, ctx, rank, size, 1024);

    /* Cleanup UCC */
    UCC_CHECK(ucc_team_destroy(team));
    UCC_CHECK(ucc_context_destroy(ctx));
    UCC_CHECK(ucc_finalize(lib));
    MPI_Finalize();
    return 0;
}
