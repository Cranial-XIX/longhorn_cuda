/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/cuda/CUDAException.h>  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK

#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_scan.cuh>

#include "online_selective_scan.h"
#include "selective_scan_common.h"
#include "static_switch.h"


template<int kNThreads_, int kNItems_, int kNRows_, bool kIsEvenLen_, bool kHasZ_, bool kHasDeltaBias_, bool kHasD_, typename input_t_, typename weight_t_>
struct Selective_Scan_online_fwd_kernel_traits {
    static_assert(kNItems_ % 4 == 0);
    static_assert(kNRows_ > 0);
    using input_t = input_t_;
    using weight_t = weight_t_;
    static constexpr int kNThreads = kNThreads_;
    // Setting MinBlocksPerMP to be 3 (instead of 2) for 128 threads improves occupancy.
    static constexpr int kMinBlocks = kNThreads < 128 ? 5 : 3;
    static constexpr int kNItems = kNItems_;
    static constexpr int kNRows = kNRows_;
    static constexpr int kNBytes = sizeof(input_t);
    static_assert(kNBytes == 2 || kNBytes == 4);
    static constexpr int kNElts = kNBytes == 4 ? 4 : std::min(8, kNItems);
    static_assert(kNItems % kNElts == 0);
    static constexpr int kNLoads = kNItems / kNElts;
    static constexpr bool kIsEvenLen = kIsEvenLen_;
    static constexpr bool kHasZ = kHasZ_;
    static constexpr bool kHasDeltaBias = kHasDeltaBias_;
    static constexpr bool kHasD = kHasD_;

    static constexpr bool kDirectIO = kIsEvenLen && kNLoads == 1;

    using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
    using scan_t = float2;
    using BlockLoadT = cub::BlockLoad<input_t, kNThreads, kNItems, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadVecT = cub::BlockLoad<vec_t, kNThreads, kNLoads,
        !kDirectIO ? cub::BLOCK_LOAD_WARP_TRANSPOSE : cub::BLOCK_LOAD_DIRECT>;
    using BlockLoadWeightT = cub::BlockLoad<input_t, kNThreads, kNItems, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadWeightVecT = cub::BlockLoad<vec_t, kNThreads, kNLoads,
        !kDirectIO ? cub::BLOCK_LOAD_WARP_TRANSPOSE  : cub::BLOCK_LOAD_DIRECT>;
    using BlockStoreT = cub::BlockStore<input_t, kNThreads, kNItems, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    using BlockStoreVecT = cub::BlockStore<vec_t, kNThreads, kNLoads,
        !kDirectIO ? cub::BLOCK_STORE_WARP_TRANSPOSE : cub::BLOCK_STORE_DIRECT>;
    // using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_RAKING_MEMOIZE>;
    // using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_RAKING>;
    using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_WARP_SCANS>;
    static constexpr int kSmemIOSize = std::max({sizeof(typename BlockLoadT::TempStorage),
                                                 sizeof(typename BlockLoadVecT::TempStorage),
                                                 (3) * sizeof(typename BlockLoadWeightT::TempStorage),
                                                 (3) * sizeof(typename BlockLoadWeightVecT::TempStorage),
                                                 sizeof(typename BlockStoreT::TempStorage),
                                                 sizeof(typename BlockStoreVecT::TempStorage)});
    static constexpr int kSmemSize = kSmemIOSize + sizeof(typename BlockScanT::TempStorage);
};


///////////////////////////////////////////////////////////////////////////////////////////



template<typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads, Ktraits::kMinBlocks)
void selective_scan_online_fwd_kernel(SSMOnlineParamsBase params) {
    constexpr bool kHasZ = Ktraits::kHasZ;
    constexpr bool kHasDeltaBias = Ktraits::kHasDeltaBias;
    constexpr bool kHasD = Ktraits::kHasD;
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr int kNItems = Ktraits::kNItems;
    constexpr int kNRows = Ktraits::kNRows;
    constexpr bool kDirectIO = Ktraits::kDirectIO;
    using input_t = typename Ktraits::input_t;
    using weight_t = float;
    using scan_t = typename Ktraits::scan_t;

    // Shared memory.
    extern __shared__ char smem_[];
    // cast to lvalue reference of expected type
    auto& smem_load = reinterpret_cast<typename Ktraits::BlockLoadT::TempStorage&>(smem_);
    auto& smem_load_weight = reinterpret_cast<typename Ktraits::BlockLoadWeightT::TempStorage&>(smem_);
    auto& smem_load_weight1 = *reinterpret_cast<typename Ktraits::BlockLoadWeightT::TempStorage*>(smem_ + sizeof(typename Ktraits::BlockLoadWeightT::TempStorage)); 
    auto& smem_load_weight2 = *reinterpret_cast<typename Ktraits::BlockLoadWeightT::TempStorage*>(smem_ + 2 * sizeof(typename Ktraits::BlockLoadWeightT::TempStorage)); 
    auto& smem_store = reinterpret_cast<typename Ktraits::BlockStoreT::TempStorage&>(smem_);
    auto& smem_scan = *reinterpret_cast<typename Ktraits::BlockScanT::TempStorage*>(smem_ + Ktraits::kSmemIOSize);
    scan_t *smem_running_prefix = reinterpret_cast<scan_t *>(smem_ + Ktraits::kSmemSize);

    const int batch_id = blockIdx.x;
    const int dim_id = blockIdx.y;
    const int group_id = dim_id / (params.dim_ngroups_ratio);
    input_t *u = reinterpret_cast<input_t *>(params.u_ptr) + batch_id * params.u_batch_stride + dim_id * kNRows * params.u_d_stride; 
    input_t *T = reinterpret_cast<input_t *>(params.T_ptr) + batch_id * params.T_batch_stride + dim_id * kNRows * params.T_d_stride; 

    // input_t *avar = reinterpret_cast<input_t *>(params.a_ptr) + batch_id * params.a_batch_stride + group_id * params.a_group_stride;
    input_t *Kvar = reinterpret_cast<input_t *>(params.K_ptr) + batch_id * params.K_batch_stride + group_id * params.K_group_stride;
    input_t *Qvar = reinterpret_cast<input_t *>(params.Q_ptr) + batch_id * params.Q_batch_stride + group_id * params.Q_group_stride;

    scan_t *x = reinterpret_cast<scan_t *>(params.x_ptr) + (batch_id * params.dim + dim_id * kNRows) * params.n_chunks * params.dstate;

    float D_val[kNRows], t_bias_val[kNRows];
    if (kHasD) {
        #pragma unroll
        for (int r = 0; r < kNRows; ++r) {
            D_val[r] = reinterpret_cast<float *>(params.D_ptr)[dim_id * kNRows + r];
        }
    }
    if (kHasDeltaBias) {
        #pragma unroll
        for (int r = 0; r < kNRows; ++r) {
            t_bias_val[r] = reinterpret_cast<float *>(params.t_bias_ptr)[dim_id * kNRows + r];
        }
    }

    constexpr int kChunkSize = kNThreads * kNItems;
    for (int chunk = 0; chunk < params.n_chunks; ++chunk) {
        input_t u_vals_load[kNRows][kNItems], T_vals_load[kNRows][kNItems];
        __syncthreads();
        #pragma unroll
        for (int r = 0; r < kNRows; ++r) {
            if constexpr (!kDirectIO) {
                if (r > 0) { __syncthreads(); }
            }
            load_input<Ktraits>(u + r * params.u_d_stride, u_vals_load[r], smem_load, params.seqlen - chunk * kChunkSize);
            if constexpr (!kDirectIO) { __syncthreads(); }
            load_input<Ktraits>(T + r * params.T_d_stride, T_vals_load[r], smem_load, params.seqlen - chunk * kChunkSize);
        }
        u += kChunkSize;
        T += kChunkSize;

        float u_vals[kNRows][kNItems], T_vals[kNRows][kNItems], out_vals[kNRows][kNItems];
        #pragma unroll
        for (int r = 0; r < kNRows; ++r) {
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                u_vals[r][i] = float(u_vals_load[r][i]);
                T_vals[r][i] = kHasDeltaBias ? float(T_vals_load[r][i]) + t_bias_val[r]: float(T_vals_load[r][i]);
                T_vals[r][i] = sigmoid(T_vals[r][i]);

                out_vals[r][i] = kHasD ? D_val[r] * u_vals[r][i] : 0.f;
            }
        }

        float K2_vals[kNItems] = {0.f};
        __syncthreads();
        for (int state_idx = 0; state_idx < params.dstate; ++state_idx) {
            float K_vals[kNItems];
            load_weight<Ktraits, false>(Kvar + state_idx * params.K_dstate_stride, K_vals,
                smem_load_weight2, (params.seqlen - chunk * kChunkSize));
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                K2_vals[i] += K_vals[i] * K_vals[i];
            }
        }
        for (int state_idx = 0; state_idx < params.dstate; ++state_idx) {
            float K_vals[kNItems], Q_vals[kNItems];
            load_weight<Ktraits, false>(Kvar + state_idx * params.K_dstate_stride, K_vals,
                smem_load_weight, (params.seqlen - chunk * kChunkSize));
            load_weight<Ktraits, false>(Qvar + state_idx * params.Q_dstate_stride, Q_vals,
                smem_load_weight1, (params.seqlen - chunk * kChunkSize));
            // load_weight<Ktraits, false>(avar + state_idx * params.a_dstate_stride, a_vals,
            //     smem_load_weight2, (params.seqlen - chunk * kChunkSize));

            #pragma unroll
            for (int r = 0; r < kNRows; ++r) {
                if (r > 0) { __syncthreads(); }  // Scan could be using the same smem
                scan_t thread_data[kNItems];
                #pragma unroll
                for (int i = 0; i < kNItems; ++i) {
                    const float T = T_vals[r][i] / (1 + T_vals[r][i] * K2_vals[i]);
                    const float forget = 1 - T * K_vals[i] * K_vals[i];
                    const float input_mat = u_vals[r][i] * T * K_vals[i];

                    thread_data[i] = make_float2( 
                        forget,
                        input_mat
                    );
                    if constexpr (!Ktraits::kIsEvenLen) {  // So that the last state is correct
                        if (threadIdx.x * kNItems + i >= params.seqlen - chunk * kChunkSize) {
                            thread_data[i] = make_float2(1.f, 0.f);
                        }
                    }
                }
                // Initialize running total
                scan_t running_prefix;
                // If we use WARP_SCAN then all lane 0 of all warps (not just thread 0) needs to read
                running_prefix = chunk > 0 && threadIdx.x % 32 == 0 ? smem_running_prefix[state_idx + r * MAX_DSTATE] : make_float2(1.f, 0.f);
                // running_prefix = chunk > 0 && threadIdx.x == 0 ? smem_running_prefix[state_idx] : make_float2(1.f, 0.f);
                SSMScanPrefixCallbackOp<float> prefix_op(running_prefix);
                Ktraits::BlockScanT(smem_scan).InclusiveScan(
                    thread_data, thread_data, SSMScanOp<float>(), prefix_op
                );
                // There's a syncthreads in the scan op, so we don't need to sync here.
                // Unless there's only 1 warp, but then it's the same thread (0) reading and writing.
                if (threadIdx.x == 0) {
                    smem_running_prefix[state_idx] = prefix_op.running_prefix;
                    x[(r * params.n_chunks + chunk) * params.dstate + state_idx] = prefix_op.running_prefix;
                }
                #pragma unroll
                for (int i = 0; i < kNItems; ++i) {
                    const float q = Q_vals[i];
                    out_vals[r][i] += thread_data[i].y * q;
                }
            }
        }

        input_t *out = reinterpret_cast<input_t *>(params.out_ptr) + batch_id * params.out_batch_stride
            + dim_id * kNRows * params.out_d_stride + chunk * kChunkSize;
        __syncthreads();
        #pragma unroll
        for (int r = 0; r < kNRows; ++r) {
            if constexpr (!kDirectIO) {
                if (r > 0) { __syncthreads(); }
            }
            store_output<Ktraits>(out + r * params.out_d_stride, out_vals[r], smem_store, params.seqlen - chunk * kChunkSize);
        }

        if constexpr (kHasZ) {
            input_t *z = reinterpret_cast<input_t *>(params.z_ptr) + batch_id * params.z_batch_stride
                + dim_id * kNRows * params.z_d_stride + chunk * kChunkSize;
            input_t *out_z = reinterpret_cast<input_t *>(params.out_z_ptr) + batch_id * params.out_z_batch_stride
                + dim_id * kNRows * params.out_z_d_stride + chunk * kChunkSize;
            #pragma unroll
            for (int r = 0; r < kNRows; ++r) {
                input_t z_vals[kNItems];
                __syncthreads();
                load_input<Ktraits>(z + r * params.z_d_stride, z_vals, smem_load, params.seqlen - chunk * kChunkSize);
                #pragma unroll
                for (int i = 0; i < kNItems; ++i) {
                    float z_val = z_vals[i];
                    out_vals[r][i] *= sigmoid(z_val, z_val); // z_val / (1 + expf(-z_val));
                }
                __syncthreads();
                store_output<Ktraits>(out_z + r * params.out_z_d_stride, out_vals[r], smem_store, params.seqlen - chunk * kChunkSize);
            }
        }

        // avar += kChunkSize;
        Kvar += kChunkSize;
        Qvar += kChunkSize;
    }
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////



template<int kNThreads, int kNItems, typename input_t>
void selective_scan_online_fwd_launch(SSMOnlineParamsBase &params, cudaStream_t stream) {
    // Only kNRows == 1 is tested for now, which ofc doesn't differ from previously when we had each block
    // processing 1 row.
    constexpr int kNRows = 1;
    BOOL_SWITCH(params.seqlen % (kNThreads * kNItems) == 0, kIsEvenLen, [&] {
        BOOL_SWITCH(params.z_ptr != nullptr , kHasZ, [&] {
            BOOL_SWITCH(params.t_bias_ptr != nullptr , kHasDeltaBias, [&] {
                BOOL_SWITCH(params.D_ptr != nullptr , kHasD, [&] {
                    using Ktraits = Selective_Scan_online_fwd_kernel_traits<kNThreads, kNItems, kNRows, kIsEvenLen, kHasZ, kHasDeltaBias, kHasD, input_t, float>;
                    // constexpr int kSmemSize = Ktraits::kSmemSize;
                    constexpr int kSmemSize = Ktraits::kSmemSize + kNRows * MAX_DSTATE * sizeof(typename Ktraits::scan_t);
                    // printf("smem_size = %d\n", kSmemSize);
                    dim3 grid(params.batch, params.dim / kNRows);
                    auto kernel = &selective_scan_online_fwd_kernel<Ktraits>;
                    if (kSmemSize >= 48 * 1024) {
                        C10_CUDA_CHECK(cudaFuncSetAttribute(
                            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
                    }
                    kernel<<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(params);
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                });
            });
        });
    });
}


template<typename input_t>
void selective_scan_online_fwd_cuda(SSMOnlineParamsBase &params, cudaStream_t stream) {
    if (params.seqlen <= 128) {
        selective_scan_online_fwd_launch<32, 4, input_t>(params, stream);
    } else if (params.seqlen <= 256) {
        selective_scan_online_fwd_launch<32, 8, input_t>(params, stream);
    } else if (params.seqlen <= 512) {
        selective_scan_online_fwd_launch<32, 16, input_t>(params, stream);
    } else if (params.seqlen <= 1024) {
        selective_scan_online_fwd_launch<64, 16, input_t>(params, stream);
    } else {
        selective_scan_online_fwd_launch<128, 16, input_t>(params, stream);
    }
}


