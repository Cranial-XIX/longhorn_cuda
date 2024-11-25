/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include <stdio.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/cuda/CUDAException.h>  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK
#include <ATen/cuda/Atomic.cuh>  // For atomicAdd on complex

#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_reduce.cuh>

#include "online_selective_scan.h"
#include "selective_scan_common.h"
#include "reverse_scan.cuh"
#include "static_switch.h"

template<typename scalar_t> __device__ __forceinline__ scalar_t conj(scalar_t x);
template<> __device__ __forceinline__ float conj<float>(float x) { return x; }
template<> __device__ __forceinline__ complex_t conj<complex_t>(complex_t x) { return std::conj(x); }


template<int kNThreads_, int kNItems_, bool kIsEvenLen_, 
         bool kHasZ_, bool kHasDeltaBias_, bool kHasD_, typename input_t_, typename weight_t_>
struct Selective_Scan_online_bwd_kernel_traits {
    static_assert(kNItems_ % 4 == 0);
    using input_t = input_t_;
    using weight_t = weight_t_;
    static constexpr int kNThreads = kNThreads_;
    static constexpr int kNItems = kNItems_;
    static constexpr int kNBytes = sizeof(input_t);
    static_assert(kNBytes == 2 || kNBytes == 4);
    static constexpr int kNElts = kNBytes == 4 ? 4 : std::min(8, kNItems);
    static_assert(kNItems % kNElts == 0);
    static constexpr int kNLoads = kNItems / kNElts;
    static constexpr bool kIsEvenLen = kIsEvenLen_;
    static constexpr bool kHasZ = kHasZ_;
    static constexpr bool kHasDeltaBias = kHasDeltaBias_;
    static constexpr bool kHasD = kHasD_;
    // Setting MinBlocksPerMP to be 3 (instead of 2) for 128 threads with float improves occupancy.
    // For complex this would lead to massive register spilling, so we keep it at 2.
    static constexpr int kMinBlocks = kNThreads == 128 && 3;
    using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
    using scan_t = float2;
    using BlockLoadT = cub::BlockLoad<input_t, kNThreads, kNItems, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadVecT = cub::BlockLoad<vec_t, kNThreads, kNLoads, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadWeightT = cub::BlockLoad<input_t, kNThreads,  kNItems, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadWeightVecT = cub::BlockLoad<vec_t, kNThreads, kNLoads, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockStoreT = cub::BlockStore<input_t, kNThreads, kNItems, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    using BlockStoreVecT = cub::BlockStore<vec_t, kNThreads, kNLoads, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    // using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_RAKING_MEMOIZE>;
    using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_RAKING>;
    // using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_WARP_SCANS>;
    using BlockReverseScanT = BlockReverseScan<scan_t, kNThreads>;
    using BlockReduceT = cub::BlockReduce<scan_t, kNThreads>;
    using BlockReduceFloatT = cub::BlockReduce<float, kNThreads>;
    using BlockReduceComplexT = cub::BlockReduce<complex_t, kNThreads>;
    using BlockExchangeT = cub::BlockExchange<float, kNThreads, kNItems>;
    static constexpr int kSmemIOSize = std::max({sizeof(typename BlockLoadT::TempStorage),
                                                 sizeof(typename BlockLoadVecT::TempStorage),
                                                 (3) * sizeof(typename BlockLoadWeightT::TempStorage),
                                                 (3) * sizeof(typename BlockLoadWeightVecT::TempStorage),
                                                 sizeof(typename BlockStoreT::TempStorage),
                                                 sizeof(typename BlockStoreVecT::TempStorage)});
    static constexpr int kSmemExchangeSize = (3) * sizeof(typename BlockExchangeT::TempStorage);
    static constexpr int kSmemReduceSize = sizeof(typename BlockReduceT::TempStorage);
    static constexpr int kSmemSize = kSmemIOSize + kSmemExchangeSize + kSmemReduceSize + sizeof(typename BlockScanT::TempStorage) + sizeof(typename BlockReverseScanT::TempStorage);
};




template<typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads, Ktraits::kMinBlocks)
void selective_scan_online_bwd_kernel(SSMOnlineParamsBwd params) {
    constexpr bool kHasZ = Ktraits::kHasZ;
    constexpr bool kHasDeltaBias = Ktraits::kHasDeltaBias;
    constexpr bool kHasD = Ktraits::kHasD;
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr int kNItems = Ktraits::kNItems;
    using input_t = typename Ktraits::input_t;
    using scan_t = typename Ktraits::scan_t;
    using weight_t = float;

    // Shared memory.
    extern __shared__ char smem_[];
    // cast to lvalue reference of expected type
    // char *smem_loadstorescan = smem_ + 2 * MAX_DSTATE * sizeof(weight_t);
    // auto& smem_load = reinterpret_cast<typename BlockLoadT::TempStorage&>(smem_ + 2 * MAX_DSTATE * sizeof(weight_t));
    // auto& smem_load = reinterpret_cast<typename BlockLoadT::TempStorage&>(smem_loadstorescan);
    auto& smem_load = reinterpret_cast<typename Ktraits::BlockLoadT::TempStorage&>(smem_);
    auto& smem_load_weight = reinterpret_cast<typename Ktraits::BlockLoadWeightT::TempStorage&>(smem_);
    auto& smem_load_weight1 = *reinterpret_cast<typename Ktraits::BlockLoadWeightT::TempStorage*>(smem_ + sizeof(typename Ktraits::BlockLoadWeightT::TempStorage));
    auto& smem_load_weight2 = *reinterpret_cast<typename Ktraits::BlockLoadWeightT::TempStorage*>(smem_ + 2 * sizeof(typename Ktraits::BlockLoadWeightT::TempStorage));
    auto& smem_store = reinterpret_cast<typename Ktraits::BlockStoreT::TempStorage&>(smem_);
    auto& smem_exchange = *reinterpret_cast<typename Ktraits::BlockExchangeT::TempStorage*>(smem_ + Ktraits::kSmemIOSize);
    auto& smem_exchange1 = *reinterpret_cast<typename Ktraits::BlockExchangeT::TempStorage*>(smem_ + Ktraits::kSmemIOSize + sizeof(typename Ktraits::BlockExchangeT::TempStorage));
    auto& smem_exchange2 = *reinterpret_cast<typename Ktraits::BlockExchangeT::TempStorage*>(smem_ + Ktraits::kSmemIOSize + 2 * (sizeof(typename Ktraits::BlockExchangeT::TempStorage)));
    auto& smem_reduce = *reinterpret_cast<typename Ktraits::BlockReduceT::TempStorage*>(reinterpret_cast<char *>(&smem_exchange) + Ktraits::kSmemExchangeSize);
    auto& smem_reduce_float = *reinterpret_cast<typename Ktraits::BlockReduceFloatT::TempStorage*>(&smem_reduce);
    auto& smem_reduce_complex = *reinterpret_cast<typename Ktraits::BlockReduceComplexT::TempStorage*>(&smem_reduce);
    auto& smem_scan = *reinterpret_cast<typename Ktraits::BlockScanT::TempStorage*>(reinterpret_cast<char *>(&smem_reduce) + Ktraits::kSmemReduceSize);
    auto& smem_reverse_scan = *reinterpret_cast<typename Ktraits::BlockReverseScanT::TempStorage*>(reinterpret_cast<char *>(&smem_scan) + sizeof(typename Ktraits::BlockScanT::TempStorage));
    weight_t *smem_delta_a = reinterpret_cast<weight_t *>(smem_ + Ktraits::kSmemSize);
    scan_t *smem_running_postfix = reinterpret_cast<scan_t *>(smem_delta_a + 3 * MAX_DSTATE + kNThreads);
    weight_t *smem_da = reinterpret_cast<weight_t *>(smem_running_postfix + MAX_DSTATE);
    weight_t *smem_db = reinterpret_cast<weight_t *>(smem_running_postfix + MAX_DSTATE * 2);

    const int batch_id = blockIdx.x;
    const int dim_id = blockIdx.y;
    const int group_id = dim_id / (params.dim_ngroups_ratio);
    input_t *u = reinterpret_cast<input_t *>(params.u_ptr) + batch_id * params.u_batch_stride + dim_id * params.u_d_stride;
    input_t *dout = reinterpret_cast<input_t *>(params.dout_ptr) + batch_id * params.dout_batch_stride + dim_id * params.dout_d_stride;
    input_t *T = reinterpret_cast<input_t *>(params.T_ptr) + batch_id * params.T_batch_stride + dim_id * params.T_d_stride; 

    input_t *Kvar = reinterpret_cast<input_t *>(params.K_ptr) + batch_id * params.K_batch_stride + group_id * params.K_group_stride;
    input_t *Qvar = reinterpret_cast<input_t *>(params.Q_ptr) + batch_id * params.Q_batch_stride + group_id * params.Q_group_stride;
    weight_t *dKvar = reinterpret_cast<weight_t *>(params.dK_ptr) + batch_id * params.dK_batch_stride + group_id * params.dK_group_stride;
    weight_t *dQvar = reinterpret_cast<weight_t *>(params.dQ_ptr) + batch_id * params.dQ_batch_stride + group_id * params.dQ_group_stride;

    float *dD = kHasD ? reinterpret_cast<float *>(params.dD_ptr) + dim_id : nullptr;
    float *dt_bias = kHasDeltaBias ? reinterpret_cast<float *>(params.dt_bias_ptr) + dim_id : nullptr;
    float D_val = kHasD ? reinterpret_cast<float *>(params.D_ptr)[dim_id] : 0.f;
    float t_bias_val = kHasDeltaBias ? reinterpret_cast<float *>(params.t_bias_ptr)[dim_id] : 0.f;
    scan_t *x = params.x_ptr == nullptr
        ? nullptr
        : reinterpret_cast<scan_t *>(params.x_ptr) + (batch_id * params.dim + dim_id) * (params.n_chunks) * params.dstate;
    float dD_val = 0;
    float dt_bias_val = 0;

    constexpr int kChunkSize = kNThreads * kNItems;
    u += (params.n_chunks - 1) * kChunkSize;
    dout += (params.n_chunks - 1) * kChunkSize;
    Kvar += (params.n_chunks - 1) * kChunkSize;
    Qvar += (params.n_chunks - 1) * kChunkSize;
    T += (params.n_chunks - 1) * kChunkSize;
    for (int chunk = params.n_chunks - 1; chunk >= 0; --chunk) {
        input_t u_vals_load[kNItems], dout_vals_load[kNItems], T_vals_load[kNItems];
        __syncthreads();
        load_input<Ktraits>(u, u_vals_load, smem_load, params.seqlen - chunk * kChunkSize);
        u -= kChunkSize;
        __syncthreads();
        load_input<Ktraits>(T, T_vals_load, smem_load, params.seqlen - chunk * kChunkSize);
        T -= kChunkSize;
        __syncthreads();
        load_input<Ktraits>(dout, dout_vals_load, smem_load, params.seqlen - chunk * kChunkSize);
        dout -= kChunkSize;

        float T_vals[kNItems], u_vals[kNItems];
        float dout_vals[kNItems];
        #pragma unroll
        for (int i = 0; i < kNItems; ++i) {
            T_vals[i] = float(T_vals_load[i]);
            T_vals[i] = kHasDeltaBias ? float(T_vals_load[i]) + t_bias_val: float(T_vals_load[i]);
            T_vals[i] = sigmoid(T_vals[i]);
            u_vals[i] = float(u_vals_load[i]);
            dout_vals[i] = float(dout_vals_load[i]);
        }

        if constexpr (kHasZ) {
            input_t *z = reinterpret_cast<input_t *>(params.z_ptr) + batch_id * params.z_batch_stride
                + dim_id * params.z_d_stride + chunk * kChunkSize;
            input_t *out = reinterpret_cast<input_t *>(params.out_ptr) + batch_id * params.out_batch_stride
                + dim_id * params.out_d_stride + chunk * kChunkSize;
            input_t *dz = reinterpret_cast<input_t *>(params.dz_ptr) + batch_id * params.dz_batch_stride
                + dim_id * params.dz_d_stride + chunk * kChunkSize;
            input_t z_vals[kNItems], out_vals_load[kNItems];
            __syncthreads();
            load_input<Ktraits>(z, z_vals, smem_load, params.seqlen - chunk * kChunkSize);
            __syncthreads();
            load_input<Ktraits>(out, out_vals_load, smem_load, params.seqlen - chunk * kChunkSize);
            float dz_vals[kNItems], z_silu_vals[kNItems], out_vals[kNItems];
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                float z_val = z_vals[i];
                float r_z_sigmoid_val = 1.0f + expf(-z_val);
                out_vals[i] = float(out_vals_load[i]);
                z_silu_vals[i] = z_val / r_z_sigmoid_val;
                dz_vals[i] = dout_vals[i] * out_vals[i] / r_z_sigmoid_val * (1.0f + z_val - z_silu_vals[i]);
                dout_vals[i] *= z_silu_vals[i];
            }
            __syncthreads();
            store_output<Ktraits>(dz, dz_vals, smem_store, params.seqlen - chunk * kChunkSize);
            if (params.out_z_ptr != nullptr) {  // Recompute and store out_z
                float out_z_vals[kNItems];
                #pragma unroll
                for (int i = 0; i < kNItems; ++i) { out_z_vals[i] = out_vals[i] * z_silu_vals[i]; }
                input_t *out_z = reinterpret_cast<input_t *>(params.out_z_ptr) + batch_id * params.out_z_batch_stride
                    + dim_id * params.out_z_d_stride + chunk * kChunkSize;
                __syncthreads();
                store_output<Ktraits>(out_z, out_z_vals, smem_store, params.seqlen - chunk * kChunkSize);
            }
        }

        float du_vals[kNItems];
        float dT_vals[kNItems] = {0.f};
        float dTK_vals[kNItems] = {0.f};
        #pragma unroll
        for (int i = 0; i < kNItems; ++i) { 
            du_vals[i] = D_val * dout_vals[i]; 
            dD_val += dout_vals[i] * u_vals[i]; 
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
        float Ts[kNItems], denoms[kNItems];
        #pragma unroll
        for (int i = 0; i < kNItems; ++i) {
            float denom = (1 + T_vals[i] * K2_vals[i]);
            float T = T_vals[i] / denom; 
            Ts[i] = T;
            denoms[i] = denom;
        }

        for (int state_idx = 0; state_idx < params.dstate; ++state_idx) {
            // Multiply the real part of A with LOG2E so we can use exp2f instead of expf.
            float K_vals[kNItems], Q_vals[kNItems];
            load_weight<Ktraits, false>(Kvar + state_idx * params.K_dstate_stride, K_vals,
                smem_load_weight, (params.seqlen - chunk * kChunkSize));
            load_weight<Ktraits, false>(Qvar + state_idx * params.Q_dstate_stride, Q_vals,
                smem_load_weight1, (params.seqlen - chunk * kChunkSize));

            scan_t thread_data[kNItems], thread_reverse_data[kNItems];
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                // float aa = K_vals[i] * K_vals[i];
                float T = Ts[i];
                float kt = T * K_vals[i];
                float forget = 1 - kt * K_vals[i];
                const float input_mat = u_vals[i] * kt;
                thread_data[i] = make_float2( 
                    forget,
                    input_mat
                );
                if (i == 0) {
                    smem_delta_a[threadIdx.x == 0 ? state_idx + (chunk % 2) * MAX_DSTATE : threadIdx.x + 2 * MAX_DSTATE] = forget;
                } else {
                    thread_reverse_data[i - 1].x = forget;
                }
                thread_reverse_data[i].y = dout_vals[i] * Q_vals[i]; 
            }
            __syncthreads(); 
            thread_reverse_data[kNItems - 1].x = threadIdx.x == kNThreads - 1
                ? (chunk == params.n_chunks - 1 ? 1.f : smem_delta_a[state_idx + ((chunk + 1) % 2) * MAX_DSTATE])
                : smem_delta_a[threadIdx.x + 1 + 2 * MAX_DSTATE];
            // Initialize running total
            scan_t running_prefix = chunk > 0 && threadIdx.x % 32 == 0 ? x[(chunk - 1) * params.dstate + state_idx] : make_float2(1.f, 0.f);
            SSMScanPrefixCallbackOp<float> prefix_op(running_prefix);
            Ktraits::BlockScanT(smem_scan).InclusiveScan(
                thread_data, thread_data, SSMScanOp<float>(), prefix_op
            );
            scan_t running_postfix = chunk < params.n_chunks - 1 && threadIdx.x % 32 == 0 ? smem_running_postfix[state_idx] : make_float2(1.f, 0.f);
            SSMScanPrefixCallbackOp<float> postfix_op(running_postfix);
            Ktraits::BlockReverseScanT(smem_reverse_scan).InclusiveReverseScan(
                thread_reverse_data, thread_reverse_data, SSMScanOp<float>(), postfix_op
            );
            if (threadIdx.x == 0) { smem_running_postfix[state_idx] = postfix_op.running_prefix; }
            float dK_vals[kNItems], dQ_vals[kNItems];
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                const float dx = thread_reverse_data[i].y;
                const float u = u_vals[i];
                const float x = thread_data[i].y;
                float T = Ts[i];


                float kt = K_vals[i] * T;
                float aa = K_vals[i] * K_vals[i];
                float forget = 1 - kt * K2_vals[i];

                const float input_mat = u_vals[i] * kt;

                du_vals[i] += dx * kt;
                dK_vals[i] = dx * T * u_vals[i];
                dT_vals[i] += dx * u_vals[i] * K_vals[i];

                const float forget_x = x - input_mat;
                const float x_minus_1 = forget_x / (forget + 1e-10f);
                float dx_dforget = dx * x_minus_1;
                dT_vals[i] += - aa * dx_dforget;

                dQ_vals[i] = dout_vals[i] * x;
                dK_vals[i] -= dx_dforget * 2 * kt;
            }
            // Block-exchange to make the atomicAdd's coalesced, otherwise they're much slower
            // Ktraits::BlockExchangeT(smem_exchange).BlockedToStriped(dB_vals, dB_vals);
            // auto &smem_exchange_C =  smem_exchange1;
            Ktraits::BlockExchangeT(smem_exchange).BlockedToStriped(dQ_vals, dQ_vals);
            Ktraits::BlockExchangeT(smem_exchange1).BlockedToStriped(dK_vals, dK_vals);
            float *dK_cur = dKvar + state_idx * params.dK_dstate_stride + chunk * kChunkSize + threadIdx.x;
            float *dQ_cur = dQvar + state_idx * params.dQ_dstate_stride + chunk * kChunkSize + threadIdx.x;
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                if (i * kNThreads < params.seqlen - chunk * kChunkSize - threadIdx.x) {
                    gpuAtomicAdd(dQ_cur + i * kNThreads, dQ_vals[i]); 
                    gpuAtomicAdd(dK_cur + i * kNThreads, dK_vals[i]); 
                }
            }
        }
        for (int i = 0; i < kNItems; ++i) {
            float denom = denoms[i]; // (1 + T_vals[i] * K2_vals[i]);
            float T2 = T_vals[i] * T_vals[i];
            dTK_vals[i] = - dT_vals[i] * (Ts[i] * Ts[i]);// (T2 / denom2);
            dT_vals[i] /= denom * denom;
            dT_vals[i] = dT_vals[i] * (T_vals[i] - T2);
            if (kHasDeltaBias) {
                dt_bias_val += dT_vals[i];
            }
        }

        input_t *du = reinterpret_cast<input_t *>(params.du_ptr) + batch_id * params.du_batch_stride
            + dim_id * params.du_d_stride + chunk * kChunkSize;
        __syncthreads();
        store_output<Ktraits>(du, du_vals, smem_store, params.seqlen - chunk * kChunkSize);
        input_t *dT = reinterpret_cast<input_t *>(params.dT_ptr) + batch_id * params.dT_batch_stride
            + dim_id * params.dT_d_stride + chunk * kChunkSize;
        __syncthreads();
        store_output<Ktraits>(dT, dT_vals, smem_store, params.seqlen - chunk * kChunkSize);
        input_t *dTK = reinterpret_cast<input_t *>(params.dTK_ptr) + batch_id * params.dTK_batch_stride
            + dim_id * params.dTK_d_stride + chunk * kChunkSize;
        __syncthreads();
        store_output<Ktraits>(dTK, dTK_vals, smem_store, params.seqlen - chunk * kChunkSize);

        Kvar -= kChunkSize;
        Qvar -= kChunkSize;
    }
    if (kHasD) {
        dD_val = Ktraits::BlockReduceFloatT(smem_reduce_float).Sum(dD_val);
        if (threadIdx.x == 0) { gpuAtomicAdd(dD, dD_val); }
    }
    if (kHasDeltaBias) {
        if (kHasD) { __syncthreads(); }
        dt_bias_val = Ktraits::BlockReduceFloatT(smem_reduce_float).Sum(dt_bias_val);
        if (threadIdx.x == 0) { gpuAtomicAdd(dt_bias, dt_bias_val); }
    }
    // __syncthreads();
    // dforget_bias_val = Ktraits::BlockReduceFloatT(smem_reduce_float).Sum(dforget_bias_val);
    // if (threadIdx.x == 0) { gpuAtomicAdd(dforget_bias, dforget_bias_val); }
}


template<int kNThreads, int kNItems, typename input_t>
void selective_scan_online_bwd_launch(SSMOnlineParamsBwd &params, cudaStream_t stream) {
    BOOL_SWITCH(params.seqlen % (kNThreads * kNItems) == 0, kIsEvenLen, [&] {
        BOOL_SWITCH(params.z_ptr != nullptr, kHasZ, [&] {
            BOOL_SWITCH(params.t_bias_ptr != nullptr , kHasDeltaBias, [&] {
                BOOL_SWITCH(params.D_ptr != nullptr , kHasD, [&] {
                    using Ktraits = Selective_Scan_online_bwd_kernel_traits<kNThreads, kNItems, kIsEvenLen, kHasZ, kHasDeltaBias, kHasD, input_t, float>;
                    constexpr int kSmemSize = Ktraits::kSmemSize + MAX_DSTATE * sizeof(typename Ktraits::scan_t) + (kNThreads + 4 * MAX_DSTATE) * sizeof(float);
                    // printf("smem_size = %d\n", kSmemSize);
                    dim3 grid(params.batch, params.dim);
                    auto kernel = &selective_scan_online_bwd_kernel<Ktraits>;
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
void selective_scan_online_bwd_cuda(SSMOnlineParamsBwd &params, cudaStream_t stream) {
    if (params.seqlen <= 128) {
        selective_scan_online_bwd_launch<32, 4, input_t>(params, stream);
    } else if (params.seqlen <= 256) {
        selective_scan_online_bwd_launch<32, 8, input_t>(params, stream);
    } else if (params.seqlen <= 512) {
        selective_scan_online_bwd_launch<32, 16, input_t>(params, stream);
    } else if (params.seqlen <= 1024) {
        selective_scan_online_bwd_launch<64, 16, input_t>(params, stream);
    } else {
        selective_scan_online_bwd_launch<128, 16, input_t>(params, stream);
    }
}
