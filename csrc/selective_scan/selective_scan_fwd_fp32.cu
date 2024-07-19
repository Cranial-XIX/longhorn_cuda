/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

// Split into multiple files to compile in paralell

#include "selective_scan_fwd_kernel.cuh"

template void selective_scan_fwd_cuda<float, float>(SSMParamsBase &params, cudaStream_t stream);
template void selective_scan_fwd_cuda<float, complex_t>(SSMParamsBase &params, cudaStream_t stream);
// template void selective_scan_A_fwd_cuda<float, float>(SSMParamsBase &params, cudaStream_t stream);
// template void selective_scan_A_fwd_cuda<float, complex_t>(SSMParamsBase &params, cudaStream_t stream);
// template void selective_scan_B_fwd_cuda<float, float>(SSMParamsBase &params, cudaStream_t stream);
// template void selective_scan_B_fwd_cuda<float, complex_t>(SSMParamsBase &params, cudaStream_t stream);
template void selective_scan_new_fwd_cuda<float, float>(SSMNewParamsBase &params, cudaStream_t stream);
template void selective_scan_decay_fwd_cuda<float, float>(SSMDecayParamsBase &params, cudaStream_t stream);
template void selective_scan_new_fwd_cuda<float, complex_t>(SSMNewParamsBase &params, cudaStream_t stream);
