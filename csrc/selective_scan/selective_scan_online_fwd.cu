/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

// Split into multiple files to compile in paralell

#include "selective_scan_online_fwd_kernel.cuh"

template void selective_scan_online_fwd_cuda<at::BFloat16>(SSMOnlineParamsBase &params, cudaStream_t stream);
// template void selective_scan_online_fwd_cuda<at::BFloat16, complex_t>(SSMOnlineParamsBase &params, cudaStream_t stream);
template void selective_scan_online_fwd_cuda<at::Half>(SSMOnlineParamsBase &params, cudaStream_t stream);
// template void selective_scan_online_fwd_cuda<at::Half, complex_t>(SSMOnlineParamsBase &params, cudaStream_t stream);
template void selective_scan_online_fwd_cuda<float>(SSMOnlineParamsBase &params, cudaStream_t stream);
// template void selective_scan_online_fwd_cuda<float, complex_t>(SSMOnlineParamsBase &params, cudaStream_t stream);