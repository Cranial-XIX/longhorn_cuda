/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

// Split into multiple files to compile in paralell

#include "selective_scan_online_bwd_kernel.cuh"

template void selective_scan_online_bwd_cuda<at::Half>(SSMOnlineParamsBwd &params, cudaStream_t stream);
// template void selective_scan_online_bwd_cuda<at::Half, complex_t>(SSMOnlineParamsBwd &params, cudaStream_t stream);
template void selective_scan_online_bwd_cuda<at::BFloat16>(SSMOnlineParamsBwd &params, cudaStream_t stream);
// template void selective_scan_online_bwd_cuda<at::BFloat16, complex_t>(SSMOnlineParamsBwd &params, cudaStream_t stream);
template void selective_scan_online_bwd_cuda<float>(SSMOnlineParamsBwd &params, cudaStream_t stream);
// template void selective_scan_online_bwd_cuda<float, complex_t>(SSMOnlineParamsBwd &params, cudaStream_t stream);
