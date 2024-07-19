/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

// Split into multiple files to compile in paralell

#include "selective_scan_bwd_kernel.cuh"

template void selective_scan_bwd_cuda<at::Half, complex_t>(SSMParamsBwd &params, cudaStream_t stream);
template void selective_scan_new_bwd_cuda<at::Half, complex_t>(SSMNewParamsBwd &params, cudaStream_t stream);
