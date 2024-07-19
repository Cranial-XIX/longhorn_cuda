/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <vector>

#include "online_selective_scan.h"

#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")

#define DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(ITYPE, NAME, ...)                    \
    if (ITYPE == at::ScalarType::Half)                                              \
    {                                                                               \
        using input_t = at::Half;                                                   \
        __VA_ARGS__();                                                              \
    }                                                                               \
    else if (ITYPE == at::ScalarType::BFloat16)                                     \
    {                                                                               \
        using input_t = at::BFloat16;                                               \
        __VA_ARGS__();                                                              \
    }                                                                               \
    else if (ITYPE == at::ScalarType::Float)                                        \
    {                                                                               \
        using input_t = float;                                                      \
        __VA_ARGS__();                                                              \
    }                                                                               \
    else                                                                            \
    {                                                                               \
        AT_ERROR(#NAME, " not implemented for input type '", toString(ITYPE), "'"); \
    }

#define DISPATCH_WTYPE_FLOAT_AND_HALF_AND_BF16(WTYPE, NAME, ...)                     \
    if (WTYPE == at::ScalarType::Half)                                               \
    {                                                                                \
        using weight_t = at::Half;                                                   \
        __VA_ARGS__();                                                               \
    }                                                                                \
    else if (WTYPE == at::ScalarType::BFloat16)                                      \
    {                                                                                \
        using weight_t = at::BFloat16;                                               \
        __VA_ARGS__();                                                               \
    }                                                                                \
    else if (WTYPE == at::ScalarType::Float)                                         \
    {                                                                                \
        using weight_t = float;                                                      \
        __VA_ARGS__();                                                               \
    }                                                                                \
    else                                                                             \
    {                                                                                \
        AT_ERROR(#NAME, " not implemented for weight type '", toString(WTYPE), "'"); \
    }

#define DISPATCH_WTYPE_FLOAT(WTYPE, NAME, ...)                                       \
    if (WTYPE == at::ScalarType::Float)                                              \
    {                                                                                \
        using weight_t = float;                                                      \
        __VA_ARGS__();                                                               \
    }                                                                                \
    else if (WTYPE == at::ScalarType::BFloat16)                                      \
    {                                                                                \
        using weight_t = float;                                                      \
        __VA_ARGS__();                                                               \
    }                                                                                \
    else if (WTYPE == at::ScalarType::Half)                                          \
    {                                                                                \
        using weight_t = float;                                                      \
        __VA_ARGS__();                                                               \
    }                                                                                \
    else                                                                             \
    {                                                                                \
        AT_ERROR(#NAME, " not implemented for weight type '", toString(WTYPE), "'"); \
    }

#define DISPATCH_WTYPE_FLOAT_AND_COMPLEX(WTYPE, NAME, ...)                           \
    if (WTYPE == at::ScalarType::Float)                                              \
    {                                                                                \
        using weight_t = float;                                                      \
        __VA_ARGS__();                                                               \
    }                                                                                \
    else if (WTYPE == at::ScalarType::BFloat16)                                      \
    {                                                                                \
        using weight_t = float;                                                      \
        __VA_ARGS__();                                                               \
    }                                                                                \
    else if (WTYPE == at::ScalarType::Half)                                          \
    {                                                                                \
        using weight_t = float;                                                      \
        __VA_ARGS__();                                                               \
    }                                                                                \
    else if (WTYPE == at::ScalarType::ComplexFloat)                                  \
    {                                                                                \
        using weight_t = c10::complex<float>;                                        \
        __VA_ARGS__();                                                               \
    }                                                                                \
    else if (WTYPE == at::ScalarType::ComplexHalf)                                   \
    {                                                                                \
        using weight_t = c10::complex<float>;                                        \
        __VA_ARGS__();                                                               \
    }                                                                                \
    else                                                                             \
    {                                                                                \
        AT_ERROR(#NAME, " not implemented for weight type '", toString(WTYPE), "'"); \
    }

template <typename input_t>
void selective_scan_online_fwd_cuda(SSMOnlineParamsBase &params, cudaStream_t stream);

template <typename input_t>
void selective_scan_online_bwd_cuda(SSMOnlineParamsBwd &params, cudaStream_t stream);
//////////////////////////////////////////////////////////////////////////////////////////////////

void set_ssm_online_params_fwd(SSMOnlineParamsBase &params,
                               // sizes
                               const size_t batch,
                               const size_t dim,
                               const size_t seqlen,
                               const size_t dstate,
                               const size_t n_groups,
                               const size_t n_chunks,
                               // device pointers
                               const at::Tensor u,
                               const at::Tensor Q,
                               const at::Tensor K,
                               const at::Tensor T,
                               const at::Tensor out,
                               const at::Tensor z,
                               const at::Tensor out_z,
                               void *D_ptr,
                               void *t_bias_ptr,
                               void *x_ptr,
                               bool has_z,
                               bool bwd=false)
{

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    params.batch = batch;
    params.dim = dim;
    params.seqlen = seqlen;
    params.dstate = dstate;
    params.n_groups = n_groups;
    params.n_chunks = n_chunks;
    params.dim_ngroups_ratio = dim / n_groups;

    // Set the pointers and strides.
    params.u_ptr = u.data_ptr();
    params.Q_ptr = Q.data_ptr();
    params.K_ptr = K.data_ptr();
    params.T_ptr = T.data_ptr();
    params.out_ptr = out.data_ptr();
    params.D_ptr = D_ptr;
    params.t_bias_ptr = t_bias_ptr;
    params.x_ptr = x_ptr;
    params.z_ptr = has_z ? z.data_ptr() : nullptr;
    // All stride are in elements, not bytes.
    params.u_batch_stride = u.stride(0);
    params.u_d_stride = u.stride(1);
    params.T_batch_stride = T.stride(0);
    params.T_d_stride = T.stride(1);

    params.K_batch_stride = K.stride(0);
    params.K_group_stride = K.stride(1);
    params.K_dstate_stride = K.stride(2);
    params.Q_batch_stride = Q.stride(0);
    params.Q_group_stride = Q.stride(1);
    params.Q_dstate_stride = Q.stride(2);

    if (has_z)
    {
        params.z_batch_stride = z.stride(0);
        params.z_d_stride = z.stride(1);
        if (!bwd) {
            params.out_z_ptr = has_z ? out_z.data_ptr() : nullptr;
            params.out_z_batch_stride = out_z.stride(0);
            params.out_z_d_stride = out_z.stride(1);
        }
    }
    params.out_batch_stride = out.stride(0);
    params.out_d_stride = out.stride(1);
}

void set_ssm_online_params_bwd(SSMOnlineParamsBwd &params,
                               // sizes
                               const size_t batch_size,
                               const size_t dim,
                               const size_t seqlen,
                               const size_t dstate,
                               const size_t n_groups,
                               const size_t n_chunks,
                               // device pointers
                               // inputs
                               const at::Tensor u,
                               const at::Tensor Q,
                               const at::Tensor K,
                               const at::Tensor T,
                               const at::Tensor out,
                               const at::Tensor z,
                               const at::Tensor out_z,
                               // gradients
                               const at::Tensor dout,
                               const at::Tensor du,
                               const at::Tensor dQ,
                               const at::Tensor dK,
                               const at::Tensor dT,
                               const at::Tensor dTK,
                               const at::Tensor dz,
                               void *dD_ptr,
                               void *D_ptr,
                               void *dt_bias_ptr,
                               void *t_bias_ptr,
                               void *x_ptr,
                               bool has_z)
{
    set_ssm_online_params_fwd(params, batch_size, dim, seqlen, dstate, n_groups, n_chunks,
                              u, Q, K, T, out, z, out_z,
                              D_ptr, t_bias_ptr, x_ptr, has_z, true);
    params.out_z_ptr = nullptr;
    // Set the pointers and strides.

    // Set the pointers and strides.
    params.du_ptr = du.data_ptr();
    params.dT_ptr = dT.data_ptr();
    params.dTK_ptr = dTK.data_ptr();
    params.dK_ptr = dK.data_ptr();
    params.dQ_ptr = dQ.data_ptr();
    params.dout_ptr = dout.data_ptr();
    params.dD_ptr = dD_ptr;
    params.dt_bias_ptr = dt_bias_ptr;
    params.dz_ptr = has_z ? dz.data_ptr() : nullptr;

    // All stride are in elements, not bytes.
    params.dout_batch_stride = dout.stride(0);
    params.dout_d_stride = dout.stride(1);
    params.du_batch_stride = du.stride(0);
    params.du_d_stride = du.stride(1);
    params.dT_batch_stride = dT.stride(0);
    params.dT_d_stride = dT.stride(1);
    params.dTK_batch_stride = dTK.stride(0);
    params.dTK_d_stride = dTK.stride(1);

    params.dK_batch_stride = dK.stride(0);
    params.dK_group_stride = dK.stride(1);
    params.dK_dstate_stride = dK.stride(2);
    params.dQ_batch_stride = dQ.stride(0);
    params.dQ_group_stride = dQ.stride(1);
    params.dQ_dstate_stride = dQ.stride(2);

    if (has_z)
    {
        params.dz_batch_stride = dz.stride(0);
        params.dz_d_stride = dz.stride(1);
    }
}

void input_validation_base(
    const at::Tensor &u,
    const at::Tensor &Q,
    const at::Tensor &K,
    const at::Tensor &T,
    const c10::optional<at::Tensor> &D_,
    const c10::optional<at::Tensor> &t_bias_,
    const c10::optional<at::Tensor> &z_)
{
    auto input_type = u.scalar_type();
    // auto weight_type = at::ScalarType::Float;
    TORCH_CHECK(input_type == at::ScalarType::Float || input_type == at::ScalarType::Half || input_type == at::ScalarType::BFloat16);

    // Check input dtype
    TORCH_CHECK(u.scalar_type() == input_type);
    TORCH_CHECK(Q.scalar_type() == input_type);
    TORCH_CHECK(K.scalar_type() == input_type);
    TORCH_CHECK(T.scalar_type() == input_type);

    // Check device
    TORCH_CHECK(u.is_cuda());
    TORCH_CHECK(Q.is_cuda());
    TORCH_CHECK(K.is_cuda());
    TORCH_CHECK(T.is_cuda());

    // Check stride
    const auto sizes = u.sizes();
    const int batch_size = sizes[0];
    const int dim = sizes[1];
    const int seqlen = sizes[2];
    const int dstate = Q.size(2);
    const int n_groups = Q.size(1);

    TORCH_CHECK(dstate <= 256, "selective_scan only supports state dimension <= 256");

    CHECK_SHAPE(u, batch_size, dim, seqlen);
    TORCH_CHECK(u.stride(-1) == 1 || u.size(-1) == 1);
    CHECK_SHAPE(T, batch_size, dim, seqlen);
    TORCH_CHECK(T.stride(-1) == 1 || T.size(-1) == 1);

    CHECK_SHAPE(Q, batch_size, n_groups, dstate, seqlen);
    TORCH_CHECK(Q.stride(-1) == 1 || Q.size(-1) == 1);
    CHECK_SHAPE(K, batch_size, n_groups, dstate, seqlen);
    TORCH_CHECK(K.stride(-1) == 1 || K.size(-1) == 1);

    if (t_bias_.has_value())
    {
        auto t_bias = t_bias_.value();
        TORCH_CHECK(t_bias.scalar_type() == at::ScalarType::Float);
        TORCH_CHECK(t_bias.is_cuda());
        TORCH_CHECK(t_bias.stride(-1) == 1 || t_bias.size(-1) == 1);
        CHECK_SHAPE(t_bias, dim);
    }
    if (D_.has_value())
    {
        auto D = D_.value();
        TORCH_CHECK(D.scalar_type() == at::ScalarType::Float);
        TORCH_CHECK(D.is_cuda());
        TORCH_CHECK(D.stride(-1) == 1 || D.size(-1) == 1);
        CHECK_SHAPE(D, dim);
    }
    if (z_.has_value())
    {
        auto z = z_.value();
        TORCH_CHECK(z.scalar_type() == input_type);
        TORCH_CHECK(z.is_cuda());
        TORCH_CHECK(z.stride(-1) == 1 || z.size(-1) == 1);
        CHECK_SHAPE(z, batch_size, dim, seqlen);
    }
}

std::vector<at::Tensor>
selective_scan_online_fwd(
    const at::Tensor &u,
    const at::Tensor &Q,
    const at::Tensor &K,
    const at::Tensor &T,
    const c10::optional<at::Tensor> &D_,
    const c10::optional<at::Tensor> &t_bias_,
    const c10::optional<at::Tensor> &z_)
{
    input_validation_base(u, Q, K, T, D_, t_bias_, z_);
    const auto sizes = u.sizes();
    const int batch_size = sizes[0];
    const int dim = sizes[1];
    const int seqlen = sizes[2];
    const int dstate = K.size(2);
    const int n_groups = K.size(1);

    at::Tensor z, out_z;
    const bool has_z = z_.has_value();
    if (has_z)
    {
        z = z_.value();
        out_z = torch::empty_like(z_.value());
    }

    const int n_chunks = (seqlen + 2048 - 1) / 2048;
    at::Tensor out = torch::empty(sizes, u.options());
    at::Tensor x = torch::empty({batch_size, dim, n_chunks, dstate * 2}, u.options().dtype(torch::kFloat32));

    SSMOnlineParamsBase params;
    set_ssm_online_params_fwd(params, batch_size, dim, seqlen, dstate, n_groups, n_chunks,
                              u, Q, K, T, out, z, out_z,
                              D_.has_value() ? D_.value().data_ptr() : nullptr,
                              t_bias_.has_value() ? t_bias_.value().data_ptr() : nullptr,
                              x.data_ptr(),
                              has_z);

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)u.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(u.scalar_type(), "selective_scan_online_fwd", [&]
                                           { selective_scan_online_fwd_cuda<input_t>(params, stream); });
    std::vector<at::Tensor> result = {out, x};
    if (has_z)
    {
        result.push_back(out_z);
    }
    return result;
}

std::vector<at::Tensor>
selective_scan_online_bwd(
    const at::Tensor &u,
    const at::Tensor &Q,
    const at::Tensor &K,
    const at::Tensor &T,
    const c10::optional<at::Tensor> &D_,
    const c10::optional<at::Tensor> &t_bias_,
    const c10::optional<at::Tensor> &z_,
    const at::Tensor &dout,
    const c10::optional<at::Tensor> &x_,
    const at::Tensor &out,
    const c10::optional<at::Tensor> &dz_)
{

    input_validation_base(u, Q, K, T, D_, t_bias_, z_);

    auto input_type = u.scalar_type();
    TORCH_CHECK(dout.scalar_type() == input_type);
    TORCH_CHECK(out.scalar_type() == input_type);

    const auto sizes = u.sizes();
    const int batch_size = sizes[0];
    const int dim = sizes[1];
    const int seqlen = sizes[2];
    const int dstate = K.size(2);
    const int n_groups = K.size(1);

    at::Tensor z, dz, out_z;

    const bool has_z = z_.has_value();
    if (has_z)
    {
        z = z_.value();
        if (dz_.has_value()) {
            dz = dz_.value();
        } else {
            dz = torch::empty_like(z);
        }
    }

    const int n_chunks = (seqlen + 2048 - 1) / 2048;
    // const int n_chunks = (seqlen + 1024 - 1) / 1024;
    if (n_chunks > 1)
    {
        TORCH_CHECK(x_.has_value());
    }
    if (x_.has_value())
    {
        auto x = x_.value();
        TORCH_CHECK(x.scalar_type() == at::ScalarType::Float);
        TORCH_CHECK(x.is_cuda());
        TORCH_CHECK(x.is_contiguous());
        CHECK_SHAPE(x, batch_size, dim, n_chunks, 2 * dstate);
    }

    at::Tensor dQ = torch::zeros_like(Q, Q.options().dtype(torch::kFloat32));
    at::Tensor dK = torch::zeros_like(K, K.options().dtype(torch::kFloat32));
    at::Tensor du = torch::empty_like(u);
    at::Tensor dT = torch::empty_like(T);
    at::Tensor dTK = torch::empty_like(T);

    at::Tensor dt_bias;
    if (t_bias_.has_value())
    {
        dt_bias = torch::zeros_like(t_bias_.value(), t_bias_.value().options().dtype(torch::kFloat32));
    }
    at::Tensor dD;
    if (D_.has_value())
    {
        dD = torch::zeros_like(D_.value(), D_.value().options().dtype(torch::kFloat32));
    }

    SSMOnlineParamsBwd params;
    set_ssm_online_params_bwd(
        params, batch_size, dim, seqlen, dstate, n_groups, n_chunks,
        u, Q, K, T, out, z, out_z,
        dout, du, dQ, dK, dT, dTK, dz,
        D_.has_value() ? dD.data_ptr() : nullptr,
        D_.has_value() ? D_.value().data_ptr() : nullptr,
        t_bias_.has_value() ? dt_bias.data_ptr() : nullptr,
        t_bias_.has_value() ? t_bias_.value().data_ptr() : nullptr,
        x_.has_value() ? x_.value().data_ptr() : nullptr,
        has_z
    );

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)u.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(u.scalar_type(), "selective_scan_online_bwd", [&] { 
        selective_scan_online_bwd_cuda<input_t>(params, stream); 
    });
    if (D_.has_value()) {
        dD = dD.to(D_.value().dtype()); 
    }
    std::vector<at::Tensor> result = {du, dQ.to(Q.dtype()), dK.to(K.dtype()), dT, dD, dz, dt_bias, dTK};
    // result.push_back(dz);
    // result.push_back(dt_bias);
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("online_fwd", &selective_scan_online_fwd, "Selective scan online forward");
    m.def("online_bwd", &selective_scan_online_bwd, "Selective scan online forward");
}
