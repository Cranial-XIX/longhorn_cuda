/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

////////////////////////////////////////////////////////////////////////////////////////////////////


struct SSMOnlineParamsBase {
    using index_t = uint32_t;

    int batch, dim, seqlen, dstate, n_groups, n_chunks;
    int dim_ngroups_ratio;

    index_t u_batch_stride;
    index_t u_d_stride;
    index_t t_batch_stride;
    index_t t_d_stride;
    index_t T_batch_stride;
    index_t T_d_stride;
    index_t K_batch_stride;
    index_t K_d_stride;
    index_t K_dstate_stride;
    index_t K_group_stride;
    index_t Q_batch_stride;
    index_t Q_d_stride;
    index_t Q_dstate_stride;
    index_t Q_group_stride;
    index_t z_batch_stride;
    index_t z_d_stride;
    index_t out_batch_stride;
    index_t out_d_stride;
    index_t out_z_batch_stride;
    index_t out_z_d_stride;

    // Common data pointers.
    void *__restrict__ u_ptr;
    void *__restrict__ t_ptr;
    void *__restrict__ T_ptr;
    void *__restrict__ b_ptr;
    void *__restrict__ K_ptr;
    void *__restrict__ Q_ptr;
    void *__restrict__ out_ptr;
    void *__restrict__ x_ptr;
    void *__restrict__ z_ptr;
    void *__restrict__ D_ptr;
    void *__restrict__ t_bias_ptr;
    void *__restrict__ out_z_ptr;
};

struct SSMOnlineParamsBwd: public SSMOnlineParamsBase {
    index_t du_batch_stride;
    index_t du_d_stride;
    index_t dT_batch_stride;
    index_t dT_d_stride;
    index_t dTK_batch_stride;
    index_t dTK_d_stride;
    index_t dK_batch_stride;
    index_t dK_d_stride;
    index_t dK_dstate_stride;
    index_t dK_group_stride;
    index_t dQ_batch_stride;
    index_t dQ_d_stride;
    index_t dQ_dstate_stride;
    index_t dQ_group_stride;
    index_t dA_d_stride;
    index_t dA_dstate_stride;
    index_t dB_d_stride;
    index_t dB_dstate_stride;
    index_t dz_batch_stride;
    index_t dz_d_stride;
    index_t dout_batch_stride;
    index_t dout_d_stride;
    index_t dout_z_batch_stride;
    index_t dout_z_d_stride;

    // Common data pointers.
    void *__restrict__ du_ptr;
    void *__restrict__ dT_ptr;
    void *__restrict__ dTK_ptr;
    void *__restrict__ dB_ptr;
    void *__restrict__ dK_ptr;
    void *__restrict__ da_ptr;
    void *__restrict__ dA_ptr;
    void *__restrict__ dQ_ptr;
    void *__restrict__ dout_ptr;
    void *__restrict__ dx_ptr;
    void *__restrict__ dz_ptr;
    void *__restrict__ dD_ptr;
    void *__restrict__ dt_bias_ptr;
    void *__restrict__ dout_z_ptr;
};
