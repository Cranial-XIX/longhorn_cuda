# Copyright (c) 2023, Tri Dao, Albert Gu.

import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.cuda.amp import custom_bwd, custom_fwd

from einops import rearrange, repeat, einsum, reduce

import fast_layer_norm
try:
    from causal_conv1d import causal_conv1d_fn
    import causal_conv1d_cuda
except ImportError:
    causal_conv1d_fn = None
    causal_conv1d_cuda = None

import selective_scan_cuda
from mamba_ssm.ops.triton.layernorm import _layer_norm_fwd, _layer_norm_bwd


@torch.jit.script
def get_dk(dTK: torch.Tensor, dK: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    return dK + dTK.sum(1, keepdim=True).unsqueeze(1) * 2.0 * K


class SelectiveScanFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                return_last_state=False):
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if z is not None and z.stride(-1) != 1:
            z = z.contiguous()
        if B.dim() == 3:
            B = rearrange(B, "b dstate l -> b 1 dstate l")
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = rearrange(C, "b dstate l -> b 1 dstate l")
            ctx.squeeze_C = True
        out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, z, delta_bias, delta_softplus)
        ctx.delta_softplus = delta_softplus
        ctx.has_z = z is not None
        last_state = x[:, :, -1, 1::2]  # (batch, dim, dstate)
        if not ctx.has_z:
            ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
            return out if not return_last_state else (out, last_state)
        else:
            ctx.save_for_backward(u, delta, A, B, C, D, z, delta_bias, x, out)
            out_z = rest[0]
            return out_z if not return_last_state else (out_z, last_state)

    @staticmethod
    def backward(ctx, dout, *args):
        if not ctx.has_z:
            u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
            z = None
            out = None
        else:
            u, delta, A, B, C, D, z, delta_bias, x, out = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        # The kernel supports passing in a pre-allocated dz (e.g., in case we want to fuse the
        # backward of selective_scan_cuda with the backward of chunk).
        # Here we just pass in None and dz will be allocated in the C++ code.
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, z, delta_bias, dout, x, out, None, ctx.delta_softplus,
            False  # option to recompute out_z, not used here
        )
        dz = rest[0] if ctx.has_z else None
        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
        return (du, ddelta, dA, dB, dC,
                dD if D is not None else None,
                dz,
                ddelta_bias if delta_bias is not None else None,
                None,
                None)


def selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                     return_last_state=False):
    """if return_last_state is True, returns (out, last_state)
    last_state has shape (batch, dim, dstate). Note that the gradient of the last state is
    not considered in the backward pass.
    """
    return SelectiveScanFn.apply(u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state)







def selective_scan_online7_fn(u, Q, K, T, D=None, t_bias=None, z=None, return_last_state=False):
    return SelectiveScanOnline7Fn.apply(u, Q, K, T, D, t_bias, z, return_last_state)


class SelectiveScanOnline7Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, u, Q, K, T, D=None, t_bias=None, z=None, return_last_state=False):
        if z is not None and z.stride(-1) != 1:
            z = z.contiguous()

        if K.dim() == 3:
            K = rearrange(K, "b dstate l -> b 1 dstate l")
            ctx.squeeze_K = True
        if Q.dim() == 3:
            Q = rearrange(Q, "b dstate l -> b 1 dstate l")
            ctx.squeeze_Q = True
        import online_selective_scan_cuda
        #import pdb; pdb.set_trace()
        out, x, *rest = online_selective_scan_cuda.online_fwd(
            u,
            Q,
            K,
            T,
            D,
            t_bias,
            z,
        )

        ctx.has_z = z is not None
        ctx.return_last_state = return_last_state
        last_state = x[:, :, -1, 1::2]  # (batch, dim, dstate)
        ctx.save_for_backward(u, Q, K, T, D, t_bias, x, z, out)
        if not ctx.has_z:
            return out if not return_last_state else (out, last_state)
        else:
            out = rest[0]
            return out if not return_last_state else (out, last_state)

    @staticmethod
    def backward(ctx, dout, *args):
        if ctx.return_last_state:
            raise NotImplementedError(
                "backward with return_last_state is not implemented"
            )
        u, Q, K, T, D, t_bias, x, z, out = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()

        import online_selective_scan_cuda
        du, dQ, dK, dT, dD, dz, dt_bias, dTK = online_selective_scan_cuda.online_bwd(
            u,
            Q,
            K,
            T,
            D,
            t_bias,
            z,
            dout,
            x,
            out,
            None
        )
        dK = get_dk(dTK, dK, K).to(K)

        dK = dK.squeeze(1) if getattr(ctx, "squeeze_K", False) else dK
        dQ = dQ.squeeze(1) if getattr(ctx, "squeeze_Q", False) else dQ
        return du, dQ, dK, dT, dD if D is not None else None, dt_bias, dz if z is not None else None, None


def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                      return_last_state=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: c(D N) or r(D N)
    B: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    C: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    out: r(B D L)
    last_state (optional): r(B D dstate) or c(B D dstate)
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3
    if A.is_complex():
        if is_variable_B:
            B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
        if is_variable_C:
            C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
    else:
        B = B.float()
        C = C.float()
    x = A.new_zeros((batch, dim, dstate))
    ys = []
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    if not is_variable_B:
        deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
    else:
        if B.dim() == 3:
            deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
        else:
            B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
            deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
    if is_variable_C and C.dim() == 4:
        C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
    last_state = None
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if not is_variable_C:
            y = torch.einsum('bdn,dn->bd', x, C)
        else:
            if C.dim() == 3:
                y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
            else:
                y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
        if i == u.shape[2] - 1:
            last_state = x
        if y.is_complex():
            y = y.real * 2
        ys.append(y)
    y = torch.stack(ys, dim=2) # (batch dim L)
    out = y if D is None else y + u * rearrange(D, "d -> d 1")
    if z is not None:
        out = out * F.silu(z)
    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out, last_state)


def selective_scan_online7_ref(u, q, k, dt, D=None, t_bias=None, z=None, return_last_state=False, eps=1e-6):
    """
    To Rui:
        we will always use real numbers, so ignore all complex logic in mamba cuda code.
    """

    """
    u:  r(B D L)
    A:  r(D N) in [0, 1], should be very close to 1
    B:  r(D N) in [0, 1], should be very close to 1
    q:  r(B N L)
    k:  r(B N L)
    dt: r(B D L), also in [0, 1]
    forget_bias: r(1) a scalar > 0

    D: r(D)
    z: r(B D L)

    out: r(B D L)
    last_state (optional): r(B D dstate) or c(B D dstate)
    """
    dtype_in = u.dtype
    u = u.float()
    q = q.float()
    k = k.float()
    dt = dt.float()
    if t_bias is not None:
        dt = dt + t_bias.float()[..., None]

    dt = torch.sigmoid(dt)
    dt = dt / (1 + dt * k.square().sum(dim=-2, keepdim=True))

    K = rearrange(k, 'b n l -> b 1 l n').pow(2)

    forget_gate = (1 - dt.unsqueeze(-1) * K)

    input_matrix = torch.einsum('bdl,bnl->bdln', (dt*u), k)

    last_state = None
    batch, dim = u.shape[:2]
    dstate = q.shape[1]

    x = q.new_zeros((batch, dim, dstate))
    ys = []
    for i in range(u.shape[2]):
        x = forget_gate[:, :, i] * x + input_matrix[:, :, i]
        y = torch.einsum('bdn,bn->bd', x, q[:, :, i])

        if i == x.shape[2] - 1:
            last_state = x

        ys.append(y)

    y = torch.stack(ys, dim=-1)  # (batch dim L)

    out = y if D is None else y + u * rearrange(D, "d -> d 1")

    if z is not None:
        out = out * F.silu(z.float())

    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out, last_state)


###############################################################################
#
# Mamba Inner Functions
#
###############################################################################


class MambaInnerFn(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
                out_proj_weight, out_proj_bias,
                A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
                C_proj_bias=None, delta_softplus=True, checkpoint_lvl=1):
        """
             xz: (batch, dim, seqlen)
        """
        assert causal_conv1d_cuda is not None, "causal_conv1d_cuda is not available. Please install causal-conv1d."
        assert checkpoint_lvl in [0, 1]
        L = xz.shape[-1]
        delta_rank = delta_proj_weight.shape[1]
        d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
        if torch.is_autocast_enabled():
            x_proj_weight = x_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            delta_proj_weight = delta_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            out_proj_weight = out_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            out_proj_bias = (out_proj_bias.to(dtype=torch.get_autocast_gpu_dtype())
                             if out_proj_bias is not None else None)
        if xz.stride(-1) != 1:
            xz = xz.contiguous()
        conv1d_weight = rearrange(conv1d_weight, "d 1 w -> d w")
        x, z = xz.chunk(2, dim=1)
        conv1d_bias = conv1d_bias.contiguous() if conv1d_bias is not None else None
        conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(
            x, conv1d_weight, conv1d_bias, None, None, None, True
        )
        # We're being very careful here about the layout, to avoid extra transposes.
        # We want delta to have d as the slowest moving dimension
        # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
        x_dbl = F.linear(rearrange(conv1d_out, 'b d l -> (b l) d'), x_proj_weight)  # (bl d)
        delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l = L)
        ctx.is_variable_B = B is None
        ctx.is_variable_C = C is None
        ctx.B_proj_bias_is_None = B_proj_bias is None
        ctx.C_proj_bias_is_None = C_proj_bias is None
        if B is None:  # variable B
            B = x_dbl[:, delta_rank:delta_rank + d_state]  # (bl dstate)
            if B_proj_bias is not None:
                B = B + B_proj_bias.to(dtype=B.dtype)
            if not A.is_complex():
                # B = rearrange(B, "(b l) dstate -> b dstate l", l=L).contiguous()
                B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            else:
                B = rearrange(B, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
        else:
            if B.stride(-1) != 1:
                B = B.contiguous()
        if C is None:  # variable C
            C = x_dbl[:, -d_state:]  # (bl dstate)
            if C_proj_bias is not None:
                C = C + C_proj_bias.to(dtype=C.dtype)
            if not A.is_complex():
                # C = rearrange(C, "(b l) dstate -> b dstate l", l=L).contiguous()
                C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            else:
                C = rearrange(C, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
        else:
            if C.stride(-1) != 1:
                C = C.contiguous()
        if D is not None:
            D = D.contiguous()
        out, scan_intermediates, out_z = selective_scan_cuda.fwd(
            conv1d_out, delta, A, B, C, D, z, delta_bias, delta_softplus
        )
        ctx.delta_softplus = delta_softplus
        ctx.out_proj_bias_is_None = out_proj_bias is None
        ctx.checkpoint_lvl = checkpoint_lvl
        if checkpoint_lvl >= 1:  # Will recompute conv1d_out and delta in the backward pass
            conv1d_out, delta = None, None
        ctx.save_for_backward(xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight,
                              delta_proj_weight, out_proj_weight, conv1d_out, delta,
                              A, B, C, D, delta_bias, scan_intermediates, out)
        return F.linear(rearrange(out_z, "b d l -> b l d"), out_proj_weight, out_proj_bias)

    @staticmethod
    @custom_bwd
    def backward(ctx, dout):
        # dout: (batch, seqlen, dim)
        assert causal_conv1d_cuda is not None, "causal_conv1d_cuda is not available. Please install causal-conv1d."
        (xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight, delta_proj_weight, out_proj_weight,
         conv1d_out, delta, A, B, C, D, delta_bias, scan_intermediates, out) = ctx.saved_tensors
        L = xz.shape[-1]
        delta_rank = delta_proj_weight.shape[1]
        d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
        x, z = xz.chunk(2, dim=1)
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        if ctx.checkpoint_lvl == 1:
            conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(
                x, conv1d_weight, conv1d_bias, None, None, None, True
            )
            delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(),
                              "d (b l) -> b d l", l = L)
        # The kernel supports passing in a pre-allocated dz (e.g., in case we want to fuse the
        # backward of selective_scan_cuda with the backward of chunk).
        dxz = torch.empty_like(xz)  # (batch, dim, seqlen)
        dx, dz = dxz.chunk(2, dim=1)
        dout = rearrange(dout, "b l e -> e (b l)")
        dout_y = rearrange(out_proj_weight.t() @ dout, "d (b l) -> b d l", l=L)
        dconv1d_out, ddelta, dA, dB, dC, dD, ddelta_bias, dz, out_z = selective_scan_cuda.bwd(
            conv1d_out, delta, A, B, C, D, z, delta_bias, dout_y, scan_intermediates, out, dz,
            ctx.delta_softplus,
            True  # option to recompute out_z
        )
        dout_proj_weight = torch.einsum("eB,dB->ed", dout, rearrange(out_z, "b d l -> d (b l)"))
        dout_proj_bias = dout.sum(dim=(0, 1)) if not ctx.out_proj_bias_is_None else None
        dD = dD if D is not None else None
        dx_dbl = torch.empty_like(x_dbl)
        dB_proj_bias = None
        if ctx.is_variable_B:
            if not A.is_complex():
                dB = rearrange(dB, "b 1 dstate l -> (b l) dstate").contiguous()
            else:
                dB = rearrange(dB, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
            dB_proj_bias = dB.sum(0) if not ctx.B_proj_bias_is_None else None
            dx_dbl[:, delta_rank:delta_rank + d_state] = dB  # (bl d)
            dB = None
        dC_proj_bias = None
        if ctx.is_variable_C:
            if not A.is_complex():
                dC = rearrange(dC, "b 1 dstate l -> (b l) dstate").contiguous()
            else:
                dC = rearrange(dC, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
            dC_proj_bias = dC.sum(0) if not ctx.C_proj_bias_is_None else None
            dx_dbl[:, -d_state:] = dC  # (bl d)
            dC = None
        ddelta = rearrange(ddelta, "b d l -> d (b l)")
        ddelta_proj_weight = torch.einsum("dB,Br->dr", ddelta, x_dbl[:, :delta_rank])
        dx_dbl[:, :delta_rank] = torch.einsum("dB,dr->Br", ddelta, delta_proj_weight)
        dconv1d_out = rearrange(dconv1d_out, "b d l -> d (b l)")
        dx_proj_weight = torch.einsum("Br,Bd->rd", dx_dbl, rearrange(conv1d_out, "b d l -> (b l) d"))
        dconv1d_out = torch.addmm(dconv1d_out, x_proj_weight.t(), dx_dbl.t(), out=dconv1d_out)
        dconv1d_out = rearrange(dconv1d_out, "d (b l) -> b d l", b=x.shape[0], l=x.shape[-1])
        # The kernel supports passing in a pre-allocated dx (e.g., in case we want to fuse the
        # backward of conv1d with the backward of chunk).
        dx, dconv1d_weight, dconv1d_bias, *_ = causal_conv1d_cuda.causal_conv1d_bwd(
            x, conv1d_weight, conv1d_bias, dconv1d_out, None, None, None, dx, False, True
        )
        dconv1d_bias = dconv1d_bias if conv1d_bias is not None else None
        dconv1d_weight = rearrange(dconv1d_weight, "d w -> d 1 w")
        return (dxz, dconv1d_weight, dconv1d_bias, dx_proj_weight, ddelta_proj_weight,
                dout_proj_weight, dout_proj_bias,
                dA, dB, dC, dD,
                ddelta_bias if delta_bias is not None else None,
                dB_proj_bias, dC_proj_bias, None)


def mamba_inner_fn(
    xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
    out_proj_weight, out_proj_bias,
    A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
    C_proj_bias=None, delta_softplus=True
):
    return MambaInnerFn.apply(xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
                              out_proj_weight, out_proj_bias,
                              A, B, C, D, delta_bias, B_proj_bias, C_proj_bias, delta_softplus)


def mamba_inner_ref(
    xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
    out_proj_weight, out_proj_bias,
    A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
    C_proj_bias=None, delta_softplus=True
):
    assert causal_conv1d_fn is not None, "causal_conv1d_fn is not available. Please install causal-conv1d."
    L = xz.shape[-1]
    delta_rank = delta_proj_weight.shape[1]
    d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
    x, z = xz.chunk(2, dim=1)
    x = causal_conv1d_fn(x, rearrange(conv1d_weight, "d 1 w -> d w"), conv1d_bias, activation="silu")
    # We're being very careful here about the layout, to avoid extra transposes.
    # We want delta to have d as the slowest moving dimension
    # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
    x_dbl = F.linear(rearrange(x, 'b d l -> (b l) d'), x_proj_weight)  # (bl d)
    delta = delta_proj_weight @ x_dbl[:, :delta_rank].t()
    delta = rearrange(delta, "d (b l) -> b d l", l=L)
    if B is None:  # variable B
        B = x_dbl[:, delta_rank:delta_rank + d_state]  # (bl d)
        if B_proj_bias is not None:
            B = B + B_proj_bias.to(dtype=B.dtype)
        if not A.is_complex():
            B = rearrange(B, "(b l) dstate -> b dstate l", l=L).contiguous()
        else:
            B = rearrange(B, "(b l) (dstate two) -> b dstate (l two)", l=L, two=2).contiguous()
    if C is None:  # variable B
        C = x_dbl[:, -d_state:]  # (bl d)
        if C_proj_bias is not None:
            C = C + C_proj_bias.to(dtype=C.dtype)
        if not A.is_complex():
            C = rearrange(C, "(b l) dstate -> b dstate l", l=L).contiguous()
        else:
            C = rearrange(C, "(b l) (dstate two) -> b dstate (l two)", l=L, two=2).contiguous()
    y = selective_scan_fn(x, delta, A, B, C, D, z=z, delta_bias=delta_bias, delta_softplus=True)
    return F.linear(rearrange(y, "b d l -> b l d"), out_proj_weight, out_proj_bias)


###############################################################################
#
# Longhorn Inner Function
#
###############################################################################


class LonghornInnerFn(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
                norm_weight, out_proj_weight, D=None, delta_bias=None):
        """
             x: (batch, dim, seqlen)
        """
        assert causal_conv1d_cuda is not None, "causal_conv1d_cuda is not available. Please install causal-conv1d."
        L = xz.shape[-1]
        R = delta_proj_weight.shape[1]
        DD = (x_proj_weight.shape[0] - R) // 2

        if torch.is_autocast_enabled():
            x_proj_weight = x_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            delta_proj_weight = delta_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())

        if xz.stride(-1) != 1:
            xz = xz.contiguous()

        conv1d_weight = rearrange(conv1d_weight, "d 1 w -> d w")
        x, z = xz.chunk(2, dim=1)
        conv1d_bias = conv1d_bias.contiguous() if conv1d_bias is not None else None
        conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(
            x, conv1d_weight, conv1d_bias, None, None, None, True
        )
        x_dbl = F.linear(rearrange(conv1d_out, 'b d l -> (b l) d'), x_proj_weight)  # (bl d)
        delta = rearrange(delta_proj_weight @ x_dbl[:, :R].t(), "d (b l) -> b d l", l = L)

        K = x_dbl[:, R:R+DD]  # (bl dstate)
        Q = x_dbl[:, -DD:]  # (bl dstate)
        K = rearrange(K, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
        Q = rearrange(Q, "(b l) dstate -> b 1 dstate l", l=L).contiguous()

        if D is not None:
            D = D.contiguous()

        import online_selective_scan_cuda
        if norm_weight is None:
            online_z = z
        else:
            online_z = None
        out, scan_intermediates, *rest = online_selective_scan_cuda.online_fwd(
            conv1d_out,
            Q,
            K,
            delta,
            D,
            delta_bias,
            online_z,
        )
        bb, ll, dd = out.shape
        o_prenorm = rearrange(out, 'b d l -> (b l) d')
        if norm_weight is not None:
            z = rearrange(z, 'b d l -> (b l) d')
            out, orsigma = fast_layer_norm.ln_fwd(o_prenorm, z, norm_weight, 1e-6)
        else:
            out = rest[0]
            out = rearrange(out, "b d l -> (b l) d")
        y = rearrange(F.linear(out, out_proj_weight), '(b l) d -> b l d', b=bb)
        if norm_weight is None:
            orsigma = None

        ctx.save_for_backward(xz, conv1d_out, conv1d_weight, conv1d_bias,
                              x_dbl[:, :R].clone(), x_proj_weight, delta_proj_weight, out_proj_weight,
                              Q, K, D, delta, delta_bias, scan_intermediates,
                              norm_weight, o_prenorm, out, orsigma)
        ctx.x_dbl_size = x_dbl.size()
        ctx.has_norm = norm_weight is not None
        return y

    @staticmethod
    @custom_bwd
    def backward(ctx, dout):
        # dout: (batch, seqlen, dim)
        assert causal_conv1d_cuda is not None, "causal_conv1d_cuda is not available. Please install causal-conv1d."

        (
            xz, conv1d_out, conv1d_weight, conv1d_bias,
            x_dbl_r, x_proj_weight, delta_proj_weight, out_proj_weight,
            Q, K, D, delta, delta_bias, scan_intermediates,
            norm_weight, o_prenorm, oz, orsigma
        ) = ctx.saved_tensors

        L = xz.shape[-1]
        R = delta_proj_weight.shape[1]
        DD = (x_proj_weight.shape[0] - R) // 2
        x, z = xz.chunk(2, dim=1)
        if dout.stride(-1) != 1:
            dout = dout.contiguous()

        dxz = torch.empty_like(xz)
        dx, dz = dxz.chunk(2, dim=1)
        dout = rearrange(dout, "b l e -> e (b l)")
        dout_proj_weight = torch.einsum("eB,Bd->ed", dout, oz)
        dout_y = out_proj_weight.t() @ dout
        # do = rearrange(out_proj_weight.t() @ dout, "d (b l) -> (b l) d", l=L)
        # do = rearrange(dout_y.contiguous(), "b d l -> (b l) d")

        bb, dd, ll = z.shape
        if ctx.has_norm:
            z = rearrange(z, 'b d l -> (b l) d').contiguous()
            do = rearrange(dout_y, "d (b l) -> (b l) d", l=L)
            do, dz_ln_out, do_rms_weight, _, _, _ = fast_layer_norm.ln_bwd(do.contiguous(), o_prenorm, z, orsigma, norm_weight)
            dz_ln_out = rearrange(dz_ln_out, '(b l) d -> b d l', b=bb).contiguous()
            online_z = None
            dz_online = None
            do = rearrange(do, '(b l) d -> b d l', b=bb).contiguous()
        else:
            do = rearrange(dout_y, 'd (b l) -> b d l', b=bb).contiguous()
            do_rms_weight = None
            online_z = z
            dz_online = dz

        import online_selective_scan_cuda
        du, dQ, dK, ddelta, dD, dz_online_out, ddelta_bias, dTK = online_selective_scan_cuda.online_bwd(
            conv1d_out,
            Q,
            K,
            delta,
            D,
            delta_bias,
            online_z, # z
            do,
            scan_intermediates,
            rearrange(o_prenorm, '(b l) d -> b d l', b=bb).contiguous(),
            dz_online
        )
        if not ctx.has_norm:
            dz = dz_online
        dx_dbl = torch.empty(ctx.x_dbl_size, dtype=x_dbl_r.dtype, device=x_dbl_r.device)

        dQ = rearrange(dQ, "b 1 dstate l -> (b l) dstate")#.contiguous()
        dx_dbl[:, -DD:].copy_(dQ)  # (bl d)
        dQ = None

        dK = get_dk(dTK, dK, K)

        dK = rearrange(dK, "b 1 dstate l -> (b l) dstate")#.contiguous()
        dx_dbl[:, R:R+DD].copy_(dK) # (bl d)
        dK = None

        ddelta = rearrange(ddelta, "b d l -> d (b l)")

        ddelta_proj_weight = torch.einsum("dB,Br->dr", ddelta, x_dbl_r)
        dx_dbl[:, :R].copy_(torch.einsum("dB,dr->Br", ddelta, delta_proj_weight))
        dconv1d_out = rearrange(du, "b d l -> d (b l)")
        dx_proj_weight = torch.einsum("Br,Bd->rd", dx_dbl, rearrange(conv1d_out, "b d l -> (b l) d"))
        dconv1d_out = torch.addmm(dconv1d_out, x_proj_weight.t(), dx_dbl.t(), out=dconv1d_out)
        dconv1d_out = rearrange(dconv1d_out, "d (b l) -> b d l", b=x.shape[0], l=x.shape[-1])
        # The kernel supports passing in a pre-allocated dx (e.g., in case we want to fuse the
        # backward of conv1d with the backward of chunk).
        _, dconv1d_weight, dconv1d_bias, *_ = causal_conv1d_cuda.causal_conv1d_bwd(
            x, conv1d_weight, conv1d_bias, dconv1d_out, None, None, None, dx, False, True
        )
        if norm_weight is not None:
            dz.copy_(dz_ln_out)
        dconv1d_bias = dconv1d_bias if conv1d_bias is not None else None
        # dconv1d_weight = rearrange(dconv1d_weight, "d w -> d 1 w")

        return (dxz, dconv1d_weight.unsqueeze(1), dconv1d_bias, dx_proj_weight, ddelta_proj_weight,
                do_rms_weight,
                dout_proj_weight,
                dD.to(D.dtype) if D is not None else None,
                ddelta_bias)


def longhorn_inner_fn(
    xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight, norm_weight, out_proj_weight,
    D=None, delta_bias=None,
):
    return LonghornInnerFn.apply(xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
                                      norm_weight, out_proj_weight, D, delta_bias)


def rms_norm_ref(x, weight, eps=1e-6):
    rstd = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)
    out = (x * rstd * weight).to(x.dtype)
    return out


def longhorn_ref(
    xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight, norm_weight, out_proj_weight,
    D=None, delta_bias=None,
):
    assert causal_conv1d_fn is not None, "causal_conv1d_fn is not available. Please install causal-conv1d."
    has_norm_weight = norm_weight is not None
    L = xz.shape[-1]
    R = delta_proj_weight.shape[1]
    DD = (x_proj_weight.shape[0] - R) // 2

    x, z = xz.chunk(2, dim=1)
    x = causal_conv1d_fn(x, rearrange(conv1d_weight, "d 1 w -> d w"), conv1d_bias, activation="silu").contiguous()
    x_dbl = F.linear(rearrange(x, 'b d l -> (b l) d'), x_proj_weight)  # (bl d)
    delta = delta_proj_weight @ x_dbl[:, :R].t()
    delta = rearrange(delta, "d (b l) -> b d l", l=L)

    K = x_dbl[:, R:R+DD]  # (bl d)
    K = rearrange(K, "(b l) dstate -> b dstate l", l=L).contiguous()

    Q = x_dbl[:, -DD:]  # (bl d)
    Q = rearrange(Q, "(b l) dstate -> b dstate l", l=L).contiguous()

    if has_norm_weight:
        online_z = None
    else:
        online_z = z

    y = selective_scan_online7_fn(x, Q.to(x), K.to(x), delta.to(x),
                                  D=D,
                                  t_bias=delta_bias,
                                  z=online_z, return_last_state=False)
    y = rearrange(y, "b d l -> b l d")
    if has_norm_weight:
        y = rms_norm_ref(y, norm_weight).to(y) * F.silu(rearrange(z, 'b d l -> b l d')).to(z)
    return F.linear(y, out_proj_weight)


###############################################################################
#
# Bidirectional Longhorn Inner Function
#
###############################################################################


def bi_longhorn_ref(
    xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight, norm_weight_forward, norm_weight_backward, out_proj_weight, D=None, delta_bias=None,
):
    assert causal_conv1d_fn is not None, "causal_conv1d_fn is not available. Please install causal-conv1d."

    L = xz.shape[-1]
    R = delta_proj_weight.shape[1]
    DD = (x_proj_weight.shape[0] - R) // 2

    x, z = xz.chunk(2, dim=1)
    x = causal_conv1d_fn(x, rearrange(conv1d_weight, "d 1 w -> d w"), conv1d_bias, activation="silu").contiguous()
    x_dbl = F.linear(rearrange(x, 'b d l -> (b l) d'), x_proj_weight)  # (bl d)
    delta = delta_proj_weight @ x_dbl[:, :R].t()
    delta = rearrange(delta, "d (b l) -> b d l", l=L)

    K = x_dbl[:, R:R+DD]  # (bl d)
    K = rearrange(K, "(b l) dstate -> b dstate l", l=L).contiguous()

    Q = x_dbl[:, -DD:]  # (bl d)
    Q = rearrange(Q, "(b l) dstate -> b dstate l", l=L).contiguous()

    y = selective_scan_online7_fn(x, Q.to(x), K.to(x), delta.to(x),
                                  D=D,
                                  t_bias=delta_bias,
                                  z=None, return_last_state=False)
    y_b = selective_scan_online7_fn(x.flip([-1]), Q.to(x).flip([-1]), K.to(x).flip([-1]), delta.to(x).flip([-1]),
                                    D=D,
                                    t_bias=delta_bias,
                                    z=None, return_last_state=False)
    y_b = y_b.flip([-1])
    y = rms_norm_ref(rearrange(y, 'b d l -> b l d'), norm_weight_forward).to(y) + rms_norm_ref(rearrange(y_b, 'b d l -> b l d'), norm_weight_backward).to(y_b)
    y = y * F.silu(rearrange(z, 'b d l -> b l d')).to(z)
    return F.linear(y, out_proj_weight)
