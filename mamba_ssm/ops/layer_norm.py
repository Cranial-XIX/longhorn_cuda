import torch
from torch.nn import init

import fast_layer_norm




class FastLayerNormFN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, g, gamma, epsilon):
        ctx.x_shape = x.shape

        x = x.contiguous()
        gamma = gamma.contiguous()
        hidden_size = gamma.numel()
        xmat = x.view((-1, hidden_size))
        ymat, rsigma = fast_layer_norm.ln_fwd(xmat, g, gamma, epsilon)
        ctx.save_for_backward(xmat, g, gamma, rsigma)
        ctx.rms_only = rms_only
        return ymat.view(x.shape)

    @staticmethod
    def backward(ctx, dy):
        # assert dy.is_contiguous()
        dy = dy.contiguous()  # this happens!
        x_or_y_mat, g, gamma, rsigma = ctx.saved_tensors
        dymat = dy.view(x_or_y_mat.shape)
        dxmat, dg, dgamma, dbeta, _, _ = fast_layer_norm.ln_bwd(dymat, x_or_y_mat,g,  rsigma, gamma)
        dx = dxmat.view(ctx.x_shape)
        return dx, dg, dgamma, None


def _fast_layer_norm(x, g, weight, epsilon):
    with torch.amp.autocast('cuda', enabled=False):
        return FastLayerNormFN.apply(x, g, weight, epsilon)


if __name__ == '__main__':
    torch.manual_seed(10)
    hidden_size = 1024
    dtype=torch.bfloat16
    x = (torch.randn(2, hidden_size, device="cuda", dtype=dtype) + 1) * 3
    g = torch.randn_like(x)
    x.requires_grad_()
    g.requires_grad_()
    weight = torch.randn(hidden_size, device="cuda", dtype=dtype)
    weight.requires_grad_()
    epsilon = 1e-10
    rms_only = True
    memory_efficient = False

    y = _fast_layer_norm(x, g, weight, epsilon)

    def rms_norm_ref(x, g, weight, eps=1e-6, dim=-1):
        dtype = x.dtype
        x = x.float()
        rstd = torch.rsqrt((x.square()).mean(dim=dim, keepdim=True) + eps)
        out = (x * rstd * weight * torch.nn.functional.silu(g)).to(dtype)
        return out

    grad_vec = torch.randn_like(y)
    y.backward(grad_vec)
    x_grad, x.grad = x.grad, None
    w_grad, weight.grad = weight.grad, None
    g_grad, g.grad = g.grad, None
    y_ref = rms_norm_ref(x, g, weight, eps=epsilon)
    y_ref.backward(grad_vec)
    print(y - y_ref)
    # print(x_grad, x.grad)
    print(x_grad - x.grad)
    print(w_grad - weight.grad)
    print(g_grad - g.grad)
