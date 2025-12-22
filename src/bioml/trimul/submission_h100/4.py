#!POPCORN leaderboard trimul

import torch
import torch.nn.functional as F
from task import input_t, output_t

def trimul_forward(x, mask,
                   norm_w, norm_b,
                   left_proj_w, right_proj_w,
                   left_gate_w, right_gate_w, out_gate_w,
                   to_out_norm_w, to_out_norm_b,
                   to_out_w):
    """
    Compiled forward pass (math only, assumes inputs/weights are correct).
    """
    d = norm_w.shape[0]
    h = left_proj_w.shape[0]

    # Fuse LayerNorm and first set of linear projections
    x_norm = F.layer_norm(x, (d,), weight=norm_w, bias=norm_b, eps=1e-5)

    stacked_w = torch.cat(
        [left_proj_w, right_proj_w, left_gate_w, right_gate_w, out_gate_w],
        dim=0
    )
    projected = F.linear(x_norm, stacked_w)  # (B, L, L, 5*h)

    lr = projected[..., :2*h]
    lr_gate = projected[..., 2*h:4*h]
    out_gate = projected[..., 4*h:]

    if mask is not None:
        lr_gated = lr * torch.sigmoid(lr_gate) * mask.unsqueeze(-1)
    else:
        lr_gated = lr * torch.sigmoid(lr_gate)

    left_gated = lr_gated[..., :h]
    right_gated = lr_gated[..., h:]

    # This works in basically the same time as hand-written code
    out = torch.einsum("bikh,bjkh->bijh", left_gated, right_gated)

    out_norm = F.layer_norm(out, (h,), weight=to_out_norm_w, bias=to_out_norm_b, eps=1e-5)
    out_gated = out_norm * torch.sigmoid(out_gate)
    final_out = F.linear(out_gated, to_out_w)

    return final_out


# Compile ONLY the math-heavy forward
trimul_forward_compile = torch.compile(trimul_forward)

def custom_kernel(data: input_t) -> output_t:
    """
    Full kernel with weight fetching, but only forward math is compiled.
    """
    input_tensor, mask, weights, config = data
    device, dtype = input_tensor.device, input_tensor.dtype

    def get_weight(key):
        w = weights[key]
        if w.device != device or w.dtype != dtype:
            w = w.to(device=device, dtype=dtype)
            weights[key] = w
        return w

    norm_w = get_weight('norm.weight')
    norm_b = get_weight('norm.bias')
    left_proj_w = get_weight('left_proj.weight')
    right_proj_w = get_weight('right_proj.weight')
    left_gate_w = get_weight('left_gate.weight')
    right_gate_w = get_weight('right_gate.weight')
    out_gate_w = get_weight('out_gate.weight')
    to_out_norm_w = get_weight('to_out_norm.weight')
    to_out_norm_b = get_weight('to_out_norm.bias')
    to_out_w = get_weight('to_out.weight')

    with torch.amp.autocast('cuda'):
        return trimul_forward_compile(
            input_tensor, mask,
            norm_w, norm_b,
            left_proj_w, right_proj_w,
            left_gate_w, right_gate_w, out_gate_w,
            to_out_norm_w, to_out_norm_b,
            to_out_w
        )
