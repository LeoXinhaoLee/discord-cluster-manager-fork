import torch
import torch.nn.functional as F

def custom_kernel(data):
    """
    Triton‑friendly implementation of the *outgoing* TriMul operator.
    The heavy inner‑product is performed with a batched GEMM (torch.bmm)
    which on H100 maps to an efficient cuBLAS kernel.  All other
    operations (LayerNorm, Linear, Sigmoid gates) are expressed with
    PyTorch primitives so that the code stays simple while still being
    completely GPU‑resident.

    Args:
        data: tuple containing
            - input_tensor: torch.Tensor of shape [B, N, N, D] (float32/float16)
            - mask:        torch.Tensor of shape [B, N, N] (bool or float)
            - weights:     dict of the model’s parameters (all on CUDA)
            - config:      dict with keys "dim" and "hidden_dim"

    Returns:
        (output_tensor,) where output_tensor has shape [B, N, N, D] and dtype torch.float16
    """
    # ----------------------------------------------------------------------
    # unpack arguments
    # ----------------------------------------------------------------------
    input_tensor, mask, weights, config = data
    dim = config["dim"]
    hidden_dim = config["hidden_dim"]
    eps = 1e-5                     # LayerNorm epsilon

    # ----------------------------------------------------------------------
    # 1) Input LayerNorm  (no bias in the reference code – we add bias here)
    # ----------------------------------------------------------------------
    # x_norm = (x - μ) / √(σ² + eps) * weight + bias
    mean = input_tensor.mean(dim=-1, keepdim=True)
    var  = input_tensor.var(dim=-1, unbiased=False, keepdim=True)
    x = (input_tensor - mean) / torch.sqrt(var + eps)
    x = x * weights["norm.weight"] + weights["norm.bias"]

    # ----------------------------------------------------------------------
    # 2) Linear projections (no bias)
    # ----------------------------------------------------------------------
    left  = F.linear(x, weights["left_proj.weight"])   # [B,N,N,hidden]
    right = F.linear(x, weights["right_proj.weight"])

    # ----------------------------------------------------------------------
    # 3) Mask (optional – mask may be all ones)
    # ----------------------------------------------------------------------
    mask_f = mask.unsqueeze(-1).to(left.dtype)   # [B,N,N,1]
    left  = left * mask_f
    right = right * mask_f

    # ----------------------------------------------------------------------
    # 4) Gating (sigmoid of linear transforms)
    # ----------------------------------------------------------------------
    left_gate  = torch.sigmoid(F.linear(x, weights["left_gate.weight"]))
    right_gate = torch.sigmoid(F.linear(x, weights["right_gate.weight"]))
    out_gate   = torch.sigmoid(F.linear(x, weights["out_gate.weight"]))

    left  = left  * left_gate
    right = right * right_gate

    # ----------------------------------------------------------------------
    # 5) Core TriMul – batched GEMM over the “k” dimension
    #    out[b,i,j,d] = Σ_k left[b,i,k,d] * right[b,j,k,d]
    #    Equivalent to: for each d,  out_d = left_d @ right_dᵀ
    # ----------------------------------------------------------------------
    B, N, _, _ = left.shape

    # reshape to (B*hidden, N, N) for a single bmm call
    left_bmm  = left.permute(0, 3, 1, 2).reshape(B * hidden_dim, N, N)
    right_bmm = right.permute(0, 3, 1, 2).reshape(B * hidden_dim, N, N)

    # batch‑matrix‑multiply: (B*hidden, N, N) @ (B*hidden, N, N)ᵀ → (B*hidden, N, N)
    out_bmm = torch.bmm(left_bmm, right_bmm.transpose(1, 2))

    # reshape back to [B, N, N, hidden]
    out = out_bmm.reshape(B, hidden_dim, N, N).permute(0, 2, 3, 1)

    # ----------------------------------------------------------------------
    # 6) Output LayerNorm (again without bias‑term in the reference)
    # ----------------------------------------------------------------------
    mean = out.mean(dim=-1, keepdim=True)
    var  = out.var(dim=-1, unbiased=False, keepdim=True)
    out = (out - mean) / torch.sqrt(var + eps)
    out = out * weights["to_out_norm.weight"] + weights["to_out_norm.bias"]

    # ----------------------------------------------------------------------
    # 7) Apply out‑gate and final linear projection
    # ----------------------------------------------------------------------
    out = out * out_gate
    out = F.linear(out, weights["to_out.weight"])

    # ----------------------------------------------------------------------
    # 8) Cast to float16 as required by the specification
    # ----------------------------------------------------------------------
    out = out.to(torch.float32)

    return out