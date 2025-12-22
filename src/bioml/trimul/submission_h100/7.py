import torch
from torch import nn, einsum
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def three_kernel(
    left_ptr,
    right_ptr,
    out_ptr,
    
    batch_size: tl.constexpr,
    seq_len: tl.constexpr,
    dim: tl.constexpr,
    
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    
):
    pid = tl.program_id(0)
    block_start_m = pid * BLOCK_SIZE_M
    block_start_n = pid * BLOCK_SIZE_N    
    offsets_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)

    mask_m = offsets_m < seq_len
    mask_n = offsets_n < seq_len
    
    left_ptrs = left_ptr + offsets_m[:, None] * seq_len * dim + offsets_n[None, :] * dim
    right_ptrs = right_ptr + offsets_m[:, None] * seq_len * dim + offsets_n[None, :] * dim
    
    left = tl.load(left_ptrs, mask=mask_m[:, None, :])
    
    
def three(left, right):
    r"""
    Args:
        left: torch.Tensor, bfloat16, Shape: [batch_size, seq_len, seq_len, dim]    
        right: torch.Tensor, bfloat16, Shape: [batch_size, seq_len, seq_len, dim]
    Returns:
        out: torch.Tensor, float32, Shape: [batch_size, seq_len, seq_len, dim]
    """

    out = torch.empty_like(left, dtype=torch.float32)
    
@torch.compile
def custom_kernel(data):
    """
    Reference implementation of TriMul using PyTorch.
    Notes:
        1. All of the F.linear are using torch.float32.
        2. Layernorm is using torch.float32.
        3. the trimul can use a lower precision, but need to transpose something for tensor cores.
    
    Args:
        data: Tuple of (input: torch.Tensor, mask: torch.Tensor, weights: Dict[str, torch.Tensor], config: Dict)
            - input: Input tensor of shape [batch_size, seq_len, seq_len, dim]
            - mask: Mask tensor of shape [batch_size, seq_len, seq_len]
            - weights: Dictionary containing model weights
            - config: Dictionary containing model configuration parameters
    """
    x, mask, weights, config = data
    hidden_dim = config["hidden_dim"]
    
    batch_size, seq_len, _, dim = x.shape

    x = F.layer_norm(x, (dim,), weight=weights['norm.weight'], bias=weights["norm.bias"])

    left = F.linear(x, weights['left_proj.weight'])
    right = F.linear(x, weights['right_proj.weight'])
    mask = mask.unsqueeze(-1)
    left = left * mask
    right = right * mask
    
    left_gate = F.linear(x, weights['left_gate.weight']).sigmoid()
    right_gate = F.linear(x, weights['right_gate.weight']).sigmoid()
    out_gate = F.linear(x, weights['out_gate.weight']).sigmoid()

    left = left * left_gate
    right = right * right_gate

    # This is O(n^3), 
    # I guess main trick is to implement with tensor cores? Perhaps it doesn't
    # do it automatically because the layout isn't supported?
    out = einsum('... i k d, ... j k d -> ... i j d', left.to(torch.bfloat16), right.to(torch.bfloat16))

    out = out.to(torch.float32)
    out = F.layer_norm(out, (hidden_dim,), weight=weights["to_out_norm.weight"], bias=weights["to_out_norm.bias"])
    out = out * out_gate
    
    return F.linear(out, weights["to_out.weight"])