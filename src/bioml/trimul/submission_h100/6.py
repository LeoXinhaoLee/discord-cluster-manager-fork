import torch
import torch.nn.functional as F
from torch import einsum
from task import input_t, output_t

@torch.compile
def trimul(x: torch.Tensor, mask: torch.Tensor,
          norm_weight: torch.Tensor, norm_bias: torch.Tensor,
          left_proj_weight: torch.Tensor, right_proj_weight: torch.Tensor,
          left_gate_weight: torch.Tensor, right_gate_weight: torch.Tensor,
          out_gate_weight: torch.Tensor,
          to_out_norm_weight: torch.Tensor, to_out_norm_bias: torch.Tensor,
          to_out_weight: torch.Tensor) -> torch.Tensor:
    """
    Functional implementation of TriMul.

    Args:
        x: [bs, seq_len, seq_len, dim]
        mask: [bs, seq_len, seq_len]
        norm_weight: Layer norm weight
        norm_bias: Layer norm bias
        left_proj_weight: Left projection weight
        right_proj_weight: Right projection weight
        left_gate_weight: Left gate weight
        right_gate_weight: Right gate weight
        out_gate_weight: Output gate weight
        to_out_norm_weight: Output layer norm weight
        to_out_norm_bias: Output layer norm bias
        to_out_weight: Final output projection weight

    Returns:
        output: [bs, seq_len, seq_len, dim]
    """
    batch_size, seq_len, _, dim = x.shape

    # Layer normalization
    x = F.layer_norm(x, [dim], norm_weight, norm_bias)

    # Fuse all linear projections with x as input
    # Concatenate all weights
    fused_weight = torch.cat([left_proj_weight, right_proj_weight,
                              left_gate_weight, right_gate_weight,
                              out_gate_weight], dim=0)

    # Single fused linear operation
    fused_output = F.linear(x, fused_weight)
    # fused_output = F.linear(x.to(torch.bfloat16), fused_weight.to(torch.bfloat16)).to(torch.float32)

    # Split the results
    hidden_dim = left_proj_weight.shape[0]
    left = fused_output[..., :hidden_dim]
    right = fused_output[..., hidden_dim:2*hidden_dim]
    left_gate = fused_output[..., 2*hidden_dim:3*hidden_dim].sigmoid()
    right_gate = fused_output[..., 3*hidden_dim:4*hidden_dim].sigmoid()
    out_gate = fused_output[..., 4*hidden_dim:5*hidden_dim].sigmoid()

    mask = mask.unsqueeze(-1)
    left = left * mask
    right = right * mask

    # Apply gates
    left = left * left_gate
    right = right * right_gate

    # Einstein summation
    out = einsum('... i k d, ... j k d -> ... i j d', left.to(torch.bfloat16), right.to(torch.bfloat16))
    # This einsum is the same as the following:
    # out = torch.zeros(batch_size, seq_len, seq_len, dim, device=x.device)

    # # Compute using nested loops
    # for b in range(batch_size):
    #     for i in range(seq_len):
    #         for j in range(seq_len):
    #             # Compute each output element
    #             for k in range(seq_len):
    #                 out[b, i, j] += left[b, i, k, :] * right[b, j, k, :]

    out = out.to(torch.float32)

    # Output normalization
    hidden_dim = out.shape[-1]
    out = F.layer_norm(out, [hidden_dim], to_out_norm_weight, to_out_norm_bias)
    out = out * out_gate

    # Final linear projection
    out = F.linear(out, to_out_weight)

    return out


def custom_kernel(data: input_t) -> output_t:
    """
    Reference implementation of TriMul using PyTorch functional interface.

    Args:
        data: Tuple of (input: torch.Tensor, mask: torch.Tensor, weights: Dict[str, torch.Tensor], config: Dict)
            - input: Input tensor of shape [batch_size, seq_len, seq_len, dim]
            - mask: Mask tensor of shape [batch_size, seq_len, seq_len]
            - weights: Dictionary containing model weights
            - config: Dictionary containing model configuration parameters
    """
    input_tensor, mask, weights, config = data

    output = trimul(
        input_tensor, mask,
        weights['norm.weight'], weights['norm.bias'],
        weights['left_proj.weight'], weights['right_proj.weight'],
        weights['left_gate.weight'], weights['right_gate.weight'],
        weights['out_gate.weight'],
        weights['to_out_norm.weight'], weights['to_out_norm.bias'],
        weights['to_out.weight']
    )

    return output
