import torch
from torch import einsum
import torch.nn.functional as F
import triton
import triton.language as tl
from task import input_t, output_t

# Original kernel for the first part of the computation.
@triton.jit
def _optimized_fused_gating_gemm_kernel(
    A_ptr,
    B_left_proj_ptr, B_left_gate_ptr,
    B_right_proj_ptr, B_right_gate_ptr,
    B_out_gate_ptr,
    Mask_ptr,
    C_left_ptr, C_right_ptr, C_out_gate_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_b_lp_k, stride_b_lp_n,
    stride_b_lg_k, stride_b_lg_n,
    stride_b_rp_k, stride_b_rp_n,
    stride_b_rg_k, stride_b_rg_n,
    stride_b_og_k, stride_b_og_n,
    stride_mask_m,
    stride_cl_m, stride_cl_n,
    stride_cr_m, stride_cr_n,
    stride_cog_m, stride_cog_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    valid_m = offs_m[:, None] < M
    valid_n = offs_n[None, :] < N

    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_lp_ptrs = B_left_proj_ptr + (offs_k[:, None] * stride_b_lp_k + offs_n[None, :] * stride_b_lp_n)
    b_lg_ptrs = B_left_gate_ptr + (offs_k[:, None] * stride_b_lg_k + offs_n[None, :] * stride_b_lg_n)
    b_rp_ptrs = B_right_proj_ptr + (offs_k[:, None] * stride_b_rp_k + offs_n[None, :] * stride_b_rp_n)
    b_rg_ptrs = B_right_gate_ptr + (offs_k[:, None] * stride_b_rg_k + offs_n[None, :] * stride_b_rg_n)
    b_og_ptrs = B_out_gate_ptr + (offs_k[:, None] * stride_b_og_k + offs_n[None, :] * stride_b_og_n)

    acc_lp = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc_lg = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc_rp = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc_rg = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc_og = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_SIZE_K):
        k_idx = k_start + offs_k
        k_mask = k_idx < K

        a = tl.load(a_ptrs + k_start * stride_ak, mask=valid_m & k_mask[None, :], other=0.0)
        b_lp = tl.load(b_lp_ptrs + k_start * stride_b_lp_k, mask=k_mask[:, None] & valid_n, other=0.0)
        b_lg = tl.load(b_lg_ptrs + k_start * stride_b_lg_k, mask=k_mask[:, None] & valid_n, other=0.0)
        b_rp = tl.load(b_rp_ptrs + k_start * stride_b_rp_k, mask=k_mask[:, None] & valid_n, other=0.0)
        b_rg = tl.load(b_rg_ptrs + k_start * stride_b_rg_k, mask=k_mask[:, None] & valid_n, other=0.0)
        b_og = tl.load(b_og_ptrs + k_start * stride_b_og_k, mask=k_mask[:, None] & valid_n, other=0.0)

        acc_lp += tl.dot(a, b_lp)
        acc_lg += tl.dot(a, b_lg)
        acc_rp += tl.dot(a, b_rp)
        acc_rg += tl.dot(a, b_rg)
        acc_og += tl.dot(a, b_og)

    mask_ptrs = Mask_ptr + offs_m * stride_mask_m
    mask_val = tl.load(mask_ptrs, mask=offs_m < M, other=0.0)

    gate_left = tl.sigmoid(acc_lg)
    c_left = acc_lp * gate_left * mask_val[:, None]
    c_left_ptrs = C_left_ptr + stride_cl_m * offs_m[:, None] + stride_cl_n * offs_n[None, :]

    gate_right = tl.sigmoid(acc_rg)
    c_right = acc_rp * gate_right * mask_val[:, None]
    c_right_ptrs = C_right_ptr + stride_cr_m * offs_m[:, None] + stride_cr_n * offs_n[None, :]

    c_out_gate = tl.sigmoid(acc_og)
    c_out_gate_ptrs = C_out_gate_ptr + stride_cog_m * offs_m[:, None] + stride_cog_n * offs_n[None, :]

    output_mask = valid_m & valid_n
    tl.store(c_left_ptrs, c_left.to(tl.float16), mask=output_mask)
    tl.store(c_right_ptrs, c_right.to(tl.float16), mask=output_mask)
    tl.store(c_out_gate_ptrs, c_out_gate, mask=output_mask)

# Optimization #6: Fused kernel for LayerNorm, element-wise multiplication, and Linear layer.
@triton.jit
def _fused_post_einsum_kernel(
    OutEinsum_ptr, OutGate_ptr,
    ToOutNormW_ptr, ToOutNormB_ptr,
    ToOutW_t_ptr,
    FinalOut_ptr,
    M, N, K,  # M=M_out, N=dim, K=hidden_dim
    stride_out_einsum_m, stride_out_einsum_k,
    stride_out_gate_m, stride_out_gate_k,
    stride_norm_w, stride_norm_b,
    stride_to_out_w_k, stride_to_out_w_n,
    stride_final_out_m, stride_final_out_n,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    LN_BLOCK_K: tl.constexpr,
    EPSILON: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # --- LayerNorm part (two-pass reduction for robustness) ---
    row_indices = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask_m = row_indices < M
    out_einsum_row_ptrs = OutEinsum_ptr + (row_indices[:, None] * stride_out_einsum_m)

    mean = tl.zeros((BLOCK_SIZE_M, 1), dtype=tl.float32)
    var = tl.zeros((BLOCK_SIZE_M, 1), dtype=tl.float32)

    # 1. Compute sum for mean
    offs_k_ln = tl.arange(0, LN_BLOCK_K)
    for k_base in range(0, tl.cdiv(K, LN_BLOCK_K)):
        k_offs = k_base * LN_BLOCK_K + offs_k_ln
        k_mask = k_offs < K
        row_block = tl.load(out_einsum_row_ptrs + k_offs[None, :] * stride_out_einsum_k,
                            mask=mask_m[:, None] & k_mask[None, :], other=0.0)
        mean += tl.sum(row_block, axis=1)[:, None]
    mean = mean / K

    # 2. Compute sum of squares for variance
    for k_base in range(0, tl.cdiv(K, LN_BLOCK_K)):
        k_offs = k_base * LN_BLOCK_K + offs_k_ln
        k_mask = k_offs < K
        row_block = tl.load(out_einsum_row_ptrs + k_offs[None, :] * stride_out_einsum_k,
                            mask=mask_m[:, None] & k_mask[None, :], other=0.0)
        row_block_minus_mean = row_block - mean
        var += tl.sum(row_block_minus_mean * row_block_minus_mean, axis=1)[:, None]
    var = var / K
    rstd = tl.math.rsqrt(var + EPSILON)

    # --- Fused GEMM part ---
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    offs_k_gemm = tl.arange(0, BLOCK_SIZE_K)
    
    to_out_w_t_ptrs = ToOutW_t_ptr + (offs_k_gemm[:, None] * stride_to_out_w_k + offs_n[None, :] * stride_to_out_w_n)

    for k_start in range(0, K, BLOCK_SIZE_K):
        k_idx = k_start + offs_k_gemm
        k_mask = k_idx < K

        # Load block of transposed weight matrix B
        b = tl.load(to_out_w_t_ptrs + k_start * stride_to_out_w_k,
                    mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)

        # Compute block of matrix A on the fly
        a_einsum_ptrs = OutEinsum_ptr + (offs_m[:, None] * stride_out_einsum_m + k_idx[None, :] * stride_out_einsum_k)
        a_einsum = tl.load(a_einsum_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=0.0)

        a_gate_ptrs = OutGate_ptr + (offs_m[:, None] * stride_out_gate_m + k_idx[None, :] * stride_out_gate_k)
        a_gate = tl.load(a_gate_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=0.0)

        norm_w_ptrs = ToOutNormW_ptr + k_idx
        norm_b_ptrs = ToOutNormB_ptr + k_idx
        gamma_block = tl.load(norm_w_ptrs, mask=k_mask, other=0.0)
        beta_block = tl.load(norm_b_ptrs, mask=k_mask, other=0.0)

        # Apply LayerNorm and gating
        a_norm = (a_einsum - mean) * rstd
        a_norm = a_norm * gamma_block[None, :] + beta_block[None, :]
        a = a_norm * a_gate

        # Accumulate result
        acc += tl.dot(a, b)

    # Store final result
    final_out_ptrs = FinalOut_ptr + (offs_m[:, None] * stride_final_out_m + offs_n[None, :] * stride_final_out_n)
    mask_out = mask_m[:, None] & (offs_n[None, :] < N)
    tl.store(final_out_ptrs, acc, mask=mask_out)

class TriMulFunctional:
    def __init__(self, dim: int, hidden_dim: int, weights: dict, device: torch.device):
        self.dim = dim
        self.hidden_dim = hidden_dim

        gemm_weights = [
            'left_proj.weight', 'right_proj.weight', 'left_gate.weight',
            'right_gate.weight', 'out_gate.weight'
        ]
        self.weights = {}
        for k, v in weights.items():
            if k in gemm_weights:
                self.weights[k] = v.to(device=device, dtype=torch.float16)
            else:
                self.weights[k] = v.to(device=device, dtype=torch.float32)

        # Pre-transpose all weights for efficient GEMM computation.
        self.weights_t = {
            k.replace('.weight', ''): v.T.contiguous()
            for k, v in self.weights.items() if k.endswith('.weight')
        }

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Perform layer normalization in float32.
        x_norm = F.layer_norm(x, (self.dim,), self.weights['norm.weight'], self.weights['norm.bias'])
        bs, seq_len, _, dim = x.shape

        # Reshape the normalized input for the fused GEMM kernel.
        M = bs * seq_len * seq_len
        x_norm_reshaped = x_norm.reshape(M, dim)
        mask_reshaped = mask.reshape(M)

        x_norm_reshaped_fp16 = x_norm_reshaped.to(torch.float16)

        left = torch.empty((M, self.hidden_dim), device=x.device, dtype=torch.float16)
        right = torch.empty((M, self.hidden_dim), device=x.device, dtype=torch.float16)
        out_gate = torch.empty((M, self.hidden_dim), device=x.device, dtype=torch.float32)

        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M'])
                               * triton.cdiv(self.hidden_dim, meta['BLOCK_SIZE_N']),)

        _optimized_fused_gating_gemm_kernel[grid](
            x_norm_reshaped_fp16,
            self.weights_t['left_proj'], self.weights_t['left_gate'],
            self.weights_t['right_proj'], self.weights_t['right_gate'],
            self.weights_t['out_gate'],
            mask_reshaped,
            left, right, out_gate,
            M, self.hidden_dim, self.dim,
            x_norm_reshaped_fp16.stride(0), x_norm_reshaped_fp16.stride(1),
            self.weights_t['left_proj'].stride(0), self.weights_t['left_proj'].stride(1),
            self.weights_t['left_gate'].stride(0), self.weights_t['left_gate'].stride(1),
            self.weights_t['right_proj'].stride(0), self.weights_t['right_proj'].stride(1),
            self.weights_t['right_gate'].stride(0), self.weights_t['right_gate'].stride(1),
            self.weights_t['out_gate'].stride(0), self.weights_t['out_gate'].stride(1),
            mask_reshaped.stride(0),
            left.stride(0), left.stride(1),
            right.stride(0), right.stride(1),
            out_gate.stride(0), out_gate.stride(1),
            BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32
        )

        left = left.reshape(bs, seq_len, seq_len, self.hidden_dim)
        right = right.reshape(bs, seq_len, seq_len, self.hidden_dim)
        out_gate = out_gate.reshape(bs, seq_len, seq_len, self.hidden_dim)

        out = einsum('...ikd,...jkd->...ijd', left, right).to(torch.float32)

        # --- Start of Fused Post-Processing ---
        M_out = bs * seq_len * seq_len
        out_reshaped = out.reshape(M_out, self.hidden_dim)
        out_gate_reshaped = out_gate.reshape(M_out, self.hidden_dim)
        
        final_out = torch.empty((M_out, self.dim), device=x.device, dtype=torch.float32)

        grid_post = lambda meta: (triton.cdiv(M_out, meta['BLOCK_SIZE_M'])
                                  * triton.cdiv(self.dim, meta['BLOCK_SIZE_N']),)

        w_norm = self.weights['to_out_norm.weight']
        b_norm = self.weights['to_out_norm.bias']
        w_out_t = self.weights_t['to_out']

        _fused_post_einsum_kernel[grid_post](
            out_reshaped, out_gate_reshaped,
            w_norm, b_norm, w_out_t,
            final_out,
            M_out, self.dim, self.hidden_dim,
            out_reshaped.stride(0), out_reshaped.stride(1),
            out_gate_reshaped.stride(0), out_gate_reshaped.stride(1),
            w_norm.stride(0), b_norm.stride(0),
            w_out_t.stride(0), w_out_t.stride(1),
            final_out.stride(0), final_out.stride(1),
            BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32,
            LN_BLOCK_K=128, # Block size for LayerNorm reduction
            EPSILON=1e-5  # Default epsilon for LayerNorm
        )
        # --- End of Fused Post-Processing ---

        return final_out.reshape(bs, seq_len, seq_len, self.dim)


def custom_kernel(data: input_t) -> output_t:
    """
    Entry point for the fused triangular multiplicative update.
    Only custom_kernel() will be imported during evaluation.
    """
    input_tensor, mask, weights, config = data

    trimul_func = TriMulFunctional(
        dim=config["dim"],
        hidden_dim=config["hidden_dim"],
        weights=weights,
        device=input_tensor.device
    )

    output = trimul_func.forward(input_tensor.to(torch.float32), mask)
    return output.to(torch.float32)