import torch
from torch import nn
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline

# CUDA kernel implementation
fused_einsum_layernorm_source = '''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#define TILE_SIZE 16
#define WARP_SIZE 32
#define EPSILON 1e-6

namespace cg = cooperative_groups;

__global__ void fused_einsum_layernorm_kernel(
    const float* left,
    const float* right,
    const float* gamma,
    const float* beta,
    float* output,
    int batch_size,
    int height,
    int width,
    int dim
) {
    // Thread and block indices
    int batch_idx = blockIdx.z;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int tid = ty * blockDim.x + tx;
    
    // Bounds check
    if (batch_idx >= batch_size || i >= height || j >= width) return;
    
    // Shared memory for tiling and reduction
    __shared__ float left_tile[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float right_tile[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float reduction_buffer[256];
    
    // Process each feature dimension
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    
    for (int d = 0; d < dim; d++) {
        float einsum_result = 0.0f;
        
        // Tiled einsum computation
        for (int tile_k = 0; tile_k < (width + TILE_SIZE - 1) / TILE_SIZE; tile_k++) {
            // Load left tile
            int k_left = tile_k * TILE_SIZE + tx;
            if (i < height && k_left < width) {
                int left_idx = batch_idx * height * width * dim + i * width * dim + k_left * dim + d;
                left_tile[ty][tx] = left[left_idx];
            } else {
                left_tile[ty][tx] = 0.0f;
            }
            
            // Load right tile (transposed access pattern)
            int k_right = tile_k * TILE_SIZE + ty;
            if (j < width && k_right < width) {
                int right_idx = batch_idx * height * width * dim + j * width * dim + k_right * dim + d;
                right_tile[tx][ty] = right[right_idx];
            } else {
                right_tile[tx][ty] = 0.0f;
            }
            
            __syncthreads();
            
            // Compute partial einsum
            for (int k = 0; k < TILE_SIZE; k++) {
                if (tile_k * TILE_SIZE + k < width) {
                    einsum_result += left_tile[ty][k] * right_tile[tx][k];
                }
            }
            
            __syncthreads();
        }
        
        // Accumulate statistics for LayerNorm
        local_sum += einsum_result;
        local_sum_sq += einsum_result * einsum_result;
        
        // Store intermediate result in shared memory for later use
        reduction_buffer[tid] = einsum_result;
        __syncthreads();
        
        // If this is the last dimension, compute LayerNorm
        if (d == dim - 1) {
            // Compute mean and variance across all dimensions
            float mean = local_sum / dim;
            float variance = (local_sum_sq / dim) - (mean * mean);
            float inv_std = rsqrtf(variance + EPSILON);
            
            // Apply LayerNorm to all computed einsum results
            for (int d_norm = 0; d_norm < dim; d_norm++) {
                // Recompute einsum for this dimension (could be optimized with more shared memory)
                float einsum_val = 0.0f;
                
                for (int tile_k = 0; tile_k < (width + TILE_SIZE - 1) / TILE_SIZE; tile_k++) {
                    // Load left tile
                    int k_left = tile_k * TILE_SIZE + tx;
                    if (i < height && k_left < width) {
                        int left_idx = batch_idx * height * width * dim + i * width * dim + k_left * dim + d_norm;
                        left_tile[ty][tx] = left[left_idx];
                    } else {
                        left_tile[ty][tx] = 0.0f;
                    }
                    
                    // Load right tile
                    int k_right = tile_k * TILE_SIZE + ty;
                    if (j < width && k_right < width) {
                        int right_idx = batch_idx * height * width * dim + j * width * dim + k_right * dim + d_norm;
                        right_tile[tx][ty] = right[right_idx];
                    } else {
                        right_tile[tx][ty] = 0.0f;
                    }
                    
                    __syncthreads();
                    
                    // Compute partial einsum
                    for (int k = 0; k < TILE_SIZE; k++) {
                        if (tile_k * TILE_SIZE + k < width) {
                            einsum_val += left_tile[ty][k] * right_tile[tx][k];
                        }
                    }
                    
                    __syncthreads();
                }
                
                // Apply LayerNorm transformation
                float normalized = (einsum_val - mean) * inv_std;
                float final_result = normalized * gamma[d_norm] + beta[d_norm];
                
                // Write output
                int out_idx = batch_idx * height * width * dim + i * width * dim + j * dim + d_norm;
                output[out_idx] = final_result;
            }
        }
    }
}

__global__ void fused_einsum_layernorm_optimized_kernel(
    const float4* left,
    const float4* right,
    const float* gamma,
    const float* beta,
    float4* output,
    int batch_size,
    int height,
    int width,
    int dim_vec
) {
    // Vectorized version for better memory throughput
    int batch_idx = blockIdx.z;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int tid = ty * blockDim.x + tx;
    
    if (batch_idx >= batch_size || i >= height || j >= width) return;
    
    __shared__ float4 left_tile[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float4 right_tile[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float sdata[256];
    
    float total_sum = 0.0f;
    float total_sum_sq = 0.0f;
    
    // Store einsum results for later LayerNorm application
    float4 einsum_results[32]; // Assuming dim_vec <= 32
    
    for (int d = 0; d < dim_vec; d++) {
        float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        
        // Tiled einsum computation
        for (int tile_k = 0; tile_k < (width + TILE_SIZE - 1) / TILE_SIZE; tile_k++) {
            // Load left tile
            int k_left = tile_k * TILE_SIZE + tx;
            if (i < height && k_left < width) {
                int left_idx = batch_idx * height * width * dim_vec + i * width * dim_vec + k_left * dim_vec + d;
                left_tile[ty][tx] = left[left_idx];
            } else {
                left_tile[ty][tx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
            
            // Load right tile
            int k_right = tile_k * TILE_SIZE + ty;
            if (j < width && k_right < width) {
                int right_idx = batch_idx * height * width * dim_vec + j * width * dim_vec + k_right * dim_vec + d;
                right_tile[tx][ty] = right[right_idx];
            } else {
                right_tile[tx][ty] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
            
            __syncthreads();
            
            // Compute partial einsum
            for (int k = 0; k < TILE_SIZE; k++) {
                if (tile_k * TILE_SIZE + k < width) {
                    float4 left_val = left_tile[ty][k];
                    float4 right_val = right_tile[tx][k];
                    
                    sum.x += left_val.x * right_val.x;
                    sum.y += left_val.y * right_val.y;
                    sum.z += left_val.z * right_val.z;
                    sum.w += left_val.w * right_val.w;
                }
            }
            
            __syncthreads();
        }
        
        // Store result and accumulate statistics
        einsum_results[d] = sum;
        total_sum += sum.x + sum.y + sum.z + sum.w;
        total_sum_sq += sum.x*sum.x + sum.y*sum.y + sum.z*sum.z + sum.w*sum.w;
    }
    
    // Compute LayerNorm statistics
    int total_elements = dim_vec * 4;
    float mean = total_sum / total_elements;
    float variance = (total_sum_sq / total_elements) - (mean * mean);
    float inv_std = rsqrtf(variance + EPSILON);
    
    // Apply LayerNorm and write output
    for (int d = 0; d < dim_vec; d++) {
        float4 result = einsum_results[d];
        
        // Normalize
        result.x = (result.x - mean) * inv_std;
        result.y = (result.y - mean) * inv_std;
        result.z = (result.z - mean) * inv_std;
        result.w = (result.w - mean) * inv_std;
        
        // Apply gamma and beta
        int gamma_idx = d * 4;
        result.x = result.x * gamma[gamma_idx] + beta[gamma_idx];
        result.y = result.y * gamma[gamma_idx + 1] + beta[gamma_idx + 1];
        result.z = result.z * gamma[gamma_idx + 2] + beta[gamma_idx + 2];
        result.w = result.w * gamma[gamma_idx + 3] + beta[gamma_idx + 3];
        
        // Write output
        int out_idx = batch_idx * height * width * dim_vec + i * width * dim_vec + j * dim_vec + d;
        output[out_idx] = result;
    }
}

torch::Tensor fused_einsum_layernorm_cuda(
    torch::Tensor left,
    torch::Tensor right,
    torch::Tensor gamma,
    torch::Tensor beta
) {
    auto batch_size = left.size(0);
    auto height = left.size(1);
    auto width = left.size(2);
    auto dim = left.size(3);
    
    auto output = torch::zeros({batch_size, height, width, dim}, left.options());
    
    if (dim % 4 == 0) {
        // Use vectorized version
        int dim_vec = dim / 4;
        
        dim3 block_size(16, 16);
        dim3 num_blocks(
            (width + block_size.x - 1) / block_size.x,
            (height + block_size.y - 1) / block_size.y,
            batch_size
        );
        
        fused_einsum_layernorm_optimized_kernel<<<num_blocks, block_size>>>(
            reinterpret_cast<const float4*>(left.data_ptr<float>()),
            reinterpret_cast<const float4*>(right.data_ptr<float>()),
            gamma.data_ptr<float>(),
            beta.data_ptr<float>(),
            reinterpret_cast<float4*>(output.data_ptr<float>()),
            batch_size,
            height,
            width,
            dim_vec
        );
    } else {
        // Use scalar version
        dim3 block_size(16, 16);
        dim3 num_blocks(
            (width + block_size.x - 1) / block_size.x,
            (height + block_size.y - 1) / block_size.y,
            batch_size
        );
        
        fused_einsum_layernorm_kernel<<<num_blocks, block_size>>>(
            left.data_ptr<float>(),
            right.data_ptr<float>(),
            gamma.data_ptr<float>(),
            beta.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            height,
            width,
            dim
        );
    }
    
    return output;
}
'''

fused_einsum_layernorm_cpp_source = "torch::Tensor fused_einsum_layernorm_cuda(torch::Tensor left, torch::Tensor right, torch::Tensor gamma, torch::Tensor beta);"

fused_einsum_layernorm = load_inline(
    name="fused_einsum_layernorm",
    cpp_sources=[fused_einsum_layernorm_cpp_source],
    cuda_sources=[fused_einsum_layernorm_source],
    functions=["fused_einsum_layernorm_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)



class TriMul(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.left_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.left_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.out_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False, dtype=torch.float32)
        self.fused_einsum_layernorm = fused_einsum_layernorm

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _, dim = x.shape
        x = self.norm(x)
        x = x.to(torch.float32)
        left = self.left_proj(x)
        right = self.right_proj(x)
        mask = mask.unsqueeze(-1)
        left = left * mask
        right = right * mask
        left_gate = self.left_gate(x).sigmoid()
        right_gate = self.right_gate(x).sigmoid()
        out_gate = self.out_gate(x).sigmoid()
        left = left * left_gate
        right = right * right_gate

        out = self.fused_einsum_layernorm.fused_einsum_layernorm_cuda(
            left, right, 
            self.to_out_norm.weight, 
            self.to_out_norm.bias
        )
        
        out = out * out_gate
        return self.to_out(out)

def custom_kernel(data: input_t) -> output_t:
    """
    CUDA-accelerated TriMul for POPCORN leaderboard.
    Args:
        data: Tuple of (input: torch.Tensor, mask: torch.Tensor, weights: Dict[str, torch.Tensor], config: Dict)
    """
    input_tensor, mask, weights, config = data
    trimul = TriMul(config["dim"], config["hidden_dim"]).to(input_tensor.device)
    
    trimul.norm.weight = nn.Parameter(weights['norm.weight'].to(torch.float32))
    trimul.left_proj.weight = nn.Parameter(weights['left_proj.weight'].to(torch.float32))
    trimul.right_proj.weight = nn.Parameter(weights['right_proj.weight'].to(torch.float32))
    trimul.left_gate.weight = nn.Parameter(weights['left_gate.weight'].to(torch.float32))
    trimul.right_gate.weight = nn.Parameter(weights['right_gate.weight'].to(torch.float32))
    trimul.out_gate.weight = nn.Parameter(weights['out_gate.weight'].to(torch.float32))
    trimul.to_out_norm.weight = nn.Parameter(weights['to_out_norm.weight'].to(torch.float32))
    trimul.to_out.weight = nn.Parameter(weights['to_out.weight'].to(torch.float32))
    trimul.norm.bias = nn.Parameter(weights['norm.bias'].to(torch.float32))
    trimul.to_out_norm.bias = nn.Parameter(weights['to_out_norm.bias'].to(torch.float32))
    
    output = trimul(input_tensor, mask).to(torch.float32)
    return output