import torch
from torch import nn
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline

einsum_mix_source = '''
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void einsum_mix_kernel(
    const float4* left,
    const float4* right,
    float4* output,
    int batch_size,
    int height,
    int width,
    int dim_vec
) {
    // Double buffering: Two sets of shared memory buffers
    __shared__ float4 left_tile_A[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float4 left_tile_B[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float4 right_tile_A[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float4 right_tile_B[TILE_SIZE][TILE_SIZE + 1];
    
    int batch_idx = blockIdx.z;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    
    if (batch_idx >= batch_size || i >= height || j >= width) return;
    
    for (int d = 0; d < dim_vec; d++) {
        // Register variables for intermediate computations
        float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 left_val_reg, right_val_reg;  // Register cache for frequently accessed values
        
        // Initialize buffer pointers
        float4 (*left_current)[TILE_SIZE + 1] = left_tile_A;
        float4 (*left_next)[TILE_SIZE + 1] = left_tile_B;
        float4 (*right_current)[TILE_SIZE + 1] = right_tile_A;
        float4 (*right_next)[TILE_SIZE + 1] = right_tile_B;
        
        int num_tiles = (width + TILE_SIZE - 1) / TILE_SIZE;
        
        // Pre-load first tile
        if (num_tiles > 0) {
            int k_left = 0 * TILE_SIZE + tx;
            if (i < height && k_left < width) {
                int left_idx = batch_idx * height * width * dim_vec + i * width * dim_vec + k_left * dim_vec + d;
                left_current[ty][tx] = left[left_idx];
            } else {
                left_current[ty][tx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
            
            int k_right = 0 * TILE_SIZE + ty;
            if (j < width && k_right < width) {
                int right_idx = batch_idx * height * width * dim_vec + j * width * dim_vec + k_right * dim_vec + d;
                right_current[tx][ty] = right[right_idx];
            } else {
                right_current[tx][ty] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
        
        __syncthreads();
        
        // Process tiles with double buffering
        for (int tile_k = 0; tile_k < num_tiles; tile_k++) {
            // Load next tile asynchronously while computing current tile
            if (tile_k + 1 < num_tiles) {
                int next_k_left = (tile_k + 1) * TILE_SIZE + tx;
                if (i < height && next_k_left < width) {
                    int left_idx = batch_idx * height * width * dim_vec + i * width * dim_vec + next_k_left * dim_vec + d;
                    left_next[ty][tx] = left[left_idx];
                } else {
                    left_next[ty][tx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                }
                
                int next_k_right = (tile_k + 1) * TILE_SIZE + ty;
                if (j < width && next_k_right < width) {
                    int right_idx = batch_idx * height * width * dim_vec + j * width * dim_vec + next_k_right * dim_vec + d;
                    right_next[tx][ty] = right[right_idx];
                } else {
                    right_next[tx][ty] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                }
            }
            
            // Compute using current tile while next tile is being loaded
            // Use register variables for the inner loop computation
            for (int k = 0; k < TILE_SIZE; k++) {
                if (tile_k * TILE_SIZE + k < width) {
                    // Load values into registers for faster access
                    left_val_reg = left_current[ty][k];
                    right_val_reg = right_current[tx][k];
                    
                    // Perform computation using register-cached values
                    sum.x += left_val_reg.x * right_val_reg.x;
                    sum.y += left_val_reg.y * right_val_reg.y;
                    sum.z += left_val_reg.z * right_val_reg.z;
                    sum.w += left_val_reg.w * right_val_reg.w;
                }
            }
            
            __syncthreads();
            
            // Swap buffers for next iteration
            float4 (*temp_left)[TILE_SIZE + 1] = left_current;
            left_current = left_next;
            left_next = temp_left;
            
            float4 (*temp_right)[TILE_SIZE + 1] = right_current;
            right_current = right_next;
            right_next = temp_right;
        }
        
        // Store final result
        if (i < height && j < width) {
            int out_idx = batch_idx * height * width * dim_vec + i * width * dim_vec + j * dim_vec + d;
            output[out_idx] = sum;
        }
    }
}

torch::Tensor einsum_mix_cuda(torch::Tensor left, torch::Tensor right) {
    auto batch_size = left.size(0);
    auto height = left.size(1);
    auto width = left.size(2);
    auto dim = left.size(3);
    
    // Ensure dimension is divisible by 4 for vectorization
    TORCH_CHECK(dim % 4 == 0, "Dimension must be divisible by 4 for vectorization");
    int dim_vec = dim / 4;
    
    auto output = torch::zeros({batch_size, height, width, dim}, left.options());
    
    dim3 block_size(16, 16);
    dim3 num_blocks(
        (width + block_size.x - 1) / block_size.x,
        (height + block_size.y - 1) / block_size.y,
        batch_size
    );
    
    einsum_mix_kernel<<<num_blocks, block_size>>>(
        reinterpret_cast<const float4*>(left.data_ptr<float>()),
        reinterpret_cast<const float4*>(right.data_ptr<float>()),
        reinterpret_cast<float4*>(output.data_ptr<float>()),
        batch_size,
        height,
        width,
        dim_vec
    );
    
    return output;
}
'''

einsum_mix_cpp_source = "torch::Tensor einsum_mix_cuda(torch::Tensor left, torch::Tensor right);"

einsum_mix = load_inline(
    name="einsum_mix",
    cpp_sources=[einsum_mix_cpp_source],
    cuda_sources=[einsum_mix_source],
    functions=["einsum_mix_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

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

        out = einsum_mix.einsum_mix_cuda(left, right)

        out = out.to(torch.float32)
        out = self.to_out_norm(out)
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