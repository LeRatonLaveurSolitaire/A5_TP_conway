#include "conway.h"
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

// Experimental 
// N'est pas le plus rapide mais possède des implémentations possible :
// - SHARED MEMORY
// - VECTORIZATION DES CACLULS PAR INSTRUCTIONS PTX


#define WARP_SIZE 8
#define NB_BLOC_Y 3

// Vectorized addition
__device__ unsigned int __vadd4(unsigned int a, unsigned int b) {
  unsigned int c;
  asm("vadd.u32.u32.u32.add %0, %1, %2;" : "=r"(c) : "r"(a), "r"(b));
  return c;
}
/*
// Vectorized && 
__device__ unsigned int __vand4(unsigned int a, unsigned int b) {
  unsigned int c;
  asm("vand.u32.u32.u32 %0, %1, %2;" : "=r"(c) : "r"(a), "r"(b));
  return c;
}

// Vectorized ||
__device__ unsigned int __vor4(unsigned int a, unsigned int b) {
  unsigned int c;
  asm("vor.u32.u32.u32 %0, %1, %2;" : "=r"(c) : "r"(a), "r"(b));
  return c;
}

// Vectorized ==
__device__ unsigned int __veq4(unsigned int a, unsigned int b) {
  unsigned int c;
  asm("veq.u32.u32.u32 %0, %1, %2;" : "=r"(c) : "r"(a), "r"(b));
  return c;
}*/

__global__ void game_of_life_kernel(unsigned int *grid, unsigned int *new_grid, short width, short height) {
  // Get the x and y coordinates of the current thread
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  int widths = blockDim.x + 2;
  int sm_x = threadIdx.x + 1;
  int sm_y = threadIdx.y + 1;

  // Calculate the number of alive neighbors
  unsigned int alive = 0;
  unsigned int c1, c2;

  __shared__ unsigned int grid_s[(WARP_SIZE+2)*(NB_BLOC_Y+2)];
  grid_s[(threadIdx.x+1) + (threadIdx.y+1) * (WARP_SIZE+2)] = grid[x + y * width];

  // Load horizontal neighbors first for coalesced accesses
  if (threadIdx.x == 0 && x > 0)
    grid_s[sm_y * widths + (sm_x - 1)] = grid[y * width + (x - 1)];
  if (threadIdx.x == blockDim.x - 1 && x < width - 1)
    grid_s[sm_y * widths + (sm_x + 1)] = grid[y * width + (x + 1)];

  // Then load vertical neighbors
  if (threadIdx.y == 0 && y > 0)
    grid_s[(sm_y - 1) * widths + sm_x] = grid[(y - 1) * width + x];
  if (threadIdx.y == blockDim.y - 1 && y < height - 1)
    grid_s[(sm_y + 1) * widths + sm_x] = grid[(y + 1) * width + x];

  // Finally load diagonal neighbors
  if (threadIdx.x == 0 && threadIdx.y == 0 && x > 0 && y > 0)
    grid_s[(sm_y - 1) * widths + (sm_x - 1)] = grid[(y - 1) * width + (x - 1)];
  if (threadIdx.x == blockDim.x - 1 && threadIdx.y == 0 && x < width - 1 && y > 0)
    grid_s[(sm_y - 1) * widths + (sm_x + 1)] = grid[(y - 1) * width + (x + 1)];
  if (threadIdx.x == 0 && threadIdx.y == blockDim.y - 1 && x > 0 && y < height - 1)
    grid_s[(sm_y + 1) * widths + (sm_x - 1)] = grid[(y + 1) * width + (x - 1)];
  if (threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.y - 1 && x < width - 1 && y < height - 1)
    grid_s[(sm_y + 1) * widths + (sm_x + 1)] = grid[(y + 1) * width + (x + 1)];

  __syncthreads();

  // Neighbor sums
  for(char j = -1; j <= 1; j++){
    for(char i = -1; i <= 1; i++){
      // Check if the neighbor is within the grid bounds and not the current cell
      if(x + i >= 0 && x + i < width && y + j >= 0 && y + j < height && (i != 0 || j != 0)) {
        asm("vadd4.u32.u32.u32 %0, %1, %2, %3;" : "=r"(alive) : "r"(alive), "r"(grid[(x + i) + (y + j) * width]), "r"(0));
        //alive = __vadd4(alive, *(unsigned int*)(grid + (x + i) + (y + j) * width));
      }
    }
  }

  // Apply the rules of Conway's Game of Life, vous pouvez me hair pour cette ligne
  //new_grid[x + y * width] = 1;//__vor4(__vand4(grid[x + y * width], __veq4(alive, 2)), __veq4(alive, 3)) || (__vand4(!grid[x + y * width], __veq4(alive, 3)));
  asm("vset2.u32.u32.eq %0, %1, %2, %3;" : "=r"(c1): "r"(alive), "r"(grid[x + y * width] << 1), "r"(0)); // alive == 2 && grid[x + y * width] == 1
  asm("vset2.u32.u32.eq %0, %1, %2, %3;" : "=r"(c2): "r"(alive), "r"(3), "r"(0)); // alive == 3
  asm("vmax4.u32.u32.u32 %0,%1,%2,%3;" : "=r"(new_grid[x + y * width]) : "r"(c1), "r"(c2), "r"(0)); // max(c1,c2) => c1 || c2
}

void game_of_life_step(torch::Tensor grid_in, torch::Tensor grid_out,
                       std::optional<torch::Stream> stream) {
  short width = grid_in.size(1);
  short height = grid_in.size(0);
  assert(grid_in.sizes() == grid_out.sizes());

  cudaStream_t cudaStream = 0;
  if (stream.has_value()) {
    cudaStream = c10::cuda::CUDAStream(stream.value()).stream();
  }


  const dim3 blockSize( WARP_SIZE,NB_BLOC_Y);
  const dim3 gridSize(width/WARP_SIZE+1,height/NB_BLOC_Y+1);

  game_of_life_kernel<<<gridSize, blockSize, 0, cudaStream>>>(
    reinterpret_cast<unsigned int*>(grid_out.data_ptr<unsigned char>()),
    reinterpret_cast<unsigned int*>(grid_out.data_ptr<unsigned char>()),
    width / 4, height);
}
