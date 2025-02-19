#include "conway.h"
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>


__global__ void game_of_life_kernel(unsigned char *grid, unsigned char *new_grid, short width, short height) {
  // Get the x and y coordinates of the current thread
  short x = blockIdx.x * blockDim.x + threadIdx.x;
  short y = blockIdx.y * blockDim.y + threadIdx.y;

  // Calculate the number of alive neighbors
  unsigned char alive = 0;

  // int pointer to the new grid at x and y
  //unsigned int* new_grid_ptr = (unsigned int*)(new_grid + x + y * width);

  // Neighbor sums
  for(char j = -1; j <= 1; j++){
    for(char i = -1; i <= 1; i++){
      // Check if the neighbor is within the grid bounds and not the current cell
      if(x + i >= 0 && x + i < width && y + j >= 0 && y + j < height && (i != 0 || j != 0)) {
        alive += grid[(x + i) + (y + j) * width];
        //alive = __vadd4(alive, *(unsigned int*)(grid + (x + i) + (y + j) * width));
      }
    }
  }

  // Apply the rules of Conway's Game of Life
  new_grid[x + y * width] = ((grid[x + y * width] && (alive == 2 || alive == 3)) || (!grid[x + y * width] && alive == 3)) ? 1 : 0;
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

  #define WARP_SIZE 32

  // const dim3 blockSize( WARP_SIZE,WARP_SIZE);
  const dim3 blockSize(WARP_SIZE, WARP_SIZE);
  const dim3 gridSize(width/WARP_SIZE,height/WARP_SIZE);

  game_of_life_kernel<<<gridSize, blockSize, 0, cudaStream>>>(
      grid_in.data_ptr<unsigned char>(), grid_out.data_ptr<unsigned char>(), width, height);
}
