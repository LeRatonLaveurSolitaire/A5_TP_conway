#include "conway.h"
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

__global__ void game_of_life_kernel(signed char *grid, signed char *new_grid, short width,
                                    short height) {
  for (int block_start_y = blockIdx.y * blockDim.y; block_start_y < height;
         block_start_y += blockDim.y * gridDim.y) {
  
    for (int block_start_x = blockIdx.x * blockDim.x; block_start_x < width;
        block_start_x += blockDim.x * gridDim.x) {

      short x = block_start_x + threadIdx.x;
      short y = block_start_y + threadIdx.y;


      // TODO: 1. Calculate the number of alive neighbors
      char alive = 0;
      for(char j = -1; j < 2; j++){
        for(char i = -1; i < 2; i++){
          if(!((x+i) < 0 || x+i> width-1 || y+j<0 || y+j>height-1 || (!i && !j)))alive += grid[(x+i)+(y*width+j*width)];
        }
      }

      // TODO: 2. Apply the rules of Conway's Game of Life
      new_grid[x+y*width] =((!(grid[x+y*width]) && (alive == 3)) || ((grid[x+y*width]) && !(alive < 2 || alive > 3)));

    }
  }
}

void game_of_life_step(torch::Tensor grid_in, torch::Tensor grid_out,
                       std::optional<torch::Stream> stream) {
  char width = grid_in.size(1);
  char height = grid_in.size(0);
  assert(grid_in.sizes() == grid_out.sizes());

  cudaStream_t cudaStream = 0;
  if (stream.has_value()) {
    cudaStream = c10::cuda::CUDAStream(stream.value()).stream();
  }

  #define WARP_SIZE 32

  // const dim3 blockSize( WARP_SIZE,WARP_SIZE);
  const dim3 blockSize( WARP_SIZE,3);
  const dim3 gridSize(width/WARP_SIZE + 1,height/3 + 1);

  game_of_life_kernel<<<gridSize, blockSize, 0, cudaStream>>>(
      grid_in.data_ptr<signed char>(), grid_out.data_ptr<signed char>(), width, height);
}
