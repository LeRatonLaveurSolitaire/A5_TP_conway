#include "conway.h"
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

__global__ void game_of_life_kernel(int *grid, int *new_grid, int width,
                                    int height) {
  for (int block_start_x = blockIdx.x * blockDim.x; block_start_x < width;
       block_start_x += blockDim.x * gridDim.x) {

    for (int block_start_y = blockIdx.y * blockDim.y; block_start_y < height;
         block_start_y += blockDim.y * gridDim.y) {

      int x = block_start_x + threadIdx.x;
      int y = block_start_y + threadIdx.y;

      if (x >= width || y >= height);
        //continue;

      // TODO: 1. Calculate the number of alive neighbors
      int alive = 0;
      for(int i = -1; i < 2; i++){
        for(int j = -1; j < 2; j++){

          if((x+i) < 0 || x+i> width-1 || y+j<0 || y+j>height-1 || (i == 0 && j == 0)){}
          else{
            alive += grid[(x+i)+(y*width+j*width)];
          }
        }
      }

      // TODO: 2. Apply the rules of Conway's Game of Life
      if(grid[x+y*width]){ // LIVE
        if (alive < 2 || alive > 3){ // KILL
          new_grid[x+y*width] = 0;
        } else {
          new_grid[x+y*width] = 1;
        }
      } else { // DED
        if(alive == 3){ // BORN
          new_grid[x+y*width] = 1;
        } else {
          new_grid[x+y*width] = 0;
        }
      }

      // TODO: 3. Write the result to the new grid
      //new_grid = grid;

      // TODO(once you pass the conformance test): measure with nvprof, and
      // check for different ways of improving performance
    }
  }
}

void game_of_life_step(torch::Tensor grid_in, torch::Tensor grid_out,
                       std::optional<torch::Stream> stream) {
  int width = grid_in.size(1);
  int height = grid_in.size(0);
  assert(grid_in.sizes() == grid_out.sizes());

  cudaStream_t cudaStream = 0;
  if (stream.has_value()) {
    cudaStream = c10::cuda::CUDAStream(stream.value()).stream();
  }

  const dim3 blockSize(1, 16);
  const dim3 gridSize(1, 1);

  game_of_life_kernel<<<gridSize, blockSize, 0, cudaStream>>>(
      grid_in.data_ptr<int>(), grid_out.data_ptr<int>(), width, height);
}
