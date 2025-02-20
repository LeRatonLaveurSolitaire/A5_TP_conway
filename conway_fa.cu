#include "conway.h"
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

// FAB EXPERIMENT

#define WARP_SIZE 32
#define NB_BLOC_Y 3

__global__ void game_of_life_kernel(unsigned char *gridOriginal, unsigned char *new_grid, short widthOriginal, short heightOriginal) {
  // Get the x and y coordinates of the current thread
  short global_x = blockIdx.x * blockDim.x + threadIdx.x;
  short global_y = blockIdx.y * blockDim.y + threadIdx.y;

	if ( global_x >= widthOriginal || global_y >= heightOriginal) return;

  // Calculate the number of alive neighbors
  unsigned char alive = 0;

	//create index to compute in the shared mem
	short width = WARP_SIZE;
	short height= NB_BLOC_Y;
	short x = threadIdx.x+1;
	short y = threadIdx.y+1;
	__shared__ int grid[(WARP_SIZE+2)*(NB_BLOC_Y+2)];
	grid[x + y * width] = gridOriginal[global_x + global_y*widthOriginal];

	if (threadIdx.x == 0 && global_x > 0)
	    grid[y*width+x - 1] = gridOriginal[global_y * width + (global_x - 1)];
	if (threadIdx.x == blockDim.x - 1 && global_x < width - 1)
	    grid[y*width+x + 1] = gridOriginal[global_y * width + (global_x + 1)];
	if (threadIdx.y == 0 && global_y > 0)
  	  grid[(y - 1)*width+x] = gridOriginal[(global_y - 1) * width + global_x];
	if (threadIdx.y == blockDim.y - 1 && global_y < height - 1)
	    grid[(y + 1)*width+x] = gridOriginal[(global_y + 1) * width + global_x];

	if (threadIdx.x == 0 && threadIdx.y == 0 && global_x > 0 && global_y > 0)
	    grid[(y - 1)*width+x - 1] = gridOriginal[(global_y - 1) * width + (global_x - 1)];
	if (threadIdx.x == blockDim.x - 1 && threadIdx.y == 0 && global_x < width - 1 && global_y > 0)
  	  grid[(y - 1)*width+x + 1] = gridOriginal[(global_y - 1) * width + (global_x + 1)];
	if (threadIdx.x == 0 && threadIdx.y == blockDim.y - 1 && global_x > 0 && global_y < height - 1)
 	   grid[(y + 1)*width+x - 1] = gridOriginal[(global_y + 1) * width + (global_x - 1)];
	if (threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.y - 1 && global_x < width - 1 && global_y < height - 1)
    grid[(y + 1)*width+x + 1] = gridOriginal[(global_y + 1) * width + (global_x + 1)];


	__syncthreads();


	

  // int pointer to the new grid at x and y
  //unsigned int* new_grid_ptr = (unsigned int*)(new_grid + x + y * width);

  // Neighbor sums
  for(char j = -1; j <= 1; j++){
    for(char i = -1; i <= 1; i++){
      // Check if the neighbor is within the grid bounds and not the current cell
      if(global_x + i >= 0 && global_x + i < widthOriginal && global_y + j >= 0 && global_y + j < heightOriginal && (i != 0 || j != 0)) {
        alive += grid[(x + i) + (y + j) * width];
        //alive = __vadd4(alive, *(unsigned int*)(grid + (x + i) + (y + j) * width));
      }
    }
  }

  // Apply the rules of Conway's Game of Life
  new_grid[global_x + global_y * widthOriginal] = ((grid[x + y * width] && (alive == 2 || alive == 3)) || (!grid[x + y * width] && alive == 3)) ? 1 : 0;
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

  // const dim3 blockSize( WARP_SIZE,WARP_SIZE);
  const dim3 blockSize(WARP_SIZE, NB_BLOC_Y);
  const dim3 gridSize(width/WARP_SIZE+1,height/NB_BLOC_Y+1);

  game_of_life_kernel<<<gridSize, blockSize, 0, cudaStream>>>(
      grid_in.data_ptr<unsigned char>(), grid_out.data_ptr<unsigned char>(), width, height);
}
