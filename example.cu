#include <iostream>
// __global__ specifier marks a GPU kernel. Returns void.
__global__

// saxpy stands for single precision a*x plus y.
// It is a combination of scalar multiplication and vector addition. 
void saxpy(int n, float a, float *x, float *y) {

  // threadIdx provides the location of the current thread in the block.
  // blockIdx provides the location of that block in the overall grid.
  // blockDim gives the number of threads in the block.
  // We can refer to a specific element in the grid by the expression below.
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  // saxpy expression
  if (i < n) y[i] = a*x[i] + y[i];
}

int main(void) {

  // left shift operator
  // convert one to binary and then add 20 zeros to get 2^20
  int N = 1<<20;

  float *x, *y, *d_x, *d_y;

  // Alocate memory on the host.
  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));

  // Allocates an array of size bytes on the device
  cudaMalloc(&d_x, N*sizeof(float));
  cudaMalloc(&d_y, N*sizeof(float));

  // fill x,y arrays with values.
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // cudaMemcpy( destination array, source array, number of bytes to transfer, direction kind )
  cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

  // Perform SAXPY on 1M elements
  // The kernel parameters blocksPerGrid and threadsPerBlock define the sizeof the problem. 
  // syntax: kernelFunctionName<<<blocksPerGrid, threadsPerBlock>>>(arguments ...)
  saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);

  // copy data from device to host.
  cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

  float maxError = 0.0f;

  for (int i = 0; i < N; i++)
    maxError = max(maxError, abs(y[i]-4.0f)); // maxError expected to be 0.0 because y[i] is 4.0f

  std::cout << "Max error: " << maxError << std::endl;


  cudaFree(d_x);
  cudaFree(d_y);
  free(x);
  free(y);
}