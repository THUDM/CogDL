#include <cuda.h>
#include <torch/types.h>

__global__ void start() { printf("start profiling\n"); }

__global__ void end() { printf("end profiling\n"); }

void startProfile() { start<<<1, 1>>>(); }
void endProfile() { end<<<1, 1>>>(); }