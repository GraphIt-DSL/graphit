#ifndef GRAPHIT_GPU_SUPPORT_H
#define GRAPHIT_GPU_SUPPORT_H
namespace gpu_runtime {
void cudaCheckLastError(void) {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
                printf("Error: %s\n", cudaGetErrorString(err));
		exit(-1);
	}
}
}

#endif
