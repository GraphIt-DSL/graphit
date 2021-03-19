#ifndef GRAPHIT_GPU_PRINTER
#define GRAPHIT_GPU_PRINTER
#include <string>

namespace gpu_runtime {
void __device__ print(int32_t val) {
	printf("%d\n", val);
}
void __device__ print(float val) {
	printf("%f\n", val);
}
void __device__ print(const char* val) {
	printf("%s\n", val);
}
}

#endif
