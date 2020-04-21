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
__device__ inline int32_t warp_bcast(int32_t mask, int32_t v, int32_t leader) {
	return __shfl_sync((uint32_t)mask, v, leader); 
}
__device__ inline int32_t atomicAggInc(int32_t *ctr) {
	int32_t lane_id = threadIdx.x % 32;
	
        int mask = __activemask();
        int leader = __ffs(mask) - 1;
        int res;
        if(lane_id == leader)
                res = atomicAdd(ctr, __popc(mask));
        res = warp_bcast(mask, res, leader);

        return (res + __popc(mask & ((1 << lane_id) - 1)));
}
template <typename T>
static bool __device__ writeMin(T *dst, T src) {
	if (*dst <= src)
		return false;
	T old_value = atomicMin(dst, src);
	bool ret = (old_value > src);
	return ret;
}
template <typename T>
static bool __device__ writeMax(T *dst, T src) {
	if (*dst >= src)
		return false;
	T old_value = atomicMax(dst, src);
	bool ret = (old_value < src);
	return ret;
}


template <typename T>
static bool __device__ writeAdd(T *dst, T src) {
	atomicAdd(dst, src);
	return true;
}
template <typename T>
static bool __device__ CAS(T *dst, T old_val, const T &new_val) {
	if (*dst != old_val)
		return false;
	return old_val == atomicCAS(dst, old_val, new_val);
}
static void __device__ parallel_memset(unsigned char* dst, unsigned char val, size_t total_bytes) {
	for (size_t index = threadIdx.x + blockDim.x * blockIdx.x; index < total_bytes; index += blockDim.x * gridDim.x)
		dst[index] = val;
}
static void __device__ parallel_memcpy(unsigned char* dst, unsigned char* src, size_t total_bytes) {
	for (size_t index = threadIdx.x + blockDim.x * blockIdx.x; index < total_bytes; index += blockDim.x * gridDim.x)
		dst[index] = src[index];
}
}

#endif
