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
	
        int32_t mask = __activemask();
        int32_t leader = __ffs(mask) - 1;
        int32_t res;
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

/*
__device__ inline int32_t upperbound(int32_t *array, int32_t len, int32_t key){
  int32_t s = 0;
  while(len>0){
    int32_t half = len>>1;
    int32_t mid = s + half;
    if(array[mid] > key){
      len = half;
    }else{
      s = mid+1;
      len = len-half-1;
    }
  }
  return s;
}*/



}
#endif
