#ifndef GRAPHIT_GPU_SUPPORT_H
#define GRAPHIT_GPU_SUPPORT_H

#define PREFIX_BLK (1024)

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

/*
__device__ int atomicAggInc(int *ptr) {
    int pred;
    int mask = __match_all_sync(__activemask(), ptr, &pred);
    int leader = __ffs(mask) – 1;    // select a leader
    int res;
    if(threadIdx.x%32 == leader)                  // leader does the update
        res = atomicAdd(ptr, __popc(mask));
    res = __shfl_sync(mask, res, leader);    // get leader’s old value
    return res + __popc(mask & ((1 << (threadIdx.x%32)) – 1)); //compute old value
}


__device__ inline int32_t atomicAggInc(int32_t *ctr) {
  unsigned int active = __activemask();
  int leader = __ffs(active) - 1;
  int change = __popc(active);
  unsigned int rank = __popc(active & __lanemask_lt());
  int32_t warp_res;
  if(rank == 0)
    warp_res = atomicAdd(ctr, change);
  warp_res = __shfl_sync(active, warp_res, leader);
  return warp_res + rank;
}
*/


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


}


#endif
