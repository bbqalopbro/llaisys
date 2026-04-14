template <int WARPS_PER_ROW>
__global__ void gemv_v3(const float *__restrict__ A,
                        const float *__restrict__ x, float *__restrict__ y,
                        int M, int K) {
  const int WARP_SIZE = 32;
  const int THREADS_PER_ROW = WARPS_PER_ROW * WARP_SIZE;

  // block 内的线程分组
  int local_thread = threadIdx.x;
  int row_in_block = local_thread / THREADS_PER_ROW;
  int thread_in_row = local_thread % THREADS_PER_ROW;
  int warp_in_row = thread_in_row / WARP_SIZE;
  int lane_id = thread_in_row % WARP_SIZE;

  int rows_per_block = blockDim.x / THREADS_PER_ROW;
  int row = blockIdx.x * rows_per_block + row_in_block;
  if (row >= M) return;

  const float *row_ptr = A + row * K;
  float sum = 0.0f;

  // 每线程 float4 向量化读取, 在分段范围内
  int k4 = K / 4;
  int global_lane = warp_in_row * WARP_SIZE + lane_id;
  for (int i = global_lane; i < k4; i += THREADS_PER_ROW) {
    float4 a4 = reinterpret_cast<const float4 *>(row_ptr)[i];
    float4 x4 = reinterpret_cast<const float4 *>(x)[i];
    sum += a4.x * x4.x + a4.y * x4.y + a4.z * x4.z + a4.w * x4.w;
  }

  // 第一级归约: warp shuffle
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  }

  // 第二级归约: shared memory (仅 WARPS_PER_ROW > 1 时需要)
  // smem 布局: [rows_per_block][WARPS_PER_ROW]
  extern __shared__ float smem[];
  if (lane_id == 0) {
    smem[row_in_block * WARPS_PER_ROW + warp_in_row] = sum;
  }
  __syncthreads();

  // 第一个 warp 的 lane 0 做最终归约
  if (warp_in_row == 0 && lane_id == 0) {
    float total = 0.0f;
    for (int w = 0; w < WARPS_PER_ROW; w++) {
      total += smem[row_in_block * WARPS_PER_ROW + w];
    }
    y[row] = total;
  }
}