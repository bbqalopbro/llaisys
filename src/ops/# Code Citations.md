# Code Citations

## License: 未知
https://github.com/ZhaoOfficial/CG/blob/ddd29cbc96109cba809bf601989b6cb4e7e36eb4/Cuda/src/chapter5/sharemem.cu

```
Good, I now have all the context. Let me answer the two questions thoroughly and then cover the remaining operators.

## 回答你的两个问题

---

### 问题 1：树形归约太 naive

**你说得对。** 当前 softmax kernel 的归约有两个已知问题：

**问题 a：最后几轮 warp divergence 严重**

```
当前代码：
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s)                             // ← 问题在这里
        sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
}
```

当 `s < 32`（一个 warp 的大小）时，同一个 warp 内只有部分线程在工作，其余线程空转但占着硬件资源。**而且 `s < 32` 时 `__syncthreads()` 是多余的**——同一个 warp 内的线程天然同步执行（SIMT 模型）。

**优化写法：warp-level 归约**

```cpp
// 当 s 缩到 32 以内时，切换到 warp shuffle 指令
// __shfl_down_sync 直接在寄存器间交换数据，不走 shared memory
for (int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (threadIdx.x < s)
        sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
}
// 最后 5 轮用 warp shuffle（寄存器级通信，比 shared memory 快 ~5x）
if (threadIdx.x < 32) {
    float val = sdata[threadIdx.x];
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1
```


## License: 未知
https://github.com/JieyangChen7/FT-MAGMA/blob/1d2e2940a0a56024cfe5d2b2898b2e64d7d294ee/magma-2.3.0/sparse/blas/zmdot_shfl.cu

```
Good, I now have all the context. Let me answer the two questions thoroughly and then cover the remaining operators.

## 回答你的两个问题

---

### 问题 1：树形归约太 naive

**你说得对。** 当前 softmax kernel 的归约有两个已知问题：

**问题 a：最后几轮 warp divergence 严重**

```
当前代码：
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s)                             // ← 问题在这里
        sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
}
```

当 `s < 32`（一个 warp 的大小）时，同一个 warp 内只有部分线程在工作，其余线程空转但占着硬件资源。**而且 `s < 32` 时 `__syncthreads()` 是多余的**——同一个 warp 内的线程天然同步执行（SIMT 模型）。

**优化写法：warp-level 归约**

```cpp
// 当 s 缩到 32 以内时，切换到 warp shuffle 指令
// __shfl_down_sync 直接在寄存器间交换数据，不走 shared memory
for (int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (threadIdx.x < s)
        sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
}
// 最后 5 轮用 warp shuffle（寄存器级通信，比 shared memory 快 ~5x）
if (threadIdx.x < 32) {
    float val = sdata[threadIdx.x];
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    if (threadIdx.x == 0
```


## License: 未知
https://github.com/ZhaoOfficial/CG/blob/ddd29cbc96109cba809bf601989b6cb4e7e36eb4/Cuda/src/chapter5/sharemem.cu

```
Good, I now have all the context. Let me answer the two questions thoroughly and then cover the remaining operators.

## 回答你的两个问题

---

### 问题 1：树形归约太 naive

**你说得对。** 当前 softmax kernel 的归约有两个已知问题：

**问题 a：最后几轮 warp divergence 严重**

```
当前代码：
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s)                             // ← 问题在这里
        sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
}
```

当 `s < 32`（一个 warp 的大小）时，同一个 warp 内只有部分线程在工作，其余线程空转但占着硬件资源。**而且 `s < 32` 时 `__syncthreads()` 是多余的**——同一个 warp 内的线程天然同步执行（SIMT 模型）。

**优化写法：warp-level 归约**

```cpp
// 当 s 缩到 32 以内时，切换到 warp shuffle 指令
// __shfl_down_sync 直接在寄存器间交换数据，不走 shared memory
for (int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (threadIdx.x < s)
        sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
}
// 最后 5 轮用 warp shuffle（寄存器级通信，比 shared memory 快 ~5x）
if (threadIdx.x < 32) {
    float val = sdata[threadIdx.x];
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1
```


## License: 未知
https://github.com/JieyangChen7/FT-MAGMA/blob/1d2e2940a0a56024cfe5d2b2898b2e64d7d294ee/magma-2.3.0/sparse/blas/zmdot_shfl.cu

```
Good, I now have all the context. Let me answer the two questions thoroughly and then cover the remaining operators.

## 回答你的两个问题

---

### 问题 1：树形归约太 naive

**你说得对。** 当前 softmax kernel 的归约有两个已知问题：

**问题 a：最后几轮 warp divergence 严重**

```
当前代码：
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s)                             // ← 问题在这里
        sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
}
```

当 `s < 32`（一个 warp 的大小）时，同一个 warp 内只有部分线程在工作，其余线程空转但占着硬件资源。**而且 `s < 32` 时 `__syncthreads()` 是多余的**——同一个 warp 内的线程天然同步执行（SIMT 模型）。

**优化写法：warp-level 归约**

```cpp
// 当 s 缩到 32 以内时，切换到 warp shuffle 指令
// __shfl_down_sync 直接在寄存器间交换数据，不走 shared memory
for (int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (threadIdx.x < s)
        sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
}
// 最后 5 轮用 warp shuffle（寄存器级通信，比 shared memory 快 ~5x）
if (threadIdx.x < 32) {
    float val = sdata[threadIdx.x];
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    if (threadIdx.x == 0
```


## License: 未知
https://github.com/ZhaoOfficial/CG/blob/ddd29cbc96109cba809bf601989b6cb4e7e36eb4/Cuda/src/chapter5/sharemem.cu

```
Good, I now have all the context. Let me answer the two questions thoroughly and then cover the remaining operators.

## 回答你的两个问题

---

### 问题 1：树形归约太 naive

**你说得对。** 当前 softmax kernel 的归约有两个已知问题：

**问题 a：最后几轮 warp divergence 严重**

```
当前代码：
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s)                             // ← 问题在这里
        sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
}
```

当 `s < 32`（一个 warp 的大小）时，同一个 warp 内只有部分线程在工作，其余线程空转但占着硬件资源。**而且 `s < 32` 时 `__syncthreads()` 是多余的**——同一个 warp 内的线程天然同步执行（SIMT 模型）。

**优化写法：warp-level 归约**

```cpp
// 当 s 缩到 32 以内时，切换到 warp shuffle 指令
// __shfl_down_sync 直接在寄存器间交换数据，不走 shared memory
for (int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (threadIdx.x < s)
        sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
}
// 最后 5 轮用 warp shuffle（寄存器级通信，比 shared memory 快 ~5x）
if (threadIdx.x < 32) {
    float val = sdata[threadIdx.x];
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1
```


## License: 未知
https://github.com/JieyangChen7/FT-MAGMA/blob/1d2e2940a0a56024cfe5d2b2898b2e64d7d294ee/magma-2.3.0/sparse/blas/zmdot_shfl.cu

```
Good, I now have all the context. Let me answer the two questions thoroughly and then cover the remaining operators.

## 回答你的两个问题

---

### 问题 1：树形归约太 naive

**你说得对。** 当前 softmax kernel 的归约有两个已知问题：

**问题 a：最后几轮 warp divergence 严重**

```
当前代码：
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s)                             // ← 问题在这里
        sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
}
```

当 `s < 32`（一个 warp 的大小）时，同一个 warp 内只有部分线程在工作，其余线程空转但占着硬件资源。**而且 `s < 32` 时 `__syncthreads()` 是多余的**——同一个 warp 内的线程天然同步执行（SIMT 模型）。

**优化写法：warp-level 归约**

```cpp
// 当 s 缩到 32 以内时，切换到 warp shuffle 指令
// __shfl_down_sync 直接在寄存器间交换数据，不走 shared memory
for (int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (threadIdx.x < s)
        sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
}
// 最后 5 轮用 warp shuffle（寄存器级通信，比 shared memory 快 ~5x）
if (threadIdx.x < 32) {
    float val = sdata[threadIdx.x];
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    if (threadIdx.x == 0
```


## License: 未知
https://github.com/ZhaoOfficial/CG/blob/ddd29cbc96109cba809bf601989b6cb4e7e36eb4/Cuda/src/chapter5/sharemem.cu

```
Good, I now have all the context. Let me answer the two questions thoroughly and then cover the remaining operators.

## 回答你的两个问题

---

### 问题 1：树形归约太 naive

**你说得对。** 当前 softmax kernel 的归约有两个已知问题：

**问题 a：最后几轮 warp divergence 严重**

```
当前代码：
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s)                             // ← 问题在这里
        sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
}
```

当 `s < 32`（一个 warp 的大小）时，同一个 warp 内只有部分线程在工作，其余线程空转但占着硬件资源。**而且 `s < 32` 时 `__syncthreads()` 是多余的**——同一个 warp 内的线程天然同步执行（SIMT 模型）。

**优化写法：warp-level 归约**

```cpp
// 当 s 缩到 32 以内时，切换到 warp shuffle 指令
// __shfl_down_sync 直接在寄存器间交换数据，不走 shared memory
for (int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (threadIdx.x < s)
        sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
}
// 最后 5 轮用 warp shuffle（寄存器级通信，比 shared memory 快 ~5x）
if (threadIdx.x < 32) {
    float val = sdata[threadIdx.x];
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1
```


## License: 未知
https://github.com/JieyangChen7/FT-MAGMA/blob/1d2e2940a0a56024cfe5d2b2898b2e64d7d294ee/magma-2.3.0/sparse/blas/zmdot_shfl.cu

```
Good, I now have all the context. Let me answer the two questions thoroughly and then cover the remaining operators.

## 回答你的两个问题

---

### 问题 1：树形归约太 naive

**你说得对。** 当前 softmax kernel 的归约有两个已知问题：

**问题 a：最后几轮 warp divergence 严重**

```
当前代码：
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s)                             // ← 问题在这里
        sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
}
```

当 `s < 32`（一个 warp 的大小）时，同一个 warp 内只有部分线程在工作，其余线程空转但占着硬件资源。**而且 `s < 32` 时 `__syncthreads()` 是多余的**——同一个 warp 内的线程天然同步执行（SIMT 模型）。

**优化写法：warp-level 归约**

```cpp
// 当 s 缩到 32 以内时，切换到 warp shuffle 指令
// __shfl_down_sync 直接在寄存器间交换数据，不走 shared memory
for (int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (threadIdx.x < s)
        sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
}
// 最后 5 轮用 warp shuffle（寄存器级通信，比 shared memory 快 ~5x）
if (threadIdx.x < 32) {
    float val = sdata[threadIdx.x];
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    if (threadIdx.x == 0
```


## License: 未知
https://github.com/ZhaoOfficial/CG/blob/ddd29cbc96109cba809bf601989b6cb4e7e36eb4/Cuda/src/chapter5/sharemem.cu

```
Good, I now have all the context. Let me answer the two questions thoroughly and then cover the remaining operators.

## 回答你的两个问题

---

### 问题 1：树形归约太 naive

**你说得对。** 当前 softmax kernel 的归约有两个已知问题：

**问题 a：最后几轮 warp divergence 严重**

```
当前代码：
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s)                             // ← 问题在这里
        sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
}
```

当 `s < 32`（一个 warp 的大小）时，同一个 warp 内只有部分线程在工作，其余线程空转但占着硬件资源。**而且 `s < 32` 时 `__syncthreads()` 是多余的**——同一个 warp 内的线程天然同步执行（SIMT 模型）。

**优化写法：warp-level 归约**

```cpp
// 当 s 缩到 32 以内时，切换到 warp shuffle 指令
// __shfl_down_sync 直接在寄存器间交换数据，不走 shared memory
for (int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (threadIdx.x < s)
        sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
}
// 最后 5 轮用 warp shuffle（寄存器级通信，比 shared memory 快 ~5x）
if (threadIdx.x < 32) {
    float val = sdata[threadIdx.x];
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1
```


## License: 未知
https://github.com/JieyangChen7/FT-MAGMA/blob/1d2e2940a0a56024cfe5d2b2898b2e64d7d294ee/magma-2.3.0/sparse/blas/zmdot_shfl.cu

```
Good, I now have all the context. Let me answer the two questions thoroughly and then cover the remaining operators.

## 回答你的两个问题

---

### 问题 1：树形归约太 naive

**你说得对。** 当前 softmax kernel 的归约有两个已知问题：

**问题 a：最后几轮 warp divergence 严重**

```
当前代码：
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s)                             // ← 问题在这里
        sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
}
```

当 `s < 32`（一个 warp 的大小）时，同一个 warp 内只有部分线程在工作，其余线程空转但占着硬件资源。**而且 `s < 32` 时 `__syncthreads()` 是多余的**——同一个 warp 内的线程天然同步执行（SIMT 模型）。

**优化写法：warp-level 归约**

```cpp
// 当 s 缩到 32 以内时，切换到 warp shuffle 指令
// __shfl_down_sync 直接在寄存器间交换数据，不走 shared memory
for (int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (threadIdx.x < s)
        sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
}
// 最后 5 轮用 warp shuffle（寄存器级通信，比 shared memory 快 ~5x）
if (threadIdx.x < 32) {
    float val = sdata[threadIdx.x];
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    if (threadIdx.x == 0
```


## License: 未知
https://github.com/ZhaoOfficial/CG/blob/ddd29cbc96109cba809bf601989b6cb4e7e36eb4/Cuda/src/chapter5/sharemem.cu

```
Good, I now have all the context. Let me answer the two questions thoroughly and then cover the remaining operators.

## 回答你的两个问题

---

### 问题 1：树形归约太 naive

**你说得对。** 当前 softmax kernel 的归约有两个已知问题：

**问题 a：最后几轮 warp divergence 严重**

```
当前代码：
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s)                             // ← 问题在这里
        sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
}
```

当 `s < 32`（一个 warp 的大小）时，同一个 warp 内只有部分线程在工作，其余线程空转但占着硬件资源。**而且 `s < 32` 时 `__syncthreads()` 是多余的**——同一个 warp 内的线程天然同步执行（SIMT 模型）。

**优化写法：warp-level 归约**

```cpp
// 当 s 缩到 32 以内时，切换到 warp shuffle 指令
// __shfl_down_sync 直接在寄存器间交换数据，不走 shared memory
for (int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (threadIdx.x < s)
        sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
}
// 最后 5 轮用 warp shuffle（寄存器级通信，比 shared memory 快 ~5x）
if (threadIdx.x < 32) {
    float val = sdata[threadIdx.x];
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1
```


## License: 未知
https://github.com/JieyangChen7/FT-MAGMA/blob/1d2e2940a0a56024cfe5d2b2898b2e64d7d294ee/magma-2.3.0/sparse/blas/zmdot_shfl.cu

```
Good, I now have all the context. Let me answer the two questions thoroughly and then cover the remaining operators.

## 回答你的两个问题

---

### 问题 1：树形归约太 naive

**你说得对。** 当前 softmax kernel 的归约有两个已知问题：

**问题 a：最后几轮 warp divergence 严重**

```
当前代码：
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s)                             // ← 问题在这里
        sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
}
```

当 `s < 32`（一个 warp 的大小）时，同一个 warp 内只有部分线程在工作，其余线程空转但占着硬件资源。**而且 `s < 32` 时 `__syncthreads()` 是多余的**——同一个 warp 内的线程天然同步执行（SIMT 模型）。

**优化写法：warp-level 归约**

```cpp
// 当 s 缩到 32 以内时，切换到 warp shuffle 指令
// __shfl_down_sync 直接在寄存器间交换数据，不走 shared memory
for (int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (threadIdx.x < s)
        sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
}
// 最后 5 轮用 warp shuffle（寄存器级通信，比 shared memory 快 ~5x）
if (threadIdx.x < 32) {
    float val = sdata[threadIdx.x];
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    if (threadIdx.x == 0
```


## License: 未知
https://github.com/ZhaoOfficial/CG/blob/ddd29cbc96109cba809bf601989b6cb4e7e36eb4/Cuda/src/chapter5/sharemem.cu

```
Good, I now have all the context. Let me answer the two questions thoroughly and then cover the remaining operators.

## 回答你的两个问题

---

### 问题 1：树形归约太 naive

**你说得对。** 当前 softmax kernel 的归约有两个已知问题：

**问题 a：最后几轮 warp divergence 严重**

```
当前代码：
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s)                             // ← 问题在这里
        sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
}
```

当 `s < 32`（一个 warp 的大小）时，同一个 warp 内只有部分线程在工作，其余线程空转但占着硬件资源。**而且 `s < 32` 时 `__syncthreads()` 是多余的**——同一个 warp 内的线程天然同步执行（SIMT 模型）。

**优化写法：warp-level 归约**

```cpp
// 当 s 缩到 32 以内时，切换到 warp shuffle 指令
// __shfl_down_sync 直接在寄存器间交换数据，不走 shared memory
for (int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (threadIdx.x < s)
        sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
}
// 最后 5 轮用 warp shuffle（寄存器级通信，比 shared memory 快 ~5x）
if (threadIdx.x < 32) {
    float val = sdata[threadIdx.x];
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1
```


## License: 未知
https://github.com/JieyangChen7/FT-MAGMA/blob/1d2e2940a0a56024cfe5d2b2898b2e64d7d294ee/magma-2.3.0/sparse/blas/zmdot_shfl.cu

```
Good, I now have all the context. Let me answer the two questions thoroughly and then cover the remaining operators.

## 回答你的两个问题

---

### 问题 1：树形归约太 naive

**你说得对。** 当前 softmax kernel 的归约有两个已知问题：

**问题 a：最后几轮 warp divergence 严重**

```
当前代码：
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s)                             // ← 问题在这里
        sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
}
```

当 `s < 32`（一个 warp 的大小）时，同一个 warp 内只有部分线程在工作，其余线程空转但占着硬件资源。**而且 `s < 32` 时 `__syncthreads()` 是多余的**——同一个 warp 内的线程天然同步执行（SIMT 模型）。

**优化写法：warp-level 归约**

```cpp
// 当 s 缩到 32 以内时，切换到 warp shuffle 指令
// __shfl_down_sync 直接在寄存器间交换数据，不走 shared memory
for (int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (threadIdx.x < s)
        sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
}
// 最后 5 轮用 warp shuffle（寄存器级通信，比 shared memory 快 ~5x）
if (threadIdx.x < 32) {
    float val = sdata[threadIdx.x];
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    if (threadIdx.x == 0
```


## License: 未知
https://github.com/ZhaoOfficial/CG/blob/ddd29cbc96109cba809bf601989b6cb4e7e36eb4/Cuda/src/chapter5/sharemem.cu

```
Good, I now have all the context. Let me answer the two questions thoroughly and then cover the remaining operators.

## 回答你的两个问题

---

### 问题 1：树形归约太 naive

**你说得对。** 当前 softmax kernel 的归约有两个已知问题：

**问题 a：最后几轮 warp divergence 严重**

```
当前代码：
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s)                             // ← 问题在这里
        sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
}
```

当 `s < 32`（一个 warp 的大小）时，同一个 warp 内只有部分线程在工作，其余线程空转但占着硬件资源。**而且 `s < 32` 时 `__syncthreads()` 是多余的**——同一个 warp 内的线程天然同步执行（SIMT 模型）。

**优化写法：warp-level 归约**

```cpp
// 当 s 缩到 32 以内时，切换到 warp shuffle 指令
// __shfl_down_sync 直接在寄存器间交换数据，不走 shared memory
for (int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (threadIdx.x < s)
        sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
}
// 最后 5 轮用 warp shuffle（寄存器级通信，比 shared memory 快 ~5x）
if (threadIdx.x < 32) {
    float val = sdata[threadIdx.x];
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1
```


## License: 未知
https://github.com/JieyangChen7/FT-MAGMA/blob/1d2e2940a0a56024cfe5d2b2898b2e64d7d294ee/magma-2.3.0/sparse/blas/zmdot_shfl.cu

```
Good, I now have all the context. Let me answer the two questions thoroughly and then cover the remaining operators.

## 回答你的两个问题

---

### 问题 1：树形归约太 naive

**你说得对。** 当前 softmax kernel 的归约有两个已知问题：

**问题 a：最后几轮 warp divergence 严重**

```
当前代码：
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s)                             // ← 问题在这里
        sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
}
```

当 `s < 32`（一个 warp 的大小）时，同一个 warp 内只有部分线程在工作，其余线程空转但占着硬件资源。**而且 `s < 32` 时 `__syncthreads()` 是多余的**——同一个 warp 内的线程天然同步执行（SIMT 模型）。

**优化写法：warp-level 归约**

```cpp
// 当 s 缩到 32 以内时，切换到 warp shuffle 指令
// __shfl_down_sync 直接在寄存器间交换数据，不走 shared memory
for (int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (threadIdx.x < s)
        sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
}
// 最后 5 轮用 warp shuffle（寄存器级通信，比 shared memory 快 ~5x）
if (threadIdx.x < 32) {
    float val = sdata[threadIdx.x];
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    if (threadIdx.x == 0
```


## License: 未知
https://github.com/ZhaoOfficial/CG/blob/ddd29cbc96109cba809bf601989b6cb4e7e36eb4/Cuda/src/chapter5/sharemem.cu

```
Good, I now have all the context. Let me answer the two questions thoroughly and then cover the remaining operators.

## 回答你的两个问题

---

### 问题 1：树形归约太 naive

**你说得对。** 当前 softmax kernel 的归约有两个已知问题：

**问题 a：最后几轮 warp divergence 严重**

```
当前代码：
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s)                             // ← 问题在这里
        sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
}
```

当 `s < 32`（一个 warp 的大小）时，同一个 warp 内只有部分线程在工作，其余线程空转但占着硬件资源。**而且 `s < 32` 时 `__syncthreads()` 是多余的**——同一个 warp 内的线程天然同步执行（SIMT 模型）。

**优化写法：warp-level 归约**

```cpp
// 当 s 缩到 32 以内时，切换到 warp shuffle 指令
// __shfl_down_sync 直接在寄存器间交换数据，不走 shared memory
for (int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (threadIdx.x < s)
        sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
}
// 最后 5 轮用 warp shuffle（寄存器级通信，比 shared memory 快 ~5x）
if (threadIdx.x < 32) {
    float val = sdata[threadIdx.x];
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1
```


## License: 未知
https://github.com/JieyangChen7/FT-MAGMA/blob/1d2e2940a0a56024cfe5d2b2898b2e64d7d294ee/magma-2.3.0/sparse/blas/zmdot_shfl.cu

```
Good, I now have all the context. Let me answer the two questions thoroughly and then cover the remaining operators.

## 回答你的两个问题

---

### 问题 1：树形归约太 naive

**你说得对。** 当前 softmax kernel 的归约有两个已知问题：

**问题 a：最后几轮 warp divergence 严重**

```
当前代码：
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s)                             // ← 问题在这里
        sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
}
```

当 `s < 32`（一个 warp 的大小）时，同一个 warp 内只有部分线程在工作，其余线程空转但占着硬件资源。**而且 `s < 32` 时 `__syncthreads()` 是多余的**——同一个 warp 内的线程天然同步执行（SIMT 模型）。

**优化写法：warp-level 归约**

```cpp
// 当 s 缩到 32 以内时，切换到 warp shuffle 指令
// __shfl_down_sync 直接在寄存器间交换数据，不走 shared memory
for (int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (threadIdx.x < s)
        sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
}
// 最后 5 轮用 warp shuffle（寄存器级通信，比 shared memory 快 ~5x）
if (threadIdx.x < 32) {
    float val = sdata[threadIdx.x];
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    if (threadIdx.x == 0
```


## License: 未知
https://github.com/ZhaoOfficial/CG/blob/ddd29cbc96109cba809bf601989b6cb4e7e36eb4/Cuda/src/chapter5/sharemem.cu

```
Good, I now have all the context. Let me answer the two questions thoroughly and then cover the remaining operators.

## 回答你的两个问题

---

### 问题 1：树形归约太 naive

**你说得对。** 当前 softmax kernel 的归约有两个已知问题：

**问题 a：最后几轮 warp divergence 严重**

```
当前代码：
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s)                             // ← 问题在这里
        sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
}
```

当 `s < 32`（一个 warp 的大小）时，同一个 warp 内只有部分线程在工作，其余线程空转但占着硬件资源。**而且 `s < 32` 时 `__syncthreads()` 是多余的**——同一个 warp 内的线程天然同步执行（SIMT 模型）。

**优化写法：warp-level 归约**

```cpp
// 当 s 缩到 32 以内时，切换到 warp shuffle 指令
// __shfl_down_sync 直接在寄存器间交换数据，不走 shared memory
for (int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (threadIdx.x < s)
        sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
}
// 最后 5 轮用 warp shuffle（寄存器级通信，比 shared memory 快 ~5x）
if (threadIdx.x < 32) {
    float val = sdata[threadIdx.x];
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1
```


## License: 未知
https://github.com/JieyangChen7/FT-MAGMA/blob/1d2e2940a0a56024cfe5d2b2898b2e64d7d294ee/magma-2.3.0/sparse/blas/zmdot_shfl.cu

```
Good, I now have all the context. Let me answer the two questions thoroughly and then cover the remaining operators.

## 回答你的两个问题

---

### 问题 1：树形归约太 naive

**你说得对。** 当前 softmax kernel 的归约有两个已知问题：

**问题 a：最后几轮 warp divergence 严重**

```
当前代码：
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s)                             // ← 问题在这里
        sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
}
```

当 `s < 32`（一个 warp 的大小）时，同一个 warp 内只有部分线程在工作，其余线程空转但占着硬件资源。**而且 `s < 32` 时 `__syncthreads()` 是多余的**——同一个 warp 内的线程天然同步执行（SIMT 模型）。

**优化写法：warp-level 归约**

```cpp
// 当 s 缩到 32 以内时，切换到 warp shuffle 指令
// __shfl_down_sync 直接在寄存器间交换数据，不走 shared memory
for (int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (threadIdx.x < s)
        sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
}
// 最后 5 轮用 warp shuffle（寄存器级通信，比 shared memory 快 ~5x）
if (threadIdx.x < 32) {
    float val = sdata[threadIdx.x];
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    if (threadIdx.x == 0
```


## License: 未知
https://github.com/ZhaoOfficial/CG/blob/ddd29cbc96109cba809bf601989b6cb4e7e36eb4/Cuda/src/chapter5/sharemem.cu

```
Good, I now have all the context. Let me answer the two questions thoroughly and then cover the remaining operators.

## 回答你的两个问题

---

### 问题 1：树形归约太 naive

**你说得对。** 当前 softmax kernel 的归约有两个已知问题：

**问题 a：最后几轮 warp divergence 严重**

```
当前代码：
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s)                             // ← 问题在这里
        sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
}
```

当 `s < 32`（一个 warp 的大小）时，同一个 warp 内只有部分线程在工作，其余线程空转但占着硬件资源。**而且 `s < 32` 时 `__syncthreads()` 是多余的**——同一个 warp 内的线程天然同步执行（SIMT 模型）。

**优化写法：warp-level 归约**

```cpp
// 当 s 缩到 32 以内时，切换到 warp shuffle 指令
// __shfl_down_sync 直接在寄存器间交换数据，不走 shared memory
for (int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (threadIdx.x < s)
        sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
}
// 最后 5 轮用 warp shuffle（寄存器级通信，比 shared memory 快 ~5x）
if (threadIdx.x < 32) {
    float val = sdata[threadIdx.x];
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1
```


## License: 未知
https://github.com/JieyangChen7/FT-MAGMA/blob/1d2e2940a0a56024cfe5d2b2898b2e64d7d294ee/magma-2.3.0/sparse/blas/zmdot_shfl.cu

```
Good, I now have all the context. Let me answer the two questions thoroughly and then cover the remaining operators.

## 回答你的两个问题

---

### 问题 1：树形归约太 naive

**你说得对。** 当前 softmax kernel 的归约有两个已知问题：

**问题 a：最后几轮 warp divergence 严重**

```
当前代码：
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s)                             // ← 问题在这里
        sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
}
```

当 `s < 32`（一个 warp 的大小）时，同一个 warp 内只有部分线程在工作，其余线程空转但占着硬件资源。**而且 `s < 32` 时 `__syncthreads()` 是多余的**——同一个 warp 内的线程天然同步执行（SIMT 模型）。

**优化写法：warp-level 归约**

```cpp
// 当 s 缩到 32 以内时，切换到 warp shuffle 指令
// __shfl_down_sync 直接在寄存器间交换数据，不走 shared memory
for (int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (threadIdx.x < s)
        sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
}
// 最后 5 轮用 warp shuffle（寄存器级通信，比 shared memory 快 ~5x）
if (threadIdx.x < 32) {
    float val = sdata[threadIdx.x];
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    if (threadIdx.x == 0
```


## License: 未知
https://github.com/ZhaoOfficial/CG/blob/ddd29cbc96109cba809bf601989b6cb4e7e36eb4/Cuda/src/chapter5/sharemem.cu

```
Good, I now have all the context. Let me answer the two questions thoroughly and then cover the remaining operators.

## 回答你的两个问题

---

### 问题 1：树形归约太 naive

**你说得对。** 当前 softmax kernel 的归约有两个已知问题：

**问题 a：最后几轮 warp divergence 严重**

```
当前代码：
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s)                             // ← 问题在这里
        sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
}
```

当 `s < 32`（一个 warp 的大小）时，同一个 warp 内只有部分线程在工作，其余线程空转但占着硬件资源。**而且 `s < 32` 时 `__syncthreads()` 是多余的**——同一个 warp 内的线程天然同步执行（SIMT 模型）。

**优化写法：warp-level 归约**

```cpp
// 当 s 缩到 32 以内时，切换到 warp shuffle 指令
// __shfl_down_sync 直接在寄存器间交换数据，不走 shared memory
for (int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (threadIdx.x < s)
        sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
}
// 最后 5 轮用 warp shuffle（寄存器级通信，比 shared memory 快 ~5x）
if (threadIdx.x < 32) {
    float val = sdata[threadIdx.x];
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    if
```


## License: 未知
https://github.com/JieyangChen7/FT-MAGMA/blob/1d2e2940a0a56024cfe5d2b2898b2e64d7d294ee/magma-2.3.0/sparse/blas/zmdot_shfl.cu

```
Good, I now have all the context. Let me answer the two questions thoroughly and then cover the remaining operators.

## 回答你的两个问题

---

### 问题 1：树形归约太 naive

**你说得对。** 当前 softmax kernel 的归约有两个已知问题：

**问题 a：最后几轮 warp divergence 严重**

```
当前代码：
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s)                             // ← 问题在这里
        sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
}
```

当 `s < 32`（一个 warp 的大小）时，同一个 warp 内只有部分线程在工作，其余线程空转但占着硬件资源。**而且 `s < 32` 时 `__syncthreads()` 是多余的**——同一个 warp 内的线程天然同步执行（SIMT 模型）。

**优化写法：warp-level 归约**

```cpp
// 当 s 缩到 32 以内时，切换到 warp shuffle 指令
// __shfl_down_sync 直接在寄存器间交换数据，不走 shared memory
for (int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (threadIdx.x < s)
        sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
}
// 最后 5 轮用 warp shuffle（寄存器级通信，比 shared memory 快 ~5x）
if (threadIdx.x < 32) {
    float val = sdata[threadIdx.x];
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    if (threadIdx.x == 0
```


## License: 未知
https://github.com/ZhaoOfficial/CG/blob/ddd29cbc96109cba809bf601989b6cb4e7e36eb4/Cuda/src/chapter5/sharemem.cu

```
Good, I now have all the context. Let me answer the two questions thoroughly and then cover the remaining operators.

## 回答你的两个问题

---

### 问题 1：树形归约太 naive

**你说得对。** 当前 softmax kernel 的归约有两个已知问题：

**问题 a：最后几轮 warp divergence 严重**

```
当前代码：
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s)                             // ← 问题在这里
        sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
}
```

当 `s < 32`（一个 warp 的大小）时，同一个 warp 内只有部分线程在工作，其余线程空转但占着硬件资源。**而且 `s < 32` 时 `__syncthreads()` 是多余的**——同一个 warp 内的线程天然同步执行（SIMT 模型）。

**优化写法：warp-level 归约**

```cpp
// 当 s 缩到 32 以内时，切换到 warp shuffle 指令
// __shfl_down_sync 直接在寄存器间交换数据，不走 shared memory
for (int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (threadIdx.x < s)
        sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
}
// 最后 5 轮用 warp shuffle（寄存器级通信，比 shared memory 快 ~5x）
if (threadIdx.x < 32) {
    float val = sdata[threadIdx.x];
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    if
```


## License: 未知
https://github.com/JieyangChen7/FT-MAGMA/blob/1d2e2940a0a56024cfe5d2b2898b2e64d7d294ee/magma-2.3.0/sparse/blas/zmdot_shfl.cu

```
Good, I now have all the context. Let me answer the two questions thoroughly and then cover the remaining operators.

## 回答你的两个问题

---

### 问题 1：树形归约太 naive

**你说得对。** 当前 softmax kernel 的归约有两个已知问题：

**问题 a：最后几轮 warp divergence 严重**

```
当前代码：
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s)                             // ← 问题在这里
        sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
}
```

当 `s < 32`（一个 warp 的大小）时，同一个 warp 内只有部分线程在工作，其余线程空转但占着硬件资源。**而且 `s < 32` 时 `__syncthreads()` 是多余的**——同一个 warp 内的线程天然同步执行（SIMT 模型）。

**优化写法：warp-level 归约**

```cpp
// 当 s 缩到 32 以内时，切换到 warp shuffle 指令
// __shfl_down_sync 直接在寄存器间交换数据，不走 shared memory
for (int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (threadIdx.x < s)
        sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
}
// 最后 5 轮用 warp shuffle（寄存器级通信，比 shared memory 快 ~5x）
if (threadIdx.x < 32) {
    float val = sdata[threadIdx.x];
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    if (threadIdx.x == 0
```

