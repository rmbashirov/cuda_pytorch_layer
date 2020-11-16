#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#define BLOCK_SIZE 512
#define BLOCK_SIZE_2D_X 32
#define BLOCK_SIZE_2D_Y 16
#define BLOCK_SIZE_3D_X 32
#define BLOCK_SIZE_3D_Y 8
#define BLOCK_SIZE_3D_Z 4

// 2d tensor axis:
// 0: yi
// 1: xi

// 3d tensor axis:
// 0: zi
// 1: yi
// 2: xi

// kernels

template <typename scalar_t>
__global__ void mult_forward_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t,2> a,
    const torch::PackedTensorAccessor32<scalar_t,2> b,
    torch::PackedTensorAccessor32<scalar_t,2> result
) {
    const int xi = threadIdx.x + blockIdx.x * blockDim.x;
    const int yi = threadIdx.y + blockIdx.y * blockDim.y;
    if (xi >= result.size(1) || yi >= result.size(0)) {
        return;
    }
    result[yi][xi] = a[yi][xi] * b[yi][xi];
}

template <typename scalar_t>
__global__ void mult_backward_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t,2> dresult,
    const torch::PackedTensorAccessor32<scalar_t,2> a,
    const torch::PackedTensorAccessor32<scalar_t,2> b,
    torch::PackedTensorAccessor32<scalar_t,2> da,
    torch::PackedTensorAccessor32<scalar_t,2> db
) {
    const int xi = threadIdx.x + blockIdx.x * blockDim.x;
    const int yi = threadIdx.y + blockIdx.y * blockDim.y;
    if (xi >= a.size(1) || yi >= a.size(0)) {
        return;
    }

    da[yi][xi] = dresult[yi][xi] * b[yi][xi];
    db[yi][xi] = dresult[yi][xi] * a[yi][xi];
}

// cpp defined functions

torch::Tensor mult_forward_cuda(
    const torch::Tensor& a,
    const torch::Tensor& b
) {
    const int gpuid = a.device().index();
    AT_CUDA_CHECK(cudaSetDevice(gpuid));
    auto options = torch::dtype(a.scalar_type()).device(torch::kCUDA, gpuid);
    const int H = a.size(0);
    const int W = a.size(1);
    const dim3 dimBlock1(BLOCK_SIZE_2D_X, BLOCK_SIZE_2D_Y);
    const dim3 dimGrid1((W - 1) / dimBlock1.x + 1, (H - 1) / dimBlock1.y + 1);
    auto result = torch::zeros({H, W}, options);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a.scalar_type(), "mult_forward_cuda_kernel", [&] {
        mult_forward_cuda_kernel<scalar_t><<<dimGrid1, dimBlock1>>>(
            a.packed_accessor32<scalar_t,2>(),
            b.packed_accessor32<scalar_t,2>(),
            result.packed_accessor32<scalar_t,2>()
        );
    });
    AT_CUDA_CHECK(cudaGetLastError());

    return result;
}

std::vector<torch::Tensor> mult_backward_cuda(
    const torch::Tensor& dresult,
    const torch::Tensor& a,
    const torch::Tensor& b
) {
    const int gpuid = a.device().index();
    AT_CUDA_CHECK(cudaSetDevice(gpuid));
    auto options = torch::dtype(a.scalar_type()).device(torch::kCUDA, gpuid);
    const int H = a.size(0);
    const int W = a.size(1);
    const dim3 dimBlock1(BLOCK_SIZE_2D_X, BLOCK_SIZE_2D_Y);
    const dim3 dimGrid1((W - 1) / dimBlock1.x + 1, (H - 1) / dimBlock1.y + 1);
    auto da = torch::zeros({H, W}, options);
    auto db = torch::zeros({H, W}, options);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a.scalar_type(), "mult_backward_cuda_kernel", [&] {
        mult_backward_cuda_kernel<scalar_t><<<dimGrid1, dimBlock1>>>(
            dresult.packed_accessor32<scalar_t,2>(),
            a.packed_accessor32<scalar_t,2>(),
            b.packed_accessor32<scalar_t,2>(),
            da.packed_accessor32<scalar_t,2>(),
            db.packed_accessor32<scalar_t,2>()
        );
    });
    AT_CUDA_CHECK(cudaGetLastError());

    return {da, db};
}
