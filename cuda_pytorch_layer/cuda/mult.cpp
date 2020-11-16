#include <torch/extension.h>
#include <vector>

// CUDA forward declarations

torch::Tensor mult_forward_cuda(
    const torch::Tensor& a,
    const torch::Tensor& b
);

std::vector<torch::Tensor> mult_backward_cuda(
    const torch::Tensor& dresult,
    const torch::Tensor& a,
    const torch::Tensor& b
);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void check_equal_dtype(const torch::Tensor& a, const torch::Tensor& b) {
    TORCH_CHECK(
        a.dtype() == b.dtype(), 
        "expected equal dtype, got ", a.dtype(), " != ", b.dtype()
    );
}

void check_equal_gpuid(const torch::Tensor& a, const torch::Tensor& b) {
    TORCH_CHECK(
        a.device().index() == b.device().index(), 
        "expected equal gpu id, got ", a.device().index(), " != ", b.device().index()
    );
}

void check_equal_shape(const torch::Tensor& a, const torch::Tensor& b) {
    TORCH_CHECK(
        a.dim() == b.dim(),
        "expected equal gpu id, got ", a.device().index(), " != ", b.device().index()
    );

    for (int i = 0; i < a.dim(); i++) {
        TORCH_CHECK(
            a.size(i) == b.size(i),
            "expected size across dimension ", i, ":", a.size(i), " != ", b.size(i)
        );
    }
}

void check_dim(const torch::Tensor& a, const int dim) {
    TORCH_CHECK(
        a.dim() == dim,
        "expected number of dimensions ", dim, ", got: ", a.dim()
    );
}

torch::Tensor mult_forward(
    const torch::Tensor& a,
    const torch::Tensor& b
) {
    CHECK_INPUT(a);
    CHECK_INPUT(b);

    check_equal_gpuid(a, b);

    check_equal_dtype(a, b);

    check_dim(a, 2);
    check_dim(b, 2);

    check_equal_shape(a, b);

    return mult_forward_cuda(a, b);
}

std::vector<torch::Tensor> mult_backward(
    const torch::Tensor& dresult,
    const torch::Tensor& a,
    const torch::Tensor& b
) {
    CHECK_INPUT(dresult);
    CHECK_INPUT(a);
    CHECK_INPUT(b);

    check_equal_gpuid(dresult, a);
    check_equal_gpuid(a, b);

    check_equal_dtype(dresult, a);
    check_equal_dtype(a, b);

    check_dim(dresult, 2);
    check_dim(a, 2);
    check_dim(b, 2);

    check_equal_shape(dresult, a);
    check_equal_shape(a, b);

    return mult_backward_cuda(dresult, a, b);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mult_forward", &mult_forward, "mult_forward (CUDA)");
    m.def("mult_backward", &mult_backward, "mult_backward (CUDA)");
}