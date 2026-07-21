#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "world_cuda_vae_ops.cuh"

#include <climits>
#include <cstdint>

namespace py = pybind11;

namespace {

void check_tensor(
        const torch::Tensor &tensor,
        at::ScalarType dtype,
        const char *name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has the wrong dtype");
}

void check_same_device(
        const torch::Tensor &a,
        const torch::Tensor &b,
        const char *a_name,
        const char *b_name) {
    TORCH_CHECK(
        a.device() == b.device(), a_name, " and ", b_name, " must be on the same device");
}

void check_conv_tensors(
        const torch::Tensor &x,
        const torch::Tensor &weight,
        const torch::Tensor &bias,
        at::ScalarType dtype) {
    check_tensor(x, dtype, "x");
    check_tensor(weight, dtype, "weight");
    check_tensor(bias, dtype, "bias");
    check_same_device(x, weight, "x", "weight");
    check_same_device(x, bias, "x", "bias");
}

void check_status(int status, const char *op) {
    TORCH_CHECK(status == 0, op, " failed");
}

void synchronize_current_stream() {
    cudaError_t status = cudaStreamSynchronize(at::cuda::getCurrentCUDAStream());
    TORCH_CHECK(
        status == cudaSuccess,
        "failed to synchronize the PyTorch CUDA stream: ",
        cudaGetErrorString(status));
}

void synchronize_production_ops() {
    cudaError_t status = cudaDeviceSynchronize();
    TORCH_CHECK(
        status == cudaSuccess,
        "failed to synchronize production VAE operators: ",
        cudaGetErrorString(status));
}

__half *half_ptr(torch::Tensor &tensor) {
    return reinterpret_cast<__half *>(tensor.data_ptr<at::Half>());
}

const __half *half_ptr(const torch::Tensor &tensor) {
    return reinterpret_cast<const __half *>(tensor.data_ptr<at::Half>());
}

WmCudaVaeConvDesc make_f32_desc(
        torch::Tensor &weight,
        torch::Tensor &bias,
        int in_channels,
        int out_channels,
        int kernel) {
    WmCudaVaeConvDesc desc{};
    desc.weight = weight.data_ptr<float>();
    desc.bias = bias.data_ptr<float>();
    desc.out_c = out_channels;
    desc.in_c = in_channels;
    desc.kernel = kernel;
    desc.has_bias = 1;
    return desc;
}

WmCudaVaeConvDesc make_f16_desc(
        torch::Tensor &weight,
        torch::Tensor &bias,
        int in_channels,
        int out_channels,
        int kernel) {
    WmCudaVaeConvDesc desc{};
    desc.weight_krsc_h = half_ptr(weight);
    desc.bias_h = half_ptr(bias);
    desc.out_c = out_channels;
    desc.in_c = in_channels;
    desc.kernel = kernel;
    desc.has_bias = 1;
    return desc;
}

void check_nchw_conv_shape(
        const torch::Tensor &x,
        const torch::Tensor &weight,
        const torch::Tensor &bias,
        int expected_kernel) {
    TORCH_CHECK(x.dim() == 4, "x must be NCHW [N,C,H,W]");
    TORCH_CHECK(weight.dim() == 4, "weight must be KCRS [Cout,Cin,K,K]");
    TORCH_CHECK(weight.size(1) == x.size(1), "input channel mismatch");
    TORCH_CHECK(
        weight.size(2) == expected_kernel && weight.size(3) == expected_kernel,
        "unexpected convolution kernel size");
    TORCH_CHECK(bias.dim() == 1 && bias.numel() == weight.size(0),
                "bias length must equal output channels");
}

void check_nhwc_conv_shape(
        const torch::Tensor &x,
        const torch::Tensor &weight,
        const torch::Tensor &bias) {
    TORCH_CHECK(x.dim() == 4, "x must be NHWC [N,H,W,C]");
    TORCH_CHECK(weight.dim() == 4, "weight must be KRSC [Cout,K,K,Cin]");
    TORCH_CHECK(weight.size(3) == x.size(3), "input channel mismatch");
    TORCH_CHECK(weight.size(1) == weight.size(2), "kernel must be square");
    TORCH_CHECK(weight.size(1) == 1 || weight.size(1) == 3,
                "only 1x1 and 3x3 kernels are supported");
    TORCH_CHECK(bias.dim() == 1 && bias.numel() == weight.size(0),
                "bias length must equal output channels");
}

torch::Tensor conv_direct_nchw_f32(
        torch::Tensor x,
        torch::Tensor weight,
        torch::Tensor bias) {
    check_conv_tensors(x, weight, bias, at::ScalarType::Float);
    TORCH_CHECK(weight.dim() == 4 && weight.size(2) == weight.size(3),
                "weight must have a square KCRS kernel");
    int kernel = static_cast<int>(weight.size(2));
    TORCH_CHECK(kernel == 1 || kernel == 3, "only 1x1 and 3x3 kernels are supported");
    check_nchw_conv_shape(x, weight, bias, kernel);
    c10::cuda::CUDAGuard device_guard(x.device());

    int batch = static_cast<int>(x.size(0));
    int in_channels = static_cast<int>(x.size(1));
    int out_channels = static_cast<int>(weight.size(0));
    int height = static_cast<int>(x.size(2));
    int width = static_cast<int>(x.size(3));
    auto out = torch::empty({batch, out_channels, height, width}, x.options());
    WmCudaVaeConvDesc desc =
        make_f32_desc(weight, bias, in_channels, out_channels, kernel);
    synchronize_current_stream();
    check_status(
        wm_cuda_vae_conv_direct_nchw_f32(
            x.data_ptr<float>(), out.data_ptr<float>(), &desc, batch, height, width),
        "wm_cuda_vae_conv_direct_nchw_f32");
    synchronize_production_ops();
    return out;
}

torch::Tensor conv_stride2_nchw_f32(
        torch::Tensor x,
        torch::Tensor weight,
        torch::Tensor bias) {
    check_conv_tensors(x, weight, bias, at::ScalarType::Float);
    check_nchw_conv_shape(x, weight, bias, 3);
    c10::cuda::CUDAGuard device_guard(x.device());

    int batch = static_cast<int>(x.size(0));
    int in_channels = static_cast<int>(x.size(1));
    int out_channels = static_cast<int>(weight.size(0));
    int height = static_cast<int>(x.size(2));
    int width = static_cast<int>(x.size(3));
    auto out = torch::empty(
        {batch, out_channels, (height + 1) / 2, (width + 1) / 2}, x.options());
    WmCudaVaeConvDesc desc = make_f32_desc(weight, bias, in_channels, out_channels, 3);
    synchronize_current_stream();
    check_status(
        wm_cuda_vae_conv_stride2_nchw_f32(
            x.data_ptr<float>(), out.data_ptr<float>(), &desc, batch, height, width),
        "wm_cuda_vae_conv_stride2_nchw_f32");
    synchronize_production_ops();
    return out;
}

torch::Tensor conv1x1_gemm_nchw_f32(
        torch::Tensor x,
        torch::Tensor weight,
        torch::Tensor bias) {
    check_conv_tensors(x, weight, bias, at::ScalarType::Float);
    check_nchw_conv_shape(x, weight, bias, 1);
    c10::cuda::CUDAGuard device_guard(x.device());

    int batch = static_cast<int>(x.size(0));
    int in_channels = static_cast<int>(x.size(1));
    int out_channels = static_cast<int>(weight.size(0));
    int height = static_cast<int>(x.size(2));
    int width = static_cast<int>(x.size(3));
    int spatial = height * width;
    auto out = torch::empty({batch, out_channels, height, width}, x.options());
    synchronize_current_stream();
    for (int frame = 0; frame < batch; ++frame) {
        const float *frame_in = x.data_ptr<float>() +
            static_cast<int64_t>(frame) * in_channels * spatial;
        float *frame_out = out.data_ptr<float>() +
            static_cast<int64_t>(frame) * out_channels * spatial;
        check_status(
            wm_cuda_vae_conv_gemm_f32(
                weight.data_ptr<float>(), frame_in, frame_out,
                out_channels, spatial, in_channels, spatial),
            "wm_cuda_vae_conv_gemm_f32(1x1)");
    }
    check_status(
        wm_cuda_vae_add_bias_nchw_f32(
            out.data_ptr<float>(), bias.data_ptr<float>(),
            batch, out_channels, height, width),
        "wm_cuda_vae_add_bias_nchw_f32(1x1)");
    synchronize_production_ops();
    return out;
}

torch::Tensor conv3x3_gemm_nchw_f32(
        torch::Tensor x,
        torch::Tensor weight,
        torch::Tensor bias) {
    check_conv_tensors(x, weight, bias, at::ScalarType::Float);
    check_nchw_conv_shape(x, weight, bias, 3);
    c10::cuda::CUDAGuard device_guard(x.device());

    int batch = static_cast<int>(x.size(0));
    int in_channels = static_cast<int>(x.size(1));
    int out_channels = static_cast<int>(weight.size(0));
    int height = static_cast<int>(x.size(2));
    int width = static_cast<int>(x.size(3));
    int spatial = height * width;
    int reduction = in_channels * 9;
    auto cols = torch::empty({reduction, spatial}, x.options());
    auto out = torch::empty({batch, out_channels, height, width}, x.options());
    synchronize_current_stream();
    for (int frame = 0; frame < batch; ++frame) {
        check_status(
            wm_cuda_vae_im2col3x3_nchw_tile_f32(
                x.data_ptr<float>(), cols.data_ptr<float>(),
                in_channels, height, width, frame, 0, spatial),
            "wm_cuda_vae_im2col3x3_nchw_tile_f32");
        float *frame_out = out.data_ptr<float>() +
            static_cast<int64_t>(frame) * out_channels * spatial;
        check_status(
            wm_cuda_vae_conv_gemm_f32(
                weight.data_ptr<float>(), cols.data_ptr<float>(), frame_out,
                out_channels, spatial, reduction, spatial),
            "wm_cuda_vae_conv_gemm_f32(3x3)");
    }
    check_status(
        wm_cuda_vae_add_bias_nchw_f32(
            out.data_ptr<float>(), bias.data_ptr<float>(),
            batch, out_channels, height, width),
        "wm_cuda_vae_add_bias_nchw_f32(3x3)");
    synchronize_production_ops();
    return out;
}

torch::Tensor conv3x3_gemm_batched_nchw_f32(
        torch::Tensor x,
        torch::Tensor weight,
        torch::Tensor bias,
        int64_t tile_columns_arg) {
    check_conv_tensors(x, weight, bias, at::ScalarType::Float);
    check_nchw_conv_shape(x, weight, bias, 3);
    TORCH_CHECK(tile_columns_arg > 0 && tile_columns_arg <= INT_MAX,
                "tile_columns is out of range");
    c10::cuda::CUDAGuard device_guard(x.device());

    int batch = static_cast<int>(x.size(0));
    int in_channels = static_cast<int>(x.size(1));
    int out_channels = static_cast<int>(weight.size(0));
    int height = static_cast<int>(x.size(2));
    int width = static_cast<int>(x.size(3));
    int spatial = height * width;
    int total_columns = batch * spatial;
    int tile_columns_max = static_cast<int>(tile_columns_arg);
    int reduction = in_channels * 9;
    auto cols = torch::empty({reduction, tile_columns_max}, x.options());
    auto tile = torch::empty({out_channels, tile_columns_max}, x.options());
    auto out = torch::empty({batch, out_channels, height, width}, x.options());
    synchronize_current_stream();
    for (int tile_start = 0; tile_start < total_columns; tile_start += tile_columns_max) {
        int tile_columns = total_columns - tile_start;
        if (tile_columns > tile_columns_max) tile_columns = tile_columns_max;
        check_status(
            wm_cuda_vae_im2col3x3_nchw_batch_tile_f32(
                x.data_ptr<float>(), cols.data_ptr<float>(),
                batch, in_channels, height, width, tile_start, tile_columns),
            "wm_cuda_vae_im2col3x3_nchw_batch_tile_f32");
        check_status(
            wm_cuda_vae_conv_gemm_f32(
                weight.data_ptr<float>(), cols.data_ptr<float>(), tile.data_ptr<float>(),
                out_channels, tile_columns, reduction, tile_columns),
            "wm_cuda_vae_conv_gemm_f32(batched 3x3)");
        check_status(
            wm_cuda_vae_scatter_conv_tile_nchw_f32(
                tile.data_ptr<float>(), out.data_ptr<float>(),
                batch, out_channels, height, width, tile_start, tile_columns),
            "wm_cuda_vae_scatter_conv_tile_nchw_f32");
    }
    check_status(
        wm_cuda_vae_add_bias_nchw_f32(
            out.data_ptr<float>(), bias.data_ptr<float>(),
            batch, out_channels, height, width),
        "wm_cuda_vae_add_bias_nchw_f32(batched 3x3)");
    synchronize_production_ops();
    return out;
}

torch::Tensor conv_nhwc_f16_impl(
        torch::Tensor x,
        torch::Tensor weight,
        torch::Tensor bias) {
    check_conv_tensors(x, weight, bias, at::ScalarType::Half);
    check_nhwc_conv_shape(x, weight, bias);
    c10::cuda::CUDAGuard device_guard(x.device());

    int batch = static_cast<int>(x.size(0));
    int height = static_cast<int>(x.size(1));
    int width = static_cast<int>(x.size(2));
    int in_channels = static_cast<int>(x.size(3));
    int out_channels = static_cast<int>(weight.size(0));
    int kernel = static_cast<int>(weight.size(1));
    auto out = torch::empty({batch, height, width, out_channels}, x.options());
    WmCudaVaeConvDesc desc =
        make_f16_desc(weight, bias, in_channels, out_channels, kernel);
    synchronize_current_stream();
    check_status(
        wm_cuda_vae_conv_nhwc_f16(
            half_ptr(x), half_ptr(out), &desc, batch, height, width),
        "wm_cuda_vae_conv_nhwc_f16");
    check_status(
        wm_cuda_vae_add_bias_nhwc_f16(
            half_ptr(out), half_ptr(bias), out.numel(), out_channels),
        "wm_cuda_vae_add_bias_nhwc_f16");
    synchronize_production_ops();
    return out;
}

torch::Tensor concat_past_nchw_f32(torch::Tensor x) {
    check_tensor(x, at::ScalarType::Float, "x");
    TORCH_CHECK(x.dim() == 4, "x must be NCHW [N,C,H,W]");
    c10::cuda::CUDAGuard device_guard(x.device());
    int batch = static_cast<int>(x.size(0));
    int channels = static_cast<int>(x.size(1));
    int height = static_cast<int>(x.size(2));
    int width = static_cast<int>(x.size(3));
    auto out = torch::empty({batch, channels * 2, height, width}, x.options());
    synchronize_current_stream();
    check_status(
        wm_cuda_vae_concat_past_nchw_f32(
            x.data_ptr<float>(), out.data_ptr<float>(),
            batch, channels, height, width),
        "wm_cuda_vae_concat_past_nchw_f32");
    synchronize_production_ops();
    return out;
}

torch::Tensor upsample2_nchw_f32(torch::Tensor x) {
    check_tensor(x, at::ScalarType::Float, "x");
    TORCH_CHECK(x.dim() == 4, "x must be NCHW [N,C,H,W]");
    c10::cuda::CUDAGuard device_guard(x.device());
    int batch = static_cast<int>(x.size(0));
    int channels = static_cast<int>(x.size(1));
    int height = static_cast<int>(x.size(2));
    int width = static_cast<int>(x.size(3));
    auto out = torch::empty({batch, channels, height * 2, width * 2}, x.options());
    synchronize_current_stream();
    check_status(
        wm_cuda_vae_upsample2_nchw_f32(
            x.data_ptr<float>(), out.data_ptr<float>(),
            batch, channels, height, width),
        "wm_cuda_vae_upsample2_nchw_f32");
    synchronize_production_ops();
    return out;
}

torch::Tensor tgrow_reshape_nchw_f32(torch::Tensor x, int64_t stride_arg) {
    check_tensor(x, at::ScalarType::Float, "x");
    TORCH_CHECK(x.dim() == 4, "x must be NCHW [N,C*stride,H,W]");
    TORCH_CHECK(stride_arg > 0 && stride_arg <= INT_MAX, "stride is out of range");
    TORCH_CHECK(x.size(1) % stride_arg == 0, "channels must be divisible by stride");
    c10::cuda::CUDAGuard device_guard(x.device());
    int batch = static_cast<int>(x.size(0));
    int stride = static_cast<int>(stride_arg);
    int channels = static_cast<int>(x.size(1) / stride);
    int height = static_cast<int>(x.size(2));
    int width = static_cast<int>(x.size(3));
    auto out = torch::empty({batch * stride, channels, height, width}, x.options());
    synchronize_current_stream();
    check_status(
        wm_cuda_vae_tgrow_reshape_nchw_f32(
            x.data_ptr<float>(), out.data_ptr<float>(),
            batch, channels, height, width, stride),
        "wm_cuda_vae_tgrow_reshape_nchw_f32");
    synchronize_production_ops();
    return out;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_direct_nchw_f32", &conv_direct_nchw_f32);
    m.def("conv_stride2_nchw_f32", &conv_stride2_nchw_f32);
    m.def("conv1x1_gemm_nchw_f32", &conv1x1_gemm_nchw_f32);
    m.def("conv3x3_gemm_nchw_f32", &conv3x3_gemm_nchw_f32);
    m.def(
        "conv3x3_gemm_batched_nchw_f32", &conv3x3_gemm_batched_nchw_f32,
        py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("tile_columns"));
    m.def("conv_nhwc_f16", &conv_nhwc_f16_impl);
    m.def("concat_past_nchw_f32", &concat_past_nchw_f32);
    m.def("upsample2_nchw_f32", &upsample2_nchw_f32);
    m.def(
        "tgrow_reshape_nchw_f32", &tgrow_reshape_nchw_f32,
        py::arg("x"), py::arg("stride"));
}
