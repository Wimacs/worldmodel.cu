// Experimental CUDA prototypes without a production operator API.
// Production Transformer and VAE parity bindings compile their runtime implementations directly.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cutlass/cutlass.h>
#include <cutlass/conv/conv2d_problem_size.h>
#include <cutlass/conv/device/implicit_gemm_convolution.h>
#include <cutlass/conv/kernel/default_conv2d_fprop.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_batched.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/layout/tensor.h>
#include <cutlass/numeric_types.h>
#include <cutlass/tensor_ref.h>

#if __has_include("kernel_forward.h")
#define WM_HAS_CUTLASS_FMHA 1
#include "kernel_forward.h"
#else
#define WM_HAS_CUTLASS_FMHA 0
#endif

#include <cmath>
#include <cstdint>
#include <vector>

#define WM_CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define WM_CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define WM_CHECK_F32(x) TORCH_CHECK((x).scalar_type() == at::ScalarType::Float, #x " must be float32")
#define WM_CHECK_F16(x) TORCH_CHECK((x).scalar_type() == at::ScalarType::Half, #x " must be float16")

namespace py = pybind11;

static int div_up_i64(int64_t a, int b) {
    return (int)((a + b - 1) / b);
}

static void check_last_cuda_error(const char *name) {
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, name, " launch failed: ", cudaGetErrorString(err));
}

static void check_cutlass_status(cutlass::Status status, const char *name) {
    TORCH_CHECK(status == cutlass::Status::kSuccess, name, " failed: ", cutlassGetStatusString(status));
}

__device__ __forceinline__ float wm_rope_phase(
        int pair_id,
        int64_t x_pos,
        int64_t y_pos,
        int64_t t_pos,
        const float *xy,
        const float *inv_t,
        int width,
        int height,
        int d_xy) {
    if (pair_id < d_xy) {
        float x = (2.0f * (float)x_pos + 1.0f) / (float)width - 1.0f;
        return x * xy[pair_id];
    }
    if (pair_id < 2 * d_xy) {
        float y = (2.0f * (float)y_pos + 1.0f) / (float)height - 1.0f;
        return y * xy[pair_id - d_xy];
    }
    return (float)t_pos * inv_t[pair_id - 2 * d_xy];
}

__global__ void ortho_rope_f32_kernel(
        const float *x,
        float *y,
        const int64_t *x_pos,
        const int64_t *y_pos,
        const int64_t *t_pos,
        const float *xy,
        const float *inv_t,
        int B,
        int H,
        int T,
        int D,
        int width,
        int height) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int half = D / 2;
    int64_t total = (int64_t)B * H * T * half;
    if (i >= total) return;

    int p = i % half;
    int64_t q = i / half;
    int t = q % T;
    q /= T;
    int h = q % H;
    int b = q / H;

    int d_xy = D / 8;
    float phase = wm_rope_phase(p, x_pos[t], y_pos[t], t_pos[t], xy, inv_t, width, height, d_xy);
    float c = cosf(phase);
    float s = sinf(phase);

    int64_t base = (((int64_t)b * H + h) * T + t) * D;
    float x0 = x[base + 2 * p];
    float x1 = x[base + 2 * p + 1];
    y[base + p] = x0 * c - x1 * s;
    y[base + half + p] = x1 * c + x0 * s;
}

__global__ void masked_attention_f32_kernel(
        const float *q,
        const float *k,
        const float *v,
        const bool *written,
        float *out,
        int B,
        int Hq,
        int Hkv,
        int Tq,
        int Tk,
        int D,
        float scale) {
    __shared__ float red[256];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int tq = row % Tq;
    int hq = (row / Tq) % Hq;
    int b = row / (Tq * Hq);
    int group = Hq / Hkv;
    int hk = hq / group;

    const float *qrow = q + (((int64_t)b * Hq + hq) * Tq + tq) * D;
    const float *kbase = k + ((int64_t)b * Hkv + hk) * Tk * D;
    const float *vbase = v + ((int64_t)b * Hkv + hk) * Tk * D;
    float *orow = out + (((int64_t)b * Hq + hq) * Tq + tq) * D;

    float acc = 0.0f;
    float m = -INFINITY;
    float l = 0.0f;

    for (int tk = 0; tk < Tk; ++tk) {
        float partial = 0.0f;
        if (written[tk]) {
            const float *krow = kbase + (int64_t)tk * D;
            for (int d = tid; d < D; d += blockDim.x) {
                partial += qrow[d] * krow[d];
            }
        }

        red[tid] = partial;
        __syncthreads();

        for (int step = blockDim.x >> 1; step > 0; step >>= 1) {
            if (tid < step) {
                red[tid] += red[tid + step];
            }
            __syncthreads();
        }

        if (!written[tk]) {
            continue;
        }

        float score = red[0] * scale;
        float new_m = fmaxf(m, score);
        float alpha = expf(m - new_m);
        float beta = expf(score - new_m);

        if (tid < D) {
            acc = acc * alpha + beta * vbase[(int64_t)tk * D + tid];
        }
        l = l * alpha + beta;
        m = new_m;
        __syncthreads();
    }

    if (tid < D) {
        orow[tid] = l > 0.0f ? acc / l : 0.0f;
    }
}
__global__ void taehv_add_bias_nhwc_f32_kernel(
        float *out,
        const float *bias,
        int64_t total,
        int C) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    out[i] += bias[i % C];
}
torch::Tensor ortho_rope_cuda(
        torch::Tensor x,
        torch::Tensor x_pos,
        torch::Tensor y_pos,
        torch::Tensor t_pos,
        torch::Tensor xy,
        torch::Tensor inv_t,
        int64_t width,
        int64_t height) {
    WM_CHECK_CUDA(x);
    WM_CHECK_CUDA(x_pos);
    WM_CHECK_CUDA(y_pos);
    WM_CHECK_CUDA(t_pos);
    WM_CHECK_CUDA(xy);
    WM_CHECK_CUDA(inv_t);
    WM_CHECK_CONTIGUOUS(x);
    WM_CHECK_CONTIGUOUS(x_pos);
    WM_CHECK_CONTIGUOUS(y_pos);
    WM_CHECK_CONTIGUOUS(t_pos);
    WM_CHECK_CONTIGUOUS(xy);
    WM_CHECK_CONTIGUOUS(inv_t);
    WM_CHECK_F32(x);
    WM_CHECK_F32(xy);
    WM_CHECK_F32(inv_t);
    TORCH_CHECK(x_pos.scalar_type() == at::ScalarType::Long, "x_pos must be int64");
    TORCH_CHECK(y_pos.scalar_type() == at::ScalarType::Long, "y_pos must be int64");
    TORCH_CHECK(t_pos.scalar_type() == at::ScalarType::Long, "t_pos must be int64");
    TORCH_CHECK(x.dim() == 4, "x must be [B,H,T,D]");
    TORCH_CHECK(x.size(3) % 8 == 0, "D must be divisible by 8");
    TORCH_CHECK(x_pos.numel() == x.size(2), "x_pos length must equal T");
    TORCH_CHECK(y_pos.numel() == x.size(2), "y_pos length must equal T");
    TORCH_CHECK(t_pos.numel() == x.size(2), "t_pos length must equal T");
    TORCH_CHECK(xy.numel() == x.size(3) / 8, "xy length must equal D/8");
    TORCH_CHECK(inv_t.numel() == x.size(3) / 4, "inv_t length must equal D/4");

    int B = (int)x.size(0);
    int H = (int)x.size(1);
    int T = (int)x.size(2);
    int D = (int)x.size(3);
    auto y = torch::empty_like(x);
    int64_t work = (int64_t)B * H * T * (D / 2);

    ortho_rope_f32_kernel<<<div_up_i64(work, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        x_pos.data_ptr<int64_t>(),
        y_pos.data_ptr<int64_t>(),
        t_pos.data_ptr<int64_t>(),
        xy.data_ptr<float>(),
        inv_t.data_ptr<float>(),
        B,
        H,
        T,
        D,
        (int)width,
        (int)height);
    check_last_cuda_error("ortho_rope_f32");
    return y;
}

torch::Tensor masked_attention_cuda(
        torch::Tensor q,
        torch::Tensor k,
        torch::Tensor v,
        torch::Tensor written,
        double scale) {
    WM_CHECK_CUDA(q);
    WM_CHECK_CUDA(k);
    WM_CHECK_CUDA(v);
    WM_CHECK_CUDA(written);
    WM_CHECK_CONTIGUOUS(q);
    WM_CHECK_CONTIGUOUS(k);
    WM_CHECK_CONTIGUOUS(v);
    WM_CHECK_CONTIGUOUS(written);
    WM_CHECK_F32(q);
    WM_CHECK_F32(k);
    WM_CHECK_F32(v);
    TORCH_CHECK(written.scalar_type() == at::ScalarType::Bool, "written must be bool");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q, k, v must be 4D");
    TORCH_CHECK(k.sizes() == v.sizes(), "k and v shapes must match");
    TORCH_CHECK(q.size(0) == k.size(0), "B mismatch");
    TORCH_CHECK(q.size(3) == k.size(3), "D mismatch");
    TORCH_CHECK(written.numel() == k.size(2), "written length must equal Tk");

    int B = (int)q.size(0);
    int Hq = (int)q.size(1);
    int Tq = (int)q.size(2);
    int D = (int)q.size(3);
    int Hkv = (int)k.size(1);
    int Tk = (int)k.size(2);
    TORCH_CHECK(Hq % Hkv == 0, "Hq must be divisible by Hkv for GQA");
    TORCH_CHECK(D > 0 && D <= 256, "masked_attention currently supports 1 <= D <= 256");

    auto out = torch::empty_like(q);
    masked_attention_f32_kernel<<<B * Hq * Tq, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        written.data_ptr<bool>(),
        out.data_ptr<float>(),
        B,
        Hq,
        Hkv,
        Tq,
        Tk,
        D,
        (float)scale);
    check_last_cuda_error("masked_attention_f32");
    return out;
}

torch::Tensor cutlass_fmha_bmhk_fp16_cuda(
        torch::Tensor q,
        torch::Tensor k,
        torch::Tensor v,
        double scale) {
#if WM_HAS_CUTLASS_FMHA
    WM_CHECK_CUDA(q);
    WM_CHECK_CUDA(k);
    WM_CHECK_CUDA(v);
    WM_CHECK_CONTIGUOUS(q);
    WM_CHECK_CONTIGUOUS(k);
    WM_CHECK_CONTIGUOUS(v);
    WM_CHECK_F16(q);
    WM_CHECK_F16(k);
    WM_CHECK_F16(v);
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q, k, v must be BMHK tensors");
    TORCH_CHECK(k.sizes() == v.sizes(), "k and v shapes must match");
    TORCH_CHECK(q.size(0) == k.size(0), "batch mismatch");
    TORCH_CHECK(q.size(2) == k.size(2), "head count mismatch");
    TORCH_CHECK(q.size(3) == 64 && k.size(3) == 64, "cutlass_fmha_bmhk_fp16 currently supports D=64");

    int B = (int)q.size(0);
    int M = (int)q.size(1);
    int H = (int)q.size(2);
    int N = (int)k.size(1);
    int D = 64;

    auto out = torch::empty({B, M, H, D}, q.options());

    using Attention = AttentionKernel<
        cutlass::half_t,
        cutlass::arch::Sm80,
        true,
        64,
        64,
        64,
        false,
        false>;

    typename Attention::Params p;
    p.query_ptr = reinterpret_cast<cutlass::half_t *>(q.data_ptr<at::Half>());
    p.key_ptr = reinterpret_cast<cutlass::half_t *>(k.data_ptr<at::Half>());
    p.value_ptr = reinterpret_cast<cutlass::half_t *>(v.data_ptr<at::Half>());
    p.output_ptr = reinterpret_cast<cutlass::half_t *>(out.data_ptr<at::Half>());
    p.output_accum_ptr = nullptr;
    p.logsumexp_ptr = nullptr;
    p.scale = (float)scale;
    p.num_heads = H;
    p.num_batches = B;
    p.head_dim = D;
    p.head_dim_value = D;
    p.num_queries = M;
    p.num_keys = N;
    p.custom_mask_type = Attention::NoCustomMask;
    p.q_strideH = D;
    p.k_strideH = D;
    p.v_strideH = D;
    p.q_strideM = H * D;
    p.k_strideM = H * D;
    p.v_strideM = H * D;
    p.q_strideB = (int64_t)M * H * D;
    p.k_strideB = (int64_t)N * H * D;
    p.v_strideB = (int64_t)N * H * D;
    p.o_strideM = H * D;

    TORCH_CHECK(Attention::check_supported(p), "CUTLASS FMHA does not support this BMHK shape");
    constexpr auto kernel_fn = attention_kernel_batched_impl<Attention>;
    int smem_bytes = sizeof(typename Attention::SharedStorage);
    if (smem_bytes > 0xc000) {
        cudaError_t attr_err = cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        TORCH_CHECK(attr_err == cudaSuccess, "cudaFuncSetAttribute failed: ", cudaGetErrorString(attr_err));
    }
    kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes, at::cuda::getCurrentCUDAStream()>>>(p);
    check_last_cuda_error("cutlass_fmha_bmhk_fp16");
    return out;
#else
    (void)q;
    (void)k;
    (void)v;
    (void)scale;
    TORCH_CHECK(false, "CUTLASS FMHA example headers were not available when building this extension");
#endif
}

torch::Tensor taehv_conv3x3_cutlass_implicit_nhwc_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {
    WM_CHECK_CUDA(x);
    WM_CHECK_CUDA(weight);
    WM_CHECK_CUDA(bias);
    WM_CHECK_CONTIGUOUS(x);
    WM_CHECK_CONTIGUOUS(weight);
    WM_CHECK_CONTIGUOUS(bias);
    WM_CHECK_F32(x);
    WM_CHECK_F32(weight);
    WM_CHECK_F32(bias);
    TORCH_CHECK(x.dim() == 4, "x must be NHWC [N,H,W,C]");
    TORCH_CHECK(weight.dim() == 4, "weight must be KRSC [Cout,3,3,Cin]");
    TORCH_CHECK(weight.size(1) == 3 && weight.size(2) == 3, "weight must be 3x3 KRSC");
    TORCH_CHECK(weight.size(3) == x.size(3), "input channel mismatch");
    TORCH_CHECK(bias.numel() == weight.size(0), "bias length must equal Cout");

    int N = (int)x.size(0);
    int H = (int)x.size(1);
    int W = (int)x.size(2);
    int C_in = (int)x.size(3);
    int C_out = (int)weight.size(0);
    auto out = torch::empty({N, H, W, C_out}, x.options());

    using Layout = cutlass::layout::TensorNHWC;
    using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
        float,
        Layout,
        float,
        Layout,
        float,
        Layout,
        float,
        cutlass::arch::OpClassSimt,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<64, 64, 8>,
        cutlass::gemm::GemmShape<32, 32, 8>,
        cutlass::gemm::GemmShape<1, 1, 1>,
        cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2,
        cutlass::arch::OpMultiplyAdd,
        cutlass::conv::IteratorAlgorithm::kAnalytic,
        cutlass::conv::StrideSupport::kUnity,
        1,
        1>::Kernel;
    using ImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

    cutlass::Tensor4DCoord input_size(N, H, W, C_in);
    cutlass::Tensor4DCoord filter_size(C_out, 3, 3, C_in);
    cutlass::Tensor4DCoord padding(1, 1, 1, 1);
    cutlass::MatrixCoord stride(1, 1);
    cutlass::MatrixCoord dilation(1, 1);
    cutlass::Tensor4DCoord output_size(N, H, W, C_out);
    cutlass::conv::Conv2dProblemSize problem(
        input_size,
        filter_size,
        padding,
        stride,
        dilation,
        output_size,
        cutlass::conv::Mode::kCrossCorrelation,
        1);

    typename ImplicitGemm::Arguments args(
        problem,
        cutlass::TensorRef<float, Layout>(
            x.data_ptr<float>(),
            Layout::packed(input_size)),
        cutlass::TensorRef<float, Layout>(
            weight.data_ptr<float>(),
            Layout::packed(filter_size)),
        cutlass::TensorRef<float, Layout>(
            out.data_ptr<float>(),
            Layout::packed(output_size)),
        cutlass::TensorRef<float, Layout>(
            out.data_ptr<float>(),
            Layout::packed(output_size)),
        {1.0f, 0.0f});

    ImplicitGemm implicit_gemm;
    check_cutlass_status(implicit_gemm.can_implement(args), "taehv_conv3x3_cutlass_implicit_nhwc.can_implement");
    size_t workspace_size = implicit_gemm.get_workspace_size(args);
    auto workspace = torch::empty({(int64_t)workspace_size}, x.options().dtype(torch::kUInt8));
    void *workspace_ptr = workspace_size ? workspace.data_ptr<uint8_t>() : nullptr;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    check_cutlass_status(implicit_gemm(args, workspace_ptr, stream), "taehv_conv3x3_cutlass_implicit_nhwc");
    check_last_cuda_error("taehv_conv3x3_cutlass_implicit_nhwc");

    taehv_add_bias_nhwc_f32_kernel<<<div_up_i64((int64_t)N * H * W * C_out, 256), 256, 0, stream>>>(
        out.data_ptr<float>(),
        bias.data_ptr<float>(),
        (int64_t)N * H * W * C_out,
        C_out);
    check_last_cuda_error("taehv_conv3x3_cutlass_implicit_nhwc_bias");
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ortho_rope", &ortho_rope_cuda, "Experimental standalone OrthoRoPE");
    m.def("masked_attention", &masked_attention_cuda, "Experimental written-mask GQA attention",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("written"), py::arg("scale"));
    m.def("cutlass_fmha_bmhk_fp16", &cutlass_fmha_bmhk_fp16_cuda,
          "Raw CUTLASS contiguous BMHK FP16 probe",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("scale"));
    m.def("taehv_conv3x3_cutlass_implicit_nhwc",
          &taehv_conv3x3_cutlass_implicit_nhwc_cuda,
          "Experimental FP32 NHWC implicit-GEMM convolution");
}
