#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "world_cuda_ops.cuh"

#include <algorithm>
#include <cstdint>
#include <vector>

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

void check_status(int status, const char *op) {
    if (status != 0) {
        cudaError_t error = cudaGetLastError();
        TORCH_CHECK(false, op, " failed: ", cudaGetErrorString(error));
    }
}

__half *half_ptr(torch::Tensor &tensor) {
    return reinterpret_cast<__half *>(tensor.data_ptr<at::Half>());
}

torch::Tensor byte_workspace(const torch::Tensor &like, size_t bytes) {
    return torch::empty(
        {static_cast<int64_t>(bytes)}, like.options().dtype(torch::kUInt8));
}

void check_linear_inputs(
        const torch::Tensor &x,
        const torch::Tensor &w,
        at::ScalarType dtype) {
    check_tensor(x, dtype, "x");
    check_tensor(w, dtype, "w");
    check_same_device(x, w, "x", "w");
    TORCH_CHECK(x.dim() == 2 && w.dim() == 2, "x and w must be 2D");
    TORCH_CHECK(x.size(1) == w.size(1), "x/w reduction dimensions must match");
}

torch::Tensor silu_cuda(torch::Tensor x) {
    check_tensor(x, at::ScalarType::Float, "x");
    c10::cuda::CUDAGuard device_guard(x.device());
    auto y = torch::empty_like(x);
    check_status(wm_cuda_silu_f32(x.data_ptr<float>(), y.data_ptr<float>(), x.numel()), "silu");
    return y;
}

torch::Tensor silu_to_half_cuda(torch::Tensor x) {
    check_tensor(x, at::ScalarType::Float, "x");
    c10::cuda::CUDAGuard device_guard(x.device());
    auto y = torch::empty(x.sizes(), x.options().dtype(torch::kFloat16));
    check_status(
        wm_cuda_silu_f32_to_f16(x.data_ptr<float>(), half_ptr(y), x.numel()),
        "silu_to_half");
    return y;
}

torch::Tensor row_major_linear_fp16_impl(
        torch::Tensor x,
        torch::Tensor w,
        int variant) {
    check_linear_inputs(x, w, at::ScalarType::Float);
    c10::cuda::CUDAGuard device_guard(x.device());
    int m = static_cast<int>(x.size(0));
    int k = static_cast<int>(x.size(1));
    int n = static_cast<int>(w.size(0));
    auto w_half = w.to(torch::kFloat16);
    auto x_half = torch::empty(x.sizes(), x.options().dtype(torch::kFloat16));
    auto y = torch::empty({m, n}, x.options());
    int status = 1;
    if (variant == 0) {
        status = wm_cuda_linear_fp16_weight_simt(
            x.data_ptr<float>(), half_ptr(x_half), half_ptr(w_half), y.data_ptr<float>(),
            m, k, n);
    } else if (variant == 1) {
        status = wm_cuda_linear_fp16_weight_tensorop(
            x.data_ptr<float>(), half_ptr(x_half), half_ptr(w_half), y.data_ptr<float>(),
            m, k, n);
    } else {
        status = wm_cuda_linear_fp16_weight_tensorop_m64n64(
            x.data_ptr<float>(), half_ptr(x_half), half_ptr(w_half), y.data_ptr<float>(),
            m, k, n);
    }
    check_status(status, "row_major_linear_fp16");
    return y;
}

torch::Tensor row_major_linear_fp16_cuda(torch::Tensor x, torch::Tensor w) {
    return row_major_linear_fp16_impl(std::move(x), std::move(w), 0);
}

torch::Tensor row_major_linear_fp16_tensorop_cuda(torch::Tensor x, torch::Tensor w) {
    return row_major_linear_fp16_impl(std::move(x), std::move(w), 1);
}

torch::Tensor row_major_linear_fp16_tensorop_m64n64_cuda(
        torch::Tensor x,
        torch::Tensor w) {
    return row_major_linear_fp16_impl(std::move(x), std::move(w), 2);
}

torch::Tensor row_major_linear_fp16_input_tensorop_m64n64_cuda(
        torch::Tensor x,
        torch::Tensor w) {
    check_linear_inputs(x, w, at::ScalarType::Half);
    c10::cuda::CUDAGuard device_guard(x.device());
    int m = static_cast<int>(x.size(0));
    int k = static_cast<int>(x.size(1));
    int n = static_cast<int>(w.size(0));
    auto y = torch::empty({m, n}, x.options().dtype(torch::kFloat32));
    check_status(
        wm_cuda_linear_fp16_input_weight_tensorop_m64n64(
            half_ptr(x), half_ptr(w), y.data_ptr<float>(), m, k, n),
        "row_major_linear_fp16_input_tensorop_m64n64");
    return y;
}

torch::Tensor row_major_linear_fp16_tensorop_m64n64_silu_half_cuda(
        torch::Tensor x,
        torch::Tensor w) {
    check_linear_inputs(x, w, at::ScalarType::Float);
    c10::cuda::CUDAGuard device_guard(x.device());
    int m = static_cast<int>(x.size(0));
    int k = static_cast<int>(x.size(1));
    int n = static_cast<int>(w.size(0));
    auto w_half = w.to(torch::kFloat16);
    auto x_half = torch::empty(x.sizes(), x.options().dtype(torch::kFloat16));
    auto y = torch::empty({m, n}, x.options().dtype(torch::kFloat16));
    check_status(
        wm_cuda_linear_fp16_weight_tensorop_m64n64_silu_half(
            x.data_ptr<float>(), half_ptr(x_half), half_ptr(w_half), half_ptr(y), m, k, n),
        "row_major_linear_fp16_tensorop_m64n64_silu_half");
    return y;
}

torch::Tensor row_major_linear_fp16_input_tensorop_m64n64_silu_half_cuda(
        torch::Tensor x,
        torch::Tensor w) {
    check_linear_inputs(x, w, at::ScalarType::Half);
    c10::cuda::CUDAGuard device_guard(x.device());
    int m = static_cast<int>(x.size(0));
    int k = static_cast<int>(x.size(1));
    int n = static_cast<int>(w.size(0));
    auto y = torch::empty({m, n}, x.options());
    check_status(
        wm_cuda_linear_fp16_input_weight_tensorop_m64n64_silu_half(
            half_ptr(x), half_ptr(w), half_ptr(y), m, k, n),
        "row_major_linear_fp16_input_tensorop_m64n64_silu_half");
    return y;
}

torch::Tensor row_major_linear_fp16_input_splitk_impl(
        torch::Tensor x,
        torch::Tensor w,
        int64_t split_k_slices,
        bool parallel) {
    check_linear_inputs(x, w, at::ScalarType::Half);
    TORCH_CHECK(split_k_slices > 0, "split_k_slices must be positive");
    c10::cuda::CUDAGuard device_guard(x.device());
    int m = static_cast<int>(x.size(0));
    int k = static_cast<int>(x.size(1));
    int n = static_cast<int>(w.size(0));
    int slices = static_cast<int>(split_k_slices);
    auto y = torch::empty({m, n}, x.options().dtype(torch::kFloat32));
    size_t bytes = parallel
        ? wm_cuda_linear_fp16_input_weight_tensorop_splitk_parallel_workspace_size(
              m, k, n, slices)
        : wm_cuda_linear_fp16_weight_tensorop_splitk_workspace_size(m, k, n, slices);
    auto workspace = byte_workspace(x, bytes);
    int status = parallel
        ? wm_cuda_linear_fp16_input_weight_tensorop_splitk_parallel(
              half_ptr(x), half_ptr(w), y.data_ptr<float>(), m, k, n, slices,
              workspace.data_ptr(), bytes)
        : wm_cuda_linear_fp16_input_weight_tensorop_splitk(
              half_ptr(x), half_ptr(w), y.data_ptr<float>(), m, k, n, slices,
              workspace.data_ptr(), bytes);
    check_status(status, parallel ? "parallel_splitk" : "splitk");
    return y;
}

torch::Tensor row_major_linear_fp16_tensorop_splitk_cuda(
        torch::Tensor x,
        torch::Tensor w,
        int64_t split_k_slices) {
    check_linear_inputs(x, w, at::ScalarType::Float);
    return row_major_linear_fp16_input_splitk_impl(
        x.to(torch::kFloat16), w.to(torch::kFloat16), split_k_slices, false);
}

torch::Tensor row_major_linear_fp16_input_tensorop_splitk_cuda(
        torch::Tensor x,
        torch::Tensor w,
        int64_t split_k_slices) {
    return row_major_linear_fp16_input_splitk_impl(
        std::move(x), std::move(w), split_k_slices, false);
}

torch::Tensor row_major_linear_fp16_input_tensorop_splitk_parallel_cuda(
        torch::Tensor x,
        torch::Tensor w,
        int64_t split_k_slices) {
    return row_major_linear_fp16_input_splitk_impl(
        std::move(x), std::move(w), split_k_slices, true);
}

torch::Tensor rms_norm_cuda(torch::Tensor x, double eps) {
    check_tensor(x, at::ScalarType::Float, "x");
    TORCH_CHECK(x.dim() >= 1 && x.size(-1) > 0, "x must have a non-empty last dimension");
    c10::cuda::CUDAGuard device_guard(x.device());
    int d = static_cast<int>(x.size(-1));
    int rows = static_cast<int>(x.numel() / d);
    auto y = torch::empty_like(x);
    check_status(
        wm_cuda_rms_norm_rows_f32(
            x.data_ptr<float>(), y.data_ptr<float>(), rows, d, static_cast<float>(eps)),
        "rms_norm");
    return y;
}

torch::Tensor ada_rms_norm_impl(
        torch::Tensor x,
        torch::Tensor scale,
        torch::Tensor bias,
        double eps,
        bool output_half) {
    check_tensor(x, at::ScalarType::Float, "x");
    check_tensor(scale, at::ScalarType::Float, "scale");
    check_tensor(bias, at::ScalarType::Float, "bias");
    check_same_device(x, scale, "x", "scale");
    check_same_device(x, bias, "x", "bias");
    TORCH_CHECK(x.dim() == 3, "x must be [B,T,D]");
    TORCH_CHECK(scale.dim() == 3 && bias.sizes() == scale.sizes(),
                "scale and bias must have shape [B,N,D]");
    int bsz = static_cast<int>(x.size(0));
    int tokens = static_cast<int>(x.size(1));
    int d = static_cast<int>(x.size(2));
    int groups = static_cast<int>(scale.size(1));
    TORCH_CHECK(scale.size(0) == bsz && scale.size(2) == d, "modulation shape mismatch");
    TORCH_CHECK(groups > 0 && tokens % groups == 0, "T must be divisible by N");
    int rows_per_group = tokens / groups;
    c10::cuda::CUDAGuard device_guard(x.device());
    auto y = torch::empty(
        x.sizes(), x.options().dtype(output_half ? torch::kFloat16 : torch::kFloat32));
    for (int b = 0; b < bsz; ++b) {
        for (int n = 0; n < groups; ++n) {
            int64_t row_offset = (static_cast<int64_t>(b) * tokens + n * rows_per_group) * d;
            int64_t mod_offset = (static_cast<int64_t>(b) * groups + n) * d;
            int status = output_half
                ? wm_cuda_ada_rms_norm_f16(
                      x.data_ptr<float>() + row_offset,
                      scale.data_ptr<float>() + mod_offset,
                      bias.data_ptr<float>() + mod_offset,
                      half_ptr(y) + row_offset,
                      rows_per_group, d, static_cast<float>(eps))
                : wm_cuda_ada_rms_norm_f32(
                      x.data_ptr<float>() + row_offset,
                      scale.data_ptr<float>() + mod_offset,
                      bias.data_ptr<float>() + mod_offset,
                      y.data_ptr<float>() + row_offset,
                      rows_per_group, d, static_cast<float>(eps));
            check_status(status, output_half ? "ada_rms_norm_half" : "ada_rms_norm");
        }
    }
    return y;
}

torch::Tensor ada_rms_norm_cuda(
        torch::Tensor x,
        torch::Tensor scale,
        torch::Tensor bias,
        double eps) {
    return ada_rms_norm_impl(
        std::move(x), std::move(scale), std::move(bias), eps, false);
}

torch::Tensor ada_rms_norm_half_cuda(
        torch::Tensor x,
        torch::Tensor scale,
        torch::Tensor bias,
        double eps) {
    return ada_rms_norm_impl(
        std::move(x), std::move(scale), std::move(bias), eps, true);
}

std::vector<torch::Tensor> qkv_rms_rope_cuda(
        torch::Tensor qkv,
        torch::Tensor x_pos,
        torch::Tensor y_pos,
        torch::Tensor t_pos,
        torch::Tensor xy,
        torch::Tensor inv_t,
        int64_t n_heads,
        int64_t n_kv_heads,
        int64_t width,
        int64_t height,
        double eps) {
    check_tensor(qkv, at::ScalarType::Float, "qkv");
    check_tensor(x_pos, at::ScalarType::Long, "x_pos");
    check_tensor(y_pos, at::ScalarType::Long, "y_pos");
    check_tensor(t_pos, at::ScalarType::Long, "t_pos");
    check_tensor(xy, at::ScalarType::Float, "xy");
    check_tensor(inv_t, at::ScalarType::Float, "inv_t");
    TORCH_CHECK(qkv.dim() == 3, "qkv must be [B,T,(Hq+2*Hkv)*D]");
    int bsz = static_cast<int>(qkv.size(0));
    int tokens = static_cast<int>(qkv.size(1));
    int heads = static_cast<int>(n_heads);
    int kv_heads = static_cast<int>(n_kv_heads);
    TORCH_CHECK(heads > 0 && kv_heads > 0 && heads % kv_heads == 0, "invalid GQA heads");
    int roles = heads + 2 * kv_heads;
    TORCH_CHECK(qkv.size(2) % roles == 0, "qkv width is not divisible by head roles");
    int d = static_cast<int>(qkv.size(2) / roles);
    TORCH_CHECK(d % 8 == 0, "head dimension must be divisible by 8");
    TORCH_CHECK(x_pos.numel() == tokens && y_pos.numel() == tokens && t_pos.numel() == tokens,
                "position lengths must equal T");
    TORCH_CHECK(xy.numel() == d / 8 && inv_t.numel() == d / 4, "RoPE table shape mismatch");
    check_same_device(qkv, x_pos, "qkv", "x_pos");
    check_same_device(qkv, y_pos, "qkv", "y_pos");
    check_same_device(qkv, t_pos, "qkv", "t_pos");
    check_same_device(qkv, xy, "qkv", "xy");
    check_same_device(qkv, inv_t, "qkv", "inv_t");
    c10::cuda::CUDAGuard device_guard(qkv.device());
    auto q = torch::empty({bsz, heads, tokens, d}, qkv.options());
    auto k = torch::empty({bsz, kv_heads, tokens, d}, qkv.options());
    auto v = torch::empty({bsz, kv_heads, tokens, d}, qkv.options());
    int64_t in_stride = static_cast<int64_t>(tokens) * roles * d;
    int64_t q_stride = static_cast<int64_t>(heads) * tokens * d;
    int64_t kv_stride = static_cast<int64_t>(kv_heads) * tokens * d;
    for (int b = 0; b < bsz; ++b) {
        check_status(
            wm_cuda_qkv_fused_rms_rope_f32(
                qkv.data_ptr<float>() + b * in_stride,
                q.data_ptr<float>() + b * q_stride,
                k.data_ptr<float>() + b * kv_stride,
                v.data_ptr<float>() + b * kv_stride,
                x_pos.data_ptr<int64_t>(), y_pos.data_ptr<int64_t>(), t_pos.data_ptr<int64_t>(),
                xy.data_ptr<float>(), inv_t.data_ptr<float>(),
                tokens, heads, kv_heads, d,
                static_cast<int>(width), static_cast<int>(height), static_cast<float>(eps)),
            "qkv_rms_rope");
    }
    return {q, k, v};
}

struct AttentionShape {
    int batch;
    int heads;
    int kv_heads;
    int tokens;
    int capacity;
    int d;
};

AttentionShape check_attention(
        const torch::Tensor &q,
        const torch::Tensor &k,
        const torch::Tensor &v,
        at::ScalarType kv_dtype) {
    check_tensor(q, at::ScalarType::Float, "q");
    check_tensor(k, kv_dtype, "k");
    check_tensor(v, kv_dtype, "v");
    check_same_device(q, k, "q", "k");
    check_same_device(q, v, "q", "v");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
                "q, k, v must be [B,H,T,D]");
    TORCH_CHECK(k.sizes() == v.sizes(), "k/v shape mismatch");
    TORCH_CHECK(q.size(0) == k.size(0) && q.size(3) == k.size(3), "q/kv shape mismatch");
    TORCH_CHECK(k.size(1) > 0 && q.size(1) % k.size(1) == 0, "invalid GQA heads");
    return {
        static_cast<int>(q.size(0)),
        static_cast<int>(q.size(1)),
        static_cast<int>(k.size(1)),
        static_cast<int>(q.size(2)),
        static_cast<int>(k.size(2)),
        static_cast<int>(q.size(3)),
    };
}

void check_indices(
        const torch::Tensor &indices,
        const torch::Tensor &q,
        int batch) {
    check_tensor(indices, at::ScalarType::Long, "indices");
    check_same_device(q, indices, "q", "indices");
    TORCH_CHECK(indices.dim() == 1 || indices.dim() == 2, "indices must be [N] or [B,N]");
    if (indices.dim() == 2) {
        TORCH_CHECK(indices.size(0) == batch, "indices batch dimension mismatch");
    }
}

const int64_t *batch_indices(const torch::Tensor &indices, int batch_index) {
    int64_t stride = indices.dim() == 2 ? indices.size(1) : 0;
    return indices.data_ptr<int64_t>() + batch_index * stride;
}

torch::Tensor attention_tokens_to_bhtd(torch::Tensor tokens) {
    // Production attention writes [B,T,H,D]; retain the historical test API [B,H,T,D].
    return tokens.permute({0, 2, 1, 3}).contiguous();
}

torch::Tensor indexed_attention_f32_impl(
        torch::Tensor q,
        torch::Tensor k,
        torch::Tensor v,
        torch::Tensor indices,
        double scale,
        bool flash) {
    AttentionShape s = check_attention(q, k, v, at::ScalarType::Float);
    check_indices(indices, q, s.batch);
    int count = static_cast<int>(indices.size(-1));
    auto counts = torch::full({s.batch}, count, q.options().dtype(torch::kInt32));
    auto out = torch::empty({s.batch, s.tokens, s.heads, s.d}, q.options());
    c10::cuda::CUDAGuard device_guard(q.device());
    int64_t q_stride = static_cast<int64_t>(s.heads) * s.tokens * s.d;
    int64_t kv_stride = static_cast<int64_t>(s.kv_heads) * s.capacity * s.d;
    int64_t out_stride = static_cast<int64_t>(s.tokens) * s.heads * s.d;
    for (int b = 0; b < s.batch; ++b) {
        int status;
        if (flash) {
            TORCH_CHECK(s.d == 64, "flash attention requires D=64");
            status = wm_cuda_indexed_attention_d64_flash_f32(
                q.data_ptr<float>() + b * q_stride,
                k.data_ptr<float>() + b * kv_stride,
                v.data_ptr<float>() + b * kv_stride,
                batch_indices(indices, b), counts.data_ptr<int>() + b,
                out.data_ptr<float>() + b * out_stride,
                s.heads, s.kv_heads, s.tokens, s.capacity, static_cast<float>(scale));
        } else {
            status = wm_cuda_indexed_attention_f32(
                q.data_ptr<float>() + b * q_stride,
                k.data_ptr<float>() + b * kv_stride,
                v.data_ptr<float>() + b * kv_stride,
                batch_indices(indices, b), counts.data_ptr<int>() + b,
                out.data_ptr<float>() + b * out_stride,
                s.heads, s.kv_heads, s.tokens, s.capacity, s.d, static_cast<float>(scale));
        }
        check_status(status, flash ? "indexed_attention_flash" : "indexed_attention");
    }
    return attention_tokens_to_bhtd(std::move(out));
}

torch::Tensor indexed_attention_cuda(
        torch::Tensor q,
        torch::Tensor k,
        torch::Tensor v,
        torch::Tensor indices,
        double scale) {
    return indexed_attention_f32_impl(
        std::move(q), std::move(k), std::move(v), std::move(indices), scale, false);
}

torch::Tensor indexed_attention_flash_cuda(
        torch::Tensor q,
        torch::Tensor k,
        torch::Tensor v,
        torch::Tensor indices,
        double scale) {
    return indexed_attention_f32_impl(
        std::move(q), std::move(k), std::move(v), std::move(indices), scale, true);
}

torch::Tensor indexed_attention_half_kv_impl(
        torch::Tensor q,
        torch::Tensor k,
        torch::Tensor v,
        torch::Tensor indices,
        double scale,
        bool flash) {
    AttentionShape s = check_attention(q, k, v, at::ScalarType::Half);
    TORCH_CHECK(s.d == 64, "half-cache attention requires D=64");
    check_indices(indices, q, s.batch);
    int count = static_cast<int>(indices.size(-1));
    auto counts = torch::full({s.batch}, count, q.options().dtype(torch::kInt32));
    auto out = torch::empty({s.batch, s.tokens, s.heads, 64}, q.options());
    c10::cuda::CUDAGuard device_guard(q.device());
    int64_t q_stride = static_cast<int64_t>(s.heads) * s.tokens * 64;
    int64_t kv_stride = static_cast<int64_t>(s.kv_heads) * s.capacity * 64;
    int64_t out_stride = static_cast<int64_t>(s.tokens) * s.heads * 64;
    for (int b = 0; b < s.batch; ++b) {
        int status = flash
            ? wm_cuda_indexed_attention_d64_flash_f16_kv(
                  q.data_ptr<float>() + b * q_stride,
                  half_ptr(k) + b * kv_stride,
                  half_ptr(v) + b * kv_stride,
                  batch_indices(indices, b), counts.data_ptr<int>() + b,
                  out.data_ptr<float>() + b * out_stride,
                  s.heads, s.kv_heads, s.tokens, s.capacity, static_cast<float>(scale))
            : wm_cuda_indexed_attention_d64_warp_f16_kv(
                  q.data_ptr<float>() + b * q_stride,
                  half_ptr(k) + b * kv_stride,
                  half_ptr(v) + b * kv_stride,
                  batch_indices(indices, b), counts.data_ptr<int>() + b,
                  out.data_ptr<float>() + b * out_stride,
                  s.heads, s.kv_heads, s.tokens, s.capacity, static_cast<float>(scale));
        check_status(status, flash ? "indexed_attention_half_kv_flash" : "indexed_attention_half_kv");
    }
    return attention_tokens_to_bhtd(std::move(out));
}

torch::Tensor indexed_attention_half_kv_cuda(
        torch::Tensor q,
        torch::Tensor k,
        torch::Tensor v,
        torch::Tensor indices,
        double scale) {
    return indexed_attention_half_kv_impl(
        std::move(q), std::move(k), std::move(v), std::move(indices), scale, false);
}

torch::Tensor indexed_attention_half_kv_flash_cuda(
        torch::Tensor q,
        torch::Tensor k,
        torch::Tensor v,
        torch::Tensor indices,
        double scale) {
    return indexed_attention_half_kv_impl(
        std::move(q), std::move(k), std::move(v), std::move(indices), scale, true);
}

torch::Tensor indexed_attention_half_kv_cutlass_impl(
        torch::Tensor q,
        torch::Tensor k,
        torch::Tensor v,
        torch::Tensor indices,
        double scale,
        bool grouped) {
    AttentionShape s = check_attention(q, k, v, at::ScalarType::Half);
    TORCH_CHECK(s.d == 64, "CUTLASS attention requires D=64");
    check_indices(indices, q, s.batch);
    int count = static_cast<int>(indices.size(-1));
    auto out = torch::empty({s.batch, s.tokens, s.heads, 64}, q.options());
    int64_t q_elems = static_cast<int64_t>(s.heads) * s.tokens * 64;
    // The grouped wrapper falls back to the per-query-head implementation for
    // small Nkv, so its compact K/V workspace must satisfy both paths.
    int compact_heads = s.heads;
    int64_t compact_elems = static_cast<int64_t>(compact_heads) * count * 64;
    int64_t score_elems = std::max<int64_t>(
        static_cast<int64_t>(s.heads) * s.tokens * count, q_elems);
    auto q_half = torch::empty({s.batch, q_elems}, q.options().dtype(torch::kFloat16));
    auto k_compact = torch::empty({s.batch, compact_elems}, q.options().dtype(torch::kFloat16));
    auto v_compact = torch::empty({s.batch, compact_elems}, q.options().dtype(torch::kFloat16));
    auto scores = torch::empty({s.batch, score_elems}, q.options());
    auto probs = torch::empty(
        {s.batch, static_cast<int64_t>(s.heads) * s.tokens * count},
        q.options().dtype(torch::kFloat16));
    c10::cuda::CUDAGuard device_guard(q.device());
    int64_t kv_stride = static_cast<int64_t>(s.kv_heads) * s.capacity * 64;
    for (int b = 0; b < s.batch; ++b) {
        int status = grouped
            ? wm_cuda_attention_d64_cutlass_grouped_f16_kv(
                  q.data_ptr<float>() + b * q_elems,
                  half_ptr(k) + b * kv_stride,
                  half_ptr(v) + b * kv_stride,
                  batch_indices(indices, b), count,
                  out.data_ptr<float>() + b * q_elems,
                  half_ptr(q_half) + b * q_elems,
                  half_ptr(k_compact) + b * compact_elems,
                  half_ptr(v_compact) + b * compact_elems,
                  scores.data_ptr<float>() + b * score_elems,
                  half_ptr(probs) + static_cast<int64_t>(b) * s.heads * s.tokens * count,
                  s.heads, s.kv_heads, s.tokens, s.capacity, static_cast<float>(scale))
            : wm_cuda_attention_d64_cutlass_f16_kv(
                  q.data_ptr<float>() + b * q_elems,
                  half_ptr(k) + b * kv_stride,
                  half_ptr(v) + b * kv_stride,
                  batch_indices(indices, b), count,
                  out.data_ptr<float>() + b * q_elems,
                  half_ptr(q_half) + b * q_elems,
                  half_ptr(k_compact) + b * compact_elems,
                  half_ptr(v_compact) + b * compact_elems,
                  scores.data_ptr<float>() + b * score_elems,
                  half_ptr(probs) + static_cast<int64_t>(b) * s.heads * s.tokens * count,
                  s.heads, s.kv_heads, s.tokens, s.capacity, static_cast<float>(scale));
        check_status(status, grouped ? "cutlass_grouped_attention" : "cutlass_attention");
    }
    return attention_tokens_to_bhtd(std::move(out));
}

torch::Tensor indexed_attention_half_kv_cutlass_cuda(
        torch::Tensor q,
        torch::Tensor k,
        torch::Tensor v,
        torch::Tensor indices,
        double scale) {
    return indexed_attention_half_kv_cutlass_impl(
        std::move(q), std::move(k), std::move(v), std::move(indices), scale, false);
}

torch::Tensor indexed_attention_half_kv_cutlass_grouped_cuda(
        torch::Tensor q,
        torch::Tensor k,
        torch::Tensor v,
        torch::Tensor indices,
        double scale) {
    return indexed_attention_half_kv_cutlass_impl(
        std::move(q), std::move(k), std::move(v), std::move(indices), scale, true);
}

torch::Tensor indexed_attention_half_kv_fmha_impl(
        torch::Tensor q,
        torch::Tensor k,
        torch::Tensor v,
        torch::Tensor indices,
        double scale,
        bool output_half) {
    TORCH_CHECK(wm_cuda_has_cutlass_fmha(), "CUTLASS FMHA headers are unavailable");
    AttentionShape s = check_attention(q, k, v, at::ScalarType::Half);
    TORCH_CHECK(s.d == 64, "FMHA attention requires D=64");
    check_indices(indices, q, s.batch);
    int count = static_cast<int>(indices.size(-1));
    int group = s.heads / s.kv_heads;
    int grouped_rows = group * s.tokens;
    int64_t out_elems = static_cast<int64_t>(s.heads) * s.tokens * 64;
    int64_t compact_elems = static_cast<int64_t>(s.kv_heads) * count * 64;
    auto out = torch::empty(
        {s.batch, s.tokens, s.heads, 64},
        q.options().dtype(output_half ? torch::kFloat16 : torch::kFloat32));
    auto out_f32 = output_half
        ? torch::empty({s.batch, out_elems}, q.options())
        : out.view({s.batch, out_elems});
    auto out_f16 = output_half
        ? out.view({s.batch, out_elems})
        : torch::empty({s.batch, out_elems}, q.options().dtype(torch::kFloat16));
    auto q_bmhd = torch::empty(
        {s.batch, grouped_rows, s.kv_heads, 64}, q.options().dtype(torch::kFloat16));
    auto k_bnhd = torch::empty({s.batch, compact_elems}, q.options().dtype(torch::kFloat16));
    auto v_bnhd = torch::empty({s.batch, compact_elems}, q.options().dtype(torch::kFloat16));
    auto out_bmhd = torch::empty_like(q_bmhd);
    c10::cuda::CUDAGuard device_guard(q.device());
    int64_t kv_stride = static_cast<int64_t>(s.kv_heads) * s.capacity * 64;
    for (int b = 0; b < s.batch; ++b) {
        check_status(
            wm_cuda_attention_d64_fmha_f16_kv(
                q.data_ptr<float>() + b * out_elems,
                half_ptr(k) + b * kv_stride,
                half_ptr(v) + b * kv_stride,
                batch_indices(indices, b), count,
                out_f32.data_ptr<float>() + b * out_elems,
                half_ptr(out_f16) + b * out_elems,
                output_half ? 1 : 0,
                half_ptr(q_bmhd) + b * out_elems,
                half_ptr(k_bnhd) + b * compact_elems,
                half_ptr(v_bnhd) + b * compact_elems,
                half_ptr(out_bmhd) + b * out_elems,
                s.heads, s.kv_heads, s.tokens, s.capacity, static_cast<float>(scale)),
            output_half ? "fmha_attention_half" : "fmha_attention");
    }
    return attention_tokens_to_bhtd(std::move(out));
}

torch::Tensor indexed_attention_half_kv_fmha_gqa_cuda(
        torch::Tensor q,
        torch::Tensor k,
        torch::Tensor v,
        torch::Tensor indices,
        double scale) {
    return indexed_attention_half_kv_fmha_impl(
        std::move(q), std::move(k), std::move(v), std::move(indices), scale, false);
}

torch::Tensor indexed_attention_half_kv_fmha_gqa_half_output_cuda(
        torch::Tensor q,
        torch::Tensor k,
        torch::Tensor v,
        torch::Tensor indices,
        double scale) {
    return indexed_attention_half_kv_fmha_impl(
        std::move(q), std::move(k), std::move(v), std::move(indices), scale, true);
}

torch::Tensor sparse_attention_half_kv_fmha_impl(
        torch::Tensor q,
        torch::Tensor k,
        torch::Tensor v,
        torch::Tensor block_ids,
        double scale,
        bool output_half) {
    TORCH_CHECK(wm_cuda_has_cutlass_fmha(), "CUTLASS FMHA headers are unavailable");
    AttentionShape s = check_attention(q, k, v, at::ScalarType::Half);
    TORCH_CHECK(s.d == 64 && s.capacity % 128 == 0, "sparse FMHA requires D=64 and 128-token blocks");
    check_tensor(block_ids, at::ScalarType::Int, "block_ids");
    check_same_device(q, block_ids, "q", "block_ids");
    TORCH_CHECK(block_ids.dim() == 1 || block_ids.dim() == 2,
                "block_ids must be [N] or [B,N]");
    if (block_ids.dim() == 2) {
        TORCH_CHECK(block_ids.size(0) == s.batch, "block_ids batch mismatch");
    }
    int block_count = static_cast<int>(block_ids.size(-1));
    int group = s.heads / s.kv_heads;
    int grouped_rows = group * s.tokens;
    int64_t out_elems = static_cast<int64_t>(s.heads) * s.tokens * 64;
    auto out = torch::empty(
        {s.batch, s.tokens, s.heads, 64},
        q.options().dtype(output_half ? torch::kFloat16 : torch::kFloat32));
    auto out_f32 = output_half
        ? torch::empty({s.batch, out_elems}, q.options())
        : out.view({s.batch, out_elems});
    auto out_f16 = output_half
        ? out.view({s.batch, out_elems})
        : torch::empty({s.batch, out_elems}, q.options().dtype(torch::kFloat16));
    auto q_bmhd = torch::empty(
        {s.batch, grouped_rows, s.kv_heads, 64}, q.options().dtype(torch::kFloat16));
    auto out_bmhd = torch::empty_like(q_bmhd);
    c10::cuda::CUDAGuard device_guard(q.device());
    int64_t kv_stride = static_cast<int64_t>(s.kv_heads) * s.capacity * 64;
    int64_t block_stride = block_ids.dim() == 2 ? block_count : 0;
    for (int b = 0; b < s.batch; ++b) {
        check_status(
            wm_cuda_attention_d64_sparse_fmha_f16_kv(
                q.data_ptr<float>() + b * out_elems,
                half_ptr(k) + b * kv_stride,
                half_ptr(v) + b * kv_stride,
                block_ids.data_ptr<int32_t>() + b * block_stride,
                block_count,
                out_f32.data_ptr<float>() + b * out_elems,
                half_ptr(out_f16) + b * out_elems,
                output_half ? 1 : 0,
                half_ptr(q_bmhd) + b * out_elems,
                half_ptr(out_bmhd) + b * out_elems,
                s.heads, s.kv_heads, s.tokens, s.capacity, static_cast<float>(scale)),
            output_half ? "sparse_fmha_attention_half" : "sparse_fmha_attention");
    }
    return attention_tokens_to_bhtd(std::move(out));
}

torch::Tensor sparse_attention_half_kv_fmha_gqa_cuda(
        torch::Tensor q,
        torch::Tensor k,
        torch::Tensor v,
        torch::Tensor block_ids,
        double scale) {
    return sparse_attention_half_kv_fmha_impl(
        std::move(q), std::move(k), std::move(v), std::move(block_ids), scale, false);
}

torch::Tensor sparse_attention_half_kv_fmha_gqa_half_output_cuda(
        torch::Tensor q,
        torch::Tensor k,
        torch::Tensor v,
        torch::Tensor block_ids,
        double scale) {
    return sparse_attention_half_kv_fmha_impl(
        std::move(q), std::move(k), std::move(v), std::move(block_ids), scale, true);
}

torch::Tensor kv_cache_upsert_impl(
        torch::Tensor cache_k,
        torch::Tensor cache_v,
        torch::Tensor written,
        torch::Tensor k,
        torch::Tensor v,
        int64_t frame_idx,
        int64_t ring_length,
        int64_t pinned_dilation,
        bool frozen,
        bool half_cache) {
    check_tensor(cache_k, half_cache ? at::ScalarType::Half : at::ScalarType::Float, "cache_k");
    check_tensor(cache_v, half_cache ? at::ScalarType::Half : at::ScalarType::Float, "cache_v");
    check_tensor(written, at::ScalarType::Bool, "written");
    check_tensor(k, at::ScalarType::Float, "k");
    check_tensor(v, at::ScalarType::Float, "v");
    TORCH_CHECK(cache_k.sizes() == cache_v.sizes() && k.sizes() == v.sizes(), "K/V shape mismatch");
    TORCH_CHECK(cache_k.dim() == 4 && k.dim() == 4, "cache and K/V must be [B,H,T,D]");
    TORCH_CHECK(cache_k.size(0) == 1 && k.size(0) == 1,
                "production cache operator is unbatched (B must be 1)");
    int heads = static_cast<int>(k.size(1));
    int tokens = static_cast<int>(k.size(2));
    int d = static_cast<int>(k.size(3));
    int ring = static_cast<int>(ring_length);
    int capacity = static_cast<int>(cache_k.size(2));
    TORCH_CHECK(cache_k.size(1) == heads && cache_k.size(3) == d, "cache/K shape mismatch");
    TORCH_CHECK(ring > 0 && capacity == ring + tokens, "cache capacity must equal ring_length + T");
    TORCH_CHECK(written.numel() == capacity, "written length mismatch");
    TORCH_CHECK(frame_idx >= 0 && pinned_dilation > 0, "invalid cache schedule");
    TORCH_CHECK(ring % tokens == 0 && (ring / tokens) % pinned_dilation == 0,
                "ring frames must be divisible by pinned_dilation");
    int64_t bucket = (frame_idx + pinned_dilation - 1) / pinned_dilation;
    int64_t num_buckets = (ring / tokens) / pinned_dilation;
    int base = static_cast<int>((bucket % num_buckets) * tokens);
    bool write_step = (frame_idx % pinned_dilation) == 0;
    auto mask_written = written.clone();
    if (write_step) mask_written.slice(0, base, base + tokens).fill_(false);
    c10::cuda::CUDAGuard device_guard(k.device());
    int status = half_cache
        ? wm_cuda_kv_cache_upsert_copy_f16(
              half_ptr(cache_k), half_ptr(cache_v), k.data_ptr<float>(), v.data_ptr<float>(),
              written.data_ptr<bool>(), heads, tokens, d, ring, base, write_step, frozen)
        : wm_cuda_kv_cache_upsert_copy_f32(
              cache_k.data_ptr<float>(), cache_v.data_ptr<float>(),
              k.data_ptr<float>(), v.data_ptr<float>(), written.data_ptr<bool>(),
              heads, tokens, d, ring, base, write_step, frozen);
    check_status(status, half_cache ? "kv_cache_upsert_half" : "kv_cache_upsert");
    return mask_written;
}

torch::Tensor kv_cache_upsert_cuda(
        torch::Tensor cache_k,
        torch::Tensor cache_v,
        torch::Tensor written,
        torch::Tensor k,
        torch::Tensor v,
        int64_t frame_idx,
        int64_t ring_length,
        int64_t pinned_dilation,
        bool frozen) {
    return kv_cache_upsert_impl(
        std::move(cache_k), std::move(cache_v), std::move(written),
        std::move(k), std::move(v), frame_idx, ring_length, pinned_dilation, frozen, false);
}

torch::Tensor kv_cache_upsert_half_cuda(
        torch::Tensor cache_k,
        torch::Tensor cache_v,
        torch::Tensor written,
        torch::Tensor k,
        torch::Tensor v,
        int64_t frame_idx,
        int64_t ring_length,
        int64_t pinned_dilation,
        bool frozen) {
    return kv_cache_upsert_impl(
        std::move(cache_k), std::move(cache_v), std::move(written),
        std::move(k), std::move(v), frame_idx, ring_length, pinned_dilation, frozen, true);
}

std::vector<torch::Tensor> cache_frame_indices_cuda(
        torch::Tensor written,
        int64_t tokens_per_frame,
        int64_t base,
        bool write_step) {
    check_tensor(written, at::ScalarType::Bool, "written");
    TORCH_CHECK(written.dim() == 1 && tokens_per_frame > 0, "invalid written/T shape");
    int capacity = static_cast<int>(written.numel());
    int tokens = static_cast<int>(tokens_per_frame);
    TORCH_CHECK(capacity % tokens == 0, "capacity must be divisible by tokens_per_frame");
    TORCH_CHECK(base >= 0 && base + tokens <= capacity && base % tokens == 0, "invalid base");
    auto indices = torch::empty({capacity}, written.options().dtype(torch::kLong));
    auto block_ids = torch::empty(
        {std::max<int64_t>(1, capacity / 128)}, written.options().dtype(torch::kInt32));
    auto count = torch::empty({1}, written.options().dtype(torch::kInt32));
    c10::cuda::CUDAGuard device_guard(written.device());
    check_status(
        wm_cuda_collect_cache_frame_indices(
            written.data_ptr<bool>(), indices.data_ptr<int64_t>(), block_ids.data_ptr<int32_t>(),
            count.data_ptr<int>(), capacity, tokens, static_cast<int>(base), write_step),
        "cache_frame_indices");
    return {indices, count};
}

torch::Tensor patchify_cuda(torch::Tensor x, torch::Tensor weight) {
    check_tensor(x, at::ScalarType::Float, "x");
    check_tensor(weight, at::ScalarType::Float, "weight");
    check_same_device(x, weight, "x", "weight");
    TORCH_CHECK(x.dim() == 4 && weight.dim() == 4, "x/weight must be NCHW/OIHW");
    int batch = static_cast<int>(x.size(0));
    int channels = static_cast<int>(x.size(1));
    int height = static_cast<int>(x.size(2));
    int width = static_cast<int>(x.size(3));
    int d = static_cast<int>(weight.size(0));
    int ph = static_cast<int>(weight.size(2));
    int pw = static_cast<int>(weight.size(3));
    TORCH_CHECK(weight.size(1) == channels && height % ph == 0 && width % pw == 0,
                "patchify shape mismatch");
    int token_h = height / ph;
    int token_w = width / pw;
    int tokens = token_h * token_w;
    auto out = torch::empty({batch, tokens, d}, x.options());
    c10::cuda::CUDAGuard device_guard(x.device());
    int64_t x_stride = static_cast<int64_t>(channels) * height * width;
    int64_t out_stride = static_cast<int64_t>(tokens) * d;
    for (int b = 0; b < batch; ++b) {
        check_status(
            wm_cuda_patchify_f32(
                x.data_ptr<float>() + b * x_stride, weight.data_ptr<float>(),
                out.data_ptr<float>() + b * out_stride,
                channels, height, width, d, ph, pw, token_h, token_w),
            "patchify");
    }
    return out;
}

torch::Tensor patchify_cutlass_cuda(torch::Tensor x, torch::Tensor weight) {
    check_tensor(x, at::ScalarType::Float, "x");
    check_tensor(weight, at::ScalarType::Float, "weight");
    check_same_device(x, weight, "x", "weight");
    TORCH_CHECK(x.dim() == 4 && weight.dim() == 4, "x/weight must be NCHW/OIHW");
    int batch = static_cast<int>(x.size(0));
    int channels = static_cast<int>(x.size(1));
    int height = static_cast<int>(x.size(2));
    int width = static_cast<int>(x.size(3));
    int d = static_cast<int>(weight.size(0));
    int ph = static_cast<int>(weight.size(2));
    int pw = static_cast<int>(weight.size(3));
    TORCH_CHECK(weight.size(1) == channels && height % ph == 0 && width % pw == 0,
                "patchify shape mismatch");
    int token_h = height / ph;
    int token_w = width / pw;
    int tokens = token_h * token_w;
    int patch_elems = channels * ph * pw;
    auto rows = torch::empty({batch, tokens, patch_elems}, x.options());
    auto out = torch::empty({batch, tokens, d}, x.options());
    c10::cuda::CUDAGuard device_guard(x.device());
    int64_t x_stride = static_cast<int64_t>(channels) * height * width;
    int64_t row_stride = static_cast<int64_t>(tokens) * patch_elems;
    int64_t out_stride = static_cast<int64_t>(tokens) * d;
    for (int b = 0; b < batch; ++b) {
        check_status(
            wm_cuda_patchify_im2row_f32(
                x.data_ptr<float>() + b * x_stride,
                rows.data_ptr<float>() + b * row_stride,
                channels, height, width, ph, pw, token_h, token_w),
            "patchify_im2row");
        check_status(
            wm_cuda_linear_f32(
                rows.data_ptr<float>() + b * row_stride, weight.data_ptr<float>(),
                out.data_ptr<float>() + b * out_stride, tokens, patch_elems, d),
            "patchify_linear");
    }
    return out;
}

torch::Tensor unpatchify_cuda(
        torch::Tensor tokens,
        torch::Tensor weight,
        torch::Tensor bias,
        int64_t channels,
        int64_t height,
        int64_t width,
        int64_t patch_h,
        int64_t patch_w) {
    check_tensor(tokens, at::ScalarType::Float, "tokens");
    check_tensor(weight, at::ScalarType::Float, "weight");
    check_tensor(bias, at::ScalarType::Float, "bias");
    TORCH_CHECK(tokens.dim() == 3 && weight.dim() == 4 && bias.dim() == 1,
                "tokens/weight/bias must be [B,T,D], [D,C,ph,pw], [C]");
    int batch = static_cast<int>(tokens.size(0));
    int token_count = static_cast<int>(tokens.size(1));
    int d = static_cast<int>(tokens.size(2));
    int c = static_cast<int>(channels);
    int h = static_cast<int>(height);
    int w = static_cast<int>(width);
    int ph = static_cast<int>(patch_h);
    int pw = static_cast<int>(patch_w);
    TORCH_CHECK(c > 0 && h % ph == 0 && w % pw == 0, "invalid output shape");
    int token_w = w / pw;
    int out_dim = c * ph * pw;
    TORCH_CHECK(token_count == (h / ph) * token_w, "token count mismatch");
    TORCH_CHECK(
        weight.size(0) == d && weight.size(1) == c &&
        weight.size(2) == ph && weight.size(3) == pw && bias.numel() == c,
                "weight/bias shape mismatch");
    auto out = torch::empty({batch, c, h, w}, tokens.options());
    c10::cuda::CUDAGuard device_guard(tokens.device());
    int64_t token_stride = static_cast<int64_t>(token_count) * d;
    int64_t out_stride = static_cast<int64_t>(c) * h * w;
    for (int b = 0; b < batch; ++b) {
        check_status(
            wm_cuda_unpatchify_f32(
                tokens.data_ptr<float>() + b * token_stride,
                weight.data_ptr<float>(), bias.data_ptr<float>(),
                out.data_ptr<float>() + b * out_stride,
                token_count, d, c, h, w, ph, pw, token_w, out_dim),
            "unpatchify");
    }
    return out;
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("silu", &silu_cuda);
    m.def("silu_to_half", &silu_to_half_cuda);
    m.def("row_major_linear_fp16", &row_major_linear_fp16_cuda);
    m.def("row_major_linear_fp16_tensorop", &row_major_linear_fp16_tensorop_cuda);
    m.def("row_major_linear_fp16_tensorop_m64n64", &row_major_linear_fp16_tensorop_m64n64_cuda);
    m.def("row_major_linear_fp16_input_tensorop_m64n64", &row_major_linear_fp16_input_tensorop_m64n64_cuda);
    m.def("row_major_linear_fp16_tensorop_m64n64_silu_half", &row_major_linear_fp16_tensorop_m64n64_silu_half_cuda);
    m.def("row_major_linear_fp16_input_tensorop_m64n64_silu_half", &row_major_linear_fp16_input_tensorop_m64n64_silu_half_cuda);
    m.def("row_major_linear_fp16_tensorop_splitk", &row_major_linear_fp16_tensorop_splitk_cuda,
          py::arg("x"), py::arg("w"), py::arg("split_k_slices"));
    m.def("row_major_linear_fp16_input_tensorop_splitk", &row_major_linear_fp16_input_tensorop_splitk_cuda,
          py::arg("x"), py::arg("w"), py::arg("split_k_slices"));
    m.def("row_major_linear_fp16_input_tensorop_splitk_parallel", &row_major_linear_fp16_input_tensorop_splitk_parallel_cuda,
          py::arg("x"), py::arg("w"), py::arg("split_k_slices"));
    m.def("rms_norm", &rms_norm_cuda, py::arg("x"), py::arg("eps") = 1.0e-6);
    m.def("ada_rms_norm", &ada_rms_norm_cuda,
          py::arg("x"), py::arg("scale"), py::arg("bias"), py::arg("eps") = 1.0e-6);
    m.def("ada_rms_norm_half", &ada_rms_norm_half_cuda,
          py::arg("x"), py::arg("scale"), py::arg("bias"), py::arg("eps") = 1.0e-6);
    m.def("qkv_rms_rope", &qkv_rms_rope_cuda,
          py::arg("qkv"), py::arg("x_pos"), py::arg("y_pos"), py::arg("t_pos"),
          py::arg("xy"), py::arg("inv_t"), py::arg("n_heads"), py::arg("n_kv_heads"),
          py::arg("width"), py::arg("height"), py::arg("eps") = 1.0e-6);
    m.def("indexed_attention", &indexed_attention_cuda);
    m.def("indexed_attention_flash", &indexed_attention_flash_cuda);
    m.def("indexed_attention_half_kv", &indexed_attention_half_kv_cuda);
    m.def("indexed_attention_half_kv_flash", &indexed_attention_half_kv_flash_cuda);
    m.def("indexed_attention_half_kv_cutlass", &indexed_attention_half_kv_cutlass_cuda);
    m.def("indexed_attention_half_kv_cutlass_grouped", &indexed_attention_half_kv_cutlass_grouped_cuda);
    m.def("indexed_attention_half_kv_fmha_gqa", &indexed_attention_half_kv_fmha_gqa_cuda);
    m.def("indexed_attention_half_kv_fmha_gqa_half_output", &indexed_attention_half_kv_fmha_gqa_half_output_cuda);
    m.def("sparse_attention_half_kv_fmha_gqa", &sparse_attention_half_kv_fmha_gqa_cuda);
    m.def("sparse_attention_half_kv_fmha_gqa_half_output", &sparse_attention_half_kv_fmha_gqa_half_output_cuda);
    m.def("kv_cache_upsert", &kv_cache_upsert_cuda);
    m.def("kv_cache_upsert_half", &kv_cache_upsert_half_cuda);
    m.def("cache_frame_indices", &cache_frame_indices_cuda);
    m.def("patchify", &patchify_cuda);
    m.def("patchify_cutlass", &patchify_cutlass_cuda);
    m.def("unpatchify", &unpatchify_cuda);
}
