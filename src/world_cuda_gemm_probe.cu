#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define CUDA_OK(expr) do { \
    cudaError_t _e = (expr); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        return 1; \
    } \
} while (0)

#define CUTLASS_OK(expr) do { \
    cutlass::Status _s = (expr); \
    if (_s != cutlass::Status::kSuccess) { \
        fprintf(stderr, "CUTLASS error %s:%d: %s\n", __FILE__, __LINE__, cutlassGetStatusString(_s)); \
        return 1; \
    } \
} while (0)

static int div_up_i64(int64_t a, int b) {
    return (int)((a + b - 1) / b);
}

__global__ static void fill_half_pattern_kernel(__half *x, int64_t n, uint32_t seed) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t v = (uint32_t)i ^ seed;
    v = v * 1664525u + 1013904223u;
    float f = ((float)(v & 2047u) / 1024.0f - 1.0f) * 0.25f;
    x[i] = __float2half_rn(f);
}

__global__ static void zero_f32_kernel(float *x, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = 0.0f;
}

static int run_simt_shape(int m, int k, int n) {
    __half *a = NULL;
    __half *b = NULL;
    float *c = NULL;
    size_t a_elems = (size_t)m * k;
    size_t b_elems = (size_t)n * k;
    size_t c_elems = (size_t)m * n;

    fprintf(stderr, "simt probe shape: M=%d K=%d N=%d\n", m, k, n);
    CUDA_OK(cudaMalloc((void **)&a, a_elems * sizeof(__half)));
    CUDA_OK(cudaMalloc((void **)&b, b_elems * sizeof(__half)));
    CUDA_OK(cudaMalloc((void **)&c, c_elems * sizeof(float)));
    fill_half_pattern_kernel<<<div_up_i64((int64_t)a_elems, 256), 256>>>(a, (int64_t)a_elems, 0x1234u);
    CUDA_OK(cudaGetLastError());
    fill_half_pattern_kernel<<<div_up_i64((int64_t)b_elems, 256), 256>>>(b, (int64_t)b_elems, 0x5678u);
    CUDA_OK(cudaGetLastError());
    zero_f32_kernel<<<div_up_i64((int64_t)c_elems, 256), 256>>>(c, (int64_t)c_elems);
    CUDA_OK(cudaGetLastError());

    using Gemm = cutlass::gemm::device::Gemm<
        cutlass::half_t,
        cutlass::layout::RowMajor,
        cutlass::half_t,
        cutlass::layout::ColumnMajor,
        float,
        cutlass::layout::RowMajor,
        float,
        cutlass::arch::OpClassSimt,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 64, 8>,
        cutlass::gemm::GemmShape<32, 64, 8>,
        cutlass::gemm::GemmShape<1, 1, 1>,
        cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2,
        1,
        1>;

    const cutlass::half_t *a_h = reinterpret_cast<const cutlass::half_t *>(a);
    const cutlass::half_t *b_h = reinterpret_cast<const cutlass::half_t *>(b);
    typename Gemm::Arguments args(
        {m, n, k},
        {a_h, k},
        {b_h, k},
        {c, n},
        {c, n},
        {1.0f, 0.0f});
    Gemm gemm;
    CUTLASS_OK(gemm(args));
    CUDA_OK(cudaDeviceSynchronize());

    float sample[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int copy_n = c_elems < 4 ? (int)c_elems : 4;
    CUDA_OK(cudaMemcpy(sample, c, (size_t)copy_n * sizeof(float), cudaMemcpyDeviceToHost));
    fprintf(stderr, "  ok sample=[%.6f %.6f %.6f %.6f]\n", sample[0], sample[1], sample[2], sample[3]);

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    return 0;
}

static int run_tensorop_shape(int m, int k, int n) {
    __half *a = NULL;
    __half *b = NULL;
    float *c = NULL;
    size_t a_elems = (size_t)m * k;
    size_t b_elems = (size_t)n * k;
    size_t c_elems = (size_t)m * n;

    fprintf(stderr, "tensorop probe shape: M=%d K=%d N=%d\n", m, k, n);
    CUDA_OK(cudaMalloc((void **)&a, a_elems * sizeof(__half)));
    CUDA_OK(cudaMalloc((void **)&b, b_elems * sizeof(__half)));
    CUDA_OK(cudaMalloc((void **)&c, c_elems * sizeof(float)));
    fill_half_pattern_kernel<<<div_up_i64((int64_t)a_elems, 256), 256>>>(a, (int64_t)a_elems, 0x1234u);
    CUDA_OK(cudaGetLastError());
    fill_half_pattern_kernel<<<div_up_i64((int64_t)b_elems, 256), 256>>>(b, (int64_t)b_elems, 0x5678u);
    CUDA_OK(cudaGetLastError());
    zero_f32_kernel<<<div_up_i64((int64_t)c_elems, 256), 256>>>(c, (int64_t)c_elems);
    CUDA_OK(cudaGetLastError());

    using Gemm = cutlass::gemm::device::Gemm<
        cutlass::half_t,
        cutlass::layout::RowMajor,
        cutlass::half_t,
        cutlass::layout::ColumnMajor,
        float,
        cutlass::layout::RowMajor,
        float,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 128, 32>,
        cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<
            float,
            128 / cutlass::sizeof_bits<float>::value,
            float,
            float>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        4>;

    const cutlass::half_t *a_h = reinterpret_cast<const cutlass::half_t *>(a);
    const cutlass::half_t *b_h = reinterpret_cast<const cutlass::half_t *>(b);
    typename Gemm::Arguments args(
        {m, n, k},
        {a_h, k},
        {b_h, k},
        {c, n},
        {c, n},
        {1.0f, 0.0f});
    Gemm gemm;
    CUTLASS_OK(gemm(args));
    CUDA_OK(cudaDeviceSynchronize());

    float sample[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int copy_n = c_elems < 4 ? (int)c_elems : 4;
    CUDA_OK(cudaMemcpy(sample, c, (size_t)copy_n * sizeof(float), cudaMemcpyDeviceToHost));
    fprintf(stderr, "  ok sample=[%.6f %.6f %.6f %.6f]\n", sample[0], sample[1], sample[2], sample[3]);

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    return 0;
}

int main(int argc, char **argv) {
    int small_only = 0;
    int use_tensorop = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--small") == 0) {
            small_only = 1;
        } else if (strcmp(argv[i], "--tensorop") == 0) {
            use_tensorop = 1;
        } else {
            fprintf(stderr, "usage: %s [--small] [--tensorop]\n", argv[0]);
            return 1;
        }
    }

    int (*run_shape)(int, int, int) = use_tensorop ? run_tensorop_shape : run_simt_shape;
    if (run_shape(1, 32, 8)) return 1;
    if (small_only) return 0;
    if (run_shape(1, 2048, 8192)) return 1;
    if (run_shape(512, 2048, 2048)) return 1;
    if (run_shape(512, 2048, 4096)) return 1;
    if (run_shape(512, 2048, 8192)) return 1;
    if (run_shape(512, 8192, 2048)) return 1;
    return 0;
}
