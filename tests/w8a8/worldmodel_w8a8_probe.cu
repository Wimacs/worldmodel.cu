#include <cuda_runtime.h>

#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination_clamp.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check_cuda(cudaError_t status, char const *what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
  }
}

void check_cutlass(cutlass::Status status, char const *what) {
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cutlassGetStatusString(status));
  }
}

template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(size_t count = 0) : count_(count) {
    if (count_) check_cuda(cudaMalloc(reinterpret_cast<void **>(&ptr_), count_ * sizeof(T)), "cudaMalloc");
  }

  ~DeviceBuffer() {
    if (ptr_) cudaFree(ptr_);
  }

  DeviceBuffer(DeviceBuffer const &) = delete;
  DeviceBuffer &operator=(DeviceBuffer const &) = delete;

  T *get() { return ptr_; }
  T const *get() const { return ptr_; }
  size_t size() const { return count_; }

 private:
  T *ptr_ = nullptr;
  size_t count_ = 0;
};

__global__ void quantize_symmetric_rows_s8_kernel(
    float const *input,
    int8_t *quantized,
    float *scales,
    int rows,
    int cols) {
  int row = static_cast<int>(blockIdx.x);
  int tid = static_cast<int>(threadIdx.x);
  if (row >= rows) return;

  float local_max = 0.0f;
  int64_t row_offset = static_cast<int64_t>(row) * cols;
  for (int col = tid; col < cols; col += static_cast<int>(blockDim.x)) {
    local_max = fmaxf(local_max, fabsf(input[row_offset + col]));
  }

  extern __shared__ float reduced[];
  reduced[tid] = local_max;
  __syncthreads();
  for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride >>= 1) {
    if (tid < stride) reduced[tid] = fmaxf(reduced[tid], reduced[tid + stride]);
    __syncthreads();
  }

  if (tid == 0) {
    // A unit scale makes an all-zero row deterministic and avoids a divide by zero.
    scales[row] = reduced[0] > 0.0f ? reduced[0] / 127.0f : 1.0f;
  }
  __syncthreads();

  float inv_scale = 1.0f / scales[row];
  for (int col = tid; col < cols; col += static_cast<int>(blockDim.x)) {
    int q = __float2int_rn(input[row_offset + col] * inv_scale);
    q = max(-127, min(127, q));
    quantized[row_offset + col] = static_cast<int8_t>(q);
  }
}

__global__ void dequantize_outer_scales_f32_kernel(
    int32_t const *accumulators,
    float const *activation_scales,
    float const *weight_scales,
    float *output,
    int rows,
    int cols) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t count = static_cast<int64_t>(rows) * cols;
  if (idx >= count) return;
  int row = static_cast<int>(idx / cols);
  int col = static_cast<int>(idx - static_cast<int64_t>(row) * cols);
  output[idx] = static_cast<float>(accumulators[idx]) * activation_scales[row] * weight_scales[col];
}

using W8A8Gemm = cutlass::gemm::device::Gemm<
    int8_t,
    cutlass::layout::RowMajor,
    int8_t,
    cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::layout::RowMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 128>,
    cutlass::gemm::GemmShape<64, 64, 128>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    cutlass::epilogue::thread::LinearCombinationClamp<int32_t, 4, int32_t, int32_t>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3>;

struct ProbeCase {
  char const *name;
  int m;
  int k;
  int n;
  bool zero_edge_rows;
};

struct Options {
  std::string selected_case = "quick";
  int custom_m = 0;
  int custom_k = 0;
  int custom_n = 0;
  int samples = 256;
  int bench_iterations = 0;
  float min_sqnr_db = 30.0f;
};

std::vector<ProbeCase> const kCases = {
    {"sanity", 32, 128, 64, true},
    {"out360", 128, 2048, 2048, false},
    {"qkv360", 128, 2048, 4096, false},
    {"fc1_360", 128, 2048, 8192, false},
    {"fc2_360", 128, 8192, 2048, false},
    {"out_main", 512, 2048, 2048, false},
    {"qkv_main", 512, 2048, 4096, false},
    {"fc1_main", 512, 2048, 8192, false},
    {"fc2_main", 512, 8192, 2048, false},
};

void print_usage(char const *argv0) {
  std::fprintf(
      stderr,
      "usage: %s [--case quick|all|NAME] [--m M --k K --n N] [--samples N] "
      "[--bench N] [--min-sqnr DB]\n",
      argv0);
}

Options parse_options(int argc, char **argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto require_value = [&](char const *name) -> char const * {
      if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
      return argv[++i];
    };
    if (arg == "--case") {
      options.selected_case = require_value("--case");
    } else if (arg == "--m") {
      options.custom_m = std::atoi(require_value("--m"));
    } else if (arg == "--k") {
      options.custom_k = std::atoi(require_value("--k"));
    } else if (arg == "--n") {
      options.custom_n = std::atoi(require_value("--n"));
    } else if (arg == "--samples") {
      options.samples = std::atoi(require_value("--samples"));
    } else if (arg == "--bench") {
      options.bench_iterations = std::atoi(require_value("--bench"));
    } else if (arg == "--min-sqnr") {
      options.min_sqnr_db = std::strtof(require_value("--min-sqnr"), nullptr);
    } else if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  if (options.samples <= 0) throw std::runtime_error("--samples must be positive");
  if (options.bench_iterations < 0) throw std::runtime_error("--bench must be non-negative");
  bool any_custom = options.custom_m || options.custom_k || options.custom_n;
  bool all_custom = options.custom_m > 0 && options.custom_k > 0 && options.custom_n > 0;
  if (any_custom && !all_custom) throw std::runtime_error("--m, --k, and --n must be supplied together");
  if (all_custom && (options.custom_k % 16) != 0) {
    throw std::runtime_error("custom K must be divisible by 16 for aligned INT8 Tensor Core loads");
  }
  return options;
}

std::vector<ProbeCase> select_cases(Options const &options) {
  if (options.custom_m > 0) {
    return {{"custom", options.custom_m, options.custom_k, options.custom_n, false}};
  }
  if (options.selected_case == "all") return kCases;
  if (options.selected_case == "quick") {
    return {kCases[0], kCases[2], kCases[4]};
  }
  for (ProbeCase const &probe : kCases) {
    if (options.selected_case == probe.name) return {probe};
  }
  throw std::runtime_error("unknown --case: " + options.selected_case);
}

void fill_inputs(ProbeCase const &probe, std::vector<float> &x, std::vector<float> &w) {
  std::mt19937 rng(0x57a8u + static_cast<unsigned>(probe.k));
  std::normal_distribution<float> normal(0.0f, 1.0f);
  for (int row = 0; row < probe.m; ++row) {
    float row_scale = std::exp2(0.25f * static_cast<float>((row % 7) - 3));
    for (int col = 0; col < probe.k; ++col) {
      x[static_cast<int64_t>(row) * probe.k + col] = normal(rng) * row_scale;
    }
  }
  for (int row = 0; row < probe.n; ++row) {
    float row_scale = std::exp2(0.20f * static_cast<float>((row % 9) - 4));
    for (int col = 0; col < probe.k; ++col) {
      w[static_cast<int64_t>(row) * probe.k + col] = normal(rng) * row_scale;
    }
  }

  if (probe.m > 1) x[static_cast<int64_t>(probe.k) + probe.k / 3] *= 10.0f;
  if (probe.n > 1) w[static_cast<int64_t>(probe.k) + probe.k / 5] *= -10.0f;
  if (probe.zero_edge_rows) {
    std::fill(x.begin(), x.begin() + probe.k, 0.0f);
    std::fill(w.begin(), w.begin() + probe.k, 0.0f);
  }
}

float measure_ms(std::function<void()> const &fn, int iterations) {
  for (int i = 0; i < 3; ++i) fn();
  check_cuda(cudaDeviceSynchronize(), "benchmark warmup");
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
  check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");
  check_cuda(cudaEventRecord(start), "cudaEventRecord(start)");
  for (int i = 0; i < iterations; ++i) fn();
  check_cuda(cudaEventRecord(stop), "cudaEventRecord(stop)");
  check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
  float elapsed = 0.0f;
  check_cuda(cudaEventElapsedTime(&elapsed, start, stop), "cudaEventElapsedTime");
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return elapsed / static_cast<float>(iterations);
}

bool run_probe(ProbeCase const &probe, Options const &options) {
  if ((probe.k % 16) != 0) {
    throw std::runtime_error(std::string(probe.name) + ": K must be divisible by 16");
  }
  int64_t x_count = static_cast<int64_t>(probe.m) * probe.k;
  int64_t w_count = static_cast<int64_t>(probe.n) * probe.k;
  int64_t y_count = static_cast<int64_t>(probe.m) * probe.n;
  std::vector<float> h_x(static_cast<size_t>(x_count));
  std::vector<float> h_w(static_cast<size_t>(w_count));
  fill_inputs(probe, h_x, h_w);

  DeviceBuffer<float> d_x(static_cast<size_t>(x_count));
  DeviceBuffer<float> d_w(static_cast<size_t>(w_count));
  DeviceBuffer<int8_t> d_qx(static_cast<size_t>(x_count));
  DeviceBuffer<int8_t> d_qw(static_cast<size_t>(w_count));
  DeviceBuffer<float> d_x_scales(static_cast<size_t>(probe.m));
  DeviceBuffer<float> d_w_scales(static_cast<size_t>(probe.n));
  DeviceBuffer<int32_t> d_acc(static_cast<size_t>(y_count));
  DeviceBuffer<float> d_y(static_cast<size_t>(y_count));

  check_cuda(cudaMemcpy(d_x.get(), h_x.data(), static_cast<size_t>(x_count) * sizeof(float), cudaMemcpyHostToDevice), "copy x");
  check_cuda(cudaMemcpy(d_w.get(), h_w.data(), static_cast<size_t>(w_count) * sizeof(float), cudaMemcpyHostToDevice), "copy w");

  auto quantize_x = [&]() {
    quantize_symmetric_rows_s8_kernel<<<probe.m, 256, 256 * sizeof(float)>>>(
        d_x.get(), d_qx.get(), d_x_scales.get(), probe.m, probe.k);
    check_cuda(cudaGetLastError(), "quantize activation rows");
  };
  auto quantize_w = [&]() {
    quantize_symmetric_rows_s8_kernel<<<probe.n, 256, 256 * sizeof(float)>>>(
        d_w.get(), d_qw.get(), d_w_scales.get(), probe.n, probe.k);
    check_cuda(cudaGetLastError(), "quantize weight rows");
  };

  quantize_x();
  quantize_w();

  typename W8A8Gemm::Arguments args(
      {probe.m, probe.n, probe.k},
      {d_qx.get(), probe.k},
      {d_qw.get(), probe.k},
      {d_acc.get(), probe.n},
      {d_acc.get(), probe.n},
      {1, 0});
  W8A8Gemm gemm;
  check_cutlass(gemm.can_implement(args), "W8A8 GEMM can_implement");
  auto run_gemm = [&]() {
    check_cutlass(gemm(args), "W8A8 GEMM");
  };
  auto dequantize = [&]() {
    int blocks = static_cast<int>((y_count + 255) / 256);
    dequantize_outer_scales_f32_kernel<<<blocks, 256>>>(
        d_acc.get(), d_x_scales.get(), d_w_scales.get(), d_y.get(), probe.m, probe.n);
    check_cuda(cudaGetLastError(), "dequantize outer scales");
  };
  run_gemm();
  dequantize();
  check_cuda(cudaDeviceSynchronize(), "probe kernels");

  std::vector<int8_t> h_qx(static_cast<size_t>(x_count));
  std::vector<int8_t> h_qw(static_cast<size_t>(w_count));
  std::vector<float> h_x_scales(static_cast<size_t>(probe.m));
  std::vector<float> h_w_scales(static_cast<size_t>(probe.n));
  std::vector<int32_t> h_acc(static_cast<size_t>(y_count));
  std::vector<float> h_y(static_cast<size_t>(y_count));
  check_cuda(cudaMemcpy(h_qx.data(), d_qx.get(), static_cast<size_t>(x_count) * sizeof(int8_t), cudaMemcpyDeviceToHost), "copy qx");
  check_cuda(cudaMemcpy(h_qw.data(), d_qw.get(), static_cast<size_t>(w_count) * sizeof(int8_t), cudaMemcpyDeviceToHost), "copy qw");
  check_cuda(cudaMemcpy(h_x_scales.data(), d_x_scales.get(), static_cast<size_t>(probe.m) * sizeof(float), cudaMemcpyDeviceToHost), "copy x scales");
  check_cuda(cudaMemcpy(h_w_scales.data(), d_w_scales.get(), static_cast<size_t>(probe.n) * sizeof(float), cudaMemcpyDeviceToHost), "copy w scales");
  check_cuda(cudaMemcpy(h_acc.data(), d_acc.get(), static_cast<size_t>(y_count) * sizeof(int32_t), cudaMemcpyDeviceToHost), "copy accumulators");
  check_cuda(cudaMemcpy(h_y.data(), d_y.get(), static_cast<size_t>(y_count) * sizeof(float), cudaMemcpyDeviceToHost), "copy output");

  if (probe.zero_edge_rows) {
    if (h_x_scales[0] != 1.0f || h_w_scales[0] != 1.0f) {
      throw std::runtime_error(std::string(probe.name) + ": zero-row scale convention failed");
    }
    for (int col = 0; col < probe.k; ++col) {
      if (h_qx[col] != 0 || h_qw[col] != 0) {
        throw std::runtime_error(std::string(probe.name) + ": zero row quantized to a non-zero value");
      }
    }
  }

  int sample_count = std::min<int64_t>(options.samples, y_count);
  int accumulator_mismatches = 0;
  float max_dequant_error = 0.0f;
  int64_t max_accumulator_abs = 0;
  long double signal_square_sum = 0.0;
  long double error_square_sum = 0.0;
  for (int sample = 0; sample < sample_count; ++sample) {
    int64_t idx = sample_count == 1
        ? 0
        : static_cast<int64_t>(sample) * (y_count - 1) / (sample_count - 1);
    int row = static_cast<int>(idx / probe.n);
    int col = static_cast<int>(idx - static_cast<int64_t>(row) * probe.n);
    int64_t accumulator_ref = 0;
    long double fp_ref = 0.0;
    int64_t x_offset = static_cast<int64_t>(row) * probe.k;
    int64_t w_offset = static_cast<int64_t>(col) * probe.k;
    for (int inner = 0; inner < probe.k; ++inner) {
      accumulator_ref += static_cast<int32_t>(h_qx[x_offset + inner]) *
                         static_cast<int32_t>(h_qw[w_offset + inner]);
      fp_ref += static_cast<long double>(h_x[x_offset + inner]) * h_w[w_offset + inner];
    }
    max_accumulator_abs = std::max<int64_t>(max_accumulator_abs, std::llabs(accumulator_ref));
    if (accumulator_ref != h_acc[idx]) ++accumulator_mismatches;
    float quant_ref = static_cast<float>(accumulator_ref) * h_x_scales[row] * h_w_scales[col];
    max_dequant_error = std::max(max_dequant_error, std::fabs(h_y[idx] - quant_ref));
    long double error = static_cast<long double>(h_y[idx]) - fp_ref;
    signal_square_sum += fp_ref * fp_ref;
    error_square_sum += error * error;
  }

  long double sqnr_db = error_square_sum > 0.0
      ? 10.0L * std::log10(signal_square_sum / error_square_sum)
      : std::numeric_limits<long double>::infinity();
  size_t x_saturated = 0;
  size_t w_saturated = 0;
  for (int8_t value : h_qx) if (value == 127 || value == -127) ++x_saturated;
  for (int8_t value : h_qw) if (value == 127 || value == -127) ++w_saturated;

  std::printf(
      "%-10s M=%d K=%d N=%d samples=%d accum_mismatch=%d max_dequant_err=%.8g "
      "sqnr=%.2Lf dB max|acc|=%lld sat_x=%.5f%% sat_w=%.5f%%\n",
      probe.name,
      probe.m,
      probe.k,
      probe.n,
      sample_count,
      accumulator_mismatches,
      max_dequant_error,
      sqnr_db,
      static_cast<long long>(max_accumulator_abs),
      100.0 * static_cast<double>(x_saturated) / static_cast<double>(x_count),
      100.0 * static_cast<double>(w_saturated) / static_cast<double>(w_count));

  if (options.bench_iterations > 0) {
    float quant_ms = measure_ms(quantize_x, options.bench_iterations);
    float gemm_ms = measure_ms(run_gemm, options.bench_iterations);
    float dequant_ms = measure_ms(dequantize, options.bench_iterations);
    auto pipeline = [&]() {
      quantize_x();
      run_gemm();
      dequantize();
    };
    float pipeline_ms = measure_ms(pipeline, options.bench_iterations);
    std::printf(
        "  bench iterations=%d activation_quant=%.5fms gemm=%.5fms dequant=%.5fms pipeline=%.5fms\n",
        options.bench_iterations,
        quant_ms,
        gemm_ms,
        dequant_ms,
        pipeline_ms);
  }

  bool pass = accumulator_mismatches == 0 &&
              max_dequant_error <= 1.0e-4f &&
              !std::isnan(static_cast<double>(sqnr_db)) &&
              sqnr_db >= options.min_sqnr_db;
  if (!pass) {
    std::fprintf(
        stderr,
        "%s FAILED (required: exact sampled accumulators, max dequant error <= 1e-4, SQNR >= %.2f dB)\n",
        probe.name,
        options.min_sqnr_db);
  }
  return pass;
}

}  // namespace

int main(int argc, char **argv) {
  try {
    Options options = parse_options(argc, argv);
    int device = 0;
    check_cuda(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
    if (properties.major < 8) {
      throw std::runtime_error("this probe is compiled for the SM80+ INT8 Tensor Core path");
    }
    std::printf(
        "device=%s compute=%d.%d policy=symmetric-s8 activation=per-row weight=per-output-channel accumulator=s32\n",
        properties.name,
        properties.major,
        properties.minor);

    bool pass = true;
    for (ProbeCase const &probe : select_cases(options)) {
      pass = run_probe(probe, options) && pass;
    }
    std::printf("W8A8 probe: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
  } catch (std::exception const &error) {
    std::fprintf(stderr, "W8A8 probe error: %s\n", error.what());
    return 1;
  }
}
