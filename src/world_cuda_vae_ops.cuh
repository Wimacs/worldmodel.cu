#ifndef WORLD_CUDA_VAE_OPS_CUH
#define WORLD_CUDA_VAE_OPS_CUH

#include <cuda_fp16.h>

#include <stdint.h>

// Backend-private descriptor for one VAE convolution. The VAE runtime owns
// every pointer; operators only consume the descriptor for a single launch.
typedef struct {
    float *weight;
    float *bias;
    __half *weight_krsc_h;
    __half *bias_h;
    int out_c;
    int in_c;
    int kernel;
    int has_bias;
} WmCudaVaeConvDesc;

int wm_cuda_vae_copy_latent_clamp_f32(
        const float *latent, float *out, int64_t n);
int wm_cuda_vae_copy_latent_clamp_nhwc_f16(
        const float *latent, __half *out, int channels, int height, int width);

int wm_cuda_vae_relu_f32(float *x, int64_t n);
int wm_cuda_vae_relu_f16(__half *x, int64_t n);
int wm_cuda_vae_add_relu_f32(
        const float *a, const float *b, float *out, int64_t n);
int wm_cuda_vae_add_relu_f16(
        const __half *a, const __half *b, __half *out, int64_t n);

int wm_cuda_vae_conv_direct_nchw_f32(
        const float *in,
        float *out,
        const WmCudaVaeConvDesc *conv,
        int batch,
        int height,
        int width);
int wm_cuda_vae_conv_stride2_nchw_f32(
        const float *in,
        float *out,
        const WmCudaVaeConvDesc *conv,
        int batch,
        int height,
        int width);
int wm_cuda_vae_conv_gemm_f32(
        const float *weight,
        const float *cols,
        float *out,
        int out_channels,
        int column_count,
        int reduction_size,
        int out_stride);
int wm_cuda_vae_conv_nhwc_f16(
        const __half *in,
        __half *out,
        const WmCudaVaeConvDesc *conv,
        int batch,
        int height,
        int width);

int wm_cuda_vae_add_bias_nchw_f32(
        float *out, const float *bias, int batch, int channels, int height, int width);
int wm_cuda_vae_add_bias_nhwc_f16(
        __half *out, const __half *bias, int64_t n, int channels);
int wm_cuda_vae_im2col3x3_nchw_tile_f32(
        const float *in,
        float *cols,
        int channels,
        int height,
        int width,
        int frame,
        int tile_start,
        int tile_cols);
int wm_cuda_vae_im2col3x3_nchw_batch_tile_f32(
        const float *in,
        float *cols,
        int batch,
        int channels,
        int height,
        int width,
        int tile_start,
        int tile_cols);
int wm_cuda_vae_scatter_conv_tile_nchw_f32(
        const float *tile,
        float *out,
        int batch,
        int channels,
        int height,
        int width,
        int tile_start,
        int tile_cols);

int wm_cuda_vae_concat_memory_nchw_f32(
        const float *cur, const float *mem, float *out, int channels, int height, int width);
int wm_cuda_vae_concat_memory_nhwc_f16(
        const __half *cur, const __half *mem, __half *out, int channels, int height, int width);
int wm_cuda_vae_concat_past_nchw_f32(
        const float *in, float *out, int batch, int channels, int height, int width);

int wm_cuda_vae_upsample2_nchw_f32(
        const float *in, float *out, int batch, int channels, int height, int width);
int wm_cuda_vae_upsample2_nhwc_f16(
        const __half *in, __half *out, int batch, int channels, int height, int width);
int wm_cuda_vae_tgrow_reshape_nchw_f32(
        const float *in,
        float *out,
        int batch,
        int channels,
        int height,
        int width,
        int stride);
int wm_cuda_vae_tgrow_reshape_nhwc_f16(
        const __half *in,
        __half *out,
        int batch,
        int channels,
        int height,
        int width,
        int stride);
int wm_cuda_vae_pixel_shuffle_u8_nchw_f32(
        const float *in, unsigned char *rgb, int height, int width);
int wm_cuda_vae_pixel_shuffle_u8_nhwc_f16(
        const __half *in, unsigned char *rgb, int height, int width);

#endif
