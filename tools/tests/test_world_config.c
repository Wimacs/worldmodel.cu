#include "world_config.h"

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <process.h>
#define TEST_GETPID _getpid
#else
#include <unistd.h>
#define TEST_GETPID getpid
#endif

static int failures = 0;
static unsigned int file_sequence = 0;

#define CHECK(condition, message) do { \
        if (!(condition)) { \
            fprintf(stderr, "FAIL: %s (line %d)\n", (message), __LINE__); \
            failures++; \
        } \
    } while (0)

static int load_text(const char *text, WorldConfig *cfg) {
    char path[160];
    FILE *f;
    size_t length = strlen(text);
    int rc;
    snprintf(path,
             sizeof(path),
             "world_config_test_%lu_%u.yaml",
             (unsigned long)TEST_GETPID(),
             file_sequence++);
    f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "could not create test config %s\n", path);
        failures++;
        return 1;
    }
    if (fwrite(text, 1, length, f) != length || fclose(f) != 0) {
        fprintf(stderr, "could not write test config %s\n", path);
        remove(path);
        failures++;
        return 1;
    }
    rc = world_config_load(cfg, path);
    remove(path);
    return rc;
}

static void expect_invalid(const char *name, const char *text) {
    WorldConfig cfg;
    if (load_text(text, &cfg) == 0) {
        fprintf(stderr, "FAIL: accepted invalid config: %s\n", name);
        failures++;
    }
}

static void test_valid_inline_config(void) {
    static const char text[] =
        "# ordinary unknown YAML remains compatible\n"
        "metadata:\n"
        "  height: this-is-not-a-top-level-field\n"
        "height: 4 # inline comment\n"
        "width: 8\n"
        "tokens_per_frame: 32\n"
        "patch: [2, 3]\n"
        "scheduler_sigmas: [1.0, 0.5, 0.0]\n"
        "value_residual: true\n"
        "prompt_conditioning: null\n"
        "new_model_option: ignored\n";
    WorldConfig cfg;
    CHECK(load_text(text, &cfg) == 0, "valid inline config should load");
    CHECK(cfg.height == 4 && cfg.width == 8, "integer fields should be parsed");
    CHECK(cfg.patch_h == 2 && cfg.patch_w == 3, "inline patch should be parsed");
    CHECK(cfg.scheduler_sigmas_count == 3, "inline schedule length should be parsed");
    CHECK(cfg.scheduler_sigmas[1] == 0.5f, "inline schedule values should be parsed");
    CHECK(cfg.value_residual == 1 && cfg.prompt_conditioning == 0, "booleans and null should be parsed");
}

static void test_valid_block_config(void) {
    static const char text[] =
        "patch: # block form\n"
        "  - 1\n"
        "  - 2\n"
        "scheduler_sigmas:\n"
        "- 1e0\n"
        "- 2.5e-1 # comment\n"
        "- 0\n"
        "value_residual: false\n"
        "prompt_conditioning: true\n";
    WorldConfig cfg;
    CHECK(load_text(text, &cfg) == 0, "valid block config should load");
    CHECK(cfg.patch_h == 1 && cfg.patch_w == 2, "block patch should be parsed");
    CHECK(cfg.scheduler_sigmas_count == 3, "block schedule should be parsed");
    CHECK(cfg.prompt_conditioning == 1, "true should enable optional feature");
}

static void test_loaded_schedule_fallback(void) {
    WorldConfig cfg;
    CHECK(load_text("height: 16\nwidth: 32\n", &cfg) == 0, "schedule omission should use safe fallback");
    CHECK(cfg.scheduler_sigmas_count == 2 && cfg.scheduler_sigmas[0] == 1.0f &&
              cfg.scheduler_sigmas[1] == 0.0f,
          "loaded config fallback schedule should be [1, 0]");
}

static void test_bad_tokens(void) {
    expect_invalid("integer suffix", "height: 16pixels\n");
    expect_invalid("integer overflow", "height: 999999999999999999999999\n");
    expect_invalid("invalid boolean", "value_residual: truthy\n");
    expect_invalid("boolean numeric alias", "prompt_conditioning: 1\n");
    expect_invalid("duplicate scalar", "height: 16\nheight: 16\n");
    expect_invalid("tokens_per_frame mismatch", "height: 4\nwidth: 8\ntokens_per_frame: 31\n");
}

static void test_bad_lists(void) {
    expect_invalid("short patch", "patch:\n- 2\nheight: 16\n");
    expect_invalid("long patch", "patch:\n- 2\n- 2\n- 2\n");
    expect_invalid("bad patch item", "patch: [2, nope]\n");
    expect_invalid("empty schedule", "scheduler_sigmas: []\n");
    expect_invalid("short schedule", "scheduler_sigmas:\n- 1\n");
    expect_invalid("non-finite schedule", "scheduler_sigmas: [1, nan, 0]\n");
    expect_invalid("negative schedule", "scheduler_sigmas: [1, -0.1, 0]\n");
    expect_invalid("increasing schedule", "scheduler_sigmas: [0.5, 0.75, 0]\n");
    expect_invalid(
        "schedule capacity overflow",
        "scheduler_sigmas: [32,31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,"
        "16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0]\n");
}

static void test_long_line(void) {
    char text[800];
    size_t i;
    WorldConfig cfg;
    memcpy(text, "unknown: ", 9);
    for (i = 9; i < sizeof(text) - 2; ++i) text[i] = 'x';
    text[sizeof(text) - 2] = '\n';
    text[sizeof(text) - 1] = 0;
    CHECK(load_text(text, &cfg) != 0, "overlong config line must be rejected");
}

static void test_validation_api(void) {
    WorldConfig cfg;
    world_config_defaults(&cfg);
    CHECK(world_config_validate(&cfg) == 0, "defaults must validate");

    cfg.n_heads = 0;
    CHECK(world_config_validate(&cfg) != 0, "zero divisor must fail validation");
    world_config_defaults(&cfg);
    cfg.d_model++;
    CHECK(world_config_validate(&cfg) != 0, "non-divisible model dimension must fail validation");
    world_config_defaults(&cfg);
    cfg.global_window++;
    CHECK(world_config_validate(&cfg) != 0, "pinned window mismatch must fail validation");
    world_config_defaults(&cfg);
    cfg.height = INT_MAX;
    CHECK(world_config_validate(&cfg) != 0, "derived int overflow must fail validation");
    world_config_defaults(&cfg);
    cfg.scheduler_sigmas[1] = NAN;
    CHECK(world_config_validate(&cfg) != 0, "non-finite direct schedule must fail validation");
    world_config_defaults(&cfg);
    cfg.d_model = 8;
    cfg.n_heads = 1;
    cfg.n_kv_heads = 1;
    cfg.mlp_ratio = INT_MAX / 8;
    cfg.n_layers = INT_MAX;
    CHECK(world_config_validate(&cfg) != 0, "multi-layer MLP address overflow must fail validation");
    CHECK(world_config_validate(NULL) != 0, "null config must fail validation");
}

int main(void) {
    test_valid_inline_config();
    test_valid_block_config();
    test_loaded_schedule_fallback();
    test_bad_tokens();
    test_bad_lists();
    test_long_line();
    test_validation_api();
    if (failures) {
        fprintf(stderr, "%d world config test(s) failed\n", failures);
        return 1;
    }
    fprintf(stderr, "world config tests passed\n");
    return 0;
}
