#include "safetensors.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <process.h>
#define test_getpid _getpid
#else
#include <unistd.h>
#define test_getpid getpid
#endif

static int failures = 0;
static unsigned int fixture_id = 0;

#define CHECK(condition)                                                                                               \
    do {                                                                                                               \
        if (!(condition)) {                                                                                            \
            fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #condition);                         \
            failures++;                                                                                                \
        }                                                                                                              \
    } while (0)

static void fixture_path(char *path, size_t path_size) {
    snprintf(path, path_size, "safetensors_test_%ld_%u.tmp", (long)test_getpid(), fixture_id++);
}

static void write_u64_le(FILE *f, uint64_t value) {
    unsigned char bytes[8];
    for (int i = 0; i < 8; ++i) bytes[i] = (unsigned char)(value >> (8 * i));
    CHECK(fwrite(bytes, 1, sizeof(bytes), f) == sizeof(bytes));
}

static int write_fixture(const char *path, const void *header, size_t header_len,
                         const void *data, size_t data_len) {
    FILE *f = fopen(path, "wb");
    if (!f) return 1;
    write_u64_le(f, (uint64_t)header_len);
    int failed = fwrite(header, 1, header_len, f) != header_len;
    if (data_len != 0) failed |= fwrite(data, 1, data_len, f) != data_len;
    failed |= fclose(f) != 0;
    return failed;
}

static int write_declared_header_fixture(const char *path, uint64_t declared_header_len,
                                         const char *available_header) {
    FILE *f = fopen(path, "wb");
    if (!f) return 1;
    write_u64_le(f, declared_header_len);
    size_t len = strlen(available_header);
    int failed = fwrite(available_header, 1, len, f) != len;
    failed |= fclose(f) != 0;
    return failed;
}

static void expect_header(const char *header, size_t data_len, int should_open) {
    char path[128];
    fixture_path(path, sizeof(path));
    unsigned char data[32] = {0};
    CHECK(data_len <= sizeof(data));
    CHECK(write_fixture(path, header, strlen(header), data, data_len) == 0);

    SafeTensors st;
    int opened = safetensors_open(&st, path) == 0;
    CHECK(opened == should_open);
    if (opened) safetensors_close(&st);
    CHECK(remove(path) == 0);
}

static void test_valid_file_and_reads(void) {
    const char *header =
        "{\"__metadata__\":{\"format\":\"pt\",\"escaped\":\"line\\nvalue\"},"
        "\"scalar\":{\"data_offsets\":[8,10],\"shape\":[],\"dtype\":\"F16\"},"
        "\"ten\\u0073or\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]},"
        "\"empty\":{\"dtype\":\"F32\",\"shape\":[0,999],\"data_offsets\":[10,10]}}   ";
    float values[2] = {1.25f, -2.5f};
    unsigned char data[10];
    memcpy(data, values, sizeof(values));
    data[8] = 0;
    data[9] = 0x3c;

    char path[128];
    fixture_path(path, sizeof(path));
    CHECK(write_fixture(path, header, strlen(header), data, sizeof(data)) == 0);

    SafeTensors st;
    CHECK(safetensors_open(&st, path) == 0);
    CHECK(st.count == 3);
    CHECK(st.header_len == strlen(header));
    CHECK(st.data_offset == UINT64_C(8) + strlen(header));
    CHECK(st.file_size == st.data_offset + sizeof(data));

    const SafeTensorEntry *tensor = safetensors_find(&st, "tensor");
    const SafeTensorEntry *scalar = safetensors_find(&st, "scalar");
    const SafeTensorEntry *empty = safetensors_find(&st, "empty");
    CHECK(tensor && tensor->ndim == 1 && tensor->shape[0] == 2);
    CHECK(scalar && scalar->ndim == 0);
    CHECK(empty && empty->ndim == 2 && empty->shape[0] == 0);

    void *raw = NULL;
    size_t bytes = 0;
    CHECK(safetensors_read_tensor(&st, tensor, &raw, &bytes) == 0);
    CHECK(bytes == sizeof(values));
    CHECK(memcmp(raw, values, sizeof(values)) == 0);
    free(raw);

    float *converted = NULL;
    size_t elements = 0;
    CHECK(safetensors_read_tensor_f32(&st, scalar, &converted, &elements) == 0);
    CHECK(elements == 1 && converted[0] == 1.0f);
    free(converted);

    CHECK(safetensors_read_tensor_f32(&st, empty, &converted, &elements) == 0);
    CHECK(elements == 0 && converted != NULL);
    free(converted);

    SafeTensorEntry tampered = *tensor;
    tampered.end--;
    CHECK(safetensors_read_tensor(&st, &tampered, &raw, &bytes) != 0);
    tampered = *tensor;
    tampered.shape[0] = 3;
    CHECK(safetensors_read_tensor(&st, &tampered, &raw, &bytes) != 0);
    CHECK(safetensors_read_tensor(&st, tensor, NULL, &bytes) != 0);
    CHECK(safetensors_read_tensor(&st, tensor, &raw, NULL) != 0);

    FILE *truncated = fopen(path, "wb");
    CHECK(truncated != NULL);
    if (truncated) CHECK(fclose(truncated) == 0);
    CHECK(safetensors_read_tensor(&st, tensor, &raw, &bytes) != 0);

    safetensors_close(&st);
    CHECK(remove(path) == 0);
}

static void test_valid_empty_tensors(void) {
    expect_header("{\"zero\":{\"dtype\":\"U8\",\"shape\":[0],\"data_offsets\":[0,0]}}", 0, 1);
    expect_header("{}", 0, 1);
    expect_header("{\"__metadata__\":{\"format\":\"pt\"}} ", 0, 1);
}

static void test_malformed_headers(void) {
    static const struct {
        const char *header;
        size_t data_len;
    } cases[] = {
        {" {\"x\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]}}", 1},
        {"{\"x\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]}}\t", 1},
        {"{\"x\":{\"dtype\":\"BOGUS\",\"shape\":[1],\"data_offsets\":[0,1]}}", 1},
        {"{\"x\":{\"dtype\":\"U8\",\"shape\":[-1],\"data_offsets\":[0,1]}}", 1},
        {"{\"x\":{\"dtype\":\"U8\",\"shape\":[1,1,1,1,1,1,1,1,1],\"data_offsets\":[0,1]}}", 1},
        {"{\"x\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[-1,0]}}", 1},
        {"{\"x\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[1,0]}}", 1},
        {"{\"x\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1,1]}}", 1},
        {"{\"x\":{\"dtype\":\"U8\",\"shape\":[2],\"data_offsets\":[0,1]}}", 1},
        {"{\"x\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}", 4},
        {"{\"x\":{\"dtype\":\"F32\",\"shape\":[9223372036854775807,2],\"data_offsets\":[0,0]}}", 0},
        {"{\"x\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,18446744073709551616]}}", 1},
        {"{\"x\":{\"dtype\":\"U8\",\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]}}", 1},
        {"{\"x\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1],\"future\":1,\"future\":2}}", 1},
        {"{\"x\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]},"
         "\"x\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]}}", 1},
        {"{\"x\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]},}", 1},
        {"{\"__metadata__\":{\"not_a_string\":1}}", 0},
        {"{\"__metadata__\":{\"duplicate\":\"a\",\"duplicate\":\"b\"}}", 0},
        {"{\"__metadata__\":{},\"__metadata__\":{}}", 0},
        {"{\"a\":{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[0,4]},"
         "\"b\":{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[8,12]}}", 12},
        {"{\"a\":{\"dtype\":\"U8\",\"shape\":[8],\"data_offsets\":[0,8]},"
         "\"b\":{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[4,8]}}", 8},
        {"{\"a\":{\"dtype\":\"U8\",\"shape\":[10],\"data_offsets\":[0,10]},"
         "\"empty\":{\"dtype\":\"U8\",\"shape\":[0],\"data_offsets\":[5,5]}}", 10},
        {"{\"x\":{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[0,4]}}", 8},
        {"{}", 1},
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        expect_header(cases[i].header, cases[i].data_len, 0);
    }
}

static void test_invalid_header_lengths(void) {
    char path[128];
    SafeTensors st;

    fixture_path(path, sizeof(path));
    CHECK(write_declared_header_fixture(path, UINT64_C(100000001), "{}") == 0);
    CHECK(safetensors_open(&st, path) != 0);
    CHECK(remove(path) == 0);

    fixture_path(path, sizeof(path));
    CHECK(write_declared_header_fixture(path, UINT64_MAX, "{}") == 0);
    CHECK(safetensors_open(&st, path) != 0);
    CHECK(remove(path) == 0);

    fixture_path(path, sizeof(path));
    CHECK(write_declared_header_fixture(path, 128, "{}") == 0);
    CHECK(safetensors_open(&st, path) != 0);
    CHECK(remove(path) == 0);
}

int main(int argc, char **argv) {
    if (argc == 2) {
        SafeTensors st;
        if (safetensors_open(&st, argv[1])) return 1;
        printf("opened %s: %d tensors, header=%" PRIu64 " bytes, file=%" PRIu64 " bytes\n",
               argv[1], st.count, st.header_len, st.file_size);
        safetensors_close(&st);
        return 0;
    }
    test_valid_file_and_reads();
    test_valid_empty_tensors();
    test_malformed_headers();
    test_invalid_header_lengths();
    if (failures != 0) {
        fprintf(stderr, "%d safetensors test(s) failed\n", failures);
        return 1;
    }
    printf("safetensors tests passed\n");
    return 0;
}
