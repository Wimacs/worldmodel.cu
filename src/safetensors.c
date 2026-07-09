#define _FILE_OFFSET_BITS 64

#include "safetensors.h"

#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define strdup _strdup
#endif

static int safetensors_fseek64(FILE *f, uint64_t off) {
#ifdef _WIN32
    return _fseeki64(f, (__int64)off, SEEK_SET);
#else
    return fseeko(f, (off_t)off, SEEK_SET);
#endif
}

static const char *skip_ws(const char *p) {
    while (*p && isspace((unsigned char)*p)) p++;
    return p;
}

static char *copy_json_string(const char **p_in) {
    const char *p = skip_ws(*p_in);
    if (*p != '"') return NULL;
    p++;
    const char *start = p;
    size_t len = 0;
    while (*p) {
        if (*p == '\\' && p[1]) {
            p += 2;
            len += 2;
            continue;
        }
        if (*p == '"') break;
        p++;
        len++;
    }
    if (*p != '"') return NULL;
    char *s = (char *)malloc(len + 1);
    if (!s) return NULL;
    memcpy(s, start, len);
    s[len] = 0;
    *p_in = p + 1;
    return s;
}

static const char *find_matching_brace(const char *p) {
    if (*p != '{') return NULL;
    int depth = 0;
    int in_string = 0;
    for (; *p; ++p) {
        if (in_string) {
            if (*p == '\\' && p[1]) {
                ++p;
            } else if (*p == '"') {
                in_string = 0;
            }
            continue;
        }
        if (*p == '"') in_string = 1;
        else if (*p == '{') depth++;
        else if (*p == '}') {
            depth--;
            if (depth == 0) return p;
        }
    }
    return NULL;
}

static int ensure_entry_cap(SafeTensors *st) {
    if (st->count < st->capacity) return 0;
    int new_cap = st->capacity ? st->capacity * 2 : 256;
    SafeTensorEntry *entries = (SafeTensorEntry *)realloc(st->entries, (size_t)new_cap * sizeof(*entries));
    if (!entries) return 1;
    st->entries = entries;
    st->capacity = new_cap;
    return 0;
}

static int parse_i64_array(const char *obj_start, const char *obj_end, const char *key, int64_t *out, int max_n, int *n_out) {
    const char *p = strstr(obj_start, key);
    if (!p || p >= obj_end) return 1;
    p = strchr(p, '[');
    if (!p || p >= obj_end) return 1;
    p++;
    int n = 0;
    while (p < obj_end && *p && *p != ']') {
        p = skip_ws(p);
        char *endp = NULL;
        long long v = strtoll(p, &endp, 10);
        if (endp == p) return 1;
        if (n < max_n) out[n++] = (int64_t)v;
        p = skip_ws(endp);
        if (*p == ',') p++;
    }
    *n_out = n;
    return 0;
}

static int parse_dtype(const char *obj_start, const char *obj_end, char dtype[16]) {
    const char *p = strstr(obj_start, "\"dtype\"");
    if (!p || p >= obj_end) return 1;
    p = strchr(p, ':');
    if (!p || p >= obj_end) return 1;
    p++;
    char *s = copy_json_string(&p);
    if (!s) return 1;
    snprintf(dtype, 16, "%s", s);
    free(s);
    return 0;
}

static int parse_entry_object(SafeTensorEntry *e, const char *obj_start, const char *obj_end) {
    if (parse_dtype(obj_start, obj_end, e->dtype)) return 1;
    if (parse_i64_array(obj_start, obj_end, "\"shape\"", e->shape, 8, &e->ndim)) return 1;

    int n_offsets = 0;
    int64_t offsets[2] = {0, 0};
    if (parse_i64_array(obj_start, obj_end, "\"data_offsets\"", offsets, 2, &n_offsets) || n_offsets != 2) return 1;
    e->begin = (uint64_t)offsets[0];
    e->end = (uint64_t)offsets[1];
    return 0;
}

static uint64_t read_u64_le(const unsigned char b[8]) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; --i) {
        v = (v << 8) | b[i];
    }
    return v;
}

int safetensors_open(SafeTensors *st, const char *path) {
    memset(st, 0, sizeof(*st));
    st->path = strdup(path);
    if (!st->path) return 1;

    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "failed to open safetensors: %s: %s\n", path, strerror(errno));
        safetensors_close(st);
        return 1;
    }

    unsigned char len_buf[8];
    if (fread(len_buf, 1, 8, f) != 8) {
        fprintf(stderr, "failed to read safetensors header length\n");
        fclose(f);
        safetensors_close(st);
        return 1;
    }
    st->header_len = read_u64_le(len_buf);
    st->header = (char *)malloc((size_t)st->header_len + 1);
    if (!st->header) {
        fclose(f);
        safetensors_close(st);
        return 1;
    }
    if (fread(st->header, 1, (size_t)st->header_len, f) != (size_t)st->header_len) {
        fprintf(stderr, "failed to read safetensors header\n");
        fclose(f);
        safetensors_close(st);
        return 1;
    }
    st->header[st->header_len] = 0;
    fclose(f);

    const char *p = st->header;
    p = skip_ws(p);
    if (*p != '{') {
        fprintf(stderr, "safetensors header is not a JSON object\n");
        safetensors_close(st);
        return 1;
    }
    p++;
    while (*p) {
        p = skip_ws(p);
        if (*p == '}') break;
        char *name = copy_json_string(&p);
        if (!name) break;
        p = skip_ws(p);
        if (*p != ':') {
            free(name);
            break;
        }
        p++;
        p = skip_ws(p);
        const char *obj_start = p;
        const char *obj_end = find_matching_brace(obj_start);
        if (!obj_end) {
            free(name);
            break;
        }
        if (strcmp(name, "__metadata__") != 0) {
            if (ensure_entry_cap(st)) {
                free(name);
                safetensors_close(st);
                return 1;
            }
            SafeTensorEntry *e = &st->entries[st->count];
            memset(e, 0, sizeof(*e));
            e->name = name;
            if (parse_entry_object(e, obj_start, obj_end)) {
                fprintf(stderr, "failed to parse tensor metadata for %s\n", name);
                safetensors_close(st);
                return 1;
            }
            st->count++;
        } else {
            free(name);
        }
        p = obj_end + 1;
        p = skip_ws(p);
        if (*p == ',') p++;
    }

    return 0;
}

void safetensors_close(SafeTensors *st) {
    if (!st) return;
    for (int i = 0; i < st->count; ++i) {
        free(st->entries[i].name);
    }
    free(st->entries);
    free(st->header);
    free(st->path);
    memset(st, 0, sizeof(*st));
}

const SafeTensorEntry *safetensors_find(const SafeTensors *st, const char *name) {
    for (int i = 0; i < st->count; ++i) {
        if (strcmp(st->entries[i].name, name) == 0) return &st->entries[i];
    }
    return NULL;
}

size_t safetensors_dtype_size(const char *dtype) {
    if (strcmp(dtype, "F32") == 0) return 4;
    if (strcmp(dtype, "BF16") == 0) return 2;
    if (strcmp(dtype, "F16") == 0) return 2;
    if (strcmp(dtype, "I64") == 0) return 8;
    if (strcmp(dtype, "I32") == 0) return 4;
    if (strcmp(dtype, "U8") == 0) return 1;
    return 0;
}

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x03ffu;
    uint32_t bits = 0;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03ffu;
            bits = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
        }
    } else if (exp == 0x1fu) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
    }
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static float bf16_to_f32(uint16_t h) {
    uint32_t bits = (uint32_t)h << 16;
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

int safetensors_read_tensor(const SafeTensors *st, const SafeTensorEntry *entry, void **data_out, size_t *bytes_out) {
    *data_out = NULL;
    *bytes_out = 0;
    if (!entry) return 1;

    uint64_t bytes64 = entry->end - entry->begin;
    if (bytes64 > (uint64_t)SIZE_MAX) return 1;
    size_t bytes = (size_t)bytes64;
    void *data = malloc(bytes ? bytes : 1);
    if (!data) return 1;

    FILE *f = fopen(st->path, "rb");
    if (!f) {
        free(data);
        return 1;
    }
    uint64_t off = 8 + st->header_len + entry->begin;
    if (safetensors_fseek64(f, off) != 0) {
        fclose(f);
        free(data);
        return 1;
    }
    int ok = fread(data, 1, bytes, f) == bytes;
    fclose(f);
    if (!ok) {
        free(data);
        return 1;
    }
    *data_out = data;
    *bytes_out = bytes;
    return 0;
}

int safetensors_read_tensor_f32(const SafeTensors *st, const SafeTensorEntry *entry, float **data_out, size_t *elems_out) {
    *data_out = NULL;
    *elems_out = 0;
    if (!entry) return 1;

    void *raw = NULL;
    size_t bytes = 0;
    if (safetensors_read_tensor(st, entry, &raw, &bytes)) return 1;

    size_t dtype_size = safetensors_dtype_size(entry->dtype);
    if (dtype_size == 0 || bytes % dtype_size != 0) {
        free(raw);
        return 1;
    }
    size_t elems = bytes / dtype_size;
    float *out = (float *)malloc(elems ? elems * sizeof(float) : sizeof(float));
    if (!out) {
        free(raw);
        return 1;
    }

    if (strcmp(entry->dtype, "F32") == 0) {
        memcpy(out, raw, elems * sizeof(float));
    } else if (strcmp(entry->dtype, "F16") == 0) {
        const uint16_t *src = (const uint16_t *)raw;
        for (size_t i = 0; i < elems; ++i) out[i] = f16_to_f32(src[i]);
    } else if (strcmp(entry->dtype, "BF16") == 0) {
        const uint16_t *src = (const uint16_t *)raw;
        for (size_t i = 0; i < elems; ++i) out[i] = bf16_to_f32(src[i]);
    } else {
        free(out);
        free(raw);
        return 1;
    }

    free(raw);
    *data_out = out;
    *elems_out = elems;
    return 0;
}

void safetensors_print_entry(const SafeTensorEntry *entry) {
    fprintf(stderr, "%s dtype=%s shape=[", entry->name, entry->dtype);
    for (int i = 0; i < entry->ndim; ++i) {
        fprintf(stderr, "%s%" PRId64, i ? "," : "", entry->shape[i]);
    }
    fprintf(stderr, "] bytes=%" PRIu64 "\n", entry->end - entry->begin);
}
