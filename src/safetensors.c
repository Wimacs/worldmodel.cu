#define _FILE_OFFSET_BITS 64
#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L
#endif

#include "safetensors.h"

#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

#define SAFETENSORS_MAX_HEADER_SIZE UINT64_C(100000000)
#define SAFETENSORS_MAX_JSON_DEPTH 64

typedef struct {
    const char *p;
    const char *end;
} JsonCursor;

static int safetensors_fseek64(FILE *f, uint64_t off) {
    if (off > (uint64_t)INT64_MAX) return 1;
#ifdef _WIN32
    return _fseeki64(f, (__int64)off, SEEK_SET);
#else
    return fseeko(f, (off_t)off, SEEK_SET);
#endif
}

static int safetensors_file_size(FILE *f, uint64_t *size_out) {
#ifdef _WIN32
    if (_fseeki64(f, 0, SEEK_END) != 0) return 1;
    __int64 pos = _ftelli64(f);
    if (pos < 0 || _fseeki64(f, 0, SEEK_SET) != 0) return 1;
#else
    if (fseeko(f, (off_t)0, SEEK_END) != 0) return 1;
    off_t pos = ftello(f);
    if (pos < 0 || fseeko(f, (off_t)0, SEEK_SET) != 0) return 1;
#endif
    *size_out = (uint64_t)pos;
    return 0;
}

static char *duplicate_string(const char *s) {
    size_t len = strlen(s);
    if (len == SIZE_MAX) return NULL;
    char *copy = (char *)malloc(len + 1);
    if (copy) memcpy(copy, s, len + 1);
    return copy;
}

static void json_skip_ws(JsonCursor *cursor) {
    while (cursor->p < cursor->end) {
        char c = *cursor->p;
        if (c != ' ' && c != '\t' && c != '\r' && c != '\n') break;
        cursor->p++;
    }
}

static int hex_digit_value(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static int parse_hex4(const char *p, const char *end, uint32_t *value_out) {
    if ((size_t)(end - p) < 4) return 1;
    uint32_t value = 0;
    for (int i = 0; i < 4; ++i) {
        int digit = hex_digit_value(p[i]);
        if (digit < 0) return 1;
        value = (value << 4) | (uint32_t)digit;
    }
    *value_out = value;
    return 0;
}

static size_t encode_utf8(uint32_t codepoint, char out[4]) {
    if (codepoint <= 0x7fu) {
        out[0] = (char)codepoint;
        return 1;
    }
    if (codepoint <= 0x7ffu) {
        out[0] = (char)(0xc0u | (codepoint >> 6));
        out[1] = (char)(0x80u | (codepoint & 0x3fu));
        return 2;
    }
    if (codepoint <= 0xffffu) {
        out[0] = (char)(0xe0u | (codepoint >> 12));
        out[1] = (char)(0x80u | ((codepoint >> 6) & 0x3fu));
        out[2] = (char)(0x80u | (codepoint & 0x3fu));
        return 3;
    }
    out[0] = (char)(0xf0u | (codepoint >> 18));
    out[1] = (char)(0x80u | ((codepoint >> 12) & 0x3fu));
    out[2] = (char)(0x80u | ((codepoint >> 6) & 0x3fu));
    out[3] = (char)(0x80u | (codepoint & 0x3fu));
    return 4;
}

static int json_parse_string(JsonCursor *cursor, char **string_out, size_t *length_out) {
    json_skip_ws(cursor);
    if (cursor->p >= cursor->end || *cursor->p != '"') return 1;

    const char *raw_start = ++cursor->p;
    const char *raw_end = raw_start;
    int escaped = 0;
    while (raw_end < cursor->end) {
        unsigned char c = (unsigned char)*raw_end;
        if (c < 0x20u) return 1;
        if (!escaped && c == '"') break;
        if (!escaped && c == '\\') escaped = 1;
        else escaped = 0;
        raw_end++;
    }
    if (raw_end >= cursor->end || escaped) return 1;

    size_t raw_len = (size_t)(raw_end - raw_start);
    char *decoded = NULL;
    if (string_out) {
        if (raw_len == SIZE_MAX) return 1;
        decoded = (char *)malloc(raw_len + 1);
        if (!decoded) return 1;
    }

    size_t decoded_len = 0;
    const char *p = raw_start;
    while (p < raw_end) {
        unsigned char c = (unsigned char)*p++;
        if (c != '\\') {
            if (decoded) decoded[decoded_len] = (char)c;
            decoded_len++;
            continue;
        }
        if (p >= raw_end) {
            free(decoded);
            return 1;
        }
        char escape = *p++;
        char replacement = 0;
        switch (escape) {
            case '"': replacement = '"'; break;
            case '\\': replacement = '\\'; break;
            case '/': replacement = '/'; break;
            case 'b': replacement = '\b'; break;
            case 'f': replacement = '\f'; break;
            case 'n': replacement = '\n'; break;
            case 'r': replacement = '\r'; break;
            case 't': replacement = '\t'; break;
            case 'u': {
                uint32_t codepoint = 0;
                if (parse_hex4(p, raw_end, &codepoint)) {
                    free(decoded);
                    return 1;
                }
                p += 4;
                if (codepoint >= 0xd800u && codepoint <= 0xdbffu) {
                    uint32_t low = 0;
                    if ((size_t)(raw_end - p) < 6 || p[0] != '\\' || p[1] != 'u' ||
                        parse_hex4(p + 2, raw_end, &low) || low < 0xdc00u || low > 0xdfffu) {
                        free(decoded);
                        return 1;
                    }
                    p += 6;
                    codepoint = 0x10000u + ((codepoint - 0xd800u) << 10) + (low - 0xdc00u);
                } else if (codepoint >= 0xdc00u && codepoint <= 0xdfffu) {
                    free(decoded);
                    return 1;
                }
                char utf8[4];
                size_t encoded_len = encode_utf8(codepoint, utf8);
                if (decoded) memcpy(decoded + decoded_len, utf8, encoded_len);
                decoded_len += encoded_len;
                continue;
            }
            default:
                free(decoded);
                return 1;
        }
        if (decoded) decoded[decoded_len] = replacement;
        decoded_len++;
    }

    if (decoded) decoded[decoded_len] = 0;
    cursor->p = raw_end + 1;
    if (string_out) *string_out = decoded;
    if (length_out) *length_out = decoded_len;
    return 0;
}

static int json_parse_u64(JsonCursor *cursor, uint64_t *value_out) {
    json_skip_ws(cursor);
    if (cursor->p >= cursor->end || *cursor->p < '0' || *cursor->p > '9') return 1;

    const char *p = cursor->p;
    uint64_t value = 0;
    if (*p == '0') {
        p++;
        if (p < cursor->end && *p >= '0' && *p <= '9') return 1;
    } else {
        while (p < cursor->end && *p >= '0' && *p <= '9') {
            uint64_t digit = (uint64_t)(*p - '0');
            if (value > (UINT64_MAX - digit) / 10u) return 1;
            value = value * 10u + digit;
            p++;
        }
    }
    cursor->p = p;
    *value_out = value;
    return 0;
}

static int json_skip_value(JsonCursor *cursor, int depth);

static int json_skip_array(JsonCursor *cursor, int depth) {
    if (depth > SAFETENSORS_MAX_JSON_DEPTH || cursor->p >= cursor->end || *cursor->p != '[') return 1;
    cursor->p++;
    json_skip_ws(cursor);
    if (cursor->p < cursor->end && *cursor->p == ']') {
        cursor->p++;
        return 0;
    }
    for (;;) {
        if (json_skip_value(cursor, depth + 1)) return 1;
        json_skip_ws(cursor);
        if (cursor->p >= cursor->end) return 1;
        if (*cursor->p == ']') {
            cursor->p++;
            return 0;
        }
        if (*cursor->p != ',') return 1;
        cursor->p++;
    }
}

static int json_skip_object(JsonCursor *cursor, int depth) {
    if (depth > SAFETENSORS_MAX_JSON_DEPTH || cursor->p >= cursor->end || *cursor->p != '{') return 1;
    cursor->p++;
    json_skip_ws(cursor);
    if (cursor->p < cursor->end && *cursor->p == '}') {
        cursor->p++;
        return 0;
    }
    for (;;) {
        if (json_parse_string(cursor, NULL, NULL)) return 1;
        json_skip_ws(cursor);
        if (cursor->p >= cursor->end || *cursor->p != ':') return 1;
        cursor->p++;
        if (json_skip_value(cursor, depth + 1)) return 1;
        json_skip_ws(cursor);
        if (cursor->p >= cursor->end) return 1;
        if (*cursor->p == '}') {
            cursor->p++;
            return 0;
        }
        if (*cursor->p != ',') return 1;
        cursor->p++;
    }
}

static int json_skip_number(JsonCursor *cursor) {
    const char *p = cursor->p;
    if (p < cursor->end && *p == '-') p++;
    if (p >= cursor->end) return 1;
    if (*p == '0') {
        p++;
        if (p < cursor->end && *p >= '0' && *p <= '9') return 1;
    } else {
        if (*p < '1' || *p > '9') return 1;
        do p++; while (p < cursor->end && *p >= '0' && *p <= '9');
    }
    if (p < cursor->end && *p == '.') {
        p++;
        if (p >= cursor->end || *p < '0' || *p > '9') return 1;
        do p++; while (p < cursor->end && *p >= '0' && *p <= '9');
    }
    if (p < cursor->end && (*p == 'e' || *p == 'E')) {
        p++;
        if (p < cursor->end && (*p == '+' || *p == '-')) p++;
        if (p >= cursor->end || *p < '0' || *p > '9') return 1;
        do p++; while (p < cursor->end && *p >= '0' && *p <= '9');
    }
    cursor->p = p;
    return 0;
}

static int json_consume_literal(JsonCursor *cursor, const char *literal) {
    size_t len = strlen(literal);
    if ((size_t)(cursor->end - cursor->p) < len || memcmp(cursor->p, literal, len) != 0) return 1;
    cursor->p += len;
    return 0;
}

static int json_skip_value(JsonCursor *cursor, int depth) {
    if (depth > SAFETENSORS_MAX_JSON_DEPTH) return 1;
    json_skip_ws(cursor);
    if (cursor->p >= cursor->end) return 1;
    switch (*cursor->p) {
        case '"': return json_parse_string(cursor, NULL, NULL);
        case '{': return json_skip_object(cursor, depth);
        case '[': return json_skip_array(cursor, depth);
        case 't': return json_consume_literal(cursor, "true");
        case 'f': return json_consume_literal(cursor, "false");
        case 'n': return json_consume_literal(cursor, "null");
        default: return json_skip_number(cursor);
    }
}

static int ensure_entry_cap(SafeTensors *st) {
    if (st->count < st->capacity) return 0;
    if (st->capacity > INT_MAX / 2) return 1;
    int new_cap = st->capacity ? st->capacity * 2 : 256;
    if ((size_t)new_cap > SIZE_MAX / sizeof(*st->entries)) return 1;
    SafeTensorEntry *entries = (SafeTensorEntry *)realloc(st->entries, (size_t)new_cap * sizeof(*entries));
    if (!entries) return 1;
    st->entries = entries;
    st->capacity = new_cap;
    return 0;
}

typedef struct {
    char **items;
    size_t count;
    size_t capacity;
} JsonKeySet;

static void json_key_set_destroy(JsonKeySet *set) {
    for (size_t i = 0; i < set->count; ++i) free(set->items[i]);
    free(set->items);
    memset(set, 0, sizeof(*set));
}

static int json_key_set_add(JsonKeySet *set, char *key) {
    if (set->count == set->capacity) {
        size_t next_capacity = set->capacity ? set->capacity * 2 : 8;
        if (next_capacity < set->capacity || next_capacity > SIZE_MAX / sizeof(*set->items)) return 1;
        char **next_items = (char **)realloc(set->items, next_capacity * sizeof(*set->items));
        if (!next_items) return 1;
        set->items = next_items;
        set->capacity = next_capacity;
    }
    set->items[set->count++] = key;
    return 0;
}

static int compare_string_ptrs(const void *a_ptr, const void *b_ptr) {
    const char *const *a = (const char *const *)a_ptr;
    const char *const *b = (const char *const *)b_ptr;
    return strcmp(*a, *b);
}

static int json_key_set_has_duplicates(JsonKeySet *set) {
    if (set->count < 2) return 0;
    qsort(set->items, set->count, sizeof(*set->items), compare_string_ptrs);
    for (size_t i = 1; i < set->count; ++i) {
        if (strcmp(set->items[i - 1], set->items[i]) == 0) return 1;
    }
    return 0;
}

size_t safetensors_dtype_size(const char *dtype) {
    if (!dtype) return 0;
    if (strcmp(dtype, "BOOL") == 0 || strcmp(dtype, "I8") == 0 || strcmp(dtype, "U8") == 0 ||
        strcmp(dtype, "F8_E4M3") == 0 || strcmp(dtype, "F8_E5M2") == 0 || strcmp(dtype, "F8_E8M0") == 0 ||
        strcmp(dtype, "F8_E4M3FNUZ") == 0 || strcmp(dtype, "F8_E5M2FNUZ") == 0) return 1;
    if (strcmp(dtype, "I16") == 0 || strcmp(dtype, "U16") == 0 || strcmp(dtype, "F16") == 0 ||
        strcmp(dtype, "BF16") == 0) return 2;
    if (strcmp(dtype, "I32") == 0 || strcmp(dtype, "U32") == 0 || strcmp(dtype, "F32") == 0) return 4;
    if (strcmp(dtype, "I64") == 0 || strcmp(dtype, "U64") == 0 || strcmp(dtype, "F64") == 0 ||
        strcmp(dtype, "C64") == 0) return 8;
    if (strcmp(dtype, "C128") == 0) return 16;
    return 0;
}

static int parse_shape(JsonCursor *cursor, SafeTensorEntry *entry) {
    json_skip_ws(cursor);
    if (cursor->p >= cursor->end || *cursor->p != '[') return 1;
    cursor->p++;
    json_skip_ws(cursor);
    if (cursor->p < cursor->end && *cursor->p == ']') {
        cursor->p++;
        entry->ndim = 0;
        return 0;
    }

    int ndim = 0;
    for (;;) {
        uint64_t dim = 0;
        if (ndim >= (int)(sizeof(entry->shape) / sizeof(entry->shape[0])) || json_parse_u64(cursor, &dim) ||
            dim > (uint64_t)INT64_MAX) return 1;
        entry->shape[ndim++] = (int64_t)dim;
        json_skip_ws(cursor);
        if (cursor->p >= cursor->end) return 1;
        if (*cursor->p == ']') {
            cursor->p++;
            entry->ndim = ndim;
            return 0;
        }
        if (*cursor->p != ',') return 1;
        cursor->p++;
    }
}

static int parse_offsets(JsonCursor *cursor, SafeTensorEntry *entry) {
    json_skip_ws(cursor);
    if (cursor->p >= cursor->end || *cursor->p != '[') return 1;
    cursor->p++;
    uint64_t begin = 0;
    uint64_t end = 0;
    if (json_parse_u64(cursor, &begin)) return 1;
    json_skip_ws(cursor);
    if (cursor->p >= cursor->end || *cursor->p != ',') return 1;
    cursor->p++;
    if (json_parse_u64(cursor, &end)) return 1;
    json_skip_ws(cursor);
    if (cursor->p >= cursor->end || *cursor->p != ']') return 1;
    cursor->p++;
    entry->begin = begin;
    entry->end = end;
    return 0;
}

static int parse_entry_object(JsonCursor *cursor, SafeTensorEntry *entry) {
    json_skip_ws(cursor);
    if (cursor->p >= cursor->end || *cursor->p != '{') return 1;
    cursor->p++;

    JsonKeySet keys = {0};
    int saw_dtype = 0;
    int saw_shape = 0;
    int saw_offsets = 0;
    int result = 1;
    json_skip_ws(cursor);
    if (cursor->p < cursor->end && *cursor->p == '}') goto done;
    for (;;) {
        char *key = NULL;
        size_t key_len = 0;
        if (json_parse_string(cursor, &key, &key_len) || strlen(key) != key_len) {
            free(key);
            goto done;
        }
        if (json_key_set_add(&keys, key)) {
            free(key);
            goto done;
        }
        json_skip_ws(cursor);
        if (cursor->p >= cursor->end || *cursor->p != ':') goto done;
        cursor->p++;

        int failed = 0;
        if (strcmp(key, "dtype") == 0) {
            char *dtype = NULL;
            size_t dtype_len = 0;
            if (saw_dtype || json_parse_string(cursor, &dtype, &dtype_len) || strlen(dtype) != dtype_len ||
                dtype_len >= sizeof(entry->dtype) || safetensors_dtype_size(dtype) == 0) {
                failed = 1;
            } else {
                memcpy(entry->dtype, dtype, dtype_len + 1);
                saw_dtype = 1;
            }
            free(dtype);
        } else if (strcmp(key, "shape") == 0) {
            if (saw_shape || parse_shape(cursor, entry)) failed = 1;
            else saw_shape = 1;
        } else if (strcmp(key, "data_offsets") == 0) {
            if (saw_offsets || parse_offsets(cursor, entry)) failed = 1;
            else saw_offsets = 1;
        } else if (json_skip_value(cursor, 1)) {
            failed = 1;
        }
        if (failed) goto done;

        json_skip_ws(cursor);
        if (cursor->p >= cursor->end) goto done;
        if (*cursor->p == '}') {
            cursor->p++;
            result = !(saw_dtype && saw_shape && saw_offsets);
            goto done;
        }
        if (*cursor->p != ',') goto done;
        cursor->p++;
    }

done:
    if (result == 0 && json_key_set_has_duplicates(&keys)) result = 1;
    json_key_set_destroy(&keys);
    return result;
}

static int parse_metadata_object(JsonCursor *cursor) {
    json_skip_ws(cursor);
    if (cursor->p >= cursor->end || *cursor->p != '{') return 1;
    cursor->p++;
    json_skip_ws(cursor);
    if (cursor->p < cursor->end && *cursor->p == '}') {
        cursor->p++;
        return 0;
    }

    JsonKeySet keys = {0};
    int failed = 0;
    for (;;) {
        char *key = NULL;
        size_t key_len = 0;
        if (json_parse_string(cursor, &key, &key_len) || strlen(key) != key_len) {
            free(key);
            failed = 1;
            break;
        }
        if (json_key_set_add(&keys, key)) {
            free(key);
            failed = 1;
            break;
        }

        json_skip_ws(cursor);
        if (cursor->p >= cursor->end || *cursor->p != ':') {
            failed = 1;
            break;
        }
        cursor->p++;
        if (json_parse_string(cursor, NULL, NULL)) {
            failed = 1;
            break;
        }
        json_skip_ws(cursor);
        if (cursor->p >= cursor->end) {
            failed = 1;
            break;
        }
        if (*cursor->p == '}') {
            cursor->p++;
            break;
        }
        if (*cursor->p != ',') {
            failed = 1;
            break;
        }
        cursor->p++;
    }
    if (!failed && json_key_set_has_duplicates(&keys)) failed = 1;
    json_key_set_destroy(&keys);
    return failed;
}

static int entry_expected_bytes(const SafeTensorEntry *entry, uint64_t *bytes_out) {
    if (!entry || entry->ndim < 0 || entry->ndim > (int)(sizeof(entry->shape) / sizeof(entry->shape[0]))) return 1;
    size_t dtype_size = safetensors_dtype_size(entry->dtype);
    if (dtype_size == 0) return 1;

    uint64_t elements = 1;
    for (int i = 0; i < entry->ndim; ++i) {
        if (entry->shape[i] < 0) return 1;
        uint64_t dim = (uint64_t)entry->shape[i];
        if (dim != 0 && elements > UINT64_MAX / dim) return 1;
        elements *= dim;
    }
    if (elements != 0 && (uint64_t)dtype_size > UINT64_MAX / elements) return 1;
    *bytes_out = elements * (uint64_t)dtype_size;
    return 0;
}

static int validate_entry(const SafeTensors *st, const SafeTensorEntry *entry) {
    if (!st || !entry || st->data_offset > st->file_size || entry->end < entry->begin) return 1;
    uint64_t data_size = st->file_size - st->data_offset;
    if (entry->end > data_size) return 1;
    uint64_t expected_bytes = 0;
    return entry_expected_bytes(entry, &expected_bytes) || expected_bytes != entry->end - entry->begin;
}

static int validate_unique_entry_names(const SafeTensors *st) {
    if (!st || st->count < 0) return 1;
    if (st->count < 2) return 0;
    if ((size_t)st->count > SIZE_MAX / sizeof(char *)) return 1;
    char **names = (char **)malloc((size_t)st->count * sizeof(*names));
    if (!names) return 1;
    for (int i = 0; i < st->count; ++i) names[i] = st->entries[i].name;
    qsort(names, (size_t)st->count, sizeof(*names), compare_string_ptrs);
    int duplicate = 0;
    for (int i = 1; i < st->count; ++i) {
        if (strcmp(names[i - 1], names[i]) == 0) {
            duplicate = 1;
            break;
        }
    }
    free(names);
    return duplicate;
}

typedef struct {
    uint64_t begin;
    uint64_t end;
} DataInterval;

static int compare_intervals(const void *a_ptr, const void *b_ptr) {
    const DataInterval *a = (const DataInterval *)a_ptr;
    const DataInterval *b = (const DataInterval *)b_ptr;
    if (a->begin < b->begin) return -1;
    if (a->begin > b->begin) return 1;
    if (a->end < b->end) return -1;
    if (a->end > b->end) return 1;
    return 0;
}

static int validate_data_coverage(const SafeTensors *st) {
    if (!st || st->data_offset > st->file_size || st->count < 0) return 1;
    uint64_t data_size = st->file_size - st->data_offset;
    if (st->count == 0) return data_size != 0;
    if ((size_t)st->count > SIZE_MAX / sizeof(DataInterval)) return 1;

    DataInterval *intervals = (DataInterval *)malloc((size_t)st->count * sizeof(*intervals));
    if (!intervals) return 1;
    for (int i = 0; i < st->count; ++i) {
        intervals[i].begin = st->entries[i].begin;
        intervals[i].end = st->entries[i].end;
    }
    qsort(intervals, (size_t)st->count, sizeof(*intervals), compare_intervals);

    uint64_t covered = 0;
    int invalid = 0;
    for (int i = 0; i < st->count; ++i) {
        uint64_t begin = intervals[i].begin;
        uint64_t end = intervals[i].end;
        if (begin == end) {
            if (begin != covered) {
                invalid = 1;
                break;
            }
            continue;
        }
        if (begin != covered || end < begin) {
            invalid = 1;
            break;
        }
        covered = end;
    }
    if (covered != data_size) invalid = 1;
    free(intervals);
    return invalid;
}

static uint64_t read_u64_le(const unsigned char b[8]) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; --i) v = (v << 8) | b[i];
    return v;
}

int safetensors_open(SafeTensors *st, const char *path) {
    if (!st || !path) return 1;
    memset(st, 0, sizeof(*st));
    st->path = duplicate_string(path);
    if (!st->path) return 1;

    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "failed to open safetensors: %s: %s\n", path, strerror(errno));
        safetensors_close(st);
        return 1;
    }
    if (safetensors_file_size(f, &st->file_size) || st->file_size < 8) {
        fprintf(stderr, "invalid safetensors file size: %s\n", path);
        fclose(f);
        safetensors_close(st);
        return 1;
    }

    unsigned char len_buf[8];
    if (fread(len_buf, 1, sizeof(len_buf), f) != sizeof(len_buf)) {
        fprintf(stderr, "failed to read safetensors header length\n");
        fclose(f);
        safetensors_close(st);
        return 1;
    }
    st->header_len = read_u64_le(len_buf);
    if (st->header_len < 2 || st->header_len > SAFETENSORS_MAX_HEADER_SIZE ||
        st->header_len > (uint64_t)(SIZE_MAX - 1) || st->header_len > UINT64_MAX - 8) {
        fprintf(stderr, "invalid safetensors header length: %" PRIu64 "\n", st->header_len);
        fclose(f);
        safetensors_close(st);
        return 1;
    }
    st->data_offset = 8 + st->header_len;
    if (st->data_offset > st->file_size) {
        fprintf(stderr, "safetensors header extends past end of file\n");
        fclose(f);
        safetensors_close(st);
        return 1;
    }

    size_t header_size = (size_t)st->header_len;
    st->header = (char *)malloc(header_size + 1);
    if (!st->header) {
        fclose(f);
        safetensors_close(st);
        return 1;
    }
    if (fread(st->header, 1, header_size, f) != header_size) {
        fprintf(stderr, "failed to read safetensors header\n");
        fclose(f);
        safetensors_close(st);
        return 1;
    }
    st->header[header_size] = 0;
    fclose(f);

    JsonCursor cursor = {st->header, st->header + header_size};
    if (cursor.p >= cursor.end || *cursor.p != '{') {
        fprintf(stderr, "safetensors header is not a JSON object\n");
        safetensors_close(st);
        return 1;
    }
    cursor.p++;
    json_skip_ws(&cursor);
    int saw_metadata = 0;
    if (cursor.p < cursor.end && *cursor.p == '}') {
        cursor.p++;
    } else {
        for (;;) {
            char *name = NULL;
            size_t name_len = 0;
            if (json_parse_string(&cursor, &name, &name_len) || strlen(name) != name_len) {
                free(name);
                fprintf(stderr, "invalid tensor name in safetensors header\n");
                safetensors_close(st);
                return 1;
            }
            json_skip_ws(&cursor);
            if (cursor.p >= cursor.end || *cursor.p != ':') {
                free(name);
                fprintf(stderr, "invalid safetensors header entry\n");
                safetensors_close(st);
                return 1;
            }
            cursor.p++;

            if (strcmp(name, "__metadata__") == 0) {
                free(name);
                if (saw_metadata || parse_metadata_object(&cursor)) {
                    fprintf(stderr, "invalid safetensors __metadata__ object\n");
                    safetensors_close(st);
                    return 1;
                }
                saw_metadata = 1;
            } else {
                if (ensure_entry_cap(st)) {
                    fprintf(stderr, "too many tensor entries: %s\n", name);
                    free(name);
                    safetensors_close(st);
                    return 1;
                }
                SafeTensorEntry *entry = &st->entries[st->count++];
                memset(entry, 0, sizeof(*entry));
                entry->name = name;
                if (parse_entry_object(&cursor, entry) || validate_entry(st, entry)) {
                    fprintf(stderr, "invalid tensor metadata for %s\n", name);
                    safetensors_close(st);
                    return 1;
                }
            }

            json_skip_ws(&cursor);
            if (cursor.p >= cursor.end) {
                fprintf(stderr, "unterminated safetensors header object\n");
                safetensors_close(st);
                return 1;
            }
            if (*cursor.p == '}') {
                cursor.p++;
                break;
            }
            if (*cursor.p != ',') {
                fprintf(stderr, "invalid separator in safetensors header\n");
                safetensors_close(st);
                return 1;
            }
            cursor.p++;
        }
    }
    while (cursor.p < cursor.end && *cursor.p == ' ') cursor.p++;
    if (cursor.p != cursor.end) {
        fprintf(stderr, "trailing non-space bytes in safetensors header padding\n");
        safetensors_close(st);
        return 1;
    }
    if (validate_unique_entry_names(st)) {
        fprintf(stderr, "duplicate tensor name in safetensors header\n");
        safetensors_close(st);
        return 1;
    }
    if (validate_data_coverage(st)) {
        fprintf(stderr, "safetensors data offsets contain overlaps, holes, or uncovered bytes\n");
        safetensors_close(st);
        return 1;
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
    if (!st || !name || st->count < 0) return NULL;
    for (int i = 0; i < st->count; ++i) {
        if (strcmp(st->entries[i].name, name) == 0) return &st->entries[i];
    }
    return NULL;
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
    if (!data_out || !bytes_out) return 1;
    *data_out = NULL;
    *bytes_out = 0;
    if (!st || !st->path || !entry || validate_entry(st, entry)) return 1;

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
    if (st->data_offset > UINT64_MAX - entry->begin) {
        fclose(f);
        free(data);
        return 1;
    }
    uint64_t off = st->data_offset + entry->begin;
    if (off > st->file_size || bytes64 > st->file_size - off) {
        fclose(f);
        free(data);
        return 1;
    }
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
    if (!data_out || !elems_out) return 1;
    *data_out = NULL;
    *elems_out = 0;
    if (!entry || (strcmp(entry->dtype, "F32") != 0 && strcmp(entry->dtype, "F16") != 0 &&
                   strcmp(entry->dtype, "BF16") != 0)) return 1;

    void *raw = NULL;
    size_t bytes = 0;
    if (safetensors_read_tensor(st, entry, &raw, &bytes)) return 1;

    size_t dtype_size = safetensors_dtype_size(entry->dtype);
    if (dtype_size == 0 || bytes % dtype_size != 0) {
        free(raw);
        return 1;
    }
    size_t elems = bytes / dtype_size;
    if (elems > SIZE_MAX / sizeof(float)) {
        free(raw);
        return 1;
    }
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
    if (!entry) return;
    fprintf(stderr, "%s dtype=%s shape=[", entry->name ? entry->name : "(unnamed)", entry->dtype);
    for (int i = 0; i < entry->ndim; ++i) {
        fprintf(stderr, "%s%" PRId64, i ? "," : "", entry->shape[i]);
    }
    uint64_t bytes = entry->end >= entry->begin ? entry->end - entry->begin : 0;
    fprintf(stderr, "] bytes=%" PRIu64 "\n", bytes);
}
