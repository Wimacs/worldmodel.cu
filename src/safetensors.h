#ifndef WORLD_SAFETENSORS_H
#define WORLD_SAFETENSORS_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    char *name;
    char dtype[16];
    int64_t shape[8];
    int ndim;
    uint64_t begin;
    uint64_t end;
} SafeTensorEntry;

typedef struct {
    char *path;
    char *header;
    uint64_t header_len;
    uint64_t data_offset;
    uint64_t file_size;
    SafeTensorEntry *entries;
    int count;
    int capacity;
} SafeTensors;

int safetensors_open(SafeTensors *st, const char *path);
void safetensors_close(SafeTensors *st);
const SafeTensorEntry *safetensors_find(const SafeTensors *st, const char *name);
size_t safetensors_dtype_size(const char *dtype);
int safetensors_read_tensor(const SafeTensors *st, const SafeTensorEntry *entry, void **data_out, size_t *bytes_out);
int safetensors_read_tensor_f32(const SafeTensors *st, const SafeTensorEntry *entry, float **data_out, size_t *elems_out);
void safetensors_print_entry(const SafeTensorEntry *entry);

#endif
