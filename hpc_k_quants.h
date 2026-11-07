#ifndef HPC_K_QUANTS_H
#define HPC_K_QUANTS_H

#include <cuda_fp16.h>
#include <cstdint>

#pragma pack(push, 1)
struct block_q2_K {
    uint8_t scales[16];
    uint8_t qs[64];
    half d;
    half dmin;
};

struct block_q3_K {
    uint8_t hmask[32];
    uint8_t qs[64];
    uint8_t scales[12];
    half d;
};

struct block_q4_K {
    half d;
    half dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};
#pragma pack(pop)

#endif // HPC_K_QUANTS_H
