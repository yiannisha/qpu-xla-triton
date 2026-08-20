#include "qpu_llama_sha256.h"

#include <string.h>

typedef struct sha256_state {
    uint32_t hash[8];
    uint64_t bit_count;
    uint8_t block[64];
    size_t block_size;
} sha256_state;

static const uint32_t round_constants[64] = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U,
    0xab1c5ed5U, 0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU,
    0x9bdc06a7U, 0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU,
    0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
    0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
    0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U, 0xa2bfe8a1U, 0xa81a664bU,
    0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U,
    0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U,
    0xc67178f2U,
};

static uint32_t rotate_right(uint32_t value, uint32_t amount) {
    return (value >> amount) | (value << (32U - amount));
}

static uint32_t load_be32(const uint8_t *source) {
    return ((uint32_t) source[0] << 24U) | ((uint32_t) source[1] << 16U) |
           ((uint32_t) source[2] << 8U) | (uint32_t) source[3];
}

static void store_be32(uint8_t *destination, uint32_t value) {
    destination[0] = (uint8_t) (value >> 24U);
    destination[1] = (uint8_t) (value >> 16U);
    destination[2] = (uint8_t) (value >> 8U);
    destination[3] = (uint8_t) value;
}

static void transform(sha256_state *state, const uint8_t block[64]) {
    uint32_t words[64];
    for (size_t index = 0; index < 16; ++index) {
        words[index] = load_be32(block + index * 4);
    }
    for (size_t index = 16; index < 64; ++index) {
        const uint32_t s0 = rotate_right(words[index - 15], 7) ^ rotate_right(words[index - 15], 18) ^
                            (words[index - 15] >> 3U);
        const uint32_t s1 = rotate_right(words[index - 2], 17) ^ rotate_right(words[index - 2], 19) ^
                            (words[index - 2] >> 10U);
        words[index] = words[index - 16] + s0 + words[index - 7] + s1;
    }

    uint32_t a = state->hash[0];
    uint32_t b = state->hash[1];
    uint32_t c = state->hash[2];
    uint32_t d = state->hash[3];
    uint32_t e = state->hash[4];
    uint32_t f = state->hash[5];
    uint32_t g = state->hash[6];
    uint32_t h = state->hash[7];
    for (size_t index = 0; index < 64; ++index) {
        const uint32_t sum1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
        const uint32_t choose = (e & f) ^ (~e & g);
        const uint32_t temporary1 = h + sum1 + choose + round_constants[index] + words[index];
        const uint32_t sum0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
        const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        const uint32_t temporary2 = sum0 + majority;
        h = g;
        g = f;
        f = e;
        e = d + temporary1;
        d = c;
        c = b;
        b = a;
        a = temporary1 + temporary2;
    }
    state->hash[0] += a;
    state->hash[1] += b;
    state->hash[2] += c;
    state->hash[3] += d;
    state->hash[4] += e;
    state->hash[5] += f;
    state->hash[6] += g;
    state->hash[7] += h;
}

static void update(sha256_state *state, const uint8_t *data, size_t size) {
    state->bit_count += (uint64_t) size * 8U;
    while (size > 0) {
        const size_t available = sizeof(state->block) - state->block_size;
        const size_t count = size < available ? size : available;
        memcpy(state->block + state->block_size, data, count);
        state->block_size += count;
        data += count;
        size -= count;
        if (state->block_size == sizeof(state->block)) {
            transform(state, state->block);
            state->block_size = 0;
        }
    }
}

void qpu_llama_sha256(const void *data, size_t size, uint8_t digest[32]) {
    sha256_state state = {
        .hash = {0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
                 0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U},
        .bit_count = 0,
        .block = {0},
        .block_size = 0,
    };
    update(&state, (const uint8_t *) data, size);
    state.block[state.block_size++] = 0x80U;
    if (state.block_size > 56) {
        memset(state.block + state.block_size, 0, sizeof(state.block) - state.block_size);
        transform(&state, state.block);
        state.block_size = 0;
    }
    memset(state.block + state.block_size, 0, 56 - state.block_size);
    for (size_t index = 0; index < 8; ++index) {
        state.block[63 - index] = (uint8_t) (state.bit_count >> (index * 8));
    }
    transform(&state, state.block);
    for (size_t index = 0; index < 8; ++index) {
        store_be32(digest + index * 4, state.hash[index]);
    }
}

void qpu_llama_sha256_hex(const void *data, size_t size, char hex[65]) {
    static const char digits[] = "0123456789abcdef";
    uint8_t digest[32];
    qpu_llama_sha256(data, size, digest);
    for (size_t index = 0; index < sizeof(digest); ++index) {
        hex[index * 2] = digits[digest[index] >> 4U];
        hex[index * 2 + 1] = digits[digest[index] & 0x0fU];
    }
    hex[64] = '\0';
}
