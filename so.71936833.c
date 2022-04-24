/*******************************************************************************
*
* stackoverflow.com/questions/71936833/nibble-shuffling-with-x64-simd
*
* Authors: Brett Hale (906839), Peter Cordes (224132)
* SPDX-License-Identifier: CC-BY-SA-4.0 OR CC0-1.0
*
*******************************************************************************/

#include <inttypes.h>
#include <stdio.h>

#include <immintrin.h>


uint64_t u4x16_sse_shuffle (uint64_t src, uint64_t idx)
{
    __m128i v_dst, v_src, v_idx, hi;

    /* u4x16 nibbles to xmm u8x16: [0:n[15], .., 0:n[0]] */

#if (1) /* Cordes: SSE2 instructions */

    v_src = _mm_cvtsi64_si128((int64_t) src);
    hi = _mm_srli_epi32(v_src, 4);
    v_src = _mm_unpacklo_epi8(v_src, hi);
    v_src = _mm_and_si128(v_src, _mm_set1_epi8(0x0f));

    v_idx = _mm_cvtsi64_si128((int64_t) idx);
    hi = _mm_srli_epi32(v_idx, 4);
    v_idx = _mm_unpacklo_epi8(v_idx, hi);
    v_idx = _mm_and_si128(v_idx, _mm_set1_epi8(0x0f));

#else   /* u64 AND + SHIFT */

    uint64_t split = UINT64_C(0x0f0f0f0f0f0f0f0f);
    uint64_t u64_lo, u64_hi;

    u64_lo = src & split;
    v_src = _mm_cvtsi64_si128((int64_t) u64_lo);
    u64_hi = (src & ~(split)) >> 4;
    hi = _mm_cvtsi64_si128((int64_t) u64_hi);
    v_src = _mm_unpacklo_epi8(v_src, hi);

    u64_lo = idx & split;
    v_idx = _mm_cvtsi64_si128((int64_t) u64_lo);
    u64_hi = (idx & ~(split)) >> 4;
    hi = _mm_cvtsi64_si128((int64_t) u64_hi);
    v_idx = _mm_unpacklo_epi8(v_idx, hi);

#endif

    /* the 'nibble' shuffle, using xmm u8x16 elements: */

    v_dst = _mm_shuffle_epi8(v_src, v_idx);

    /* recombine nibbles: [0:n15, 0:n14, .., 0:n1, 0:n0]
     * as: [[127:64 = any], [63:0 = n15:n14, .., n1:n0]] */

#if (1) /* Cordes recombine: pmaddubsw + packuswb */

    __m128i m_mul = _mm_set1_epi16(0x1001);

    v_dst = _mm_maddubs_epi16(v_dst, m_mul);
    v_dst = _mm_packus_epi16(v_dst, v_dst);

#else   /* recombine: SHIFT + OR + SHUFFLE: */

    __m128i m_odd = _mm_set_epi64x(
        INT64_C(-1), INT64_C(0x0f0d0b0907050301));

    hi = _mm_slli_epi64(v_dst, 4);
    v_dst = _mm_bslli_si128(v_dst, 1);

    /* clang-14.0.1 replacing bslli_si128 with pshufb! */
    v_dst = _mm_shuffle_epi8(_mm_or_si128(v_dst, hi), m_odd);

#endif

    return ((uint64_t) (_mm_cvtsi128_si64(v_dst)));
}


int main (void)
{
    /* test vectors: */

#define u64_identity UINT64_C(0xFEDCBA9876543210);
#define u64_inv_gold UINT64_C(0x9E3779B97F4A7C15);
#define u64_rnd_nrpt UINT64_C(0xB74E05C2FD83169A);

    uint64_t src = u64_inv_gold;
    uint64_t idx = u64_rnd_nrpt;

    fprintf(stdout, "src: 0x%016" PRIX64 "\n", src);
    fprintf(stdout, "idx: 0x%016" PRIX64 "\n", idx);

    uint64_t dst = 0;

    for (int i = 0; i < 16; i++)
    {
        uint8_t index = (idx >> (i * 4)) & 0xf;
        dst |= ((src >> (index * 4)) & 0xf) << (i * 4);
    }

    fprintf(stdout, "dst (serial) : 0x%016" PRIX64 "\n", dst);

    dst = u4x16_sse_shuffle(src, idx);
    fprintf(stdout, "dst (SSSE3+) : 0x%016" PRIX64 "\n", dst);

    return (0);
}

/******************************************************************************/
