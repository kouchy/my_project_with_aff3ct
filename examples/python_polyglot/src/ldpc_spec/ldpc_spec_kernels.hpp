// Hand-specialized 5G NR QC-LDPC decoder kernels (horizontal layered BP + offset min-sum).
//
// Same algorithm and the same flat, precomputed Tanner-graph tables as the Julia (`ldpc_bp_opt.jl`) and
// Python/Numba (`ldpc_bp_opt.py`) "OPT" decoders, so that the C++ baseline differs from them only by language:
//   - `chk_off[c] .. chk_off[c + 1]`: edges of check node c (one layer = one check node),
//   - `chk_to_var[k]`               : variable node of edge k,
//   - `info_pos[i]`                 : codeword position of information bit i.
//
// Two execution modes:
//   - scalar      : one frame per call (plain C++, compiler auto-vectorization only);
//   - inter-frame : W frames per call, one frame per SIMD lane, hand-written intrinsics for
//                   AVX-512 (F+BW), AVX2, SSE4.1 (x86-64) and NEON (ARM64). W = register width / sizeof(T).
//
// This header has no dependency on AFF3CT / StreamPU (see `Decoder_LDPC_spec.hpp` for the module wrapper).

#ifndef LDPC_SPEC_KERNELS_HPP_
#define LDPC_SPEC_KERNELS_HPP_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>
#include <vector>

#if defined(__AVX512F__) && defined(__AVX512BW__)
#include <immintrin.h>
#define LDPC_SPEC_AVX512 1
#elif defined(__AVX2__)
#include <immintrin.h>
#define LDPC_SPEC_AVX2 1
#elif defined(__SSE4_1__)
#include <smmintrin.h>
#define LDPC_SPEC_SSE41 1
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define LDPC_SPEC_NEON 1
#else
#define LDPC_SPEC_NO_SIMD 1
#endif

namespace ldpc_spec
{

// -------------------------------------------------------------------------------------------------
// Aligned storage
// -------------------------------------------------------------------------------------------------
// LOC80-BEGIN CPP_SPEC.module
template<typename T, std::size_t A = 64>
struct Aligned_allocator
{
    using value_type = T;
    template<typename U>
    struct rebind
    {
        using other = Aligned_allocator<U, A>;
    };
    Aligned_allocator() noexcept = default;
    template<typename U>
    Aligned_allocator(const Aligned_allocator<U, A>&) noexcept
    {
    }
    T* allocate(std::size_t n)
    {
        void* p = nullptr;
        const std::size_t bytes = ((n * sizeof(T) + A - 1) / A) * A;
        if (posix_memalign(&p, A, bytes ? bytes : A) != 0) throw std::bad_alloc();
        return static_cast<T*>(p);
    }
    void deallocate(T* p, std::size_t) noexcept { std::free(p); }
    template<typename U>
    bool operator==(const Aligned_allocator<U, A>&) const noexcept
    {
        return true;
    }
    template<typename U>
    bool operator!=(const Aligned_allocator<U, A>&) const noexcept
    {
        return false;
    }
};
template<typename T>
using avector = std::vector<T, Aligned_allocator<T>>;
// LOC80-END CPP_SPEC.module

// -------------------------------------------------------------------------------------------------
// Flat Tanner graph (built once per parity-check matrix, shared read-only by all decoder clones)
// -------------------------------------------------------------------------------------------------
// LOC80-BEGIN CPP_SPEC.module
struct Graph
{
    int n_var = 0;
    int n_chk = 0;
    int K = 0;
    int max_chk_deg = 0;
    std::vector<int32_t> chk_off;    // n_chk + 1
    std::vector<int32_t> chk_to_var; // n_edges
    std::vector<int32_t> info_pos;   // K

    // `chk_cols[c]` lists the variable nodes connected to check node c (AFF3CT `get_col_to_rows()` order).
    template<class Cols, class Pos>
    Graph(const int n_var, const Cols& chk_cols, const Pos& info_bits_pos)
      : n_var(n_var)
      , n_chk((int)chk_cols.size())
      , K((int)info_bits_pos.size())
    {
        chk_off.resize(n_chk + 1, 0);
        for (int c = 0; c < n_chk; c++)
        {
            const int deg = (int)chk_cols[c].size();
            chk_off[c + 1] = chk_off[c] + deg;
            max_chk_deg = std::max(max_chk_deg, deg);
            for (const auto v : chk_cols[c])
                chk_to_var.push_back((int32_t)v);
        }
        for (const auto p : info_bits_pos)
            info_pos.push_back((int32_t)p);
    }
};
// LOC80-END CPP_SPEC.module

// -------------------------------------------------------------------------------------------------
// Scalar arithmetic helpers (float32 or saturated int16)
// -------------------------------------------------------------------------------------------------
// LOC80-BEGIN CPP_SPEC_SCALAR
inline float
sub_s(float a, float b)
{
    return a - b;
}
inline float
add_s(float a, float b)
{
    return a + b;
}
inline int16_t
sub_s(int16_t a, int16_t b)
{
    return (int16_t)std::min(32767, std::max(-32768, (int)a - (int)b));
}
inline int16_t
add_s(int16_t a, int16_t b)
{
    return (int16_t)std::min(32767, std::max(-32768, (int)a + (int)b));
}
inline float
abs_s(float a)
{
    return std::fabs(a);
}
inline int16_t
abs_s(int16_t a)
{
    return (int16_t)(a < 0 ? -a : a); // wraps on -32768, like Julia's abs(::Int16)
}
inline float
clamp_s(float a, float)
{
    return a;
}
inline int16_t
clamp_s(int16_t a, int16_t sat)
{
    return std::min<int16_t>(sat, std::max<int16_t>((int16_t)-sat, a));
}
// LOC80-END CPP_SPEC_SCALAR
// LOC80-BEGIN CPP_SPEC.common
template<typename T>
constexpr T
max_val()
{
    return std::numeric_limits<T>::max();
}
// LOC80-END CPP_SPEC.common
// LOC80-BEGIN CPP_SPEC_SCALAR

// -------------------------------------------------------------------------------------------------
// Scalar kernel: 1 frame, n_ite layered iterations (no early stopping)
// -------------------------------------------------------------------------------------------------
template<typename T>
void
decode_scalar(const Graph& g, const T* Y_N, T* vn, T* msg, T* con, const int n_ite, const T off, const T sat)
{
    const int32_t* __restrict off_c = g.chk_off.data();
    const int32_t* __restrict c2v = g.chk_to_var.data();
    std::copy(Y_N, Y_N + g.n_var, vn);
    std::fill(msg, msg + g.chk_off[g.n_chk], T(0));

    for (int ite = 0; ite < n_ite; ite++)
        for (int c = 0; c < g.n_chk; c++)
        {
            const int k0 = off_c[c], k1 = off_c[c + 1];
            bool sign = false;
            T min1 = max_val<T>(), min2 = max_val<T>();
            for (int k = k0, i = 0; k < k1; k++, i++)
            {
                const T x = sub_s(vn[c2v[k]], msg[k]);
                const T a = abs_s(x);
                con[i] = x;
                sign ^= (x < T(0));
                min2 = std::min(min2, std::max(a, min1));
                min1 = std::min(min1, a);
            }
            const T cst1 = std::max(T(0), T(min2 - off));
            const T cst2 = std::max(T(0), T(min1 - off));
            for (int k = k0, i = 0; k < k1; k++, i++)
            {
                const T x = con[i];
                const T r = (abs_s(x) == min1) ? cst1 : cst2;
                const T m = clamp_s((sign ^ (x < T(0))) ? T(-r) : r, sat);
                msg[k] = m;
                vn[c2v[k]] = add_s(x, m);
            }
        }
}

template<typename T, typename B>
void
hard_decision_scalar(const Graph& g, const T* vn, B* V_K)
{
    for (int i = 0; i < g.K; i++)
        V_K[i] = vn[g.info_pos[i]] < T(0) ? B(1) : B(0);
}
// LOC80-END CPP_SPEC_SCALAR

// -------------------------------------------------------------------------------------------------
// SIMD operation sets (one frame per lane). Each set provides:
//   V: vector type, S: sign accumulator type, W: lanes,
//   load/store/set1/zero/add/sub/min/max/abs, sign_init/sign_acc, oms_out.
// -------------------------------------------------------------------------------------------------
// LOC80-BEGIN CPP_SPEC_INTER
template<typename T>
struct Ops;
// LOC80-END CPP_SPEC_INTER

#if defined(LDPC_SPEC_AVX512)
// LOC80-BEGIN CPP_SPEC_OPS_AVX512
template<>
struct Ops<float>
{
    using T = float;
    using V = __m512;
    using S = __mmask16;
    static constexpr int W = 16;
    static V load(const T* p) { return _mm512_load_ps(p); }
    static void store(T* p, V v) { _mm512_store_ps(p, v); }
    static V set1(T x) { return _mm512_set1_ps(x); }
    static V add(V a, V b) { return _mm512_add_ps(a, b); }
    static V sub(V a, V b) { return _mm512_sub_ps(a, b); }
    static V min(V a, V b) { return _mm512_min_ps(a, b); }
    static V max(V a, V b) { return _mm512_max_ps(a, b); }
    static V abs(V a) { return _mm512_abs_ps(a); }
    static S neg(V x) { return _mm512_cmp_ps_mask(x, _mm512_setzero_ps(), _CMP_LT_OQ); }
    static S sign_init() { return 0; }
    static S sign_acc(S s, V x) { return (S)(s ^ neg(x)); }
    static V oms_out(V a, V m1, V c1, V c2, S s, V x, V, V)
    {
        const V mag = _mm512_mask_blend_ps(_mm512_cmp_ps_mask(a, m1, _CMP_EQ_OQ), c2, c1);
        const __m512i mi = _mm512_castps_si512(mag);
        return _mm512_castsi512_ps(
          _mm512_mask_xor_epi32(mi, (S)(s ^ neg(x)), mi, _mm512_set1_epi32((int)0x80000000)));
    }
};
template<>
struct Ops<int16_t>
{
    using T = int16_t;
    using V = __m512i;
    using S = __m512i;
    static constexpr int W = 32;
    static V load(const T* p) { return _mm512_load_si512((const void*)p); }
    static void store(T* p, V v) { _mm512_store_si512((void*)p, v); }
    static V set1(T x) { return _mm512_set1_epi16(x); }
    static V add(V a, V b) { return _mm512_adds_epi16(a, b); }
    static V sub(V a, V b) { return _mm512_subs_epi16(a, b); }
    static V min(V a, V b) { return _mm512_min_epi16(a, b); }
    static V max(V a, V b) { return _mm512_max_epi16(a, b); }
    static V abs(V a) { return _mm512_abs_epi16(a); }
    static S sign_init() { return _mm512_setzero_si512(); }
    static S sign_acc(S s, V x) { return _mm512_xor_si512(s, _mm512_srai_epi16(x, 15)); }
    static V oms_out(V a, V m1, V c1, V c2, S s, V x, V sat, V nsat)
    {
        const V mag = _mm512_mask_blend_epi16(_mm512_cmpeq_epi16_mask(a, m1), c2, c1);
        const V sg = _mm512_xor_si512(s, _mm512_srai_epi16(x, 15));
        const V r = _mm512_sub_epi16(_mm512_xor_si512(mag, sg), sg);
        return _mm512_min_epi16(_mm512_max_epi16(r, nsat), sat);
    }
};
// LOC80-END CPP_SPEC_OPS_AVX512
#elif defined(LDPC_SPEC_AVX2)
template<>
struct Ops<float>
{
    using T = float;
    using V = __m256;
    using S = __m256;
    static constexpr int W = 8;
    static V sgn() { return _mm256_castsi256_ps(_mm256_set1_epi32((int)0x80000000)); }
    static V load(const T* p) { return _mm256_load_ps(p); }
    static void store(T* p, V v) { _mm256_store_ps(p, v); }
    static V set1(T x) { return _mm256_set1_ps(x); }
    static V add(V a, V b) { return _mm256_add_ps(a, b); }
    static V sub(V a, V b) { return _mm256_sub_ps(a, b); }
    static V min(V a, V b) { return _mm256_min_ps(a, b); }
    static V max(V a, V b) { return _mm256_max_ps(a, b); }
    static V abs(V a) { return _mm256_andnot_ps(sgn(), a); }
    static V neg(V x) { return _mm256_and_ps(_mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LT_OQ), sgn()); }
    static S sign_init() { return _mm256_setzero_ps(); }
    static S sign_acc(S s, V x) { return _mm256_xor_ps(s, neg(x)); }
    static V oms_out(V a, V m1, V c1, V c2, S s, V x, V, V)
    {
        const V mag = _mm256_blendv_ps(c2, c1, _mm256_cmp_ps(a, m1, _CMP_EQ_OQ));
        return _mm256_xor_ps(mag, _mm256_xor_ps(s, neg(x)));
    }
};
template<>
struct Ops<int16_t>
{
    using T = int16_t;
    using V = __m256i;
    using S = __m256i;
    static constexpr int W = 16;
    static V load(const T* p) { return _mm256_load_si256((const __m256i*)p); }
    static void store(T* p, V v) { _mm256_store_si256((__m256i*)p, v); }
    static V set1(T x) { return _mm256_set1_epi16(x); }
    static V add(V a, V b) { return _mm256_adds_epi16(a, b); }
    static V sub(V a, V b) { return _mm256_subs_epi16(a, b); }
    static V min(V a, V b) { return _mm256_min_epi16(a, b); }
    static V max(V a, V b) { return _mm256_max_epi16(a, b); }
    static V abs(V a) { return _mm256_abs_epi16(a); }
    static S sign_init() { return _mm256_setzero_si256(); }
    static S sign_acc(S s, V x) { return _mm256_xor_si256(s, _mm256_srai_epi16(x, 15)); }
    static V oms_out(V a, V m1, V c1, V c2, S s, V x, V sat, V nsat)
    {
        const V mag = _mm256_blendv_epi8(c2, c1, _mm256_cmpeq_epi16(a, m1));
        const V sg = _mm256_xor_si256(s, _mm256_srai_epi16(x, 15));
        const V r = _mm256_sub_epi16(_mm256_xor_si256(mag, sg), sg);
        return _mm256_min_epi16(_mm256_max_epi16(r, nsat), sat);
    }
};
#elif defined(LDPC_SPEC_SSE41)
template<>
struct Ops<float>
{
    using T = float;
    using V = __m128;
    using S = __m128;
    static constexpr int W = 4;
    static V sgn() { return _mm_castsi128_ps(_mm_set1_epi32((int)0x80000000)); }
    static V load(const T* p) { return _mm_load_ps(p); }
    static void store(T* p, V v) { _mm_store_ps(p, v); }
    static V set1(T x) { return _mm_set1_ps(x); }
    static V add(V a, V b) { return _mm_add_ps(a, b); }
    static V sub(V a, V b) { return _mm_sub_ps(a, b); }
    static V min(V a, V b) { return _mm_min_ps(a, b); }
    static V max(V a, V b) { return _mm_max_ps(a, b); }
    static V abs(V a) { return _mm_andnot_ps(sgn(), a); }
    static V neg(V x) { return _mm_and_ps(_mm_cmplt_ps(x, _mm_setzero_ps()), sgn()); }
    static S sign_init() { return _mm_setzero_ps(); }
    static S sign_acc(S s, V x) { return _mm_xor_ps(s, neg(x)); }
    static V oms_out(V a, V m1, V c1, V c2, S s, V x, V, V)
    {
        const V mag = _mm_blendv_ps(c2, c1, _mm_cmpeq_ps(a, m1));
        return _mm_xor_ps(mag, _mm_xor_ps(s, neg(x)));
    }
};
template<>
struct Ops<int16_t>
{
    using T = int16_t;
    using V = __m128i;
    using S = __m128i;
    static constexpr int W = 8;
    static V load(const T* p) { return _mm_load_si128((const __m128i*)p); }
    static void store(T* p, V v) { _mm_store_si128((__m128i*)p, v); }
    static V set1(T x) { return _mm_set1_epi16(x); }
    static V add(V a, V b) { return _mm_adds_epi16(a, b); }
    static V sub(V a, V b) { return _mm_subs_epi16(a, b); }
    static V min(V a, V b) { return _mm_min_epi16(a, b); }
    static V max(V a, V b) { return _mm_max_epi16(a, b); }
    static V abs(V a) { return _mm_abs_epi16(a); }
    static S sign_init() { return _mm_setzero_si128(); }
    static S sign_acc(S s, V x) { return _mm_xor_si128(s, _mm_srai_epi16(x, 15)); }
    static V oms_out(V a, V m1, V c1, V c2, S s, V x, V sat, V nsat)
    {
        const V mag = _mm_blendv_epi8(c2, c1, _mm_cmpeq_epi16(a, m1));
        const V sg = _mm_xor_si128(s, _mm_srai_epi16(x, 15));
        const V r = _mm_sub_epi16(_mm_xor_si128(mag, sg), sg);
        return _mm_min_epi16(_mm_max_epi16(r, nsat), sat);
    }
};
#elif defined(LDPC_SPEC_NEON)
// LOC80-BEGIN CPP_SPEC_OPS_NEON
template<>
struct Ops<float>
{
    using T = float;
    using V = float32x4_t;
    using S = uint32x4_t;
    static constexpr int W = 4;
    static V load(const T* p) { return vld1q_f32(p); }
    static void store(T* p, V v) { vst1q_f32(p, v); }
    static V set1(T x) { return vdupq_n_f32(x); }
    static V add(V a, V b) { return vaddq_f32(a, b); }
    static V sub(V a, V b) { return vsubq_f32(a, b); }
    static V min(V a, V b) { return vminq_f32(a, b); }
    static V max(V a, V b) { return vmaxq_f32(a, b); }
    static V abs(V a) { return vabsq_f32(a); }
    static S neg(V x) { return vandq_u32(vcltq_f32(x, vdupq_n_f32(0.f)), vdupq_n_u32(0x80000000u)); }
    static S sign_init() { return vdupq_n_u32(0); }
    static S sign_acc(S s, V x) { return veorq_u32(s, neg(x)); }
    static V oms_out(V a, V m1, V c1, V c2, S s, V x, V, V)
    {
        const V mag = vbslq_f32(vceqq_f32(a, m1), c1, c2);
        return vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(mag), veorq_u32(s, neg(x))));
    }
};
template<>
struct Ops<int16_t>
{
    using T = int16_t;
    using V = int16x8_t;
    using S = int16x8_t;
    static constexpr int W = 8;
    static V load(const T* p) { return vld1q_s16(p); }
    static void store(T* p, V v) { vst1q_s16(p, v); }
    static V set1(T x) { return vdupq_n_s16(x); }
    static V add(V a, V b) { return vqaddq_s16(a, b); }
    static V sub(V a, V b) { return vqsubq_s16(a, b); }
    static V min(V a, V b) { return vminq_s16(a, b); }
    static V max(V a, V b) { return vmaxq_s16(a, b); }
    static V abs(V a) { return vabsq_s16(a); }
    static S sign_init() { return vdupq_n_s16(0); }
    static S sign_acc(S s, V x) { return veorq_s16(s, vshrq_n_s16(x, 15)); }
    static V oms_out(V a, V m1, V c1, V c2, S s, V x, V sat, V nsat)
    {
        const V mag = vbslq_s16(vceqq_s16(a, m1), c1, c2);
        const V sg = veorq_s16(s, vshrq_n_s16(x, 15));
        const V r = vsubq_s16(veorq_s16(mag, sg), sg);
        return vminq_s16(vmaxq_s16(r, nsat), sat);
    }
};
// LOC80-END CPP_SPEC_OPS_NEON
#endif

#if !defined(LDPC_SPEC_NO_SIMD)
// LOC80-BEGIN CPP_SPEC_INTER.module
constexpr bool simd_available = true;
template<typename T>
constexpr int
simd_width()
{
    return Ops<T>::W;
}
// LOC80-END CPP_SPEC_INTER.module
#else
constexpr bool simd_available = false;
template<typename T>
constexpr int
simd_width()
{
    return 1;
}
#endif

// -------------------------------------------------------------------------------------------------
// Inter-frame SIMD kernel: W frames, lane f = frame f. Layout: vn[v * W + f], msg[k * W + f].
// -------------------------------------------------------------------------------------------------
#if !defined(LDPC_SPEC_NO_SIMD)
// LOC80-BEGIN CPP_SPEC_INTER
template<typename T>
void
decode_inter(const Graph& g, T* vn, T* msg, T* con, const int n_ite, const T off, const T sat)
{
    using O = Ops<T>;
    using V = typename O::V;
    using S = typename O::S;
    constexpr int W = O::W;
    const int32_t* __restrict off_c = g.chk_off.data();
    const int32_t* __restrict c2v = g.chk_to_var.data();
    const V zero = O::set1(T(0)), vmax = O::set1(max_val<T>()), voff = O::set1(off);
    const V vsat = O::set1(sat), vnsat = O::set1(T(-sat));
    std::fill(msg, msg + (std::size_t)g.chk_off[g.n_chk] * W, T(0));

    for (int ite = 0; ite < n_ite; ite++)
        for (int c = 0; c < g.n_chk; c++)
        {
            const int k0 = off_c[c], k1 = off_c[c + 1];
            S sign = O::sign_init();
            V min1 = vmax, min2 = vmax;
            for (int k = k0, i = 0; k < k1; k++, i++)
            {
                const V x = O::sub(O::load(vn + (std::size_t)c2v[k] * W), O::load(msg + (std::size_t)k * W));
                const V a = O::abs(x);
                O::store(con + i * W, x);
                sign = O::sign_acc(sign, x);
                min2 = O::min(min2, O::max(a, min1));
                min1 = O::min(min1, a);
            }
            const V cst1 = O::max(zero, O::sub(min2, voff));
            const V cst2 = O::max(zero, O::sub(min1, voff));
            for (int k = k0, i = 0; k < k1; k++, i++)
            {
                const V x = O::load(con + i * W);
                const V m = O::oms_out(O::abs(x), min1, cst1, cst2, sign, x, vsat, vnsat);
                O::store(msg + (std::size_t)k * W, m);
                O::store(vn + (std::size_t)c2v[k] * W, O::add(x, m));
            }
        }
}
#endif

// Frame-major (W frames x N) -> lane-interleaved (N x W)
template<typename T>
void
transpose_in(const T* Y_N, T* vn, const int N, const int W)
{
    for (int f = 0; f < W; f++)
    {
        const T* src = Y_N + (std::size_t)f * N;
        for (int v = 0; v < N; v++)
            vn[(std::size_t)v * W + f] = src[v];
    }
}

template<typename T, typename B>
void
hard_decision_inter(const Graph& g, const T* vn, B* V_K, const int W)
{
    for (int f = 0; f < W; f++)
        for (int i = 0; i < g.K; i++)
            V_K[(std::size_t)f * g.K + i] = vn[(std::size_t)g.info_pos[i] * W + f] < T(0) ? B(1) : B(0);
}
// LOC80-END CPP_SPEC_INTER

} // namespace ldpc_spec

#endif // LDPC_SPEC_KERNELS_HPP_
