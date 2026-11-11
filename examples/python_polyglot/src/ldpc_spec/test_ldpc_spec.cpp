// Standalone check of the specialized LDPC kernels: inter-frame SIMD output must be bit-identical,
// lane by lane, to the scalar kernel (which mirrors the Julia/Numba OPT decoders). Also prints timings.
// Build: c++ -O3 -march=native -std=c++17 test_ldpc_spec.cpp -o test_ldpc_spec
#include <chrono>
#include <cstdio>
#include <random>
#include <set>
#include "ldpc_spec_kernels.hpp"

using namespace ldpc_spec;

template<typename T>
static int run(const Graph& g, const int n_ite, const T off, const T sat, std::mt19937& rng)
{
    const int W = simd_width<T>(), N = g.n_var, E = g.chk_off[g.n_chk];
    std::normal_distribution<float> noise(0.f, 1.f);
    avector<T> Y((size_t)W * N);
    for (auto& y : Y)
    {
        const float l = 2.f * (1.f + 0.8f * noise(rng));
        if constexpr (std::is_same<T, float>::value) y = l;
        else y = (T)std::max(-31.f, std::min(31.f, std::round(l * 4.f)));
    }
    // scalar reference, frame by frame
    std::vector<int32_t> ref((size_t)W * g.K);
    avector<T> vn(N), msg(E), con(g.max_chk_deg);
    auto t0 = std::chrono::steady_clock::now();
    for (int f = 0; f < W; f++)
    {
        decode_scalar<T>(g, Y.data() + (size_t)f * N, vn.data(), msg.data(), con.data(), n_ite, off, sat);
        hard_decision_scalar<T, int32_t>(g, vn.data(), ref.data() + (size_t)f * g.K);
    }
    auto t1 = std::chrono::steady_clock::now();
    int n_err = 0;
#if !defined(LDPC_SPEC_NO_SIMD)
    std::vector<int32_t> out((size_t)W * g.K);
    avector<T> vnI((size_t)N * W), msgI((size_t)E * W), conI((size_t)g.max_chk_deg * W);
    auto t2 = std::chrono::steady_clock::now();
    transpose_in<T>(Y.data(), vnI.data(), N, W);
    decode_inter<T>(g, vnI.data(), msgI.data(), conI.data(), n_ite, off, sat);
    hard_decision_inter<T, int32_t>(g, vnI.data(), out.data(), W);
    auto t3 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < out.size(); i++) n_err += (out[i] != ref[i]);
    // compare soft values too (lane by lane)
    for (int f = 0; f < W; f++)
    {
        decode_scalar<T>(g, Y.data() + (size_t)f * N, vn.data(), msg.data(), con.data(), n_ite, off, sat);
        for (int v = 0; v < N; v++)
        {
            const T a = vn[v], b = vnI[(size_t)v * W + f];
            if (!(a == b)) n_err++;
        }
    }
    const double us_s = std::chrono::duration<double, std::micro>(t1 - t0).count() / W;
    const double us_i = std::chrono::duration<double, std::micro>(t3 - t2).count() / W;
    std::printf("  %-7s W=%2d  mismatches=%d  scalar %.1f us/frame  inter %.2f us/frame  (x%.1f)\n",
                sizeof(T) == 4 ? "float32" : "int16", W, n_err, us_s, us_i, us_s / us_i);
#else
    std::printf("  no SIMD ISA available, scalar only\n");
#endif
    return n_err;
}

int main()
{
    std::mt19937 rng(42);
    const int n_var = 1100, n_chk = 836 / 2; // random sparse graph, check degrees 3..19
    std::vector<std::vector<uint32_t>> cols(n_chk);
    std::uniform_int_distribution<int> deg(3, 19), var(0, n_var - 1);
    for (auto& c : cols)
    {
        std::set<uint32_t> s;
        const int d = deg(rng);
        while ((int)s.size() < d) s.insert((uint32_t)var(rng));
        c.assign(s.begin(), s.end());
    }
    std::vector<uint32_t> pos(128);
    for (int i = 0; i < 128; i++) pos[i] = (uint32_t)(i * 3);
    Graph g(n_var, cols, pos);
#if defined(LDPC_SPEC_AVX512)
    std::printf("ISA: AVX-512\n");
#elif defined(LDPC_SPEC_AVX2)
    std::printf("ISA: AVX2\n");
#elif defined(LDPC_SPEC_SSE41)
    std::printf("ISA: SSE4.1\n");
#elif defined(LDPC_SPEC_NEON)
    std::printf("ISA: NEON\n");
#endif
    int err = 0;
    err += run<float>(g, 10, 0.15f, 0.f, rng);
    err += run<int16_t>(g, 10, (int16_t)1, (int16_t)2047, rng);
    std::printf(err ? "FAILED\n" : "OK\n");
    return err != 0;
}
