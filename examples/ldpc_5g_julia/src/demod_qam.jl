# Square Gray QAM demodulator in Julia, used in the receiver stage of the mixed-language pipeline
# (task `demodulate`, sockets CP [1 per frame], Y_N1 [2*ceil(N/bps) channel samples per frame], Y_N2 [N LLRs]).
#
# Mirrors AFF3CT's Modem_generic<..., Constellation_QAM, MAX>::demodulate: for each bit b of each received symbol y,
#   LLR_b = MAX_{j : bit b of j = 0} ( -|y - c_j|^2 / (2 sigma^2) ) - MAX_{j : bit b of j = 1} ( ... )
# with MAX = max (max-log, algo 0), max_linear (max + max(0, 0.301 - |a - b|/2), algo 1) or max* (exact log-MAP,
# algo 2), the points c_j being AFF3CT's own constellation (passed by the Python control plane), visited in the same
# order and in Float32. No @fastmath and no muladd: the arithmetic is the same as the Numba demodulator.

struct QAMDemod
    c_re::Vector{Float32}
    c_im::Vector{Float32}
    bps::Int
    algo::Int
    d::Vector{Float32}   # scratch (constellation metrics); the state is deep-copied per worker thread
end

QAMDemod(c_re::Vector{Float32}, c_im::Vector{Float32}, bps::Int, algo::Int) =
    QAMDemod(c_re, c_im, bps, algo, zeros(Float32, length(c_re)))

# LOC80-BEGIN JL_DEMOD_QAM
@inline vmax(::Val{0}, a::Float32, b::Float32) = a >= b ? a : b
@inline function vmax(::Val{1}, a::Float32, b::Float32)
    m = a >= b ? a : b
    c = 0.301f0 - 0.5f0 * abs(a - b)
    return m + (c > 0.0f0 ? c : 0.0f0)
end
@inline function vmax(::Val{2}, a::Float32, b::Float32)
    m = a >= b ? a : b
    return m + log1p(exp(-abs(a - b)))
end

function _demod_qam!(::Val{A}, st::QAMDemod, CP, Y1, Y2, n_fra::Int) where {A}
    M = length(st.c_re)
    bps = st.bps
    N = div(length(Y2), n_fra)
    Nm = div(length(Y1), n_fra)
    d = st.d
    @inbounds for f in 0:(n_fra - 1)
        s = CP[f + 1]
        inv_s2 = 1.0f0 / (2.0f0 * s * s)
        for k in 0:(div(N + bps - 1, bps) - 1)
            yr = Y1[f * Nm + 2k + 1]
            yi = Y1[f * Nm + 2k + 2]
            for j in 1:M
                dr = yr - st.c_re[j]
                di = yi - st.c_im[j]
                d[j] = -((dr * dr) + (di * di)) * inv_s2
            end
            for b in 0:(bps - 1)
                n = k * bps + b
                n >= N && break
                L0 = -Inf32
                L1 = -Inf32
                for j in 0:(M - 1)
                    if ((j >> b) & 1) == 0
                        L0 = vmax(Val(A), L0, d[j + 1])
                    else
                        L1 = vmax(Val(A), L1, d[j + 1])
                    end
                end
                Y2[f * N + n + 1] = L0 - L1
            end
        end
    end
    return nothing
end
# LOC80-END JL_DEMOD_QAM

function demod_qam_julia(st::QAMDemod, sck_CP::AbstractVector{Float32}, sck_Y_N1::AbstractVector{Float32},
                         sck_Y_N2::AbstractVector{Float32}, frame_id::UInt32 = UInt32(0),
                         n_frames_per_wave::UInt32 = UInt32(1))::Int32
    n_fra = Int(n_frames_per_wave)
    if st.algo == 1
        _demod_qam!(Val(1), st, sck_CP, sck_Y_N1, sck_Y_N2, n_fra)
    elseif st.algo == 2
        _demod_qam!(Val(2), st, sck_CP, sck_Y_N1, sck_Y_N2, n_fra)
    else
        _demod_qam!(Val(0), st, sck_CP, sck_Y_N1, sck_Y_N2, n_fra)
    end
    return Int32(0)
end

