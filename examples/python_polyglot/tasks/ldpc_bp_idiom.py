"""
LDPC Belief Propagation decoder in Python / Numba for StreamPU (`Stateful_core`)
Tier "IDIOM" (`ldpc_bp_idiom.py`): idiomatic Numba relying on LLVM auto-vectorization.

Horizontal layered schedule + Offset Min-Sum, fixed number of iterations (no syndrome), float32 or
int16 (fixed-point, saturating arithmetic). Bit-exact with the `opt` tier (`ldpc_bp_opt.py`).

Inter-frame mode: the W frames of a wave are processed together, one frame per "lane". All the
per-variable / per-branch buffers have shape (n, W) (C order, lanes innermost) so that every inner
`for w in range(W)` loop is a contiguous, independent loop that LLVM can vectorize.
No `llvmlite` intrinsics, no `@intrinsic`, no explicit IR. Scalar mode is the same code with W = 1.
"""

import numpy as np
from numba import njit, types


# LOC80-BEGIN PY_IDIOM_INTER
@njit(nogil=True, fastmath=True, inline="always")
def _sat_i16(x):
    """Saturate a (widened) integer to the int16 range."""
    return np.int16(min(max(x, -32768), 32767))


@njit(nogil=True, fastmath=True)
def decode_siho_idiom_f32(chk_offsets, chk_to_var, info_bits_pos, offset, n_ite,
                          post, msg, contrib, sign, min1, min2, cst1, cst2, Y_N_wave, V_K_wave):
    W, N = Y_N_wave.shape
    K = info_bits_pos.shape[0]
    zero = np.float32(0.0)
    for w in range(W):  # frames -> lanes layout
        for v in range(N):
            post[v, w] = Y_N_wave[w, v]
    msg[:] = zero
    for _ in range(n_ite):
        for c in range(chk_offsets.shape[0] - 1):
            k0, k1 = chk_offsets[c], chk_offsets[c + 1]
            sign[:] = 0
            min1[:] = np.float32(3.4028235e38)
            min2[:] = np.float32(3.4028235e38)
            # 1. variable-to-check messages, sign product and the two smallest magnitudes
            for k in range(k0, k1):
                u = chk_to_var[k]
                for w in range(W):
                    v = post[u, w] - msg[k, w]
                    contrib[k - k0, w] = v
                    sign[w] ^= v < zero
                    min2[w] = min(min2[w], max(abs(v), min1[w]))
                    min1[w] = min(min1[w], abs(v))
            for w in range(W):
                cst1[w] = max(zero, min2[w] - offset)
                cst2[w] = max(zero, min1[w] - offset)
            # 2. new check-to-variable messages and posterior update
            for k in range(k0, k1):
                u = chk_to_var[k]
                for w in range(W):
                    v = contrib[k - k0, w]
                    mag = cst1[w] if abs(v) == min1[w] else cst2[w]
                    m = -mag if sign[w] ^ (v < zero) else mag
                    msg[k, w] = m
                    post[u, w] = v + m
    for w in range(W):  # hard decision, lanes -> frames layout
        for i in range(K):
            V_K_wave[w, i] = post[info_bits_pos[i], w] < zero
    return 0


@njit(nogil=True, fastmath=True)
def decode_siho_idiom_i16(chk_offsets, chk_to_var, info_bits_pos, offset, sat_val, n_ite,
                          post, msg, contrib, sign, min1, min2, cst1, cst2, Y_N_wave, V_K_wave):
    W, N = Y_N_wave.shape
    K = info_bits_pos.shape[0]
    for w in range(W):  # frames -> lanes layout
        for v in range(N):
            post[v, w] = Y_N_wave[w, v]
    msg[:] = 0
    for _ in range(n_ite):
        for c in range(chk_offsets.shape[0] - 1):
            k0, k1 = chk_offsets[c], chk_offsets[c + 1]
            sign[:] = 0
            min1[:] = 32767
            min2[:] = 32767
            # 1. variable-to-check messages (saturating sub), sign product and the two smallest magnitudes
            for k in range(k0, k1):
                u = chk_to_var[k]
                for w in range(W):
                    v = _sat_i16(np.int32(post[u, w]) - np.int32(msg[k, w]))
                    a = np.int16(abs(v))  # int16 abs wraps on -32768 (as the opt tier)
                    contrib[k - k0, w] = v
                    sign[w] ^= v < 0
                    min2[w] = min(min2[w], max(a, min1[w]))
                    min1[w] = min(min1[w], a)
            for w in range(W):
                cst1[w] = max(0, _sat_i16(np.int32(min2[w]) - offset))
                cst2[w] = max(0, _sat_i16(np.int32(min1[w]) - offset))
            # 2. new messages clamped to +-sat_val, posterior updated with a saturating add
            for k in range(k0, k1):
                u = chk_to_var[k]
                for w in range(W):
                    v = contrib[k - k0, w]
                    mag = cst1[w] if np.int16(abs(v)) == min1[w] else cst2[w]
                    m = min(max(-mag if sign[w] ^ (v < 0) else mag, -sat_val), sat_val)
                    msg[k, w] = m
                    post[u, w] = _sat_i16(np.int32(v) + np.int32(m))
    for w in range(W):  # hard decision, lanes -> frames layout
        for i in range(K):
            V_K_wave[w, i] = post[info_bits_pos[i], w] < 0
    return 0
# LOC80-END PY_IDIOM_INTER


# Ahead-of-execution compilation (mirroring Julia `precompile(...)`)
_I32_1D = types.int32[::1]
_U8_1D = types.uint8[::1]
_F32_1D, _F32_2D = types.float32[::1], types.float32[:, ::1]
_I16_1D, _I16_2D = types.int16[::1], types.int16[:, ::1]
decode_siho_idiom_f32.compile((
    _I32_1D, _I32_1D, _I32_1D, types.float32, types.int32,
    _F32_2D, _F32_2D, _F32_2D, _U8_1D, _F32_1D, _F32_1D, _F32_1D, _F32_1D,
    _F32_2D, types.int32[:, ::1],
))
decode_siho_idiom_i16.compile((
    _I32_1D, _I32_1D, _I32_1D, types.int16, types.int16, types.int32,
    _I16_2D, _I16_2D, _I16_2D, _U8_1D, _I16_1D, _I16_1D, _I16_1D, _I16_1D,
    _I16_2D, _I16_2D,
))
