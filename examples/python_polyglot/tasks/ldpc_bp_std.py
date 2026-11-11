"""
Generic Modular LDPC Belief Propagation Decoder in Python / Numba for StreamPU (`Stateful_core`)
Tier 2 (`ldpc_bp_std.py`): Idiomatic, Zero-Allocation High-Performance Numba (~350 lines).

Mirrors `ldpc_bp_std.jl` using standard Numba performance idioms without any low-level LLVM intrinsics:
  - Zero `llvmlite` IR intrinsics, zero `@intrinsic`, zero raw pointer arithmetic.
  - `@njit(nogil=True, fastmath=True)` with flat CSR/CSC Tanner graph arrays.
  - Pre-allocated per-thread 2D workspace buffers (`(n_branches, W)` and `(N_cw, W)` where `W = 1`
    in Scalar mode and `W = n_frames` in `INTER` mode) for zero heap allocations on the hot path.
  - Explicit ahead-of-execution `.compile(...)` precompilation at module load.
"""

import math
import numpy as np
from numba import njit, types


# LOC80-BEGIN PY_STD
@njit(nogil=True, fastmath=True, inline="always")
def _sat_i16(x: np.int32, sat: np.int16) -> np.int16:
    s = np.int32(sat)
    if x > s:
        return sat
    if x < -s:
        return np.int16(-s)
    return np.int16(x)


@njit(nogil=True, fastmath=True, inline="always")
def _add_sat_i16(a: np.int16, b: np.int16, sat: np.int16) -> np.int16:
    return _sat_i16(np.int32(a) + np.int32(b), sat)


@njit(nogil=True, fastmath=True, inline="always")
def _sub_sat_i16(a: np.int16, b: np.int16) -> np.int16:
    return _sat_i16(np.int32(a) - np.int32(b), np.int16(32767))
    # LOC80-END PY_STD


@njit(nogil=True, fastmath=True, inline="always")
def _update_var_nodes_f32(
    var_offsets: np.ndarray,
    Y_N: np.ndarray,
    c2v: np.ndarray,
    v2c: np.ndarray,
    post: np.ndarray,
    acc_w: np.ndarray,
    write_v2c: bool,
) -> None:
    n_var = var_offsets.shape[0] - 1
    W = Y_N.shape[1]
    for v in range(n_var):
        e_start, e_stop = var_offsets[v], var_offsets[v + 1]
        for w in range(W):
            acc_w[w] = Y_N[v, w]
        for e in range(e_start, e_stop):
            for w in range(W):
                acc_w[w] = np.float32(acc_w[w] + c2v[e, w])
        if write_v2c:
            for e in range(e_start, e_stop):
                for w in range(W):
                    v2c[e, w] = np.float32(acc_w[w] - c2v[e, w])
        else:
            for w in range(W):
                post[v, w] = acc_w[w]


@njit(nogil=True, fastmath=True, inline="always")
def _update_var_nodes_i16(
    var_offsets: np.ndarray,
    Y_N: np.ndarray,
    c2v: np.ndarray,
    v2c: np.ndarray,
    post: np.ndarray,
    acc_w: np.ndarray,
    write_v2c: bool,
) -> None:
    n_var = var_offsets.shape[0] - 1
    W = Y_N.shape[1]
    max_i16 = np.int16(32767)
    for v in range(n_var):
        e_start, e_stop = var_offsets[v], var_offsets[v + 1]
        for w in range(W):
            acc_w[w] = Y_N[v, w]
        for e in range(e_start, e_stop):
            for w in range(W):
                acc_w[w] = _add_sat_i16(acc_w[w], c2v[e, w], max_i16)
        if write_v2c:
            for e in range(e_start, e_stop):
                for w in range(W):
                    v2c[e, w] = _sub_sat_i16(acc_w[w], c2v[e, w])
        else:
            for w in range(W):
                post[v, w] = acc_w[w]


@njit(nogil=True, fastmath=True, inline="always")
def _check_syndrome_f32(
    chk_offsets: np.ndarray, chk_to_var: np.ndarray, post: np.ndarray, synd_w: np.ndarray
) -> bool:
    n_chk = chk_offsets.shape[0] - 1
    W = post.shape[1]
    for c in range(n_chk):
        for w in range(W):
            synd_w[w] = np.uint8(0)
        for k in range(chk_offsets[c], chk_offsets[c + 1]):
            u = chk_to_var[k]
            for w in range(W):
                synd_w[w] ^= np.uint8(1) if post[u, w] < np.float32(0.0) else np.uint8(0)
        for w in range(W):
            if synd_w[w] != np.uint8(0):
                return False
    return True


@njit(nogil=True, fastmath=True, inline="always")
def _check_syndrome_i16(
    chk_offsets: np.ndarray, chk_to_var: np.ndarray, post: np.ndarray, synd_w: np.ndarray
) -> bool:
    n_chk = chk_offsets.shape[0] - 1
    W = post.shape[1]
    for c in range(n_chk):
        for w in range(W):
            synd_w[w] = np.uint8(0)
        for k in range(chk_offsets[c], chk_offsets[c + 1]):
            u = chk_to_var[k]
            for w in range(W):
                synd_w[w] ^= np.uint8(1) if post[u, w] < np.int16(0) else np.uint8(0)
        for w in range(W):
            if synd_w[w] != np.uint8(0):
                return False
    return True


# 1. Flooding + SPA (Float32)
@njit(nogil=True, fastmath=True, inline="always")
def _step_flooding_spa_f32(
    chk_offsets: np.ndarray,
    transpose: np.ndarray,
    v2c: np.ndarray,
    c2v: np.ndarray,
    vals: np.ndarray,
    sng_w: np.ndarray,
    prd_w: np.ndarray,
    nz_w: np.ndarray,
) -> None:
    max_tanh = np.float32(1.0) - np.float32(1.1920929e-7)
    n_chk = chk_offsets.shape[0] - 1
    W = v2c.shape[1]
    for c in range(n_chk):
        k_start, k_stop = chk_offsets[c], chk_offsets[c + 1]
        deg = k_stop - k_start
        for w in range(W):
            sng_w[w] = np.uint8(0)
            prd_w[w] = np.float32(1.0)
            nz_w[w] = np.int32(0)
        for idx in range(deg):
            t_k = transpose[k_start + idx]
            for w in range(W):
                v_val = v2c[t_k, w]
                res = np.float32(math.tanh(abs(float(v_val)) * 0.5))
                vals[idx, w] = res
                if res > np.float32(0.0):
                    if math.copysign(1.0, float(v_val)) < 0.0:
                        sng_w[w] ^= np.uint8(1)
                    prd_w[w] = np.float32(prd_w[w] * res)
                else:
                    nz_w[w] += np.int32(1)
        for idx in range(deg):
            t_k = transpose[k_start + idx]
            for w in range(W):
                v_val = v2c[t_k, w]
                val_v = vals[idx, w]
                isz = val_v == np.float32(0.0)
                nz = nz_w[w]
                if nz == 0 or (nz == 1 and isz):
                    rt = prd_w[w] if isz else np.float32(prd_w[w] / val_v)
                    if rt > max_tanh:
                        rt = max_tanh
                    mag = np.float32(2.0 * math.atanh(float(rt)))
                    v_neg = np.uint8(0) if isz else (np.uint8(1) if math.copysign(1.0, float(v_val)) < 0.0 else np.uint8(0))
                    s_out = sng_w[w] ^ v_neg
                    c2v[t_k, w] = -mag if s_out != np.uint8(0) else mag
                else:
                    c2v[t_k, w] = np.float32(0.0)


# 2a. Flooding + OMS (Float32)
@njit(nogil=True, fastmath=True, inline="always")
def _step_flooding_oms_f32(
    chk_offsets: np.ndarray,
    transpose: np.ndarray,
    offset: np.float32,
    v2c: np.ndarray,
    c2v: np.ndarray,
    sng_w: np.ndarray,
    min1_w: np.ndarray,
    min2_w: np.ndarray,
    cst1_w: np.ndarray,
    cst2_w: np.ndarray,
) -> None:
    n_chk = chk_offsets.shape[0] - 1
    W = v2c.shape[1]
    inf_f32 = np.float32(3.4028235e38)
    for c in range(n_chk):
        k_start, k_stop = chk_offsets[c], chk_offsets[c + 1]
        for w in range(W):
            sng_w[w] = np.uint8(0)
            min1_w[w] = inf_f32
            min2_w[w] = inf_f32
        for k in range(k_start, k_stop):
            t_k = transpose[k]
            for w in range(W):
                v_val = v2c[t_k, w]
                v_abs = np.float32(-v_val) if v_val < np.float32(0.0) else v_val
                sng_w[w] ^= np.uint8(1) if v_val < np.float32(0.0) else np.uint8(0)
                m1 = min1_w[w]
                m2 = min2_w[w]
                mx = v_abs if v_abs > m1 else m1
                min2_w[w] = mx if mx < m2 else m2
                min1_w[w] = v_abs if v_abs < m1 else m1
        for w in range(W):
            d1 = np.float32(min2_w[w] - offset)
            d2 = np.float32(min1_w[w] - offset)
            cst1_w[w] = d1 if d1 > np.float32(0.0) else np.float32(0.0)
            cst2_w[w] = d2 if d2 > np.float32(0.0) else np.float32(0.0)
        for k in range(k_start, k_stop):
            t_k = transpose[k]
            for w in range(W):
                v_val = v2c[t_k, w]
                v_abs = np.float32(-v_val) if v_val < np.float32(0.0) else v_val
                mag = cst1_w[w] if v_abs == min1_w[w] else cst2_w[w]
                s_out = sng_w[w] ^ (np.uint8(1) if v_val < np.float32(0.0) else np.uint8(0))
                c2v[t_k, w] = -mag if s_out != np.uint8(0) else mag


# 2b. Flooding + OMS (Int16)
@njit(nogil=True, fastmath=True, inline="always")
def _step_flooding_oms_i16(
    chk_offsets: np.ndarray,
    transpose: np.ndarray,
    offset: np.int16,
    sat_val: np.int16,
    v2c: np.ndarray,
    c2v: np.ndarray,
    sng_w: np.ndarray,
    min1_w: np.ndarray,
    min2_w: np.ndarray,
    cst1_w: np.ndarray,
    cst2_w: np.ndarray,
) -> None:
    n_chk = chk_offsets.shape[0] - 1
    W = v2c.shape[1]
    max_i16 = np.int16(32767)
    for c in range(n_chk):
        k_start, k_stop = chk_offsets[c], chk_offsets[c + 1]
        for w in range(W):
            sng_w[w] = np.uint8(0)
            min1_w[w] = max_i16
            min2_w[w] = max_i16
        for k in range(k_start, k_stop):
            t_k = transpose[k]
            for w in range(W):
                v_val = v2c[t_k, w]
                v_abs = np.int16(-v_val) if v_val < np.int16(0) else v_val
                sng_w[w] ^= np.uint8(1) if v_val < np.int16(0) else np.uint8(0)
                m1 = min1_w[w]
                m2 = min2_w[w]
                mx = v_abs if v_abs > m1 else m1
                min2_w[w] = mx if mx < m2 else m2
                min1_w[w] = v_abs if v_abs < m1 else m1
        for w in range(W):
            d1 = _sub_sat_i16(min2_w[w], offset)
            d2 = _sub_sat_i16(min1_w[w], offset)
            cst1_w[w] = d1 if d1 > np.int16(0) else np.int16(0)
            cst2_w[w] = d2 if d2 > np.int16(0) else np.int16(0)
        for k in range(k_start, k_stop):
            t_k = transpose[k]
            for w in range(W):
                v_val = v2c[t_k, w]
                v_abs = np.int16(-v_val) if v_val < np.int16(0) else v_val
                mag = cst1_w[w] if v_abs == min1_w[w] else cst2_w[w]
                s_out = sng_w[w] ^ (np.uint8(1) if v_val < np.int16(0) else np.uint8(0))
                res = np.int16(-mag) if s_out != np.uint8(0) else mag
                c2v[t_k, w] = _sat_i16(np.int32(res), sat_val)


# 3. Horizontal Layered + SPA (Float32)
@njit(nogil=True, fastmath=True, inline="always")
def _step_layered_spa_f32(
    chk_offsets: np.ndarray,
    chk_to_var: np.ndarray,
    vn: np.ndarray,
    msg: np.ndarray,
    vals: np.ndarray,
    contrib: np.ndarray,
    sng_w: np.ndarray,
    prd_w: np.ndarray,
    nz_w: np.ndarray,
) -> None:
    max_tanh = np.float32(1.0) - np.float32(1.1920929e-7)
    n_chk = chk_offsets.shape[0] - 1
    W = vn.shape[1]
    for c in range(n_chk):
        k_start, k_stop = chk_offsets[c], chk_offsets[c + 1]
        deg = k_stop - k_start
        for w in range(W):
            sng_w[w] = np.uint8(0)
            prd_w[w] = np.float32(1.0)
            nz_w[w] = np.int32(0)
        for idx in range(deg):
            k = k_start + idx
            u = chk_to_var[k]
            for w in range(W):
                v_val = np.float32(vn[u, w] - msg[k, w])
                contrib[idx, w] = v_val
                res = np.float32(math.tanh(abs(float(v_val)) * 0.5))
                vals[idx, w] = res
                if res > np.float32(0.0):
                    if math.copysign(1.0, float(v_val)) < 0.0:
                        sng_w[w] ^= np.uint8(1)
                    prd_w[w] = np.float32(prd_w[w] * res)
                else:
                    nz_w[w] += np.int32(1)
        for idx in range(deg):
            k = k_start + idx
            u = chk_to_var[k]
            for w in range(W):
                v_val = contrib[idx, w]
                val_v = vals[idx, w]
                isz = val_v == np.float32(0.0)
                nz = nz_w[w]
                if nz == 0 or (nz == 1 and isz):
                    rt = prd_w[w] if isz else np.float32(prd_w[w] / val_v)
                    if rt > max_tanh:
                        rt = max_tanh
                    mag = np.float32(2.0 * math.atanh(float(rt)))
                    v_neg = np.uint8(0) if isz else (np.uint8(1) if math.copysign(1.0, float(v_val)) < 0.0 else np.uint8(0))
                    s_out = sng_w[w] ^ v_neg
                    new_msg = -mag if s_out != np.uint8(0) else mag
                else:
                    new_msg = np.float32(0.0)
                msg[k, w] = new_msg
                vn[u, w] = np.float32(v_val + new_msg)


# 4a. Horizontal Layered + OMS (Float32)
# LOC80-BEGIN PY_STD
@njit(nogil=True, fastmath=True, inline="always")
def _step_layered_oms_f32(
    chk_offsets: np.ndarray,
    chk_to_var: np.ndarray,
    offset: np.float32,
    vn: np.ndarray,
    msg: np.ndarray,
    contrib: np.ndarray,
    sng_w: np.ndarray,
    min1_w: np.ndarray,
    min2_w: np.ndarray,
    cst1_w: np.ndarray,
    cst2_w: np.ndarray,
) -> None:
    n_chk = chk_offsets.shape[0] - 1
    W = vn.shape[1]
    inf_f32 = np.float32(3.4028235e38)
    for c in range(n_chk):
        k_start, k_stop = chk_offsets[c], chk_offsets[c + 1]
        deg = k_stop - k_start
        for w in range(W):
            sng_w[w] = np.uint8(0)
            min1_w[w] = inf_f32
            min2_w[w] = inf_f32
        for idx in range(deg):
            k = k_start + idx
            u = chk_to_var[k]
            for w in range(W):
                v_val = np.float32(vn[u, w] - msg[k, w])
                contrib[idx, w] = v_val
                v_abs = np.float32(-v_val) if v_val < np.float32(0.0) else v_val
                sng_w[w] ^= np.uint8(1) if v_val < np.float32(0.0) else np.uint8(0)
                m1 = min1_w[w]
                m2 = min2_w[w]
                mx = v_abs if v_abs > m1 else m1
                min2_w[w] = mx if mx < m2 else m2
                min1_w[w] = v_abs if v_abs < m1 else m1
        for w in range(W):
            d1 = np.float32(min2_w[w] - offset)
            d2 = np.float32(min1_w[w] - offset)
            cst1_w[w] = d1 if d1 > np.float32(0.0) else np.float32(0.0)
            cst2_w[w] = d2 if d2 > np.float32(0.0) else np.float32(0.0)
        for idx in range(deg):
            k = k_start + idx
            u = chk_to_var[k]
            for w in range(W):
                v_val = contrib[idx, w]
                v_abs = np.float32(-v_val) if v_val < np.float32(0.0) else v_val
                mag = cst1_w[w] if v_abs == min1_w[w] else cst2_w[w]
                s_out = sng_w[w] ^ (np.uint8(1) if v_val < np.float32(0.0) else np.uint8(0))
                new_msg = -mag if s_out != np.uint8(0) else mag
                msg[k, w] = new_msg
                vn[u, w] = np.float32(v_val + new_msg)


# 4b. Horizontal Layered + OMS (Int16)
@njit(nogil=True, fastmath=True, inline="always")
def _step_layered_oms_i16(
    chk_offsets: np.ndarray,
    chk_to_var: np.ndarray,
    offset: np.int16,
    sat_val: np.int16,
    vn: np.ndarray,
    msg: np.ndarray,
    contrib: np.ndarray,
    sng_w: np.ndarray,
    min1_w: np.ndarray,
    min2_w: np.ndarray,
    cst1_w: np.ndarray,
    cst2_w: np.ndarray,
) -> None:
    n_chk = chk_offsets.shape[0] - 1
    W = vn.shape[1]
    max_i16 = np.int16(32767)
    for c in range(n_chk):
        k_start, k_stop = chk_offsets[c], chk_offsets[c + 1]
        deg = k_stop - k_start
        for w in range(W):
            sng_w[w] = np.uint8(0)
            min1_w[w] = max_i16
            min2_w[w] = max_i16
        for idx in range(deg):
            k = k_start + idx
            u = chk_to_var[k]
            for w in range(W):
                v_val = _sub_sat_i16(vn[u, w], msg[k, w])
                contrib[idx, w] = v_val
                v_abs = np.int16(-v_val) if v_val < np.int16(0) else v_val
                sng_w[w] ^= np.uint8(1) if v_val < np.int16(0) else np.uint8(0)
                m1 = min1_w[w]
                m2 = min2_w[w]
                mx = v_abs if v_abs > m1 else m1
                min2_w[w] = mx if mx < m2 else m2
                min1_w[w] = v_abs if v_abs < m1 else m1
        for w in range(W):
            d1 = _sub_sat_i16(min2_w[w], offset)
            d2 = _sub_sat_i16(min1_w[w], offset)
            cst1_w[w] = d1 if d1 > np.int16(0) else np.int16(0)
            cst2_w[w] = d2 if d2 > np.int16(0) else np.int16(0)
        for idx in range(deg):
            k = k_start + idx
            u = chk_to_var[k]
            for w in range(W):
                v_val = contrib[idx, w]
                v_abs = np.int16(-v_val) if v_val < np.int16(0) else v_val
                mag = cst1_w[w] if v_abs == min1_w[w] else cst2_w[w]
                s_out = sng_w[w] ^ (np.uint8(1) if v_val < np.int16(0) else np.uint8(0))
                res = np.int16(-mag) if s_out != np.uint8(0) else mag
                new_msg = _sat_i16(np.int32(res), sat_val)
                msg[k, w] = new_msg
                vn[u, w] = _add_sat_i16(v_val, new_msg, sat_val)
                # LOC80-END PY_STD


# ====================================================================================================
# TOP-LEVEL ENTRY POINTS (`decode_siho_std_f32` / `decode_siho_std_i16`)
# ====================================================================================================

# LOC80-BEGIN PY_STD
@njit(nogil=True, fastmath=True)
def decode_siho_std_f32(
    var_offsets: np.ndarray,
    chk_offsets: np.ndarray,
    transpose: np.ndarray,
    chk_to_var: np.ndarray,
    info_bits_pos: np.ndarray,
    is_layered: bool,
    is_oms: bool,
    offset: np.float32,
    n_ite: int,
    enable_syndrome: bool,
    syndrome_depth: int,
    Y_N_buf: np.ndarray,
    post: np.ndarray,
    c2v: np.ndarray,
    v2c: np.ndarray,
    vals: np.ndarray,
    contrib: np.ndarray,
    sng_w: np.ndarray,
    min1_w: np.ndarray,
    min2_w: np.ndarray,
    cst1_w: np.ndarray,
    cst2_w: np.ndarray,
    prd_w: np.ndarray,
    nz_w: np.ndarray,
    synd_w: np.ndarray,
    Y_N_wave: np.ndarray,
    V_K_wave: np.ndarray,
) -> int:
    K = info_bits_pos.shape[0]
    N = var_offsets.shape[0] - 1
    W = Y_N_wave.shape[0]
    target_in = post if is_layered else Y_N_buf
    for w in range(W):
        for v in range(N):
            target_in[v, w] = Y_N_wave[w, v]
    c2v.fill(np.float32(0.0))

    cur_depth = 0
    valid = True
    ite_done = 0
    for ite in range(1, n_ite + 1):
    # LOC80-END PY_STD
        if not is_layered:
            _update_var_nodes_f32(var_offsets, Y_N_buf, c2v, v2c, post, min1_w, True)
            if not is_oms:
                _step_flooding_spa_f32(chk_offsets, transpose, v2c, c2v, vals, sng_w, prd_w, nz_w)
            else:
                _step_flooding_oms_f32(chk_offsets, transpose, offset, v2c, c2v, sng_w, min1_w, min2_w, cst1_w, cst2_w)
        else:
            if not is_oms:
                _step_layered_spa_f32(chk_offsets, chk_to_var, post, c2v, vals, contrib, sng_w, prd_w, nz_w)
            else:
                # LOC80-BEGIN PY_STD
                _step_layered_oms_f32(chk_offsets, chk_to_var, offset, post, c2v, contrib, sng_w, min1_w, min2_w, cst1_w, cst2_w)
                # LOC80-END PY_STD

        if enable_syndrome and (is_layered or ite != n_ite):
            if not is_layered:
                _update_var_nodes_f32(var_offsets, Y_N_buf, c2v, v2c, post, min1_w, False)
            synd = _check_syndrome_f32(chk_offsets, chk_to_var, post, synd_w)
            cur_depth = ((cur_depth + 1) % syndrome_depth) if synd else 0
            valid = synd and (cur_depth == 0)
            if valid:
                ite_done = ite
                break
        if ite == n_ite:
            ite_done = n_ite

    if not is_layered and ite_done == n_ite:
        _update_var_nodes_f32(var_offsets, Y_N_buf, c2v, v2c, post, min1_w, False)

    # LOC80-BEGIN PY_STD
    for w in range(W):
        for i in range(K):
            V_K_wave[w, i] = np.int32(1) if post[info_bits_pos[i], w] < np.float32(0.0) else np.int32(0)
    return 0 if valid else 1
    # LOC80-END PY_STD


# LOC80-BEGIN PY_STD
@njit(nogil=True, fastmath=True)
def decode_siho_std_i16(
    var_offsets: np.ndarray,
    chk_offsets: np.ndarray,
    transpose: np.ndarray,
    chk_to_var: np.ndarray,
    info_bits_pos: np.ndarray,
    is_layered: bool,
    offset: np.int16,
    sat_val: np.int16,
    n_ite: int,
    enable_syndrome: bool,
    syndrome_depth: int,
    Y_N_buf: np.ndarray,
    post: np.ndarray,
    c2v: np.ndarray,
    v2c: np.ndarray,
    contrib: np.ndarray,
    sng_w: np.ndarray,
    min1_w: np.ndarray,
    min2_w: np.ndarray,
    cst1_w: np.ndarray,
    cst2_w: np.ndarray,
    synd_w: np.ndarray,
    Y_N_wave: np.ndarray,
    V_K_wave: np.ndarray,
) -> int:
    K = info_bits_pos.shape[0]
    N = var_offsets.shape[0] - 1
    W = Y_N_wave.shape[0]
    target_in = post if is_layered else Y_N_buf
    for w in range(W):
        for v in range(N):
            target_in[v, w] = Y_N_wave[w, v]
    c2v.fill(np.int16(0))

    cur_depth = 0
    valid = True
    ite_done = 0
    for ite in range(1, n_ite + 1):
    # LOC80-END PY_STD
        if not is_layered:
            _update_var_nodes_i16(var_offsets, Y_N_buf, c2v, v2c, post, min1_w, True)
            _step_flooding_oms_i16(chk_offsets, transpose, offset, sat_val, v2c, c2v, sng_w, min1_w, min2_w, cst1_w, cst2_w)
        else:
            # LOC80-BEGIN PY_STD
            _step_layered_oms_i16(chk_offsets, chk_to_var, offset, sat_val, post, c2v, contrib, sng_w, min1_w, min2_w, cst1_w, cst2_w)
            # LOC80-END PY_STD

        if enable_syndrome and (is_layered or ite != n_ite):
            if not is_layered:
                _update_var_nodes_i16(var_offsets, Y_N_buf, c2v, v2c, post, min1_w, False)
            synd = _check_syndrome_i16(chk_offsets, chk_to_var, post, synd_w)
            cur_depth = ((cur_depth + 1) % syndrome_depth) if synd else 0
            valid = synd and (cur_depth == 0)
            if valid:
                ite_done = ite
                break
        if ite == n_ite:
            ite_done = n_ite

    if not is_layered and ite_done == n_ite:
        _update_var_nodes_i16(var_offsets, Y_N_buf, c2v, v2c, post, min1_w, False)

    # LOC80-BEGIN PY_STD
    for w in range(W):
        for i in range(K):
            V_K_wave[w, i] = np.int16(1) if post[info_bits_pos[i], w] < np.int16(0) else np.int16(0)
    return 0 if valid else 1
    # LOC80-END PY_STD


# Ahead-of-execution compilation for Float32 and Int16 entry points (mirroring Julia `precompile(...)`)
_I32_1D = types.int32[::1]
_F32_1D = types.float32[::1]
_F32_2D = types.float32[:, ::1]
_I16_1D = types.int16[::1]
_I16_2D = types.int16[:, ::1]
_U8_1D = types.uint8[::1]

decode_siho_std_f32.compile((
    _I32_1D, _I32_1D, _I32_1D, _I32_1D, _I32_1D,
    types.boolean, types.boolean, types.float32,
    types.int64, types.boolean, types.int64,
    _F32_2D, _F32_2D, _F32_2D, _F32_2D, _F32_2D, _F32_2D,
    _U8_1D, _F32_1D, _F32_1D, _F32_1D, _F32_1D, _F32_1D, _I32_1D, _U8_1D,
    _F32_2D, types.int32[:, ::1],
))

decode_siho_std_i16.compile((
    _I32_1D, _I32_1D, _I32_1D, _I32_1D, _I32_1D,
    types.boolean, types.int16, types.int16,
    types.int64, types.boolean, types.int64,
    _I16_2D, _I16_2D, _I16_2D, _I16_2D, _I16_2D,
    _U8_1D, _I16_1D, _I16_1D, _I16_1D, _I16_1D, _U8_1D,
    _I16_2D, _I16_2D,
))

