"""
Generic Modular LDPC Belief Propagation Decoder in Python / Numba for StreamPU (`Stateful_core`)
Tier 1 (`ldpc_bp_naive.py`): Compact, Textbook Algorithmic Implementation (~200 lines).

Mirrors `ldpc_bp_naive.jl` and prioritizes maximum readability and closeness to the
mathematical equations:
  - Standard `@njit(nogil=True)` without `fastmath=True` or `llvmlite` intrinsics.
  - Directly expresses Flooding (`BP_FLOODING`) and Horizontal Layered (`BP_HORIZONTAL_LAYERED`)
    schedules with Sum-Product (`SPA`) and Offset Min-Sum (`OMS`) check-node updates,
    allocating `incoming` / `outgoing` edge vectors per check node.
  - In inter-frame mode (`INTER`, W frames per wave), loops sequentially over each frame `w in range(W)`.
"""

import math
import numpy as np
from numba import njit, types


@njit(nogil=True)
def _sat_i16(x: np.int32, sat: np.int16) -> np.int16:
    s = np.int32(sat)
    if x > s:
        return sat
    if x < -s:
        return np.int16(-s)
    return np.int16(x)


@njit(nogil=True)
def _check_node_update_f32(v2c: np.ndarray, is_oms: bool, offset: np.float32) -> np.ndarray:
    deg = v2c.shape[0]
    c2v = np.empty(deg, dtype=np.float32)
    if not is_oms:  # Sum-Product Algorithm (SPA)
        max_tanh = np.float32(1.0) - np.float32(1.1920929e-7)
        t_vals = np.empty(deg, dtype=np.float32)
        for j in range(deg):
            t_vals[j] = np.float32(math.tanh(abs(float(v2c[j])) * 0.5))
        for i in range(deg):
            prod_ext = np.float32(1.0)
            sgn_ext = False
            for j in range(deg):
                if j != i:
                    prod_ext = np.float32(prod_ext * t_vals[j])
                    sgn_ext ^= math.copysign(1.0, float(v2c[j])) < 0.0
            clamped = prod_ext if prod_ext < max_tanh else max_tanh
            mag = np.float32(2.0 * math.atanh(float(clamped)))
            c2v[i] = -mag if sgn_ext else mag
    else:  # Offset Min-Sum (OMS)
        mags = np.abs(v2c)
        tot_sgn = False
        for j in range(deg):
            tot_sgn ^= v2c[j] < np.float32(0.0)
        for i in range(deg):
            min_ext = np.float32(1e30)
            for j in range(deg):
                if j != i and mags[j] < min_ext:
                    min_ext = mags[j]
            mag = max(np.float32(0.0), min_ext - offset)
            sgn = tot_sgn ^ (v2c[i] < np.float32(0.0))
            c2v[i] = -mag if sgn else mag
    return c2v


@njit(nogil=True)
def _check_node_update_i16(v2c: np.ndarray, is_oms: bool, offset: np.int16, sat_val: np.int16) -> np.ndarray:
    deg = v2c.shape[0]
    c2v = np.empty(deg, dtype=np.int16)
    if not is_oms:  # Sum-Product Algorithm (SPA)
        max_tanh = np.float32(1.0) - np.float32(1.1920929e-7)
        t_vals = np.empty(deg, dtype=np.float32)
        for j in range(deg):
            t_vals[j] = np.float32(math.tanh(abs(float(v2c[j])) * 0.5))
        for i in range(deg):
            prod_ext = np.float32(1.0)
            sgn_ext = False
            for j in range(deg):
                if j != i:
                    prod_ext = np.float32(prod_ext * t_vals[j])
                    sgn_ext ^= v2c[j] < np.int16(0)
            clamped = prod_ext if prod_ext < max_tanh else max_tanh
            mag = np.float32(2.0 * math.atanh(float(clamped)))
            val = np.int32(-mag if sgn_ext else mag)
            c2v[i] = _sat_i16(val, sat_val)
    else:  # Offset Min-Sum (OMS)
        mags = np.empty(deg, dtype=np.int16)
        tot_sgn = False
        for j in range(deg):
            vj = v2c[j]
            mags[j] = np.int16(-vj) if vj < np.int16(0) else vj
            tot_sgn ^= vj < np.int16(0)
        for i in range(deg):
            min_ext = np.int16(32767)
            for j in range(deg):
                if j != i and mags[j] < min_ext:
                    min_ext = mags[j]
            diff = np.int32(min_ext) - np.int32(offset)
            mag = np.int16(diff) if diff > 0 else np.int16(0)
            sgn = tot_sgn ^ (v2c[i] < np.int16(0))
            c2v[i] = _sat_i16(np.int32(-mag) if sgn else np.int32(mag), sat_val)
    return c2v


@njit(nogil=True)
def _check_syndrome_f32(chk_offsets: np.ndarray, chk_to_var: np.ndarray, post: np.ndarray) -> bool:
    n_chk = chk_offsets.shape[0] - 1
    for c in range(n_chk):
        parity = False
        for k in range(chk_offsets[c], chk_offsets[c + 1]):
            parity ^= post[chk_to_var[k]] < np.float32(0.0)
        if parity:
            return False
    return True


@njit(nogil=True)
def _check_syndrome_i16(chk_offsets: np.ndarray, chk_to_var: np.ndarray, post: np.ndarray) -> bool:
    n_chk = chk_offsets.shape[0] - 1
    for c in range(n_chk):
        parity = False
        for k in range(chk_offsets[c], chk_offsets[c + 1]):
            parity ^= post[chk_to_var[k]] < np.int16(0)
        if parity:
            return False
    return True


@njit(nogil=True)
def _decode_single_frame_f32(
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
    c2v: np.ndarray,
    v2c: np.ndarray,
    post: np.ndarray,
    Y_N: np.ndarray,
    V_K: np.ndarray,
) -> bool:
    n_var = var_offsets.shape[0] - 1
    n_chk = chk_offsets.shape[0] - 1
    c2v.fill(np.float32(0.0))
    post[:] = Y_N
    cur_depth = 0
    valid = True

    for ite in range(1, n_ite + 1):
        if not is_layered:  # Flooding schedule
            for v in range(n_var):
                e_start, e_stop = var_offsets[v], var_offsets[v + 1]
                tot = Y_N[v]
                for e in range(e_start, e_stop):
                    tot = np.float32(tot + c2v[e])
                post[v] = tot
                for e in range(e_start, e_stop):
                    v2c[e] = np.float32(tot - c2v[e])
            for c in range(n_chk):
                k_start, k_stop = chk_offsets[c], chk_offsets[c + 1]
                deg = k_stop - k_start
                incoming = np.empty(deg, dtype=np.float32)
                for idx in range(deg):
                    incoming[idx] = v2c[transpose[k_start + idx]]
                outgoing = _check_node_update_f32(incoming, is_oms, offset)
                for idx in range(deg):
                    c2v[transpose[k_start + idx]] = outgoing[idx]
            if (enable_syndrome and ite != n_ite) or ite == n_ite:
                for v in range(n_var):
                    tot = Y_N[v]
                    for e in range(var_offsets[v], var_offsets[v + 1]):
                        tot = np.float32(tot + c2v[e])
                    post[v] = tot
        else:  # Horizontal Layered schedule
            for c in range(n_chk):
                k_start, k_stop = chk_offsets[c], chk_offsets[c + 1]
                deg = k_stop - k_start
                incoming = np.empty(deg, dtype=np.float32)
                for idx in range(deg):
                    k = k_start + idx
                    incoming[idx] = np.float32(post[chk_to_var[k]] - c2v[k])
                outgoing = _check_node_update_f32(incoming, is_oms, offset)
                for idx in range(deg):
                    k = k_start + idx
                    c2v[k] = outgoing[idx]
                    post[chk_to_var[k]] = np.float32(incoming[idx] + outgoing[idx])

        if enable_syndrome and (is_layered or ite != n_ite):
            synd = _check_syndrome_f32(chk_offsets, chk_to_var, post)
            cur_depth = ((cur_depth + 1) % syndrome_depth) if synd else 0
            valid = synd and (cur_depth == 0)
            if valid:
                break

    for i in range(info_bits_pos.shape[0]):
        V_K[i] = np.int32(1) if post[info_bits_pos[i]] < np.float32(0.0) else np.int32(0)
    return valid


@njit(nogil=True)
def _decode_single_frame_i16(
    var_offsets: np.ndarray,
    chk_offsets: np.ndarray,
    transpose: np.ndarray,
    chk_to_var: np.ndarray,
    info_bits_pos: np.ndarray,
    is_layered: bool,
    is_oms: bool,
    offset: np.int16,
    sat_val: np.int16,
    n_ite: int,
    enable_syndrome: bool,
    syndrome_depth: int,
    c2v: np.ndarray,
    v2c: np.ndarray,
    post: np.ndarray,
    Y_N: np.ndarray,
    V_K: np.ndarray,
) -> bool:
    n_var = var_offsets.shape[0] - 1
    n_chk = chk_offsets.shape[0] - 1
    max_i16 = np.int16(32767)
    c2v.fill(np.int16(0))
    post[:] = Y_N
    cur_depth = 0
    valid = True

    for ite in range(1, n_ite + 1):
        if not is_layered:  # Flooding schedule
            for v in range(n_var):
                e_start, e_stop = var_offsets[v], var_offsets[v + 1]
                tot = np.int32(Y_N[v])
                for e in range(e_start, e_stop):
                    tot += np.int32(c2v[e])
                post[v] = _sat_i16(tot, max_i16)
                for e in range(e_start, e_stop):
                    v2c[e] = _sat_i16(tot - np.int32(c2v[e]), max_i16)
            for c in range(n_chk):
                k_start, k_stop = chk_offsets[c], chk_offsets[c + 1]
                deg = k_stop - k_start
                incoming = np.empty(deg, dtype=np.int16)
                for idx in range(deg):
                    incoming[idx] = v2c[transpose[k_start + idx]]
                outgoing = _check_node_update_i16(incoming, is_oms, offset, sat_val)
                for idx in range(deg):
                    c2v[transpose[k_start + idx]] = outgoing[idx]
            if (enable_syndrome and ite != n_ite) or ite == n_ite:
                for v in range(n_var):
                    tot = np.int32(Y_N[v])
                    for e in range(var_offsets[v], var_offsets[v + 1]):
                        tot += np.int32(c2v[e])
                    post[v] = _sat_i16(tot, max_i16)
        else:  # Horizontal Layered schedule
            for c in range(n_chk):
                k_start, k_stop = chk_offsets[c], chk_offsets[c + 1]
                deg = k_stop - k_start
                incoming = np.empty(deg, dtype=np.int16)
                for idx in range(deg):
                    k = k_start + idx
                    incoming[idx] = _sat_i16(np.int32(post[chk_to_var[k]]) - np.int32(c2v[k]), max_i16)
                outgoing = _check_node_update_i16(incoming, is_oms, offset, sat_val)
                for idx in range(deg):
                    k = k_start + idx
                    c2v[k] = outgoing[idx]
                    post[chk_to_var[k]] = _sat_i16(np.int32(incoming[idx]) + np.int32(outgoing[idx]), sat_val)

        if enable_syndrome and (is_layered or ite != n_ite):
            synd = _check_syndrome_i16(chk_offsets, chk_to_var, post)
            cur_depth = ((cur_depth + 1) % syndrome_depth) if synd else 0
            valid = synd and (cur_depth == 0)
            if valid:
                break

    for i in range(info_bits_pos.shape[0]):
        V_K[i] = np.int16(1) if post[info_bits_pos[i]] < np.int16(0) else np.int16(0)
    return valid


@njit(nogil=True)
def decode_siho_naive_f32(
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
    c2v: np.ndarray,
    v2c: np.ndarray,
    post: np.ndarray,
    Y_N_wave: np.ndarray,
    V_K_wave: np.ndarray,
) -> int:
    W = Y_N_wave.shape[0]
    all_valid = True
    for w in range(W):
        all_valid &= _decode_single_frame_f32(
            var_offsets, chk_offsets, transpose, chk_to_var, info_bits_pos,
            is_layered, is_oms, offset, n_ite, enable_syndrome, syndrome_depth,
            c2v, v2c, post, Y_N_wave[w], V_K_wave[w],
        )
    return 0 if all_valid else 1


@njit(nogil=True)
def decode_siho_naive_i16(
    var_offsets: np.ndarray,
    chk_offsets: np.ndarray,
    transpose: np.ndarray,
    chk_to_var: np.ndarray,
    info_bits_pos: np.ndarray,
    is_layered: bool,
    is_oms: bool,
    offset: np.int16,
    sat_val: np.int16,
    n_ite: int,
    enable_syndrome: bool,
    syndrome_depth: int,
    c2v: np.ndarray,
    v2c: np.ndarray,
    post: np.ndarray,
    Y_N_wave: np.ndarray,
    V_K_wave: np.ndarray,
) -> int:
    W = Y_N_wave.shape[0]
    all_valid = True
    for w in range(W):
        all_valid &= _decode_single_frame_i16(
            var_offsets, chk_offsets, transpose, chk_to_var, info_bits_pos,
            is_layered, is_oms, offset, sat_val, n_ite, enable_syndrome, syndrome_depth,
            c2v, v2c, post, Y_N_wave[w], V_K_wave[w],
        )
    return 0 if all_valid else 1


# Ahead-of-execution compilation for Float32 and Int16 entry points (mirroring Julia `precompile(...)`)
_I32_1D = types.int32[::1]
decode_siho_naive_f32.compile((
    _I32_1D, _I32_1D, _I32_1D, _I32_1D, _I32_1D,
    types.boolean, types.boolean, types.float32,
    types.int64, types.boolean, types.int64,
    types.float32[::1], types.float32[::1], types.float32[::1],
    types.float32[:, ::1], types.int32[:, ::1],
))
decode_siho_naive_i16.compile((
    _I32_1D, _I32_1D, _I32_1D, _I32_1D, _I32_1D,
    types.boolean, types.boolean, types.int16, types.int16,
    types.int64, types.boolean, types.int64,
    types.int16[::1], types.int16[::1], types.int16[::1],
    types.int16[:, ::1], types.int16[:, ::1],
))

