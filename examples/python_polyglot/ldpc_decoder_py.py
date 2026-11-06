"""Generic Modular LDPC Belief Propagation Decoder in Python (NumPy & Numba) for StreamPU (`Stateful_core`).

Mirrors AFF3CT's C++ template architecture and the Julia parametric-dispatch architecture by decoupling
four orthogonal axes:
  1. Scheduling (`dec_type`):
       - `"BP_FLOODING"`           (Flooding schedule / BP-F)
       - `"BP_HORIZONTAL_LAYERED"` (Horizontal Layered schedule / BP-HL)
  2. Check-Node Update Rule (`dec_implem`):
       - `"SPA"` (Sum-Product Algorithm, `f32`)
       - `"OMS"` (Offset Min-Sum, `f32` or `i16`)
  3. Representation (`q_type`):
       - `"f32"` (`np.float32` LLRs, `np.int32` decoded bits)
       - `"i16"` (`np.int16` fixed-point LLRs with `[-sat_val, +sat_val]` saturation, `np.int16` decoded bits)
  4. Vectorization / Execution Mode (`simd_strategy`):
       - `""`      (Scalar mode: 1 frame per wave)
       - `"INTER"` (Inter-frame SIMD mode across `W = n_frames_per_wave` frames simultaneously, stored with `W`
                    contiguous along the innermost row-major dimension `(N_cw, W)` / `(n_branches, W)`)
"""

import math
import numpy as np
import streampu as spu
import py_aff3ct


# ====================================================================================================
# 1. NUMBA JIT KERNELS (`@njit(nogil=True, fastmath=True, cache=True)`)
# ====================================================================================================

_NUMBA_KERNELS = None


def _get_numba_kernels():
    global _NUMBA_KERNELS
    if _NUMBA_KERNELS is not None:
        return _NUMBA_KERNELS

    from numba import njit

    MAX_TANH_F32 = np.float32(1.0) - np.float32(np.finfo(np.float32).eps)
    MAX_F32 = np.float32(np.finfo(np.float32).max)
    MAX_I16 = np.int16(32767)
    MIN_I16 = np.int16(-32768)

    @njit(nogil=True, fastmath=True)
    def _sat_i16(val_i32, limit_i16):
        lim = np.int32(limit_i16)
        v1 = lim if val_i32 > lim else val_i32
        v2 = -lim if v1 < -lim else v1
        return np.int16(v2)

    @njit(nogil=True, fastmath=True)
    def _clamp_i16(val_i32):
        v1 = np.int32(32767) if val_i32 > np.int32(32767) else val_i32
        v2 = np.int32(-32768) if v1 < np.int32(-32768) else v1
        return np.int16(v2)

    @njit(nogil=True, fastmath=True)
    def _check_syndrome_scalar_f32(chk_offsets, chk_to_var, post):
        n_chk = len(chk_offsets) - 1
        for c in range(n_chk):
            k_start = chk_offsets[c]
            k_stop = chk_offsets[c + 1]
            sign = False
            for k in range(k_start, k_stop):
                if post[chk_to_var[k]] < np.float32(0.0):
                    sign = not sign
            if sign:
                return False
        return True

    @njit(nogil=True, fastmath=True)
    def _check_syndrome_scalar_i16(chk_offsets, chk_to_var, post):
        n_chk = len(chk_offsets) - 1
        for c in range(n_chk):
            k_start = chk_offsets[c]
            k_stop = chk_offsets[c + 1]
            sign = False
            for k in range(k_start, k_stop):
                if post[chk_to_var[k]] < np.int16(0):
                    sign = not sign
            if sign:
                return False
        return True

    @njit(nogil=True, fastmath=True)
    def _check_syndrome_inter_f32(chk_offsets, chk_to_var, post_inter, synd_w, W):
        n_chk = len(chk_offsets) - 1
        for c in range(n_chk):
            k_start = chk_offsets[c]
            k_stop = chk_offsets[c + 1]
            for w in range(W):
                synd_w[w] = np.uint8(0)
            for k in range(k_start, k_stop):
                u = chk_to_var[k]
                for w in range(W):
                    synd_w[w] ^= np.uint8(1) if post_inter[u, w] < np.float32(0.0) else np.uint8(0)
            any_fail = np.uint8(0)
            for w in range(W):
                any_fail |= synd_w[w]
            if any_fail != np.uint8(0):
                return False
        return True

    @njit(nogil=True, fastmath=True)
    def _check_syndrome_inter_i16(chk_offsets, chk_to_var, post_inter, synd_w, W):
        n_chk = len(chk_offsets) - 1
        for c in range(n_chk):
            k_start = chk_offsets[c]
            k_stop = chk_offsets[c + 1]
            for w in range(W):
                synd_w[w] = np.uint8(0)
            for k in range(k_start, k_stop):
                u = chk_to_var[k]
                for w in range(W):
                    synd_w[w] ^= np.uint8(1) if post_inter[u, w] < np.int16(0) else np.uint8(0)
            any_fail = np.uint8(0)
            for w in range(W):
                any_fail |= synd_w[w]
            if any_fail != np.uint8(0):
                return False
        return True

    # ------------------------------------------------------------------
    # SCALAR NUMBA KERNELS (1 frame per call)
    # ------------------------------------------------------------------

    @njit(nogil=True, fastmath=True)
    def decode_flooding_spa_f32_scalar(
        var_offsets, chk_offsets, transpose, chk_to_var, info_bits_pos,
        n_ite, enable_syndrome, syndrome_depth,
        msg_chk_to_var, msg_var_to_chk, post, values,
        Y_N, V_K
    ):
        n_var = len(var_offsets) - 1
        n_chk = len(chk_offsets) - 1
        K = len(info_bits_pos)
        msg_chk_to_var.fill(np.float32(0.0))
        cur_synd_depth = 0
        valid_synd = True
        ite_done = 0

        for ite in range(1, n_ite + 1):
            for v in range(n_var):
                e_start = var_offsets[v]
                e_stop = var_offsets[v + 1]
                s = np.float32(0.0)
                for e in range(e_start, e_stop):
                    s += msg_chk_to_var[e]
                tmp = Y_N[v] + s
                for e in range(e_start, e_stop):
                    msg_var_to_chk[e] = tmp - msg_chk_to_var[e]

            for c in range(n_chk):
                k_start = chk_offsets[c]
                k_stop = chk_offsets[c + 1]
                sign = False
                product = np.float32(1.0)
                n_zeros = 0
                idx = 0
                for k in range(k_start, k_stop):
                    var_val = msg_var_to_chk[transpose[k]]
                    var_abs = abs(var_val)
                    res = np.float32(math.tanh(var_abs * np.float32(0.5)))
                    if res == np.float32(0.0):
                        n_zeros += 1
                    else:
                        if math.copysign(1.0, var_val) < 0.0:
                            sign = not sign
                        product *= res
                    values[idx] = res
                    idx += 1

                if n_zeros == 0:
                    idx = 0
                    for k in range(k_start, k_stop):
                        t_k = transpose[k]
                        var_val = msg_var_to_chk[t_k]
                        val_v = values[idx]
                        res_tmp = product / val_v
                        if not (res_tmp < np.float32(1.0)):
                            res_tmp = MAX_TANH_F32
                        res_abs = np.float32(2.0) * np.float32(math.atanh(res_tmp))
                        res_sng = sign ^ (math.copysign(1.0, var_val) < 0.0)
                        msg_chk_to_var[t_k] = -res_abs if res_sng else res_abs
                        idx += 1
                elif n_zeros == 1:
                    res_tmp = product if product < np.float32(1.0) else MAX_TANH_F32
                    res_abs = np.float32(2.0) * np.float32(math.atanh(res_tmp))
                    res_val = -res_abs if sign else res_abs
                    idx = 0
                    for k in range(k_start, k_stop):
                        t_k = transpose[k]
                        msg_chk_to_var[t_k] = res_val if values[idx] == np.float32(0.0) else np.float32(0.0)
                        idx += 1
                else:
                    for k in range(k_start, k_stop):
                        msg_chk_to_var[transpose[k]] = np.float32(0.0)

            if enable_syndrome and ite != n_ite:
                for v in range(n_var):
                    e_start = var_offsets[v]
                    e_stop = var_offsets[v + 1]
                    s = np.float32(0.0)
                    for e in range(e_start, e_stop):
                        s += msg_chk_to_var[e]
                    post[v] = Y_N[v] + s
                synd = _check_syndrome_scalar_f32(chk_offsets, chk_to_var, post)
                cur_synd_depth = ((cur_synd_depth + 1) % syndrome_depth) if synd else 0
                valid_synd = synd and (cur_synd_depth == 0)
                if valid_synd:
                    ite_done = ite
                    break
            if ite == n_ite:
                ite_done = n_ite

        if ite_done == n_ite:
            for v in range(n_var):
                e_start = var_offsets[v]
                e_stop = var_offsets[v + 1]
                s = np.float32(0.0)
                for e in range(e_start, e_stop):
                    s += msg_chk_to_var[e]
                post[v] = Y_N[v] + s

        for i in range(K):
            V_K[i] = np.int32(1) if post[info_bits_pos[i]] < np.float32(0.0) else np.int32(0)
        return 0 if valid_synd else 1

    @njit(nogil=True, fastmath=True)
    def decode_flooding_oms_f32_scalar(
        var_offsets, chk_offsets, transpose, chk_to_var, info_bits_pos,
        n_ite, enable_syndrome, syndrome_depth, offset,
        msg_chk_to_var, msg_var_to_chk, post,
        Y_N, V_K
    ):
        n_var = len(var_offsets) - 1
        n_chk = len(chk_offsets) - 1
        K = len(info_bits_pos)
        off = np.float32(offset)
        msg_chk_to_var.fill(np.float32(0.0))
        cur_synd_depth = 0
        valid_synd = True
        ite_done = 0

        for ite in range(1, n_ite + 1):
            for v in range(n_var):
                e_start = var_offsets[v]
                e_stop = var_offsets[v + 1]
                s = np.float32(0.0)
                for e in range(e_start, e_stop):
                    s += msg_chk_to_var[e]
                tmp = Y_N[v] + s
                for e in range(e_start, e_stop):
                    msg_var_to_chk[e] = tmp - msg_chk_to_var[e]

            for c in range(n_chk):
                k_start = chk_offsets[c]
                k_stop = chk_offsets[c + 1]
                sign = False
                min1 = MAX_F32
                min2 = MAX_F32
                for k in range(k_start, k_stop):
                    var_val = msg_var_to_chk[transpose[k]]
                    var_abs = abs(var_val)
                    if var_val < np.float32(0.0):
                        sign = not sign
                    m_max = var_abs if var_abs > min1 else min1
                    if m_max < min2:
                        min2 = m_max
                    if var_abs < min1:
                        min1 = var_abs

                cst1 = min2 - off
                if cst1 < np.float32(0.0):
                    cst1 = np.float32(0.0)
                cst2 = min1 - off
                if cst2 < np.float32(0.0):
                    cst2 = np.float32(0.0)

                for k in range(k_start, k_stop):
                    t_k = transpose[k]
                    var_val = msg_var_to_chk[t_k]
                    var_abs = abs(var_val)
                    res_abs = cst1 if var_abs == min1 else cst2
                    res_sng = sign ^ (var_val < np.float32(0.0))
                    msg_chk_to_var[t_k] = -res_abs if res_sng else res_abs

            if enable_syndrome and ite != n_ite:
                for v in range(n_var):
                    e_start = var_offsets[v]
                    e_stop = var_offsets[v + 1]
                    s = np.float32(0.0)
                    for e in range(e_start, e_stop):
                        s += msg_chk_to_var[e]
                    post[v] = Y_N[v] + s
                synd = _check_syndrome_scalar_f32(chk_offsets, chk_to_var, post)
                cur_synd_depth = ((cur_synd_depth + 1) % syndrome_depth) if synd else 0
                valid_synd = synd and (cur_synd_depth == 0)
                if valid_synd:
                    ite_done = ite
                    break
            if ite == n_ite:
                ite_done = n_ite

        if ite_done == n_ite:
            for v in range(n_var):
                e_start = var_offsets[v]
                e_stop = var_offsets[v + 1]
                s = np.float32(0.0)
                for e in range(e_start, e_stop):
                    s += msg_chk_to_var[e]
                post[v] = Y_N[v] + s

        for i in range(K):
            V_K[i] = np.int32(1) if post[info_bits_pos[i]] < np.float32(0.0) else np.int32(0)
        return 0 if valid_synd else 1

    @njit(nogil=True, fastmath=True)
    def decode_flooding_oms_i16_scalar(
        var_offsets, chk_offsets, transpose, chk_to_var, info_bits_pos,
        n_ite, enable_syndrome, syndrome_depth, offset_i16, sat_val_i16,
        msg_chk_to_var, msg_var_to_chk, post,
        Y_N, V_K
    ):
        n_var = len(var_offsets) - 1
        n_chk = len(chk_offsets) - 1
        K = len(info_bits_pos)
        off = np.int16(offset_i16)
        msg_chk_to_var.fill(np.int16(0))
        cur_synd_depth = 0
        valid_synd = True
        ite_done = 0

        for ite in range(1, n_ite + 1):
            for v in range(n_var):
                e_start = var_offsets[v]
                e_stop = var_offsets[v + 1]
                s = np.int32(0)
                for e in range(e_start, e_stop):
                    s += np.int32(msg_chk_to_var[e])
                tmp = np.int32(Y_N[v]) + s
                for e in range(e_start, e_stop):
                    msg_var_to_chk[e] = _clamp_i16(tmp - np.int32(msg_chk_to_var[e]))

            for c in range(n_chk):
                k_start = chk_offsets[c]
                k_stop = chk_offsets[c + 1]
                sign = False
                min1 = MAX_I16
                min2 = MAX_I16
                for k in range(k_start, k_stop):
                    var_val = msg_var_to_chk[transpose[k]]
                    var_abs = -var_val if var_val < np.int16(0) else var_val
                    if var_val < np.int16(0):
                        sign = not sign
                    m_max = var_abs if var_abs > min1 else min1
                    if m_max < min2:
                        min2 = m_max
                    if var_abs < min1:
                        min1 = var_abs

                c1 = np.int32(min2) - np.int32(off)
                cst1 = np.int16(c1) if c1 > 0 else np.int16(0)
                c2 = np.int32(min1) - np.int32(off)
                cst2 = np.int16(c2) if c2 > 0 else np.int16(0)

                for k in range(k_start, k_stop):
                    t_k = transpose[k]
                    var_val = msg_var_to_chk[t_k]
                    var_abs = -var_val if var_val < np.int16(0) else var_val
                    res_abs = cst1 if var_abs == min1 else cst2
                    res_sng = sign ^ (var_val < np.int16(0))
                    res = -np.int32(res_abs) if res_sng else np.int32(res_abs)
                    msg_chk_to_var[t_k] = _sat_i16(res, sat_val_i16)

            if enable_syndrome and ite != n_ite:
                for v in range(n_var):
                    e_start = var_offsets[v]
                    e_stop = var_offsets[v + 1]
                    s = np.int32(0)
                    for e in range(e_start, e_stop):
                        s += np.int32(msg_chk_to_var[e])
                    post[v] = _clamp_i16(np.int32(Y_N[v]) + s)
                synd = _check_syndrome_scalar_i16(chk_offsets, chk_to_var, post)
                cur_synd_depth = ((cur_synd_depth + 1) % syndrome_depth) if synd else 0
                valid_synd = synd and (cur_synd_depth == 0)
                if valid_synd:
                    ite_done = ite
                    break
            if ite == n_ite:
                ite_done = n_ite

        if ite_done == n_ite:
            for v in range(n_var):
                e_start = var_offsets[v]
                e_stop = var_offsets[v + 1]
                s = np.int32(0)
                for e in range(e_start, e_stop):
                    s += np.int32(msg_chk_to_var[e])
                post[v] = _clamp_i16(np.int32(Y_N[v]) + s)

        for i in range(K):
            V_K[i] = np.int16(1) if post[info_bits_pos[i]] < np.int16(0) else np.int16(0)
        return 0 if valid_synd else 1

    @njit(nogil=True, fastmath=True)
    def decode_layered_spa_f32_scalar(
        chk_offsets, chk_to_var, info_bits_pos,
        n_ite, enable_syndrome, syndrome_depth,
        messages, var_nodes, values, contrib,
        Y_N, V_K
    ):
        n_var = len(var_nodes)
        n_chk = len(chk_offsets) - 1
        K = len(info_bits_pos)
        for v in range(n_var):
            var_nodes[v] = Y_N[v]
        messages.fill(np.float32(0.0))
        cur_synd_depth = 0
        valid_synd = True

        for _ in range(n_ite):
            for c in range(n_chk):
                k_start = chk_offsets[c]
                k_stop = chk_offsets[c + 1]
                sign = False
                product = np.float32(1.0)
                n_zeros = 0
                idx = 0
                for k in range(k_start, k_stop):
                    u = chk_to_var[k]
                    var_val = var_nodes[u] - messages[k]
                    contrib[idx] = var_val
                    var_abs = abs(var_val)
                    res = np.float32(math.tanh(var_abs * np.float32(0.5)))
                    if res == np.float32(0.0):
                        n_zeros += 1
                    else:
                        if math.copysign(1.0, var_val) < 0.0:
                            sign = not sign
                        product *= res
                    values[idx] = res
                    idx += 1

                if n_zeros == 0:
                    idx = 0
                    for k in range(k_start, k_stop):
                        u = chk_to_var[k]
                        var_val = contrib[idx]
                        val_v = values[idx]
                        res_tmp = product / val_v
                        if not (res_tmp < np.float32(1.0)):
                            res_tmp = MAX_TANH_F32
                        res_abs = np.float32(2.0) * np.float32(math.atanh(res_tmp))
                        res_sng = sign ^ (math.copysign(1.0, var_val) < 0.0)
                        new_msg = -res_abs if res_sng else res_abs
                        messages[k] = new_msg
                        var_nodes[u] = var_val + new_msg
                        idx += 1
                elif n_zeros == 1:
                    res_tmp = product if product < np.float32(1.0) else MAX_TANH_F32
                    res_abs = np.float32(2.0) * np.float32(math.atanh(res_tmp))
                    res_val = -res_abs if sign else res_abs
                    idx = 0
                    for k in range(k_start, k_stop):
                        u = chk_to_var[k]
                        var_val = contrib[idx]
                        new_msg = res_val if values[idx] == np.float32(0.0) else np.float32(0.0)
                        messages[k] = new_msg
                        var_nodes[u] = var_val + new_msg
                        idx += 1
                else:
                    idx = 0
                    for k in range(k_start, k_stop):
                        u = chk_to_var[k]
                        messages[k] = np.float32(0.0)
                        var_nodes[u] = contrib[idx]
                        idx += 1

            if enable_syndrome:
                synd = _check_syndrome_scalar_f32(chk_offsets, chk_to_var, var_nodes)
                cur_synd_depth = ((cur_synd_depth + 1) % syndrome_depth) if synd else 0
                valid_synd = synd and (cur_synd_depth == 0)
                if valid_synd:
                    break

        for i in range(K):
            V_K[i] = np.int32(1) if var_nodes[info_bits_pos[i]] < np.float32(0.0) else np.int32(0)
        return 0 if valid_synd else 1

    @njit(nogil=True, fastmath=True)
    def decode_layered_oms_f32_scalar(
        chk_offsets, chk_to_var, info_bits_pos,
        n_ite, enable_syndrome, syndrome_depth, offset,
        messages, var_nodes, contrib,
        Y_N, V_K
    ):
        n_var = len(var_nodes)
        n_chk = len(chk_offsets) - 1
        K = len(info_bits_pos)
        off = np.float32(offset)
        for v in range(n_var):
            var_nodes[v] = Y_N[v]
        messages.fill(np.float32(0.0))
        cur_synd_depth = 0
        valid_synd = True

        for _ in range(n_ite):
            for c in range(n_chk):
                k_start = chk_offsets[c]
                k_stop = chk_offsets[c + 1]
                sign = False
                min1 = MAX_F32
                min2 = MAX_F32
                idx = 0
                for k in range(k_start, k_stop):
                    u = chk_to_var[k]
                    var_val = var_nodes[u] - messages[k]
                    contrib[idx] = var_val
                    var_abs = abs(var_val)
                    if var_val < np.float32(0.0):
                        sign = not sign
                    m_max = var_abs if var_abs > min1 else min1
                    if m_max < min2:
                        min2 = m_max
                    if var_abs < min1:
                        min1 = var_abs
                    idx += 1

                cst1 = min2 - off
                if cst1 < np.float32(0.0):
                    cst1 = np.float32(0.0)
                cst2 = min1 - off
                if cst2 < np.float32(0.0):
                    cst2 = np.float32(0.0)

                idx = 0
                for k in range(k_start, k_stop):
                    u = chk_to_var[k]
                    var_val = contrib[idx]
                    var_abs = abs(var_val)
                    res_abs = cst1 if var_abs == min1 else cst2
                    res_sng = sign ^ (var_val < np.float32(0.0))
                    new_msg = -res_abs if res_sng else res_abs
                    messages[k] = new_msg
                    var_nodes[u] = var_val + new_msg
                    idx += 1

            if enable_syndrome:
                synd = _check_syndrome_scalar_f32(chk_offsets, chk_to_var, var_nodes)
                cur_synd_depth = ((cur_synd_depth + 1) % syndrome_depth) if synd else 0
                valid_synd = synd and (cur_synd_depth == 0)
                if valid_synd:
                    break

        for i in range(K):
            V_K[i] = np.int32(1) if var_nodes[info_bits_pos[i]] < np.float32(0.0) else np.int32(0)
        return 0 if valid_synd else 1

    @njit(nogil=True, fastmath=True)
    def decode_layered_oms_i16_scalar(
        chk_offsets, chk_to_var, info_bits_pos,
        n_ite, enable_syndrome, syndrome_depth, offset_i16, sat_val_i16,
        messages, var_nodes, contrib,
        Y_N, V_K
    ):
        n_var = len(var_nodes)
        n_chk = len(chk_offsets) - 1
        K = len(info_bits_pos)
        off = np.int16(offset_i16)
        for v in range(n_var):
            var_nodes[v] = Y_N[v]
        messages.fill(np.int16(0))
        cur_synd_depth = 0
        valid_synd = True

        for _ in range(n_ite):
            for c in range(n_chk):
                k_start = chk_offsets[c]
                k_stop = chk_offsets[c + 1]
                sign = False
                min1 = MAX_I16
                min2 = MAX_I16
                idx = 0
                for k in range(k_start, k_stop):
                    u = chk_to_var[k]
                    var_val = _clamp_i16(np.int32(var_nodes[u]) - np.int32(messages[k]))
                    contrib[idx] = var_val
                    var_abs = -var_val if var_val < np.int16(0) else var_val
                    if var_val < np.int16(0):
                        sign = not sign
                    m_max = var_abs if var_abs > min1 else min1
                    if m_max < min2:
                        min2 = m_max
                    if var_abs < min1:
                        min1 = var_abs
                    idx += 1

                c1 = np.int32(min2) - np.int32(off)
                cst1 = np.int16(c1) if c1 > 0 else np.int16(0)
                c2 = np.int32(min1) - np.int32(off)
                cst2 = np.int16(c2) if c2 > 0 else np.int16(0)

                idx = 0
                for k in range(k_start, k_stop):
                    u = chk_to_var[k]
                    var_val = contrib[idx]
                    var_abs = -var_val if var_val < np.int16(0) else var_val
                    res_abs = cst1 if var_abs == min1 else cst2
                    res_sng = sign ^ (var_val < np.int16(0))
                    res = -np.int32(res_abs) if res_sng else np.int32(res_abs)
                    new_msg = _sat_i16(res, sat_val_i16)
                    messages[k] = new_msg
                    var_nodes[u] = _clamp_i16(np.int32(var_val) + np.int32(new_msg))
                    idx += 1

            if enable_syndrome:
                synd = _check_syndrome_scalar_i16(chk_offsets, chk_to_var, var_nodes)
                cur_synd_depth = ((cur_synd_depth + 1) % syndrome_depth) if synd else 0
                valid_synd = synd and (cur_synd_depth == 0)
                if valid_synd:
                    break

        for i in range(K):
            V_K[i] = np.int16(1) if var_nodes[info_bits_pos[i]] < np.int16(0) else np.int16(0)
        return 0 if valid_synd else 1

    # ------------------------------------------------------------------
    # INTER-FRAME SIMD NUMBA KERNELS (`W` frames contiguous in axis 1)
    # ------------------------------------------------------------------

    @njit(nogil=True, fastmath=True)
    def decode_flooding_spa_f32_inter(
        var_offsets, chk_offsets, transpose, chk_to_var, info_bits_pos,
        n_ite, enable_syndrome, syndrome_depth,
        Y_N_inter, post_inter, msg_chk_to_var_inter, msg_var_to_chk_inter,
        values_inter, sign_w, prod_w, nzeros_w, synd_w,
        Y_N_wave, V_K_wave
    ):
        W = Y_N_wave.shape[0]
        n_var = len(var_offsets) - 1
        n_chk = len(chk_offsets) - 1
        K = len(info_bits_pos)

        for w in range(W):
            for v in range(n_var):
                Y_N_inter[v, w] = Y_N_wave[w, v]
        msg_chk_to_var_inter.fill(np.float32(0.0))

        cur_synd_depth = 0
        valid_synd = True
        ite_done = 0

        for ite in range(1, n_ite + 1):
            for v in range(n_var):
                e_start = var_offsets[v]
                e_stop = var_offsets[v + 1]
                for w in range(W):
                    post_inter[v, w] = Y_N_inter[v, w]
                for e in range(e_start, e_stop):
                    for w in range(W):
                        post_inter[v, w] += msg_chk_to_var_inter[e, w]
                for e in range(e_start, e_stop):
                    for w in range(W):
                        msg_var_to_chk_inter[e, w] = post_inter[v, w] - msg_chk_to_var_inter[e, w]

            for c in range(n_chk):
                k_start = chk_offsets[c]
                k_stop = chk_offsets[c + 1]
                for w in range(W):
                    sign_w[w] = np.uint8(0)
                    prod_w[w] = np.float32(1.0)
                    nzeros_w[w] = np.int32(0)

                idx = 0
                for k in range(k_start, k_stop):
                    t_k = transpose[k]
                    for w in range(W):
                        var_val = msg_var_to_chk_inter[t_k, w]
                        var_abs = abs(var_val)
                        res = np.float32(math.tanh(var_abs * np.float32(0.5)))
                        is_z = (res == np.float32(0.0))
                        sng = np.uint8(1) if math.copysign(1.0, var_val) < 0.0 else np.uint8(0)
                        sign_w[w] = sign_w[w] if is_z else (sign_w[w] ^ sng)
                        prod_w[w] = prod_w[w] if is_z else (prod_w[w] * res)
                        nzeros_w[w] = (nzeros_w[w] + np.int32(1)) if is_z else nzeros_w[w]
                        values_inter[idx, w] = res
                    idx += 1

                idx = 0
                for k in range(k_start, k_stop):
                    t_k = transpose[k]
                    for w in range(W):
                        nz = nzeros_w[w]
                        val_v = values_inter[idx, w]
                        is_z = (val_v == np.float32(0.0))
                        if nz == 0:
                            var_val = msg_var_to_chk_inter[t_k, w]
                            res_tmp = prod_w[w] / val_v
                            if not (res_tmp < np.float32(1.0)):
                                res_tmp = MAX_TANH_F32
                            res_abs = np.float32(2.0) * np.float32(math.atanh(res_tmp))
                            sng = sign_w[w] ^ (np.uint8(1) if math.copysign(1.0, var_val) < 0.0 else np.uint8(0))
                            msg_chk_to_var_inter[t_k, w] = -res_abs if sng != 0 else res_abs
                        elif nz == 1 and is_z:
                            res_tmp = prod_w[w] if prod_w[w] < np.float32(1.0) else MAX_TANH_F32
                            res_abs = np.float32(2.0) * np.float32(math.atanh(res_tmp))
                            msg_chk_to_var_inter[t_k, w] = -res_abs if sign_w[w] != 0 else res_abs
                        else:
                            msg_chk_to_var_inter[t_k, w] = np.float32(0.0)
                    idx += 1

            if enable_syndrome and ite != n_ite:
                for v in range(n_var):
                    e_start = var_offsets[v]
                    e_stop = var_offsets[v + 1]
                    for w in range(W):
                        post_inter[v, w] = Y_N_inter[v, w]
                    for e in range(e_start, e_stop):
                        for w in range(W):
                            post_inter[v, w] += msg_chk_to_var_inter[e, w]
                synd = _check_syndrome_inter_f32(chk_offsets, chk_to_var, post_inter, synd_w, W)
                cur_synd_depth = ((cur_synd_depth + 1) % syndrome_depth) if synd else 0
                valid_synd = synd and (cur_synd_depth == 0)
                if valid_synd:
                    ite_done = ite
                    break
            if ite == n_ite:
                ite_done = n_ite

        if ite_done == n_ite:
            for v in range(n_var):
                e_start = var_offsets[v]
                e_stop = var_offsets[v + 1]
                for w in range(W):
                    post_inter[v, w] = Y_N_inter[v, w]
                for e in range(e_start, e_stop):
                    for w in range(W):
                        post_inter[v, w] += msg_chk_to_var_inter[e, w]

        for w in range(W):
            for i in range(K):
                V_K_wave[w, i] = np.int32(1) if post_inter[info_bits_pos[i], w] < np.float32(0.0) else np.int32(0)
        return 0 if valid_synd else 1

    @njit(nogil=True, fastmath=True)
    def decode_flooding_oms_f32_inter(
        var_offsets, chk_offsets, transpose, chk_to_var, info_bits_pos,
        n_ite, enable_syndrome, syndrome_depth, offset,
        Y_N_inter, post_inter, msg_chk_to_var_inter, msg_var_to_chk_inter,
        sign_w, min1_w, min2_w, cst1_w, cst2_w, synd_w,
        Y_N_wave, V_K_wave
    ):
        W = Y_N_wave.shape[0]
        n_var = len(var_offsets) - 1
        n_chk = len(chk_offsets) - 1
        K = len(info_bits_pos)
        off = np.float32(offset)

        for w in range(W):
            for v in range(n_var):
                Y_N_inter[v, w] = Y_N_wave[w, v]
        msg_chk_to_var_inter.fill(np.float32(0.0))

        cur_synd_depth = 0
        valid_synd = True
        ite_done = 0

        for ite in range(1, n_ite + 1):
            for v in range(n_var):
                e_start = var_offsets[v]
                e_stop = var_offsets[v + 1]
                for w in range(W):
                    post_inter[v, w] = Y_N_inter[v, w]
                for e in range(e_start, e_stop):
                    for w in range(W):
                        post_inter[v, w] += msg_chk_to_var_inter[e, w]
                for e in range(e_start, e_stop):
                    for w in range(W):
                        msg_var_to_chk_inter[e, w] = post_inter[v, w] - msg_chk_to_var_inter[e, w]

            for c in range(n_chk):
                k_start = chk_offsets[c]
                k_stop = chk_offsets[c + 1]
                for w in range(W):
                    sign_w[w] = np.uint8(0)
                    min1_w[w] = MAX_F32
                    min2_w[w] = MAX_F32

                for k in range(k_start, k_stop):
                    t_k = transpose[k]
                    for w in range(W):
                        var_val = msg_var_to_chk_inter[t_k, w]
                        var_abs = abs(var_val)
                        sign_w[w] ^= np.uint8(1) if var_val < np.float32(0.0) else np.uint8(0)
                        m1 = min1_w[w]
                        m2 = min2_w[w]
                        m_max = var_abs if var_abs > m1 else m1
                        min2_w[w] = m_max if m_max < m2 else m2
                        min1_w[w] = var_abs if var_abs < m1 else m1

                for w in range(W):
                    c1 = min2_w[w] - off
                    cst1_w[w] = c1 if c1 > np.float32(0.0) else np.float32(0.0)
                    c2 = min1_w[w] - off
                    cst2_w[w] = c2 if c2 > np.float32(0.0) else np.float32(0.0)

                for k in range(k_start, k_stop):
                    t_k = transpose[k]
                    for w in range(W):
                        var_val = msg_var_to_chk_inter[t_k, w]
                        var_abs = abs(var_val)
                        res_abs = cst1_w[w] if var_abs == min1_w[w] else cst2_w[w]
                        sng = sign_w[w] ^ (np.uint8(1) if var_val < np.float32(0.0) else np.uint8(0))
                        msg_chk_to_var_inter[t_k, w] = -res_abs if sng != 0 else res_abs

            if enable_syndrome and ite != n_ite:
                for v in range(n_var):
                    e_start = var_offsets[v]
                    e_stop = var_offsets[v + 1]
                    for w in range(W):
                        post_inter[v, w] = Y_N_inter[v, w]
                    for e in range(e_start, e_stop):
                        for w in range(W):
                            post_inter[v, w] += msg_chk_to_var_inter[e, w]
                synd = _check_syndrome_inter_f32(chk_offsets, chk_to_var, post_inter, synd_w, W)
                cur_synd_depth = ((cur_synd_depth + 1) % syndrome_depth) if synd else 0
                valid_synd = synd and (cur_synd_depth == 0)
                if valid_synd:
                    ite_done = ite
                    break
            if ite == n_ite:
                ite_done = n_ite

        if ite_done == n_ite:
            for v in range(n_var):
                e_start = var_offsets[v]
                e_stop = var_offsets[v + 1]
                for w in range(W):
                    post_inter[v, w] = Y_N_inter[v, w]
                for e in range(e_start, e_stop):
                    for w in range(W):
                        post_inter[v, w] += msg_chk_to_var_inter[e, w]

        for w in range(W):
            for i in range(K):
                V_K_wave[w, i] = np.int32(1) if post_inter[info_bits_pos[i], w] < np.float32(0.0) else np.int32(0)
        return 0 if valid_synd else 1

    @njit(nogil=True, fastmath=True)
    def decode_flooding_oms_i16_inter(
        var_offsets, chk_offsets, transpose, chk_to_var, info_bits_pos,
        n_ite, enable_syndrome, syndrome_depth, offset_i16, sat_val_i16,
        Y_N_inter, post_inter, msg_chk_to_var_inter, msg_var_to_chk_inter,
        sign_w, min1_w, min2_w, cst1_w, cst2_w, synd_w,
        Y_N_wave, V_K_wave
    ):
        W = Y_N_wave.shape[0]
        n_var = len(var_offsets) - 1
        n_chk = len(chk_offsets) - 1
        K = len(info_bits_pos)
        off = np.int16(offset_i16)

        for w in range(W):
            for v in range(n_var):
                Y_N_inter[v, w] = Y_N_wave[w, v]
        msg_chk_to_var_inter.fill(np.int16(0))

        cur_synd_depth = 0
        valid_synd = True
        ite_done = 0

        for ite in range(1, n_ite + 1):
            for v in range(n_var):
                e_start = var_offsets[v]
                e_stop = var_offsets[v + 1]
                for w in range(W):
                    post_inter[v, w] = Y_N_inter[v, w]
                for e in range(e_start, e_stop):
                    for w in range(W):
                        post_inter[v, w] = _clamp_i16(np.int32(post_inter[v, w]) + np.int32(msg_chk_to_var_inter[e, w]))
                for e in range(e_start, e_stop):
                    for w in range(W):
                        msg_var_to_chk_inter[e, w] = _clamp_i16(np.int32(post_inter[v, w]) - np.int32(msg_chk_to_var_inter[e, w]))

            for c in range(n_chk):
                k_start = chk_offsets[c]
                k_stop = chk_offsets[c + 1]
                for w in range(W):
                    sign_w[w] = np.uint8(0)
                    min1_w[w] = MAX_I16
                    min2_w[w] = MAX_I16

                for k in range(k_start, k_stop):
                    t_k = transpose[k]
                    for w in range(W):
                        var_val = msg_var_to_chk_inter[t_k, w]
                        var_abs = -var_val if var_val < np.int16(0) else var_val
                        sign_w[w] ^= np.uint8(1) if var_val < np.int16(0) else np.uint8(0)
                        m1 = min1_w[w]
                        m2 = min2_w[w]
                        m_max = var_abs if var_abs > m1 else m1
                        min2_w[w] = m_max if m_max < m2 else m2
                        min1_w[w] = var_abs if var_abs < m1 else m1

                for w in range(W):
                    c1 = np.int32(min2_w[w]) - np.int32(off)
                    cst1_w[w] = np.int16(c1) if c1 > 0 else np.int16(0)
                    c2 = np.int32(min1_w[w]) - np.int32(off)
                    cst2_w[w] = np.int16(c2) if c2 > 0 else np.int16(0)

                for k in range(k_start, k_stop):
                    t_k = transpose[k]
                    for w in range(W):
                        var_val = msg_var_to_chk_inter[t_k, w]
                        var_abs = -var_val if var_val < np.int16(0) else var_val
                        res_abs = cst1_w[w] if var_abs == min1_w[w] else cst2_w[w]
                        sng = sign_w[w] ^ (np.uint8(1) if var_val < np.int16(0) else np.uint8(0))
                        res = -np.int32(res_abs) if sng != 0 else np.int32(res_abs)
                        msg_chk_to_var_inter[t_k, w] = _sat_i16(res, sat_val_i16)

            if enable_syndrome and ite != n_ite:
                for v in range(n_var):
                    e_start = var_offsets[v]
                    e_stop = var_offsets[v + 1]
                    for w in range(W):
                        post_inter[v, w] = Y_N_inter[v, w]
                    for e in range(e_start, e_stop):
                        for w in range(W):
                            post_inter[v, w] = _clamp_i16(np.int32(post_inter[v, w]) + np.int32(msg_chk_to_var_inter[e, w]))
                synd = _check_syndrome_inter_i16(chk_offsets, chk_to_var, post_inter, synd_w, W)
                cur_synd_depth = ((cur_synd_depth + 1) % syndrome_depth) if synd else 0
                valid_synd = synd and (cur_synd_depth == 0)
                if valid_synd:
                    ite_done = ite
                    break
            if ite == n_ite:
                ite_done = n_ite

        if ite_done == n_ite:
            for v in range(n_var):
                e_start = var_offsets[v]
                e_stop = var_offsets[v + 1]
                for w in range(W):
                    post_inter[v, w] = Y_N_inter[v, w]
                for e in range(e_start, e_stop):
                    for w in range(W):
                        post_inter[v, w] = _clamp_i16(np.int32(post_inter[v, w]) + np.int32(msg_chk_to_var_inter[e, w]))

        for w in range(W):
            for i in range(K):
                V_K_wave[w, i] = np.int16(1) if post_inter[info_bits_pos[i], w] < np.int16(0) else np.int16(0)
        return 0 if valid_synd else 1

    @njit(nogil=True, fastmath=True)
    def decode_layered_spa_f32_inter(
        chk_offsets, chk_to_var, info_bits_pos,
        n_ite, enable_syndrome, syndrome_depth,
        var_nodes_inter, messages_inter, values_inter, contrib_inter,
        sign_w, prod_w, nzeros_w, synd_w,
        Y_N_wave, V_K_wave
    ):
        W = Y_N_wave.shape[0]
        n_var = var_nodes_inter.shape[0]
        n_chk = len(chk_offsets) - 1
        K = len(info_bits_pos)

        for w in range(W):
            for v in range(n_var):
                var_nodes_inter[v, w] = Y_N_wave[w, v]
        messages_inter.fill(np.float32(0.0))

        cur_synd_depth = 0
        valid_synd = True

        for _ in range(n_ite):
            for c in range(n_chk):
                k_start = chk_offsets[c]
                k_stop = chk_offsets[c + 1]
                for w in range(W):
                    sign_w[w] = np.uint8(0)
                    prod_w[w] = np.float32(1.0)
                    nzeros_w[w] = np.int32(0)

                idx = 0
                for k in range(k_start, k_stop):
                    u = chk_to_var[k]
                    for w in range(W):
                        var_val = var_nodes_inter[u, w] - messages_inter[k, w]
                        contrib_inter[idx, w] = var_val
                        var_abs = abs(var_val)
                        res = np.float32(math.tanh(var_abs * np.float32(0.5)))
                        is_z = (res == np.float32(0.0))
                        sng = np.uint8(1) if math.copysign(1.0, var_val) < 0.0 else np.uint8(0)
                        sign_w[w] = sign_w[w] if is_z else (sign_w[w] ^ sng)
                        prod_w[w] = prod_w[w] if is_z else (prod_w[w] * res)
                        nzeros_w[w] = (nzeros_w[w] + np.int32(1)) if is_z else nzeros_w[w]
                        values_inter[idx, w] = res
                    idx += 1

                idx = 0
                for k in range(k_start, k_stop):
                    u = chk_to_var[k]
                    for w in range(W):
                        nz = nzeros_w[w]
                        val_v = values_inter[idx, w]
                        var_val = contrib_inter[idx, w]
                        is_z = (val_v == np.float32(0.0))
                        if nz == 0:
                            res_tmp = prod_w[w] / val_v
                            if not (res_tmp < np.float32(1.0)):
                                res_tmp = MAX_TANH_F32
                            res_abs = np.float32(2.0) * np.float32(math.atanh(res_tmp))
                            sng = sign_w[w] ^ (np.uint8(1) if math.copysign(1.0, var_val) < 0.0 else np.uint8(0))
                            new_msg = -res_abs if sng != 0 else res_abs
                        elif nz == 1 and is_z:
                            res_tmp = prod_w[w] if prod_w[w] < np.float32(1.0) else MAX_TANH_F32
                            res_abs = np.float32(2.0) * np.float32(math.atanh(res_tmp))
                            new_msg = -res_abs if sign_w[w] != 0 else res_abs
                        else:
                            new_msg = np.float32(0.0)
                        messages_inter[k, w] = new_msg
                        var_nodes_inter[u, w] = var_val + new_msg
                    idx += 1

            if enable_syndrome:
                synd = _check_syndrome_inter_f32(chk_offsets, chk_to_var, var_nodes_inter, synd_w, W)
                cur_synd_depth = ((cur_synd_depth + 1) % syndrome_depth) if synd else 0
                valid_synd = synd and (cur_synd_depth == 0)
                if valid_synd:
                    break

        for w in range(W):
            for i in range(K):
                V_K_wave[w, i] = np.int32(1) if var_nodes_inter[info_bits_pos[i], w] < np.float32(0.0) else np.int32(0)
        return 0 if valid_synd else 1

    @njit(nogil=True, fastmath=True)
    def decode_layered_oms_f32_inter(
        chk_offsets, chk_to_var, info_bits_pos,
        n_ite, enable_syndrome, syndrome_depth, offset,
        var_nodes_inter, messages_inter, contrib_inter,
        sign_w, min1_w, min2_w, cst1_w, cst2_w, synd_w,
        Y_N_wave, V_K_wave
    ):
        W = Y_N_wave.shape[0]
        n_var = var_nodes_inter.shape[0]
        n_chk = len(chk_offsets) - 1
        K = len(info_bits_pos)
        off = np.float32(offset)

        for w in range(W):
            for v in range(n_var):
                var_nodes_inter[v, w] = Y_N_wave[w, v]
        messages_inter.fill(np.float32(0.0))

        cur_synd_depth = 0
        valid_synd = True

        for _ in range(n_ite):
            for c in range(n_chk):
                k_start = chk_offsets[c]
                k_stop = chk_offsets[c + 1]
                for w in range(W):
                    sign_w[w] = np.uint8(0)
                    min1_w[w] = MAX_F32
                    min2_w[w] = MAX_F32

                idx = 0
                for k in range(k_start, k_stop):
                    u = chk_to_var[k]
                    for w in range(W):
                        var_val = var_nodes_inter[u, w] - messages_inter[k, w]
                        contrib_inter[idx, w] = var_val
                        var_abs = abs(var_val)
                        sign_w[w] ^= np.uint8(1) if var_val < np.float32(0.0) else np.uint8(0)
                        m1 = min1_w[w]
                        m2 = min2_w[w]
                        m_max = var_abs if var_abs > m1 else m1
                        min2_w[w] = m_max if m_max < m2 else m2
                        min1_w[w] = var_abs if var_abs < m1 else m1
                    idx += 1

                for w in range(W):
                    c1 = min2_w[w] - off
                    cst1_w[w] = c1 if c1 > np.float32(0.0) else np.float32(0.0)
                    c2 = min1_w[w] - off
                    cst2_w[w] = c2 if c2 > np.float32(0.0) else np.float32(0.0)

                idx = 0
                for k in range(k_start, k_stop):
                    u = chk_to_var[k]
                    for w in range(W):
                        var_val = contrib_inter[idx, w]
                        var_abs = abs(var_val)
                        res_abs = cst1_w[w] if var_abs == min1_w[w] else cst2_w[w]
                        sng = sign_w[w] ^ (np.uint8(1) if var_val < np.float32(0.0) else np.uint8(0))
                        new_msg = -res_abs if sng != 0 else res_abs
                        messages_inter[k, w] = new_msg
                        var_nodes_inter[u, w] = var_val + new_msg
                    idx += 1

            if enable_syndrome:
                synd = _check_syndrome_inter_f32(chk_offsets, chk_to_var, var_nodes_inter, synd_w, W)
                cur_synd_depth = ((cur_synd_depth + 1) % syndrome_depth) if synd else 0
                valid_synd = synd and (cur_synd_depth == 0)
                if valid_synd:
                    break

        for w in range(W):
            for i in range(K):
                V_K_wave[w, i] = np.int32(1) if var_nodes_inter[info_bits_pos[i], w] < np.float32(0.0) else np.int32(0)
        return 0 if valid_synd else 1

    @njit(nogil=True, fastmath=True)
    def decode_layered_oms_i16_inter(
        chk_offsets, chk_to_var, info_bits_pos,
        n_ite, enable_syndrome, syndrome_depth, offset_i16, sat_val_i16,
        var_nodes_inter, messages_inter, contrib_inter,
        sign_w, min1_w, min2_w, cst1_w, cst2_w, synd_w,
        Y_N_wave, V_K_wave
    ):
        W = Y_N_wave.shape[0]
        n_var = var_nodes_inter.shape[0]
        n_chk = len(chk_offsets) - 1
        K = len(info_bits_pos)
        off = np.int16(offset_i16)

        for w in range(W):
            for v in range(n_var):
                var_nodes_inter[v, w] = Y_N_wave[w, v]
        messages_inter.fill(np.int16(0))

        cur_synd_depth = 0
        valid_synd = True

        for _ in range(n_ite):
            for c in range(n_chk):
                k_start = chk_offsets[c]
                k_stop = chk_offsets[c + 1]
                for w in range(W):
                    sign_w[w] = np.uint8(0)
                    min1_w[w] = MAX_I16
                    min2_w[w] = MAX_I16

                idx = 0
                for k in range(k_start, k_stop):
                    u = chk_to_var[k]
                    for w in range(W):
                        var_val = _clamp_i16(np.int32(var_nodes_inter[u, w]) - np.int32(messages_inter[k, w]))
                        contrib_inter[idx, w] = var_val
                        var_abs = -var_val if var_val < np.int16(0) else var_val
                        sign_w[w] ^= np.uint8(1) if var_val < np.int16(0) else np.uint8(0)
                        m1 = min1_w[w]
                        m2 = min2_w[w]
                        m_max = var_abs if var_abs > m1 else m1
                        min2_w[w] = m_max if m_max < m2 else m2
                        min1_w[w] = var_abs if var_abs < m1 else m1
                    idx += 1

                for w in range(W):
                    c1 = np.int32(min2_w[w]) - np.int32(off)
                    cst1_w[w] = np.int16(c1) if c1 > 0 else np.int16(0)
                    c2 = np.int32(min1_w[w]) - np.int32(off)
                    cst2_w[w] = np.int16(c2) if c2 > 0 else np.int16(0)

                idx = 0
                for k in range(k_start, k_stop):
                    u = chk_to_var[k]
                    for w in range(W):
                        var_val = contrib_inter[idx, w]
                        var_abs = -var_val if var_val < np.int16(0) else var_val
                        res_abs = cst1_w[w] if var_abs == min1_w[w] else cst2_w[w]
                        sng = sign_w[w] ^ (np.uint8(1) if var_val < np.int16(0) else np.uint8(0))
                        res = -np.int32(res_abs) if sng != 0 else np.int32(res_abs)
                        new_msg = _sat_i16(res, sat_val_i16)
                        messages_inter[k, w] = new_msg
                        var_nodes_inter[u, w] = _clamp_i16(np.int32(var_val) + np.int32(new_msg))
                    idx += 1

            if enable_syndrome:
                synd = _check_syndrome_inter_i16(chk_offsets, chk_to_var, var_nodes_inter, synd_w, W)
                cur_synd_depth = ((cur_synd_depth + 1) % syndrome_depth) if synd else 0
                valid_synd = synd and (cur_synd_depth == 0)
                if valid_synd:
                    break

        for w in range(W):
            for i in range(K):
                V_K_wave[w, i] = np.int16(1) if var_nodes_inter[info_bits_pos[i], w] < np.int16(0) else np.int16(0)
        return 0 if valid_synd else 1

    _NUMBA_KERNELS = {
        ("BP_FLOODING", "SPA", "f32", ""): decode_flooding_spa_f32_scalar,
        ("BP_FLOODING", "OMS", "f32", ""): decode_flooding_oms_f32_scalar,
        ("BP_FLOODING", "OMS", "i16", ""): decode_flooding_oms_i16_scalar,
        ("BP_HORIZONTAL_LAYERED", "SPA", "f32", ""): decode_layered_spa_f32_scalar,
        ("BP_HORIZONTAL_LAYERED", "OMS", "f32", ""): decode_layered_oms_f32_scalar,
        ("BP_HORIZONTAL_LAYERED", "OMS", "i16", ""): decode_layered_oms_i16_scalar,
        ("BP_FLOODING", "SPA", "f32", "INTER"): decode_flooding_spa_f32_inter,
        ("BP_FLOODING", "OMS", "f32", "INTER"): decode_flooding_oms_f32_inter,
        ("BP_FLOODING", "OMS", "i16", "INTER"): decode_flooding_oms_i16_inter,
        ("BP_HORIZONTAL_LAYERED", "SPA", "f32", "INTER"): decode_layered_spa_f32_inter,
        ("BP_HORIZONTAL_LAYERED", "OMS", "f32", "INTER"): decode_layered_oms_f32_inter,
        ("BP_HORIZONTAL_LAYERED", "OMS", "i16", "INTER"): decode_layered_oms_i16_inter,
    }
    return _NUMBA_KERNELS


# ====================================================================================================
# 2. PURE PYTHON / NUMPY KERNELS (CPython GIL & Free-Threaded No-GIL Reference)
# ====================================================================================================

def _numpy_decode_layered(mdl, Y_N_wave, V_K_wave):
    """Vectorized NumPy Horizontal Layered BP decoder (processes all frames in `Y_N_wave` simultaneously)."""
    W = Y_N_wave.shape[0]
    is_i16 = mdl.is_i16
    chk_offsets = mdl.chk_offsets
    chk_to_var = mdl.chk_to_var
    n_chk = len(chk_offsets) - 1
    max_tanh = np.float32(1.0) - np.float32(np.finfo(np.float32).eps)

    # Shape (n_var, W) and (n_branches, W)
    var_nodes = Y_N_wave.T.copy()
    messages = np.zeros((mdl.n_branches, W), dtype=np.int16 if is_i16 else np.float32)

    cur_synd_depth = 0
    valid_synd = True

    for _ in range(mdl.n_ite):
        for c in range(n_chk):
            k_start = int(chk_offsets[c])
            k_stop = int(chk_offsets[c + 1])
            u_idx = chk_to_var[k_start:k_stop]
            if is_i16:
                contrib = np.clip(
                    var_nodes[u_idx].astype(np.int32) - messages[k_start:k_stop].astype(np.int32),
                    -32768,
                    32767,
                ).astype(np.int16)
                abs_c = np.abs(contrib)
                sgn_c = contrib < 0
                tot_sgn = np.bitwise_xor.reduce(sgn_c, axis=0)
                part = np.partition(abs_c, 1, axis=0)
                min1 = part[0]
                min2 = part[1]
                cst1 = np.maximum(np.int16(0), (min2.astype(np.int32) - int(mdl.offset)).astype(np.int16))
                cst2 = np.maximum(np.int16(0), (min1.astype(np.int32) - int(mdl.offset)).astype(np.int16))
                res_abs = np.where(abs_c == min1, cst1, cst2)
                res_sng = np.bitwise_xor(tot_sgn, sgn_c)
                new_msg = np.where(res_sng, -res_abs.astype(np.int32), res_abs.astype(np.int32))
                new_msg = np.clip(new_msg, -int(mdl.sat_val_i16), int(mdl.sat_val_i16)).astype(np.int16)
                messages[k_start:k_stop] = new_msg
                var_nodes[u_idx] = np.clip(
                    contrib.astype(np.int32) + new_msg.astype(np.int32), -32768, 32767
                ).astype(np.int16)
            else:
                contrib = var_nodes[u_idx] - messages[k_start:k_stop]
                abs_c = np.abs(contrib)
                if mdl.dec_implem == "OMS":
                    sgn_c = contrib < np.float32(0.0)
                    tot_sgn = np.bitwise_xor.reduce(sgn_c, axis=0)
                    part = np.partition(abs_c, 1, axis=0)
                    min1 = part[0]
                    min2 = part[1]
                    cst1 = np.maximum(np.float32(0.0), min2 - np.float32(mdl.offset))
                    cst2 = np.maximum(np.float32(0.0), min1 - np.float32(mdl.offset))
                    res_abs = np.where(abs_c == min1, cst1, cst2)
                    res_sng = np.bitwise_xor(tot_sgn, sgn_c)
                    new_msg = np.where(res_sng, -res_abs, res_abs)
                else:
                    vals = np.tanh(abs_c * np.float32(0.5))
                    is_z = vals == np.float32(0.0)
                    sgn_c = np.signbit(contrib) & (~is_z)
                    tot_sgn = np.bitwise_xor.reduce(sgn_c, axis=0)
                    n_zeros = np.sum(is_z, axis=0)
                    safe_vals = np.where(is_z, np.float32(1.0), vals)
                    prod = np.prod(safe_vals, axis=0)
                    res_tmp = np.where(is_z, prod, prod / safe_vals)
                    res_tmp = np.minimum(res_tmp, max_tanh)
                    res_abs = np.float32(2.0) * np.arctanh(res_tmp)
                    res_sng = np.where(is_z, tot_sgn, np.bitwise_xor(tot_sgn, np.signbit(contrib)))
                    new_msg = np.where(res_sng, -res_abs, res_abs)
                    active = (n_zeros == 0) | ((n_zeros == 1) & is_z)
                    new_msg = np.where(active, new_msg, np.float32(0.0))
                messages[k_start:k_stop] = new_msg
                var_nodes[u_idx] = contrib + new_msg

        if mdl.enable_syndrome:
            synd_ok = True
            for c in range(n_chk):
                k_start = int(chk_offsets[c])
                k_stop = int(chk_offsets[c + 1])
                s = np.bitwise_xor.reduce(var_nodes[chk_to_var[k_start:k_stop]] < 0, axis=0)
                if np.any(s):
                    synd_ok = False
                    break
            cur_synd_depth = ((cur_synd_depth + 1) % mdl.syndrome_depth) if synd_ok else 0
            valid_synd = synd_ok and (cur_synd_depth == 0)
            if valid_synd:
                break

    out_dtype = np.int16 if is_i16 else np.int32
    V_K_wave[:] = (var_nodes[mdl.info_bits_pos].T < 0).astype(out_dtype)
    return 0 if valid_synd else 1


def _numpy_decode_flooding(mdl, Y_N_wave, V_K_wave):
    """Vectorized NumPy Flooding BP decoder (processes all frames in `Y_N_wave` simultaneously)."""
    W = Y_N_wave.shape[0]
    is_i16 = mdl.is_i16
    var_offsets = mdl.var_offsets
    chk_offsets = mdl.chk_offsets
    transpose = mdl.transpose
    chk_to_var = mdl.chk_to_var
    n_var = len(var_offsets) - 1
    n_chk = len(chk_offsets) - 1
    max_tanh = np.float32(1.0) - np.float32(np.finfo(np.float32).eps)

    Y_N_T = Y_N_wave.T.copy()
    dtype = np.int16 if is_i16 else np.float32
    msg_chk_to_var = np.zeros((mdl.n_branches, W), dtype=dtype)
    msg_var_to_chk = np.zeros((mdl.n_branches, W), dtype=dtype)
    post = np.zeros((n_var, W), dtype=dtype)

    cur_synd_depth = 0
    valid_synd = True
    ite_done = 0

    for ite in range(1, mdl.n_ite + 1):
        for v in range(n_var):
            e_start = int(var_offsets[v])
            e_stop = int(var_offsets[v + 1])
            if is_i16:
                s = np.sum(msg_chk_to_var[e_start:e_stop].astype(np.int32), axis=0)
                tmp = Y_N_T[v].astype(np.int32) + s
                msg_var_to_chk[e_start:e_stop] = np.clip(
                    tmp - msg_chk_to_var[e_start:e_stop].astype(np.int32), -32768, 32767
                ).astype(np.int16)
            else:
                s = np.sum(msg_chk_to_var[e_start:e_stop], axis=0)
                tmp = Y_N_T[v] + s
                msg_var_to_chk[e_start:e_stop] = tmp - msg_chk_to_var[e_start:e_stop]

        for c in range(n_chk):
            k_start = int(chk_offsets[c])
            k_stop = int(chk_offsets[c + 1])
            t_idx = transpose[k_start:k_stop]
            contrib = msg_var_to_chk[t_idx]
            abs_c = np.abs(contrib)
            if is_i16:
                sgn_c = contrib < 0
                tot_sgn = np.bitwise_xor.reduce(sgn_c, axis=0)
                part = np.partition(abs_c, 1, axis=0)
                min1 = part[0]
                min2 = part[1]
                cst1 = np.maximum(np.int16(0), (min2.astype(np.int32) - int(mdl.offset)).astype(np.int16))
                cst2 = np.maximum(np.int16(0), (min1.astype(np.int32) - int(mdl.offset)).astype(np.int16))
                res_abs = np.where(abs_c == min1, cst1, cst2)
                res_sng = np.bitwise_xor(tot_sgn, sgn_c)
                new_msg = np.where(res_sng, -res_abs.astype(np.int32), res_abs.astype(np.int32))
                msg_chk_to_var[t_idx] = np.clip(new_msg, -int(mdl.sat_val_i16), int(mdl.sat_val_i16)).astype(np.int16)
            elif mdl.dec_implem == "OMS":
                sgn_c = contrib < np.float32(0.0)
                tot_sgn = np.bitwise_xor.reduce(sgn_c, axis=0)
                part = np.partition(abs_c, 1, axis=0)
                min1 = part[0]
                min2 = part[1]
                cst1 = np.maximum(np.float32(0.0), min2 - np.float32(mdl.offset))
                cst2 = np.maximum(np.float32(0.0), min1 - np.float32(mdl.offset))
                res_abs = np.where(abs_c == min1, cst1, cst2)
                res_sng = np.bitwise_xor(tot_sgn, sgn_c)
                msg_chk_to_var[t_idx] = np.where(res_sng, -res_abs, res_abs)
            else:
                vals = np.tanh(abs_c * np.float32(0.5))
                is_z = vals == np.float32(0.0)
                sgn_c = np.signbit(contrib) & (~is_z)
                tot_sgn = np.bitwise_xor.reduce(sgn_c, axis=0)
                n_zeros = np.sum(is_z, axis=0)
                safe_vals = np.where(is_z, np.float32(1.0), vals)
                prod = np.prod(safe_vals, axis=0)
                res_tmp = np.where(is_z, prod, prod / safe_vals)
                res_tmp = np.minimum(res_tmp, max_tanh)
                res_abs = np.float32(2.0) * np.arctanh(res_tmp)
                res_sng = np.where(is_z, tot_sgn, np.bitwise_xor(tot_sgn, np.signbit(contrib)))
                new_msg = np.where(res_sng, -res_abs, res_abs)
                active = (n_zeros == 0) | ((n_zeros == 1) & is_z)
                msg_chk_to_var[t_idx] = np.where(active, new_msg, np.float32(0.0))

        if mdl.enable_syndrome and ite != mdl.n_ite:
            for v in range(n_var):
                e_start = int(var_offsets[v])
                e_stop = int(var_offsets[v + 1])
                if is_i16:
                    s = np.sum(msg_chk_to_var[e_start:e_stop].astype(np.int32), axis=0)
                    post[v] = np.clip(Y_N_T[v].astype(np.int32) + s, -32768, 32767).astype(np.int16)
                else:
                    post[v] = Y_N_T[v] + np.sum(msg_chk_to_var[e_start:e_stop], axis=0)
            synd_ok = True
            for c in range(n_chk):
                k_start = int(chk_offsets[c])
                k_stop = int(chk_offsets[c + 1])
                s = np.bitwise_xor.reduce(post[chk_to_var[k_start:k_stop]] < 0, axis=0)
                if np.any(s):
                    synd_ok = False
                    break
            cur_synd_depth = ((cur_synd_depth + 1) % mdl.syndrome_depth) if synd_ok else 0
            valid_synd = synd_ok and (cur_synd_depth == 0)
            if valid_synd:
                ite_done = ite
                break
        if ite == mdl.n_ite:
            ite_done = mdl.n_ite

    if ite_done == mdl.n_ite:
        for v in range(n_var):
            e_start = int(var_offsets[v])
            e_stop = int(var_offsets[v + 1])
            if is_i16:
                s = np.sum(msg_chk_to_var[e_start:e_stop].astype(np.int32), axis=0)
                post[v] = np.clip(Y_N_T[v].astype(np.int32) + s, -32768, 32767).astype(np.int16)
            else:
                post[v] = Y_N_T[v] + np.sum(msg_chk_to_var[e_start:e_stop], axis=0)

    out_dtype = np.int16 if is_i16 else np.int32
    V_K_wave[:] = (post[mdl.info_bits_pos].T < 0).astype(out_dtype)
    return 0 if valid_synd else 1


# ====================================================================================================
# 3. STREAMPU `Stateful_core` MODULE (`DecoderLDPCPython`)
# ====================================================================================================

class DecoderLDPCPython(spu.Stateful_core):
    """Generic LDPC BP Decoder in Python (`NumPy` or `Numba @njit(nogil=True)`) as a StreamPU `Stateful_core` module."""

    def __init__(
        self,
        code: py_aff3ct.LDPC_Code,
        dec_type: str = "BP_FLOODING",
        dec_implem: str = "SPA",
        q_type: str = "f32",
        simd_strategy: str = "",
        n_ite: int = 20,
        offset: float = 0.0,
        enable_syndrome: bool = True,
        syndrome_depth: int = 1,
        n_frames: int = 1,
        use_numba: bool = True,
    ):
        super().__init__()
        self.dec_type = dec_type
        self.dec_implem = dec_implem
        self.q_type = "i16" if q_type in ("i16", "int16", "Int16") else "f32"
        self.is_i16 = self.q_type == "i16"
        self.simd_strategy = "INTER" if simd_strategy == "INTER" else ""
        self.single_wave = self.simd_strategy == "INTER"
        self.n_ite = int(n_ite)
        self.offset = float(offset)
        self.enable_syndrome = bool(enable_syndrome)
        self.syndrome_depth = int(syndrome_depth)
        self.use_numba = bool(use_numba)

        self.K = int(code.K)
        self.N_cw = int(code.N_cw)
        self.n_branches = int(code.n_connections)
        self.max_chk_degree = int(code.cols_max_degree)
        self.sat_val_i16 = np.int16(code.sat_val_i16)

        # Shared read-only CSR/CSC Tanner graph arrays across all cloned threads
        (
            self.var_offsets,
            self.chk_offsets,
            self.transpose,
            self.chk_to_var,
            self.info_bits_pos,
        ) = code.get_tanner_graph_arrays()

        backend_tag = "Numba" if self.use_numba else "NumPy"
        self.name = f"Decoder_LDPC_{self.dec_type}_{self.dec_implem}_{self.q_type}_{backend_tag}"

        if self.single_wave:
            py_aff3ct.set_single_wave(self, True)
        self.n_frames = n_frames

        self._allocate_workspace(n_frames if self.single_wave else 1)

        if self.use_numba:
            kernels = _get_numba_kernels()
            key = (self.dec_type, self.dec_implem, self.q_type, self.simd_strategy)
            if key not in kernels:
                raise ValueError(f"Unsupported Numba LDPC configuration: {key}")
            self._kernel = kernels[key]
            # Warm up JIT compilation before multi-threaded Pipeline / Sequence execution
            self._warmup_numba()
        else:
            self._kernel = None

        task = self.create_task("decode_siho")
        task.replicable = True
        in_dtype = spu.int16 if self.is_i16 else spu.float32
        out_dtype = spu.int16 if self.is_i16 else spu.int32
        self.create_socket_in(task, "Y_N", self.N_cw, in_dtype)
        self.create_socket_out(task, "V_K", self.K, out_dtype)

        if self.use_numba:
            self.create_codelet(task, DecoderLDPCPython._decode_numba)
        else:
            self.create_codelet(task, DecoderLDPCPython._decode_numpy)

    def _allocate_workspace(self, W: int):
        dtype = np.int16 if self.is_i16 else np.float32
        W = max(1, int(W))
        self._work_W = W
        if self.single_wave:
            self.Y_N_inter = np.zeros((self.N_cw, W), dtype=dtype)
            self.post_inter = np.zeros((self.N_cw, W), dtype=dtype)
            self.msg_chk_to_var_inter = np.zeros((self.n_branches, W), dtype=dtype)
            self.msg_var_to_chk_inter = np.zeros((self.n_branches, W), dtype=dtype)
            self.values_inter = np.zeros((self.max_chk_degree, W), dtype=np.float32)
            self.contrib_inter = np.zeros((self.max_chk_degree, W), dtype=dtype)
            self.sign_w = np.zeros(W, dtype=np.uint8)
            self.min1_w = np.zeros(W, dtype=dtype)
            self.min2_w = np.zeros(W, dtype=dtype)
            self.cst1_w = np.zeros(W, dtype=dtype)
            self.cst2_w = np.zeros(W, dtype=dtype)
            self.prod_w = np.zeros(W, dtype=np.float32)
            self.nzeros_w = np.zeros(W, dtype=np.int32)
            self.synd_w = np.zeros(W, dtype=np.uint8)
        else:
            self.msg_chk_to_var = np.zeros(self.n_branches, dtype=dtype)
            self.msg_var_to_chk = np.zeros(self.n_branches, dtype=dtype)
            self.post = np.zeros(self.N_cw, dtype=dtype)
            self.values = np.zeros(self.max_chk_degree, dtype=np.float32)
            self.contrib = np.zeros(self.max_chk_degree, dtype=dtype)

    def _warmup_numba(self):
        in_dtype = np.int16 if self.is_i16 else np.float32
        out_dtype = np.int16 if self.is_i16 else np.int32
        W = self._work_W
        dummy_in = np.zeros((W, self.N_cw), dtype=in_dtype)
        dummy_out = np.zeros((W, self.K), dtype=out_dtype)
        self._run_numba_wave(dummy_in, dummy_out)

    def _run_numba_wave(self, Y_N_wave, V_K_wave):
        if self.single_wave:
            W = Y_N_wave.shape[0]
            if W != self._work_W:
                self._allocate_workspace(W)
            if self.dec_type == "BP_FLOODING":
                if self.dec_implem == "SPA":
                    return self._kernel(
                        self.var_offsets, self.chk_offsets, self.transpose, self.chk_to_var, self.info_bits_pos,
                        self.n_ite, self.enable_syndrome, self.syndrome_depth,
                        self.Y_N_inter, self.post_inter, self.msg_chk_to_var_inter, self.msg_var_to_chk_inter,
                        self.values_inter, self.sign_w, self.prod_w, self.nzeros_w, self.synd_w,
                        Y_N_wave, V_K_wave,
                    )
                elif not self.is_i16:
                    return self._kernel(
                        self.var_offsets, self.chk_offsets, self.transpose, self.chk_to_var, self.info_bits_pos,
                        self.n_ite, self.enable_syndrome, self.syndrome_depth, np.float32(self.offset),
                        self.Y_N_inter, self.post_inter, self.msg_chk_to_var_inter, self.msg_var_to_chk_inter,
                        self.sign_w, self.min1_w, self.min2_w, self.cst1_w, self.cst2_w, self.synd_w,
                        Y_N_wave, V_K_wave,
                    )
                else:
                    return self._kernel(
                        self.var_offsets, self.chk_offsets, self.transpose, self.chk_to_var, self.info_bits_pos,
                        self.n_ite, self.enable_syndrome, self.syndrome_depth, np.int16(self.offset), self.sat_val_i16,
                        self.Y_N_inter, self.post_inter, self.msg_chk_to_var_inter, self.msg_var_to_chk_inter,
                        self.sign_w, self.min1_w, self.min2_w, self.cst1_w, self.cst2_w, self.synd_w,
                        Y_N_wave, V_K_wave,
                    )
            else:
                if self.dec_implem == "SPA":
                    return self._kernel(
                        self.chk_offsets, self.chk_to_var, self.info_bits_pos,
                        self.n_ite, self.enable_syndrome, self.syndrome_depth,
                        self.post_inter, self.msg_chk_to_var_inter, self.values_inter, self.contrib_inter,
                        self.sign_w, self.prod_w, self.nzeros_w, self.synd_w,
                        Y_N_wave, V_K_wave,
                    )
                elif not self.is_i16:
                    return self._kernel(
                        self.chk_offsets, self.chk_to_var, self.info_bits_pos,
                        self.n_ite, self.enable_syndrome, self.syndrome_depth, np.float32(self.offset),
                        self.post_inter, self.msg_chk_to_var_inter, self.contrib_inter,
                        self.sign_w, self.min1_w, self.min2_w, self.cst1_w, self.cst2_w, self.synd_w,
                        Y_N_wave, V_K_wave,
                    )
                else:
                    return self._kernel(
                        self.chk_offsets, self.chk_to_var, self.info_bits_pos,
                        self.n_ite, self.enable_syndrome, self.syndrome_depth, np.int16(self.offset), self.sat_val_i16,
                        self.post_inter, self.msg_chk_to_var_inter, self.contrib_inter,
                        self.sign_w, self.min1_w, self.min2_w, self.cst1_w, self.cst2_w, self.synd_w,
                        Y_N_wave, V_K_wave,
                    )
        else:
            Y_N_1d = Y_N_wave[0]
            V_K_1d = V_K_wave[0]
            if self.dec_type == "BP_FLOODING":
                if self.dec_implem == "SPA":
                    return self._kernel(
                        self.var_offsets, self.chk_offsets, self.transpose, self.chk_to_var, self.info_bits_pos,
                        self.n_ite, self.enable_syndrome, self.syndrome_depth,
                        self.msg_chk_to_var, self.msg_var_to_chk, self.post, self.values,
                        Y_N_1d, V_K_1d,
                    )
                elif not self.is_i16:
                    return self._kernel(
                        self.var_offsets, self.chk_offsets, self.transpose, self.chk_to_var, self.info_bits_pos,
                        self.n_ite, self.enable_syndrome, self.syndrome_depth, np.float32(self.offset),
                        self.msg_chk_to_var, self.msg_var_to_chk, self.post,
                        Y_N_1d, V_K_1d,
                    )
                else:
                    return self._kernel(
                        self.var_offsets, self.chk_offsets, self.transpose, self.chk_to_var, self.info_bits_pos,
                        self.n_ite, self.enable_syndrome, self.syndrome_depth, np.int16(self.offset), self.sat_val_i16,
                        self.msg_chk_to_var, self.msg_var_to_chk, self.post,
                        Y_N_1d, V_K_1d,
                    )
            else:
                if self.dec_implem == "SPA":
                    return self._kernel(
                        self.chk_offsets, self.chk_to_var, self.info_bits_pos,
                        self.n_ite, self.enable_syndrome, self.syndrome_depth,
                        self.msg_chk_to_var, self.post, self.values, self.contrib,
                        Y_N_1d, V_K_1d,
                    )
                elif not self.is_i16:
                    return self._kernel(
                        self.chk_offsets, self.chk_to_var, self.info_bits_pos,
                        self.n_ite, self.enable_syndrome, self.syndrome_depth, np.float32(self.offset),
                        self.msg_chk_to_var, self.post, self.contrib,
                        Y_N_1d, V_K_1d,
                    )
                else:
                    return self._kernel(
                        self.chk_offsets, self.chk_to_var, self.info_bits_pos,
                        self.n_ite, self.enable_syndrome, self.syndrome_depth, np.int16(self.offset), self.sat_val_i16,
                        self.msg_chk_to_var, self.post, self.contrib,
                        Y_N_1d, V_K_1d,
                    )

    @staticmethod
    def _decode_numba(mdl, tsk, frame_id):
        return mdl._run_numba_wave(tsk["Y_N"].wave, tsk["V_K"].wave)

    @staticmethod
    def _decode_numpy(mdl, tsk, frame_id):
        if mdl.dec_type == "BP_HORIZONTAL_LAYERED":
            return _numpy_decode_layered(mdl, tsk["Y_N"].wave, tsk["V_K"].wave)
        return _numpy_decode_flooding(mdl, tsk["Y_N"].wave, tsk["V_K"].wave)

    def clone(self):
        other = type(self).__new__(type(self))
        spu.Stateful_core.__init__(other, self)
        other.dec_type = self.dec_type
        other.dec_implem = self.dec_implem
        other.q_type = self.q_type
        other.is_i16 = self.is_i16
        other.simd_strategy = self.simd_strategy
        other.single_wave = self.single_wave
        other.n_ite = self.n_ite
        other.offset = self.offset
        other.enable_syndrome = self.enable_syndrome
        other.syndrome_depth = self.syndrome_depth
        other.use_numba = self.use_numba
        other.K = self.K
        other.N_cw = self.N_cw
        other.n_branches = self.n_branches
        other.max_chk_degree = self.max_chk_degree
        other.sat_val_i16 = self.sat_val_i16
        # Share immutable CSR/CSC graph arrays across thread clones
        other.var_offsets = self.var_offsets
        other.chk_offsets = self.chk_offsets
        other.transpose = self.transpose
        other.chk_to_var = self.chk_to_var
        other.info_bits_pos = self.info_bits_pos
        other._kernel = self._kernel
        # Allocate independent per-thread workspace buffers
        other._allocate_workspace(self._work_W)
        return other

