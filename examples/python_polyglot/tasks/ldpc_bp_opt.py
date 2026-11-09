"""
Generic Modular LDPC Belief Propagation Decoder in Python / Numba for StreamPU (`Stateful_core`)
Tier 3 (`ldpc_bp_opt.py`): Hardware-Optimized Numba + `llvmlite` SIMD Vector Implementation.

Mirrors `ldpc_bp_opt.jl` with explicit LLVM IR vector intrinsics (`llvmlite.ir.VectorType`):
  - Scalar mode (`W = 1`):
      * Branchless Cephes polynomial `_tanh_cephes_f32` and `_atanh_cephes_f32` with IEEE-754
        `@intrinsic` bitcasts (avoiding libm call overhead and enabling full inlining).
      * Hardware saturating `@llvm.sadd.sat.i16` / `@llvm.ssub.sat.i16` for 16-bit fixed-point.
  - Inter-frame SIMD mode (`INTER`, `W in (4, 8, 16)`):
      * Register-resident `<W x float>` and `<W x i16>` LLVM SIMD vector operations via
        `types.UniTuple` (`instcombine` folds tuple insert/extract away completely, emitting
        pure NEON / AVX2 / AVX-512 vector instructions).
      * Vectorized Cephes `<W x float>` `vtanh_cephes` and `vatanh_cephes` in LLVM IR.
      * Vectorized SPA and OMS check-node updates (`vspa_in`, `vspa_out_prep`, `vspa_out_finish`,
        `voms_in`, `voms_cst`, `voms_out`) and horizontal `@llvm.vector.reduce.or` syndrome checks.
  - Ahead-of-execution `.compile(...)` precompilation at module load.
"""

import numpy as np
from numba import njit, types
from numba.extending import intrinsic
from llvmlite import ir


def _decl_intrinsic(builder, name, ret_ty, arg_tys):
    fnty = ir.FunctionType(ret_ty, arg_tys)
    try:
        return builder.module.get_global(name)
    except KeyError:
        return ir.Function(builder.module, fnty, name=name)


# ====================================================================================================
# 1. SCALAR LLVM INTRINSICS & CEPHES POLYNOMIALS (`W = 1`)
# ====================================================================================================

@intrinsic
def _bitcast_f32_to_i32(typingctx, x_ty):
    def codegen(context, builder, sig, args):
        return builder.bitcast(args[0], ir.IntType(32))
    return types.int32(types.float32), codegen


@intrinsic
def _bitcast_i32_to_f32(typingctx, x_ty):
    def codegen(context, builder, sig, args):
        return builder.bitcast(args[0], ir.FloatType())
    return types.float32(types.int32), codegen


@intrinsic
def _sadd_sat_i16(typingctx, a_ty, b_ty):
    def codegen(context, builder, sig, args):
        i16 = ir.IntType(16)
        fn = _decl_intrinsic(builder, "llvm.sadd.sat.i16", i16, [i16, i16])
        return builder.call(fn, [args[0], args[1]])
    return types.int16(types.int16, types.int16), codegen


@intrinsic
def _ssub_sat_i16(typingctx, a_ty, b_ty):
    def codegen(context, builder, sig, args):
        i16 = ir.IntType(16)
        fn = _decl_intrinsic(builder, "llvm.ssub.sat.i16", i16, [i16, i16])
        return builder.call(fn, [args[0], args[1]])
    return types.int16(types.int16, types.int16), codegen


@njit(nogil=True, fastmath=True, inline="always")
def _clamp_sat_i16(x: np.int16, sat: np.int16) -> np.int16:
    neg_sat = np.int16(-sat)
    if x > sat:
        return sat
    if x < neg_sat:
        return neg_sat
    return x


@njit(nogil=True, fastmath=True, inline="always")
def _add_sat_val_i16(a: np.int16, b: np.int16, sat: np.int16) -> np.int16:
    return _clamp_sat_i16(_sadd_sat_i16(a, b), sat)


@njit(nogil=True, fastmath=True, inline="always")
def _tanh_cephes_f32(x: np.float32) -> np.float32:
    xbits = _bitcast_f32_to_i32(x)
    sgn_bit = xbits & np.int32(-2147483648)
    z = _bitcast_i32_to_f32(xbits & np.int32(0x7FFFFFFF))
    z2 = np.float32(z * z)
    p = np.float32(-5.70498872745e-3)
    p = np.float32(p * z2 + np.float32(2.06390887954e-2))
    p = np.float32(p * z2 + np.float32(-5.37397155531e-2))
    p = np.float32(p * z2 + np.float32(1.33314422036e-1))
    p = np.float32(p * z2 + np.float32(-3.33332819422e-1))
    y_small = np.float32(z * z2 * p + z)

    x2 = np.float32(2.0) * z
    if x2 > np.float32(88.3762626647949):
        x2 = np.float32(88.3762626647949)
    n_int = np.int32(x2 * np.float32(1.44269504088896341) + np.float32(0.5))
    fx = np.float32(n_int)
    x2 = np.float32(x2 - fx * np.float32(0.693359375))
    x2 = np.float32(x2 - fx * np.float32(-2.12194440e-4))
    ze = np.float32(x2 * x2)
    pe = np.float32(1.9875691500e-4)
    pe = np.float32(pe * x2 + np.float32(1.3981999507e-3))
    pe = np.float32(pe * x2 + np.float32(8.3334519073e-3))
    pe = np.float32(pe * x2 + np.float32(4.1665795894e-2))
    pe = np.float32(pe * x2 + np.float32(1.6666665459e-1))
    pe = np.float32(pe * x2 + np.float32(5.0000001201e-1))
    ye = np.float32(pe * ze + x2 + np.float32(1.0))
    pow2n = _bitcast_i32_to_f32((n_int + np.int32(0x7F)) << np.int32(23))
    e = np.float32(ye * pow2n)
    y_med = np.float32(np.float32(1.0) - np.float32(2.0) / (e + np.float32(1.0)))

    y = y_small if z < np.float32(0.625) else y_med
    if z > np.float32(19.061547465398498):
        y = np.float32(1.0)
    return _bitcast_i32_to_f32(_bitcast_f32_to_i32(y) | sgn_bit)


@njit(nogil=True, fastmath=True, inline="always")
def _atanh_cephes_f32(x: np.float32) -> np.float32:
    xbits = _bitcast_f32_to_i32(x)
    sgn_bit = xbits & np.int32(-2147483648)
    z = _bitcast_i32_to_f32(xbits & np.int32(0x7FFFFFFF))
    z2 = np.float32(z * z)
    p = np.float32(1.81740078349e-1)
    p = np.float32(p * z2 + np.float32(8.24370301058e-2))
    p = np.float32(p * z2 + np.float32(1.46691431730e-1))
    p = np.float32(p * z2 + np.float32(1.99782164500e-1))
    p = np.float32(p * z2 + np.float32(3.33337300303e-1))
    y_small = np.float32(z * z2 * p + z)

    r = np.float32((np.float32(1.0) + z) / (np.float32(1.0) - z))
    r_bits = _bitcast_f32_to_i32(r)
    exp_i = (r_bits >> np.int32(23)) - np.int32(0x7E)
    xm = _bitcast_i32_to_f32((r_bits & np.int32(0x007FFFFF)) | np.int32(0x3F000000))
    if xm < np.float32(0.707106781186547524):
        xm = np.float32(xm + xm)
        exp_i -= np.int32(1)
    fe = np.float32(exp_i)
    xm1 = np.float32(xm - np.float32(1.0))
    x2 = np.float32(xm1 * xm1)
    pl = np.float32(7.0376836292e-2)
    pl = np.float32(pl * xm1 + np.float32(-1.1514610310e-1))
    pl = np.float32(pl * xm1 + np.float32(1.1676998740e-1))
    pl = np.float32(pl * xm1 + np.float32(-1.2420140846e-1))
    pl = np.float32(pl * xm1 + np.float32(1.4249322787e-1))
    pl = np.float32(pl * xm1 + np.float32(-1.6668057665e-1))
    pl = np.float32(pl * xm1 + np.float32(2.0000714765e-1))
    pl = np.float32(pl * xm1 + np.float32(-2.4999993993e-1))
    pl = np.float32(pl * xm1 + np.float32(3.3333331174e-1))
    yl = np.float32(pl * xm1 * x2)
    yl = np.float32(yl + fe * np.float32(-2.12194440e-4))
    yl = np.float32(yl - np.float32(0.5) * x2)
    yl = np.float32(xm1 + yl + fe * np.float32(0.693359375))
    y_med = np.float32(np.float32(0.5) * yl)

    y = y_small if z < np.float32(0.5) else y_med
    if z >= np.float32(1.0):
        y = _bitcast_i32_to_f32(np.int32(0x7F800000))
    return _bitcast_i32_to_f32(_bitcast_f32_to_i32(y) | sgn_bit)


@njit(nogil=True, fastmath=True)
def decode_siho_opt_scalar_f32(
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
    post: np.ndarray,
    c2v: np.ndarray,
    v2c: np.ndarray,
    vals: np.ndarray,
    contrib: np.ndarray,
    Y_N_wave: np.ndarray,
    V_K_wave: np.ndarray,
) -> int:
    K = info_bits_pos.shape[0]
    n_var = var_offsets.shape[0] - 1
    n_chk = chk_offsets.shape[0] - 1
    Y_N = Y_N_wave[0]
    V_K = V_K_wave[0]
    max_tanh = np.float32(1.0) - np.float32(1.1920929e-7)
    inf_f32 = np.float32(3.4028235e38)

    c2v.fill(np.float32(0.0))
    if is_layered:
        for v in range(n_var):
            post[v] = Y_N[v]

    cur_depth = 0
    valid = True
    ite_done = 0
    for ite in range(1, n_ite + 1):
        if not is_layered:
            for v in range(n_var):
                e_start, e_stop = var_offsets[v], var_offsets[v + 1]
                acc = Y_N[v]
                for e in range(e_start, e_stop):
                    acc = np.float32(acc + c2v[e])
                for e in range(e_start, e_stop):
                    v2c[e] = np.float32(acc - c2v[e])
            if not is_oms:
                for c in range(n_chk):
                    k_start, k_stop = chk_offsets[c], chk_offsets[c + 1]
                    deg = k_stop - k_start
                    sng = np.int32(0)
                    prd = np.float32(1.0)
                    nz = 0
                    for idx in range(deg):
                        v_val = v2c[transpose[k_start + idx]]
                        res = _tanh_cephes_f32(np.float32(abs(v_val) * np.float32(0.5)))
                        vals[idx] = res
                        if res > np.float32(0.0):
                            sng ^= _bitcast_f32_to_i32(v_val)
                            prd = np.float32(prd * res)
                        else:
                            nz += 1
                    for idx in range(deg):
                        t_k = transpose[k_start + idx]
                        v_val = v2c[t_k]
                        val_v = vals[idx]
                        isz = val_v == np.float32(0.0)
                        if nz == 0 or (nz == 1 and isz):
                            rt = prd if isz else np.float32(prd / val_v)
                            if rt > max_tanh:
                                rt = max_tanh
                            mag = np.float32(np.float32(2.0) * _atanh_cephes_f32(rt))
                            v_sng = np.int32(0) if isz else _bitcast_f32_to_i32(v_val)
                            s_bit = (sng ^ v_sng) & np.int32(-2147483648)
                            c2v[t_k] = _bitcast_i32_to_f32(_bitcast_f32_to_i32(mag) | s_bit)
                        else:
                            c2v[t_k] = np.float32(0.0)
            else:
                for c in range(n_chk):
                    k_start, k_stop = chk_offsets[c], chk_offsets[c + 1]
                    sng = False
                    min1 = inf_f32
                    min2 = inf_f32
                    for k in range(k_start, k_stop):
                        v_val = v2c[transpose[k]]
                        v_abs = np.float32(-v_val) if v_val < np.float32(0.0) else v_val
                        sng ^= v_val < np.float32(0.0)
                        mx = v_abs if v_abs > min1 else min1
                        min2 = mx if mx < min2 else min2
                        min1 = v_abs if v_abs < min1 else min1
                    d1 = np.float32(min2 - offset)
                    d2 = np.float32(min1 - offset)
                    cst1 = d1 if d1 > np.float32(0.0) else np.float32(0.0)
                    cst2 = d2 if d2 > np.float32(0.0) else np.float32(0.0)
                    for k in range(k_start, k_stop):
                        t_k = transpose[k]
                        v_val = v2c[t_k]
                        v_abs = np.float32(-v_val) if v_val < np.float32(0.0) else v_val
                        mag = cst1 if v_abs == min1 else cst2
                        s_out = sng ^ (v_val < np.float32(0.0))
                        c2v[t_k] = -mag if s_out else mag
        else:
            if not is_oms:
                for c in range(n_chk):
                    k_start, k_stop = chk_offsets[c], chk_offsets[c + 1]
                    deg = k_stop - k_start
                    sng = np.int32(0)
                    prd = np.float32(1.0)
                    nz = 0
                    for idx in range(deg):
                        k = k_start + idx
                        v_val = np.float32(post[chk_to_var[k]] - c2v[k])
                        contrib[idx] = v_val
                        res = _tanh_cephes_f32(np.float32(abs(v_val) * np.float32(0.5)))
                        vals[idx] = res
                        if res > np.float32(0.0):
                            sng ^= _bitcast_f32_to_i32(v_val)
                            prd = np.float32(prd * res)
                        else:
                            nz += 1
                    for idx in range(deg):
                        k = k_start + idx
                        u = chk_to_var[k]
                        v_val = contrib[idx]
                        val_v = vals[idx]
                        isz = val_v == np.float32(0.0)
                        if nz == 0 or (nz == 1 and isz):
                            rt = prd if isz else np.float32(prd / val_v)
                            if rt > max_tanh:
                                rt = max_tanh
                            mag = np.float32(np.float32(2.0) * _atanh_cephes_f32(rt))
                            v_sng = np.int32(0) if isz else _bitcast_f32_to_i32(v_val)
                            s_bit = (sng ^ v_sng) & np.int32(-2147483648)
                            new_msg = _bitcast_i32_to_f32(_bitcast_f32_to_i32(mag) | s_bit)
                        else:
                            new_msg = np.float32(0.0)
                        c2v[k] = new_msg
                        post[u] = np.float32(v_val + new_msg)
            else:
                for c in range(n_chk):
                    k_start, k_stop = chk_offsets[c], chk_offsets[c + 1]
                    deg = k_stop - k_start
                    sng = False
                    min1 = inf_f32
                    min2 = inf_f32
                    for idx in range(deg):
                        k = k_start + idx
                        v_val = np.float32(post[chk_to_var[k]] - c2v[k])
                        contrib[idx] = v_val
                        v_abs = np.float32(-v_val) if v_val < np.float32(0.0) else v_val
                        sng ^= v_val < np.float32(0.0)
                        mx = v_abs if v_abs > min1 else min1
                        min2 = mx if mx < min2 else min2
                        min1 = v_abs if v_abs < min1 else min1
                    d1 = np.float32(min2 - offset)
                    d2 = np.float32(min1 - offset)
                    cst1 = d1 if d1 > np.float32(0.0) else np.float32(0.0)
                    cst2 = d2 if d2 > np.float32(0.0) else np.float32(0.0)
                    for idx in range(deg):
                        k = k_start + idx
                        u = chk_to_var[k]
                        v_val = contrib[idx]
                        v_abs = np.float32(-v_val) if v_val < np.float32(0.0) else v_val
                        mag = cst1 if v_abs == min1 else cst2
                        s_out = sng ^ (v_val < np.float32(0.0))
                        new_msg = -mag if s_out else mag
                        c2v[k] = new_msg
                        post[u] = np.float32(v_val + new_msg)

        if enable_syndrome and (is_layered or ite != n_ite):
            if not is_layered:
                for v in range(n_var):
                    acc = Y_N[v]
                    for e in range(var_offsets[v], var_offsets[v + 1]):
                        acc = np.float32(acc + c2v[e])
                    post[v] = acc
            synd = True
            for c in range(n_chk):
                parity = False
                for k in range(chk_offsets[c], chk_offsets[c + 1]):
                    parity ^= post[chk_to_var[k]] < np.float32(0.0)
                if parity:
                    synd = False
                    break
            cur_depth = ((cur_depth + 1) % syndrome_depth) if synd else 0
            valid = synd and (cur_depth == 0)
            if valid:
                ite_done = ite
                break
        if ite == n_ite:
            ite_done = n_ite

    if not is_layered and ite_done == n_ite:
        for v in range(n_var):
            acc = Y_N[v]
            for e in range(var_offsets[v], var_offsets[v + 1]):
                acc = np.float32(acc + c2v[e])
            post[v] = acc

    for i in range(K):
        V_K[i] = np.int32(1) if post[info_bits_pos[i]] < np.float32(0.0) else np.int32(0)
    return 0 if valid else 1


@njit(nogil=True, fastmath=True)
def decode_siho_opt_scalar_i16(
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
    post: np.ndarray,
    c2v: np.ndarray,
    v2c: np.ndarray,
    contrib: np.ndarray,
    Y_N_wave: np.ndarray,
    V_K_wave: np.ndarray,
) -> int:
    K = info_bits_pos.shape[0]
    n_var = var_offsets.shape[0] - 1
    n_chk = chk_offsets.shape[0] - 1
    Y_N = Y_N_wave[0]
    V_K = V_K_wave[0]
    max_i16 = np.int16(32767)

    c2v.fill(np.int16(0))
    if is_layered:
        for v in range(n_var):
            post[v] = Y_N[v]

    cur_depth = 0
    valid = True
    ite_done = 0
    for ite in range(1, n_ite + 1):
        if not is_layered:
            for v in range(n_var):
                e_start, e_stop = var_offsets[v], var_offsets[v + 1]
                acc = Y_N[v]
                for e in range(e_start, e_stop):
                    acc = _sadd_sat_i16(acc, c2v[e])
                for e in range(e_start, e_stop):
                    v2c[e] = _ssub_sat_i16(acc, c2v[e])
            for c in range(n_chk):
                k_start, k_stop = chk_offsets[c], chk_offsets[c + 1]
                sng = np.int16(0)
                min1 = max_i16
                min2 = max_i16
                for k in range(k_start, k_stop):
                    v_val = v2c[transpose[k]]
                    v_abs = np.int16(-v_val) if v_val < np.int16(0) else v_val
                    sng ^= v_val
                    mx = v_abs if v_abs > min1 else min1
                    min2 = mx if mx < min2 else min2
                    min1 = v_abs if v_abs < min1 else min1
                d1 = _ssub_sat_i16(min2, offset)
                d2 = _ssub_sat_i16(min1, offset)
                cst1 = d1 if d1 > np.int16(0) else np.int16(0)
                cst2 = d2 if d2 > np.int16(0) else np.int16(0)
                for k in range(k_start, k_stop):
                    t_k = transpose[k]
                    v_val = v2c[t_k]
                    v_abs = np.int16(-v_val) if v_val < np.int16(0) else v_val
                    mag = cst1 if v_abs == min1 else cst2
                    s_out = (sng ^ v_val) < np.int16(0)
                    c2v[t_k] = _clamp_sat_i16(np.int16(-mag) if s_out else mag, sat_val)
        else:
            for c in range(n_chk):
                k_start, k_stop = chk_offsets[c], chk_offsets[c + 1]
                deg = k_stop - k_start
                sng = np.int16(0)
                min1 = max_i16
                min2 = max_i16
                for idx in range(deg):
                    k = k_start + idx
                    v_val = _ssub_sat_i16(post[chk_to_var[k]], c2v[k])
                    contrib[idx] = v_val
                    v_abs = np.int16(-v_val) if v_val < np.int16(0) else v_val
                    sng ^= v_val
                    mx = v_abs if v_abs > min1 else min1
                    min2 = mx if mx < min2 else min2
                    min1 = v_abs if v_abs < min1 else min1
                d1 = _ssub_sat_i16(min2, offset)
                d2 = _ssub_sat_i16(min1, offset)
                cst1 = d1 if d1 > np.int16(0) else np.int16(0)
                cst2 = d2 if d2 > np.int16(0) else np.int16(0)
                for idx in range(deg):
                    k = k_start + idx
                    u = chk_to_var[k]
                    v_val = contrib[idx]
                    v_abs = np.int16(-v_val) if v_val < np.int16(0) else v_val
                    mag = cst1 if v_abs == min1 else cst2
                    s_out = (sng ^ v_val) < np.int16(0)
                    new_msg = _clamp_sat_i16(np.int16(-mag) if s_out else mag, sat_val)
                    c2v[k] = new_msg
                    post[u] = _add_sat_val_i16(v_val, new_msg, sat_val)

        if enable_syndrome and (is_layered or ite != n_ite):
            if not is_layered:
                for v in range(n_var):
                    acc = Y_N[v]
                    for e in range(var_offsets[v], var_offsets[v + 1]):
                        acc = _sadd_sat_i16(acc, c2v[e])
                    post[v] = acc
            synd = True
            for c in range(n_chk):
                parity = np.int16(0)
                for k in range(chk_offsets[c], chk_offsets[c + 1]):
                    parity ^= post[chk_to_var[k]]
                if parity < np.int16(0):
                    synd = False
                    break
            cur_depth = ((cur_depth + 1) % syndrome_depth) if synd else 0
            valid = synd and (cur_depth == 0)
            if valid:
                ite_done = ite
                break
        if ite == n_ite:
            ite_done = n_ite

    if not is_layered and ite_done == n_ite:
        for v in range(n_var):
            acc = Y_N[v]
            for e in range(var_offsets[v], var_offsets[v + 1]):
                acc = _sadd_sat_i16(acc, c2v[e])
            post[v] = acc

    for i in range(K):
        V_K[i] = np.int16(1) if post[info_bits_pos[i]] < np.int16(0) else np.int16(0)
    return 0 if valid else 1


# ====================================================================================================
# 2. INTER-FRAME LLVM SIMD VECTOR FACTORY (`<W x float>` and `<W x i16>`)
# ====================================================================================================
#
# Why `vreg_f32()` / `vreg_i32()` / `vreg_i16()` (`alloca <W x T>` in entry block):
#   Numba's Python type system lacks a native SIMD vector type. Returning `UniTuple(float32, W)` from
#   `@intrinsic` works within a single basic block, but for loop-carried variables (`acc`, `m1`, `m2`,
#   `sng`, `prd`, `nz`), Numba allocates `[W x float]` (`ir.ArrayType`) on the stack, which LLVM's SROA
#   pass splits into `W` scalar PHI nodes (emitting `mov.s` lane unpack/pack instructions).
#   Allocating `ir.VectorType(T, W)` in the function entry block via `vreg_*()` and operating on it via
#   `<W x T>` loads/stores allows LLVM's `instcombine` + `SROA`/`mem2reg` to promote every `vreg_*()`
#   directly into hardware SIMD SSA registers (`q0..q31` on NEON, `ymm` on AVX2, `zmm` on AVX-512) with
#   zero lane shuffling and zero stack traffic.

def _make_f32_simd_kernel(W: int):
    vf32_ty = ir.VectorType(ir.FloatType(), W)
    vi32_ty = ir.VectorType(ir.IntType(32), W)
    pf32_ty = ir.PointerType(vf32_ty)
    pi32_ty = ir.PointerType(vi32_ty)

    def _vconst_f32(val):
        c = ir.FloatType()(float(val))
        return ir.Constant(vf32_ty, [c] * W)

    def _vconst_i32(val):
        c = ir.IntType(32)(int(val))
        return ir.Constant(vi32_ty, [c] * W)

    def _bcast_scalar_f32(builder, val):
        i32 = ir.IntType(32)
        u = ir.Constant(vf32_ty, ir.Undefined)
        v0 = builder.insert_element(u, val, i32(0))
        return builder.shuffle_vector(v0, u, ir.Constant(vi32_ty, [i32(0)] * W))

    def _bcast_scalar_i32(builder, val):
        i32 = ir.IntType(32)
        u = ir.Constant(vi32_ty, ir.Undefined)
        v0 = builder.insert_element(u, val, i32(0))
        return builder.shuffle_vector(v0, u, ir.Constant(vi32_ty, [i32(0)] * W))

    @intrinsic
    def vreg_f32(typingctx):
        def codegen(context, builder, sig, args):
            with builder.goto_entry_block():
                ptr = builder.alloca(vf32_ty)
            return builder.ptrtoint(ptr, ir.IntType(64))
        return types.int64(), codegen

    @intrinsic
    def vreg_i32(typingctx):
        def codegen(context, builder, sig, args):
            with builder.goto_entry_block():
                ptr = builder.alloca(vi32_ty)
            return builder.ptrtoint(ptr, ir.IntType(64))
        return types.int64(), codegen

    @intrinsic
    def vload_f32(typingctx, dst_ty, arr_ty, row_ty):
        def codegen(context, builder, sig, args):
            dst = builder.inttoptr(args[0], pf32_ty)
            data_ptr = builder.extract_value(args[1], 4)
            idx = builder.mul(args[2], ir.Constant(args[2].type, W))
            vptr = builder.bitcast(builder.gep(data_ptr, [idx]), pf32_ty)
            builder.store(builder.load(vptr, align=16), dst)
            return context.get_dummy_value()
        return types.none(types.int64, arr_ty, row_ty), codegen

    @intrinsic
    def vstore_f32(typingctx, arr_ty, row_ty, src_ty):
        def codegen(context, builder, sig, args):
            data_ptr = builder.extract_value(args[0], 4)
            idx = builder.mul(args[1], ir.Constant(args[1].type, W))
            vptr = builder.bitcast(builder.gep(data_ptr, [idx]), pf32_ty)
            src = builder.inttoptr(args[2], pf32_ty)
            builder.store(builder.load(src), vptr, align=16)
            return context.get_dummy_value()
        return types.none(arr_ty, row_ty, types.int64), codegen

    @intrinsic
    def vset_f32(typingctx, dst_ty, val_ty):
        def codegen(context, builder, sig, args):
            dst = builder.inttoptr(args[0], pf32_ty)
            builder.store(_bcast_scalar_f32(builder, args[1]), dst)
            return context.get_dummy_value()
        return types.none(types.int64, types.float32), codegen

    @intrinsic
    def vset_i32(typingctx, dst_ty, val_ty):
        def codegen(context, builder, sig, args):
            dst = builder.inttoptr(args[0], pi32_ty)
            builder.store(_bcast_scalar_i32(builder, args[1]), dst)
            return context.get_dummy_value()
        return types.none(types.int64, types.int32), codegen

    @intrinsic
    def vadd_f32(typingctx, dst_ty, a_ty, b_ty):
        def codegen(context, builder, sig, args):
            dst = builder.inttoptr(args[0], pf32_ty)
            va = builder.load(builder.inttoptr(args[1], pf32_ty))
            vb = builder.load(builder.inttoptr(args[2], pf32_ty))
            builder.store(builder.fadd(va, vb, flags=["fast"]), dst)
            return context.get_dummy_value()
        return types.none(types.int64, types.int64, types.int64), codegen

    @intrinsic
    def vsub_f32(typingctx, dst_ty, a_ty, b_ty):
        def codegen(context, builder, sig, args):
            dst = builder.inttoptr(args[0], pf32_ty)
            va = builder.load(builder.inttoptr(args[1], pf32_ty))
            vb = builder.load(builder.inttoptr(args[2], pf32_ty))
            builder.store(builder.fsub(va, vb, flags=["fast"]), dst)
            return context.get_dummy_value()
        return types.none(types.int64, types.int64, types.int64), codegen

    @intrinsic
    def voms_in_f32(typingctx, v_ty, sng_ty, m1_ty, m2_ty):
        def codegen(context, builder, sig, args):
            v = builder.load(builder.inttoptr(args[0], pf32_ty))
            p_sng = builder.inttoptr(args[1], pi32_ty)
            p_m1 = builder.inttoptr(args[2], pf32_ty)
            p_m2 = builder.inttoptr(args[3], pf32_ty)
            sng = builder.load(p_sng)
            m1 = builder.load(p_m1)
            m2 = builder.load(p_m2)
            vbits = builder.bitcast(v, vi32_ty)
            sng_out = builder.xor(sng, vbits)
            f_abs = _decl_intrinsic(builder, f"llvm.fabs.v{W}f32", vf32_ty, [vf32_ty])
            f_min = _decl_intrinsic(builder, f"llvm.minnum.v{W}f32", vf32_ty, [vf32_ty, vf32_ty])
            f_max = _decl_intrinsic(builder, f"llvm.maxnum.v{W}f32", vf32_ty, [vf32_ty, vf32_ty])
            va = builder.call(f_abs, [v])
            mx = builder.call(f_max, [va, m1])
            m2_out = builder.call(f_min, [m2, mx])
            m1_out = builder.call(f_min, [m1, va])
            builder.store(sng_out, p_sng)
            builder.store(m1_out, p_m1)
            builder.store(m2_out, p_m2)
            return context.get_dummy_value()
        return types.none(types.int64, types.int64, types.int64, types.int64), codegen

    @intrinsic
    def voms_cst_f32(typingctx, c1_ty, c2_ty, m1_ty, m2_ty, off_ty):
        def codegen(context, builder, sig, args):
            p_c1 = builder.inttoptr(args[0], pf32_ty)
            p_c2 = builder.inttoptr(args[1], pf32_ty)
            m1 = builder.load(builder.inttoptr(args[2], pf32_ty))
            m2 = builder.load(builder.inttoptr(args[3], pf32_ty))
            off = builder.load(builder.inttoptr(args[4], pf32_ty))
            f_max = _decl_intrinsic(builder, f"llvm.maxnum.v{W}f32", vf32_ty, [vf32_ty, vf32_ty])
            z = _vconst_f32(0.0)
            builder.store(builder.call(f_max, [z, builder.fsub(m2, off, flags=["fast"])]), p_c1)
            builder.store(builder.call(f_max, [z, builder.fsub(m1, off, flags=["fast"])]), p_c2)
            return context.get_dummy_value()
        return types.none(types.int64, types.int64, types.int64, types.int64, types.int64), codegen

    @intrinsic
    def voms_out_f32(typingctx, dst_ty, v_ty, sng_ty, m1_ty, c1_ty, c2_ty):
        def codegen(context, builder, sig, args):
            dst = builder.inttoptr(args[0], pf32_ty)
            v = builder.load(builder.inttoptr(args[1], pf32_ty))
            sng = builder.load(builder.inttoptr(args[2], pi32_ty))
            m1 = builder.load(builder.inttoptr(args[3], pf32_ty))
            c1 = builder.load(builder.inttoptr(args[4], pf32_ty))
            c2 = builder.load(builder.inttoptr(args[5], pf32_ty))
            f_abs = _decl_intrinsic(builder, f"llvm.fabs.v{W}f32", vf32_ty, [vf32_ty])
            va = builder.call(f_abs, [v])
            eq = builder.fcmp_ordered("==", va, m1)
            mag = builder.select(eq, c1, c2)
            vbits = builder.bitcast(v, vi32_ty)
            s_out = builder.and_(builder.xor(sng, vbits), _vconst_i32(-2147483648))
            mag_bits = builder.bitcast(mag, vi32_ty)
            res = builder.bitcast(builder.or_(mag_bits, s_out), vf32_ty)
            builder.store(res, dst)
            return context.get_dummy_value()
        return types.none(types.int64, types.int64, types.int64, types.int64, types.int64, types.int64), codegen

    def _emit_tanh_cephes(builder, x):
        f_abs = _decl_intrinsic(builder, f"llvm.fabs.v{W}f32", vf32_ty, [vf32_ty])
        f_min = _decl_intrinsic(builder, f"llvm.minnum.v{W}f32", vf32_ty, [vf32_ty, vf32_ty])
        f_floor = _decl_intrinsic(builder, f"llvm.floor.v{W}f32", vf32_ty, [vf32_ty])
        f_copysign = _decl_intrinsic(builder, f"llvm.copysign.v{W}f32", vf32_ty, [vf32_ty, vf32_ty])
        z = builder.call(f_abs, [x])
        z2 = builder.fmul(z, z, flags=["fast"])
        p = _vconst_f32(-5.70498872745e-3)
        p = builder.fadd(builder.fmul(p, z2, flags=["fast"]), _vconst_f32(2.06390887954e-2), flags=["fast"])
        p = builder.fadd(builder.fmul(p, z2, flags=["fast"]), _vconst_f32(-5.37397155531e-2), flags=["fast"])
        p = builder.fadd(builder.fmul(p, z2, flags=["fast"]), _vconst_f32(1.33314422036e-1), flags=["fast"])
        p = builder.fadd(builder.fmul(p, z2, flags=["fast"]), _vconst_f32(-3.33332819422e-1), flags=["fast"])
        y_small = builder.fadd(builder.fmul(builder.fmul(z, z2, flags=["fast"]), p, flags=["fast"]), z, flags=["fast"])
        x2 = builder.fmul(_vconst_f32(2.0), z, flags=["fast"])
        x2 = builder.call(f_min, [x2, _vconst_f32(88.3762626647949)])
        fx = builder.fadd(builder.fmul(x2, _vconst_f32(1.44269504088896341), flags=["fast"]), _vconst_f32(0.5), flags=["fast"])
        fx = builder.call(f_floor, [fx])
        x2 = builder.fsub(x2, builder.fmul(fx, _vconst_f32(0.693359375), flags=["fast"]), flags=["fast"])
        x2 = builder.fsub(x2, builder.fmul(fx, _vconst_f32(-2.12194440e-4), flags=["fast"]), flags=["fast"])
        z_e = builder.fmul(x2, x2, flags=["fast"])
        pe = _vconst_f32(1.9875691500e-4)
        pe = builder.fadd(builder.fmul(pe, x2, flags=["fast"]), _vconst_f32(1.3981999507e-3), flags=["fast"])
        pe = builder.fadd(builder.fmul(pe, x2, flags=["fast"]), _vconst_f32(8.3334519073e-3), flags=["fast"])
        pe = builder.fadd(builder.fmul(pe, x2, flags=["fast"]), _vconst_f32(4.1665795894e-2), flags=["fast"])
        pe = builder.fadd(builder.fmul(pe, x2, flags=["fast"]), _vconst_f32(1.6666665459e-1), flags=["fast"])
        pe = builder.fadd(builder.fmul(pe, x2, flags=["fast"]), _vconst_f32(5.0000001201e-1), flags=["fast"])
        ye = builder.fadd(builder.fadd(builder.fmul(pe, z_e, flags=["fast"]), x2, flags=["fast"]), _vconst_f32(1.0), flags=["fast"])
        n_int = builder.fptosi(fx, vi32_ty)
        pow2n = builder.bitcast(builder.shl(builder.add(n_int, _vconst_i32(0x7F)), _vconst_i32(23)), vf32_ty)
        e = builder.fmul(ye, pow2n, flags=["fast"])
        y_med = builder.fsub(_vconst_f32(1.0), builder.fdiv(_vconst_f32(2.0), builder.fadd(e, _vconst_f32(1.0), flags=["fast"]), flags=["fast"]), flags=["fast"])
        y = builder.select(builder.fcmp_ordered("<", z, _vconst_f32(0.625)), y_small, y_med)
        y = builder.select(builder.fcmp_ordered(">", z, _vconst_f32(19.061547465398498)), _vconst_f32(1.0), y)
        return builder.call(f_copysign, [y, x])

    def _emit_atanh_cephes(builder, x):
        f_abs = _decl_intrinsic(builder, f"llvm.fabs.v{W}f32", vf32_ty, [vf32_ty])
        f_copysign = _decl_intrinsic(builder, f"llvm.copysign.v{W}f32", vf32_ty, [vf32_ty, vf32_ty])
        z = builder.call(f_abs, [x])
        z2 = builder.fmul(z, z, flags=["fast"])
        p = _vconst_f32(1.81740078349e-1)
        p = builder.fadd(builder.fmul(p, z2, flags=["fast"]), _vconst_f32(8.24370301058e-2), flags=["fast"])
        p = builder.fadd(builder.fmul(p, z2, flags=["fast"]), _vconst_f32(1.46691431730e-1), flags=["fast"])
        p = builder.fadd(builder.fmul(p, z2, flags=["fast"]), _vconst_f32(1.99782164500e-1), flags=["fast"])
        p = builder.fadd(builder.fmul(p, z2, flags=["fast"]), _vconst_f32(3.33337300303e-1), flags=["fast"])
        y_small = builder.fadd(builder.fmul(builder.fmul(z, z2, flags=["fast"]), p, flags=["fast"]), z, flags=["fast"])
        r = builder.fdiv(builder.fadd(_vconst_f32(1.0), z, flags=["fast"]), builder.fsub(_vconst_f32(1.0), z, flags=["fast"]), flags=["fast"])
        r_bits = builder.bitcast(r, vi32_ty)
        exp_i = builder.sub(builder.ashr(r_bits, _vconst_i32(23)), _vconst_i32(0x7E))
        xm = builder.bitcast(builder.or_(builder.and_(r_bits, _vconst_i32(0x007FFFFF)), _vconst_i32(0x3F000000)), vf32_ty)
        lt_s = builder.fcmp_ordered("<", xm, _vconst_f32(0.707106781186547524))
        xm_adj = builder.select(lt_s, builder.fadd(xm, xm, flags=["fast"]), xm)
        exp_adj = builder.select(lt_s, builder.sub(exp_i, _vconst_i32(1)), exp_i)
        fe = builder.sitofp(exp_adj, vf32_ty)
        xm1 = builder.fsub(xm_adj, _vconst_f32(1.0), flags=["fast"])
        x2 = builder.fmul(xm1, xm1, flags=["fast"])
        pl = _vconst_f32(7.0376836292e-2)
        pl = builder.fadd(builder.fmul(pl, xm1, flags=["fast"]), _vconst_f32(-1.1514610310e-1), flags=["fast"])
        pl = builder.fadd(builder.fmul(pl, xm1, flags=["fast"]), _vconst_f32(1.1676998740e-1), flags=["fast"])
        pl = builder.fadd(builder.fmul(pl, xm1, flags=["fast"]), _vconst_f32(-1.2420140846e-1), flags=["fast"])
        pl = builder.fadd(builder.fmul(pl, xm1, flags=["fast"]), _vconst_f32(1.4249322787e-1), flags=["fast"])
        pl = builder.fadd(builder.fmul(pl, xm1, flags=["fast"]), _vconst_f32(-1.6668057665e-1), flags=["fast"])
        pl = builder.fadd(builder.fmul(pl, xm1, flags=["fast"]), _vconst_f32(2.0000714765e-1), flags=["fast"])
        pl = builder.fadd(builder.fmul(pl, xm1, flags=["fast"]), _vconst_f32(-2.4999993993e-1), flags=["fast"])
        pl = builder.fadd(builder.fmul(pl, xm1, flags=["fast"]), _vconst_f32(3.3333331174e-1), flags=["fast"])
        yl = builder.fmul(pl, builder.fmul(xm1, x2, flags=["fast"]), flags=["fast"])
        yl = builder.fadd(yl, builder.fmul(fe, _vconst_f32(-2.12194440e-4), flags=["fast"]), flags=["fast"])
        yl = builder.fsub(yl, builder.fmul(_vconst_f32(0.5), x2, flags=["fast"]), flags=["fast"])
        yl = builder.fadd(xm1, yl, flags=["fast"])
        yl = builder.fadd(yl, builder.fmul(fe, _vconst_f32(0.693359375), flags=["fast"]), flags=["fast"])
        y_med = builder.fmul(_vconst_f32(0.5), yl, flags=["fast"])
        y = builder.select(builder.fcmp_ordered("<", z, _vconst_f32(0.5)), y_small, y_med)
        y = builder.select(builder.fcmp_ordered(">=", z, _vconst_f32(1.0)), _vconst_f32(float("inf")), y)
        return builder.call(f_copysign, [y, x])

    @intrinsic
    def vspa_in_f32(typingctx, res_ty, v_ty, sng_ty, prd_ty, nz_ty):
        def codegen(context, builder, sig, args):
            p_res = builder.inttoptr(args[0], pf32_ty)
            v = builder.load(builder.inttoptr(args[1], pf32_ty))
            p_sng = builder.inttoptr(args[2], pi32_ty)
            p_prd = builder.inttoptr(args[3], pf32_ty)
            p_nz = builder.inttoptr(args[4], pi32_ty)
            sng = builder.load(p_sng)
            prd = builder.load(p_prd)
            nz = builder.load(p_nz)
            f_abs = _decl_intrinsic(builder, f"llvm.fabs.v{W}f32", vf32_ty, [vf32_ty])
            half_abs = builder.fmul(builder.call(f_abs, [v]), _vconst_f32(0.5), flags=["fast"])
            res = _emit_tanh_cephes(builder, half_abs)
            builder.store(res, p_res)
            gt0 = builder.fcmp_ordered(">", res, _vconst_f32(0.0))
            vbits = builder.bitcast(v, vi32_ty)
            sng_out = builder.select(gt0, builder.xor(sng, vbits), sng)
            prd_out = builder.select(gt0, builder.fmul(prd, res, flags=["fast"]), prd)
            nz_out = builder.select(gt0, nz, builder.add(nz, _vconst_i32(1)))
            builder.store(sng_out, p_sng)
            builder.store(prd_out, p_prd)
            builder.store(nz_out, p_nz)
            return context.get_dummy_value()
        return types.none(types.int64, types.int64, types.int64, types.int64, types.int64), codegen

    @intrinsic
    def vspa_out_f32(typingctx, dst_ty, v_ty, val_ty, sng_ty, prd_ty, nz_ty):
        def codegen(context, builder, sig, args):
            dst = builder.inttoptr(args[0], pf32_ty)
            v = builder.load(builder.inttoptr(args[1], pf32_ty))
            val_v = builder.load(builder.inttoptr(args[2], pf32_ty))
            sng = builder.load(builder.inttoptr(args[3], pi32_ty))
            prd = builder.load(builder.inttoptr(args[4], pf32_ty))
            nz = builder.load(builder.inttoptr(args[5], pi32_ty))
            isz = builder.fcmp_ordered("==", val_v, _vconst_f32(0.0))
            q = builder.fdiv(prd, val_v, flags=["fast"])
            rt0 = builder.select(isz, prd, q)
            f_min = _decl_intrinsic(builder, f"llvm.minnum.v{W}f32", vf32_ty, [vf32_ty, vf32_ty])
            rt = builder.call(f_min, [rt0, _vconst_f32(1.0 - 1.1920929e-7)])
            atanh_v = _emit_atanh_cephes(builder, rt)
            mag = builder.fmul(_vconst_f32(2.0), atanh_v, flags=["fast"])
            vbits = builder.bitcast(v, vi32_ty)
            v_sng = builder.select(isz, _vconst_i32(0), vbits)
            s_bit = builder.and_(builder.xor(sng, v_sng), _vconst_i32(-2147483648))
            signed_mag = builder.bitcast(builder.or_(builder.bitcast(mag, vi32_ty), s_bit), vf32_ty)
            nz0 = builder.icmp_unsigned("==", nz, _vconst_i32(0))
            nz1 = builder.icmp_unsigned("==", nz, _vconst_i32(1))
            keep = builder.or_(nz0, builder.and_(nz1, isz))
            res = builder.select(keep, signed_mag, _vconst_f32(0.0))
            builder.store(res, dst)
            return context.get_dummy_value()
        return types.none(types.int64, types.int64, types.int64, types.int64, types.int64, types.int64), codegen

    @intrinsic
    def vsynd_xor_f32(typingctx, synd_ty, pv_ty):
        def codegen(context, builder, sig, args):
            p_s = builder.inttoptr(args[0], pi32_ty)
            s = builder.load(p_s)
            pv = builder.load(builder.inttoptr(args[1], pf32_ty))
            pbits = builder.bitcast(pv, vi32_ty)
            builder.store(builder.xor(s, pbits), p_s)
            return context.get_dummy_value()
        return types.none(types.int64, types.int64), codegen

    @intrinsic
    def vsynd_any_neg_i32(typingctx, synd_ty):
        def codegen(context, builder, sig, args):
            s = builder.load(builder.inttoptr(args[0], pi32_ty))
            s_neg = builder.and_(s, _vconst_i32(-2147483648))
            i32 = ir.IntType(32)
            f_red = _decl_intrinsic(builder, f"llvm.vector.reduce.or.v{W}i32", i32, [vi32_ty])
            r = builder.call(f_red, [s_neg])
            return builder.icmp_unsigned("!=", r, i32(0))
        return types.boolean(types.int64), codegen

    @njit(nogil=True, fastmath=True)
    def _vupdate_var_f32(var_offsets, Y_N_buf, c2v, v2c, post, write_v2c):
        n_var = var_offsets.shape[0] - 1
        acc = vreg_f32()
        tmp = vreg_f32()
        out = vreg_f32()
        for v in range(n_var):
            e_start, e_stop = var_offsets[v], var_offsets[v + 1]
            vload_f32(acc, Y_N_buf, v)
            for e in range(e_start, e_stop):
                vload_f32(tmp, c2v, e)
                vadd_f32(acc, acc, tmp)
            if write_v2c:
                for e in range(e_start, e_stop):
                    vload_f32(tmp, c2v, e)
                    vsub_f32(out, acc, tmp)
                    vstore_f32(v2c, e, out)
            else:
                vstore_f32(post, v, acc)

    @njit(nogil=True, fastmath=True)
    def _vstep_flooding_spa_f32(chk_offsets, transpose, v2c, c2v, vals):
        n_chk = chk_offsets.shape[0] - 1
        sng = vreg_i32()
        prd = vreg_f32()
        nz = vreg_i32()
        vv = vreg_f32()
        res = vreg_f32()
        val_v = vreg_f32()
        nmsg = vreg_f32()
        for c in range(n_chk):
            k_start, k_stop = chk_offsets[c], chk_offsets[c + 1]
            deg = k_stop - k_start
            vset_i32(sng, np.int32(0))
            vset_f32(prd, np.float32(1.0))
            vset_i32(nz, np.int32(0))
            for idx in range(deg):
                vload_f32(vv, v2c, transpose[k_start + idx])
                vspa_in_f32(res, vv, sng, prd, nz)
                vstore_f32(vals, idx, res)
            for idx in range(deg):
                tk = transpose[k_start + idx]
                vload_f32(vv, v2c, tk)
                vload_f32(val_v, vals, idx)
                vspa_out_f32(nmsg, vv, val_v, sng, prd, nz)
                vstore_f32(c2v, tk, nmsg)

    @njit(nogil=True, fastmath=True)
    def _vstep_flooding_oms_f32(chk_offsets, transpose, offset, v2c, c2v):
        n_chk = chk_offsets.shape[0] - 1
        off_v = vreg_f32()
        sng = vreg_i32()
        m1 = vreg_f32()
        m2 = vreg_f32()
        c1 = vreg_f32()
        c2 = vreg_f32()
        vv = vreg_f32()
        nmsg = vreg_f32()
        vset_f32(off_v, offset)
        for c in range(n_chk):
            k_start, k_stop = chk_offsets[c], chk_offsets[c + 1]
            vset_i32(sng, np.int32(0))
            vset_f32(m1, np.float32(3.4028235e38))
            vset_f32(m2, np.float32(3.4028235e38))
            for k in range(k_start, k_stop):
                vload_f32(vv, v2c, transpose[k])
                voms_in_f32(vv, sng, m1, m2)
            voms_cst_f32(c1, c2, m1, m2, off_v)
            for k in range(k_start, k_stop):
                tk = transpose[k]
                vload_f32(vv, v2c, tk)
                voms_out_f32(nmsg, vv, sng, m1, c1, c2)
                vstore_f32(c2v, tk, nmsg)

    @njit(nogil=True, fastmath=True)
    def _vstep_layered_spa_f32(chk_offsets, chk_to_var, post, c2v, vals, contrib):
        n_chk = chk_offsets.shape[0] - 1
        sng = vreg_i32()
        prd = vreg_f32()
        nz = vreg_i32()
        pu = vreg_f32()
        ck = vreg_f32()
        vv = vreg_f32()
        res = vreg_f32()
        val_v = vreg_f32()
        nmsg = vreg_f32()
        for c in range(n_chk):
            k_start, k_stop = chk_offsets[c], chk_offsets[c + 1]
            deg = k_stop - k_start
            vset_i32(sng, np.int32(0))
            vset_f32(prd, np.float32(1.0))
            vset_i32(nz, np.int32(0))
            for idx in range(deg):
                k = k_start + idx
                u = chk_to_var[k]
                vload_f32(pu, post, u)
                vload_f32(ck, c2v, k)
                vsub_f32(vv, pu, ck)
                vstore_f32(contrib, idx, vv)
                vspa_in_f32(res, vv, sng, prd, nz)
                vstore_f32(vals, idx, res)
            for idx in range(deg):
                k = k_start + idx
                u = chk_to_var[k]
                vload_f32(vv, contrib, idx)
                vload_f32(val_v, vals, idx)
                vspa_out_f32(nmsg, vv, val_v, sng, prd, nz)
                vstore_f32(c2v, k, nmsg)
                vadd_f32(pu, vv, nmsg)
                vstore_f32(post, u, pu)

    @njit(nogil=True, fastmath=True)
    def _vstep_layered_oms_f32(chk_offsets, chk_to_var, offset, post, c2v, contrib):
        n_chk = chk_offsets.shape[0] - 1
        off_v = vreg_f32()
        sng = vreg_i32()
        m1 = vreg_f32()
        m2 = vreg_f32()
        c1 = vreg_f32()
        c2 = vreg_f32()
        pu = vreg_f32()
        ck = vreg_f32()
        vv = vreg_f32()
        nmsg = vreg_f32()
        vset_f32(off_v, offset)
        for c in range(n_chk):
            k_start, k_stop = chk_offsets[c], chk_offsets[c + 1]
            deg = k_stop - k_start
            vset_i32(sng, np.int32(0))
            vset_f32(m1, np.float32(3.4028235e38))
            vset_f32(m2, np.float32(3.4028235e38))
            for idx in range(deg):
                k = k_start + idx
                u = chk_to_var[k]
                vload_f32(pu, post, u)
                vload_f32(ck, c2v, k)
                vsub_f32(vv, pu, ck)
                vstore_f32(contrib, idx, vv)
                voms_in_f32(vv, sng, m1, m2)
            voms_cst_f32(c1, c2, m1, m2, off_v)
            for idx in range(deg):
                k = k_start + idx
                u = chk_to_var[k]
                vload_f32(vv, contrib, idx)
                voms_out_f32(nmsg, vv, sng, m1, c1, c2)
                vstore_f32(c2v, k, nmsg)
                vadd_f32(pu, vv, nmsg)
                vstore_f32(post, u, pu)

    @njit(nogil=True, fastmath=True)
    def _vcheck_syndrome_f32(chk_offsets, chk_to_var, post) -> bool:
        n_chk = chk_offsets.shape[0] - 1
        s = vreg_i32()
        pv = vreg_f32()
        for c in range(n_chk):
            vset_i32(s, np.int32(0))
            for k in range(chk_offsets[c], chk_offsets[c + 1]):
                vload_f32(pv, post, chk_to_var[k])
                vsynd_xor_f32(s, pv)
            if vsynd_any_neg_i32(s):
                return False
        return True

    @njit(nogil=True, fastmath=True)
    def kernel_f32(
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
        Y_N_wave: np.ndarray,
        V_K_wave: np.ndarray,
    ) -> int:
        K = info_bits_pos.shape[0]
        n_var = var_offsets.shape[0] - 1
        target_in = post if is_layered else Y_N_buf
        for w in range(W):
            for v in range(n_var):
                target_in[v, w] = Y_N_wave[w, v]

        z_f = vreg_f32()
        vset_f32(z_f, np.float32(0.0))
        for e in range(c2v.shape[0]):
            vstore_f32(c2v, e, z_f)

        cur_depth = 0
        valid = True
        ite_done = 0
        for ite in range(1, n_ite + 1):
            if not is_layered:
                _vupdate_var_f32(var_offsets, Y_N_buf, c2v, v2c, post, True)
                if not is_oms:
                    _vstep_flooding_spa_f32(chk_offsets, transpose, v2c, c2v, vals)
                else:
                    _vstep_flooding_oms_f32(chk_offsets, transpose, offset, v2c, c2v)
            else:
                if not is_oms:
                    _vstep_layered_spa_f32(chk_offsets, chk_to_var, post, c2v, vals, contrib)
                else:
                    _vstep_layered_oms_f32(chk_offsets, chk_to_var, offset, post, c2v, contrib)

            if enable_syndrome and (is_layered or ite != n_ite):
                if not is_layered:
                    _vupdate_var_f32(var_offsets, Y_N_buf, c2v, v2c, post, False)
                synd_ok = _vcheck_syndrome_f32(chk_offsets, chk_to_var, post)
                cur_depth = ((cur_depth + 1) % syndrome_depth) if synd_ok else 0
                valid = synd_ok and (cur_depth == 0)
                if valid:
                    ite_done = ite
                    break
            if ite == n_ite:
                ite_done = n_ite

        if not is_layered and ite_done == n_ite:
            _vupdate_var_f32(var_offsets, Y_N_buf, c2v, v2c, post, False)

        for w in range(W):
            for i in range(K):
                V_K_wave[w, i] = np.int32(1) if post[info_bits_pos[i], w] < np.float32(0.0) else np.int32(0)
        return 0 if valid else 1

    return kernel_f32


def _make_i16_simd_kernel(W: int):
    vi16_ty = ir.VectorType(ir.IntType(16), W)
    pi16_ty = ir.PointerType(vi16_ty)

    def _vconst_i16(val):
        c = ir.IntType(16)(int(val))
        return ir.Constant(vi16_ty, [c] * W)

    def _bcast_scalar_i16(builder, val):
        i32 = ir.IntType(32)
        u = ir.Constant(vi16_ty, ir.Undefined)
        v0 = builder.insert_element(u, val, i32(0))
        return builder.shuffle_vector(v0, u, ir.Constant(ir.VectorType(i32, W), [i32(0)] * W))

    @intrinsic
    def vreg_i16(typingctx):
        def codegen(context, builder, sig, args):
            with builder.goto_entry_block():
                ptr = builder.alloca(vi16_ty)
            return builder.ptrtoint(ptr, ir.IntType(64))
        return types.int64(), codegen

    @intrinsic
    def vload_i16(typingctx, dst_ty, arr_ty, row_ty):
        def codegen(context, builder, sig, args):
            dst = builder.inttoptr(args[0], pi16_ty)
            data_ptr = builder.extract_value(args[1], 4)
            idx = builder.mul(args[2], ir.Constant(args[2].type, W))
            vptr = builder.bitcast(builder.gep(data_ptr, [idx]), pi16_ty)
            builder.store(builder.load(vptr, align=16), dst)
            return context.get_dummy_value()
        return types.none(types.int64, arr_ty, row_ty), codegen

    @intrinsic
    def vstore_i16(typingctx, arr_ty, row_ty, src_ty):
        def codegen(context, builder, sig, args):
            data_ptr = builder.extract_value(args[0], 4)
            idx = builder.mul(args[1], ir.Constant(args[1].type, W))
            vptr = builder.bitcast(builder.gep(data_ptr, [idx]), pi16_ty)
            src = builder.inttoptr(args[2], pi16_ty)
            builder.store(builder.load(src), vptr, align=16)
            return context.get_dummy_value()
        return types.none(arr_ty, row_ty, types.int64), codegen

    @intrinsic
    def vset_i16(typingctx, dst_ty, val_ty):
        def codegen(context, builder, sig, args):
            dst = builder.inttoptr(args[0], pi16_ty)
            builder.store(_bcast_scalar_i16(builder, args[1]), dst)
            return context.get_dummy_value()
        return types.none(types.int64, types.int16), codegen

    @intrinsic
    def vsadd_i16(typingctx, dst_ty, a_ty, b_ty):
        def codegen(context, builder, sig, args):
            dst = builder.inttoptr(args[0], pi16_ty)
            va = builder.load(builder.inttoptr(args[1], pi16_ty))
            vb = builder.load(builder.inttoptr(args[2], pi16_ty))
            fn = _decl_intrinsic(builder, f"llvm.sadd.sat.v{W}i16", vi16_ty, [vi16_ty, vi16_ty])
            builder.store(builder.call(fn, [va, vb]), dst)
            return context.get_dummy_value()
        return types.none(types.int64, types.int64, types.int64), codegen

    @intrinsic
    def vssub_i16(typingctx, dst_ty, a_ty, b_ty):
        def codegen(context, builder, sig, args):
            dst = builder.inttoptr(args[0], pi16_ty)
            va = builder.load(builder.inttoptr(args[1], pi16_ty))
            vb = builder.load(builder.inttoptr(args[2], pi16_ty))
            fn = _decl_intrinsic(builder, f"llvm.ssub.sat.v{W}i16", vi16_ty, [vi16_ty, vi16_ty])
            builder.store(builder.call(fn, [va, vb]), dst)
            return context.get_dummy_value()
        return types.none(types.int64, types.int64, types.int64), codegen

    @intrinsic
    def vsadd_sat_val_i16(typingctx, dst_ty, a_ty, b_ty, sat_ty, neg_sat_ty):
        def codegen(context, builder, sig, args):
            dst = builder.inttoptr(args[0], pi16_ty)
            va = builder.load(builder.inttoptr(args[1], pi16_ty))
            vb = builder.load(builder.inttoptr(args[2], pi16_ty))
            sat_v = builder.load(builder.inttoptr(args[3], pi16_ty))
            neg_sat_v = builder.load(builder.inttoptr(args[4], pi16_ty))
            f_add = _decl_intrinsic(builder, f"llvm.sadd.sat.v{W}i16", vi16_ty, [vi16_ty, vi16_ty])
            f_min = _decl_intrinsic(builder, f"llvm.smin.v{W}i16", vi16_ty, [vi16_ty, vi16_ty])
            f_max = _decl_intrinsic(builder, f"llvm.smax.v{W}i16", vi16_ty, [vi16_ty, vi16_ty])
            s = builder.call(f_add, [va, vb])
            s = builder.call(f_min, [s, sat_v])
            s = builder.call(f_max, [s, neg_sat_v])
            builder.store(s, dst)
            return context.get_dummy_value()
        return types.none(types.int64, types.int64, types.int64, types.int64, types.int64), codegen

    @intrinsic
    def voms_in_i16(typingctx, v_ty, sng_ty, m1_ty, m2_ty):
        def codegen(context, builder, sig, args):
            v = builder.load(builder.inttoptr(args[0], pi16_ty))
            p_sng = builder.inttoptr(args[1], pi16_ty)
            p_m1 = builder.inttoptr(args[2], pi16_ty)
            p_m2 = builder.inttoptr(args[3], pi16_ty)
            sng = builder.load(p_sng)
            m1 = builder.load(p_m1)
            m2 = builder.load(p_m2)
            sng_out = builder.xor(sng, v)
            f_abs = _decl_intrinsic(builder, f"llvm.abs.v{W}i16", vi16_ty, [vi16_ty, ir.IntType(1)])
            f_min = _decl_intrinsic(builder, f"llvm.smin.v{W}i16", vi16_ty, [vi16_ty, vi16_ty])
            f_max = _decl_intrinsic(builder, f"llvm.smax.v{W}i16", vi16_ty, [vi16_ty, vi16_ty])
            va = builder.call(f_abs, [v, ir.IntType(1)(0)])
            mx = builder.call(f_max, [va, m1])
            m2_out = builder.call(f_min, [m2, mx])
            m1_out = builder.call(f_min, [m1, va])
            builder.store(sng_out, p_sng)
            builder.store(m1_out, p_m1)
            builder.store(m2_out, p_m2)
            return context.get_dummy_value()
        return types.none(types.int64, types.int64, types.int64, types.int64), codegen

    @intrinsic
    def voms_cst_i16(typingctx, c1_ty, c2_ty, m1_ty, m2_ty, off_ty):
        def codegen(context, builder, sig, args):
            p_c1 = builder.inttoptr(args[0], pi16_ty)
            p_c2 = builder.inttoptr(args[1], pi16_ty)
            m1 = builder.load(builder.inttoptr(args[2], pi16_ty))
            m2 = builder.load(builder.inttoptr(args[3], pi16_ty))
            off = builder.load(builder.inttoptr(args[4], pi16_ty))
            f_sub = _decl_intrinsic(builder, f"llvm.ssub.sat.v{W}i16", vi16_ty, [vi16_ty, vi16_ty])
            f_max = _decl_intrinsic(builder, f"llvm.smax.v{W}i16", vi16_ty, [vi16_ty, vi16_ty])
            z = _vconst_i16(0)
            builder.store(builder.call(f_max, [z, builder.call(f_sub, [m2, off])]), p_c1)
            builder.store(builder.call(f_max, [z, builder.call(f_sub, [m1, off])]), p_c2)
            return context.get_dummy_value()
        return types.none(types.int64, types.int64, types.int64, types.int64, types.int64), codegen

    @intrinsic
    def voms_out_i16(typingctx, dst_ty, v_ty, sng_ty, m1_ty, c1_ty, c2_ty, sat_ty, neg_sat_ty):
        def codegen(context, builder, sig, args):
            dst = builder.inttoptr(args[0], pi16_ty)
            v = builder.load(builder.inttoptr(args[1], pi16_ty))
            sng = builder.load(builder.inttoptr(args[2], pi16_ty))
            m1 = builder.load(builder.inttoptr(args[3], pi16_ty))
            c1 = builder.load(builder.inttoptr(args[4], pi16_ty))
            c2 = builder.load(builder.inttoptr(args[5], pi16_ty))
            sat_v = builder.load(builder.inttoptr(args[6], pi16_ty))
            neg_sat_v = builder.load(builder.inttoptr(args[7], pi16_ty))
            f_abs = _decl_intrinsic(builder, f"llvm.abs.v{W}i16", vi16_ty, [vi16_ty, ir.IntType(1)])
            f_min = _decl_intrinsic(builder, f"llvm.smin.v{W}i16", vi16_ty, [vi16_ty, vi16_ty])
            f_max = _decl_intrinsic(builder, f"llvm.smax.v{W}i16", vi16_ty, [vi16_ty, vi16_ty])
            va = builder.call(f_abs, [v, ir.IntType(1)(0)])
            eq = builder.icmp_unsigned("==", va, m1)
            mag = builder.select(eq, c1, c2)
            s_out = builder.xor(sng, v)
            z = _vconst_i16(0)
            is_neg = builder.icmp_signed("<", s_out, z)
            res = builder.select(is_neg, builder.sub(z, mag), mag)
            res = builder.call(f_min, [res, sat_v])
            res = builder.call(f_max, [res, neg_sat_v])
            builder.store(res, dst)
            return context.get_dummy_value()
        return types.none(types.int64, types.int64, types.int64, types.int64, types.int64, types.int64, types.int64, types.int64), codegen

    @intrinsic
    def vsynd_xor_i16(typingctx, synd_ty, pv_ty):
        def codegen(context, builder, sig, args):
            p_s = builder.inttoptr(args[0], pi16_ty)
            s = builder.load(p_s)
            pv = builder.load(builder.inttoptr(args[1], pi16_ty))
            builder.store(builder.xor(s, pv), p_s)
            return context.get_dummy_value()
        return types.none(types.int64, types.int64), codegen

    @intrinsic
    def vsynd_any_neg_i16(typingctx, synd_ty):
        def codegen(context, builder, sig, args):
            s = builder.load(builder.inttoptr(args[0], pi16_ty))
            s_neg = builder.and_(s, _vconst_i16(-32768))
            i16 = ir.IntType(16)
            f_red = _decl_intrinsic(builder, f"llvm.vector.reduce.or.v{W}i16", i16, [vi16_ty])
            r = builder.call(f_red, [s_neg])
            return builder.icmp_unsigned("!=", r, i16(0))
        return types.boolean(types.int64), codegen

    @njit(nogil=True, fastmath=True)
    def _vupdate_var_i16(var_offsets, Y_N_buf, c2v, v2c, post, write_v2c):
        n_var = var_offsets.shape[0] - 1
        acc = vreg_i16()
        tmp = vreg_i16()
        out = vreg_i16()
        for v in range(n_var):
            e_start, e_stop = var_offsets[v], var_offsets[v + 1]
            vload_i16(acc, Y_N_buf, v)
            for e in range(e_start, e_stop):
                vload_i16(tmp, c2v, e)
                vsadd_i16(acc, acc, tmp)
            if write_v2c:
                for e in range(e_start, e_stop):
                    vload_i16(tmp, c2v, e)
                    vssub_i16(out, acc, tmp)
                    vstore_i16(v2c, e, out)
            else:
                vstore_i16(post, v, acc)

    @njit(nogil=True, fastmath=True)
    def _vstep_flooding_oms_i16(chk_offsets, transpose, offset, sat_val, v2c, c2v):
        n_chk = chk_offsets.shape[0] - 1
        off_v = vreg_i16()
        sat_v = vreg_i16()
        neg_sat_v = vreg_i16()
        sng = vreg_i16()
        m1 = vreg_i16()
        m2 = vreg_i16()
        c1 = vreg_i16()
        c2 = vreg_i16()
        vv = vreg_i16()
        nmsg = vreg_i16()
        vset_i16(off_v, offset)
        vset_i16(sat_v, sat_val)
        vset_i16(neg_sat_v, np.int16(-sat_val))
        for c in range(n_chk):
            k_start, k_stop = chk_offsets[c], chk_offsets[c + 1]
            vset_i16(sng, np.int16(0))
            vset_i16(m1, np.int16(32767))
            vset_i16(m2, np.int16(32767))
            for k in range(k_start, k_stop):
                vload_i16(vv, v2c, transpose[k])
                voms_in_i16(vv, sng, m1, m2)
            voms_cst_i16(c1, c2, m1, m2, off_v)
            for k in range(k_start, k_stop):
                tk = transpose[k]
                vload_i16(vv, v2c, tk)
                voms_out_i16(nmsg, vv, sng, m1, c1, c2, sat_v, neg_sat_v)
                vstore_i16(c2v, tk, nmsg)

    @njit(nogil=True, fastmath=True)
    def _vstep_layered_oms_i16(chk_offsets, chk_to_var, offset, sat_val, post, c2v, contrib):
        n_chk = chk_offsets.shape[0] - 1
        off_v = vreg_i16()
        sat_v = vreg_i16()
        neg_sat_v = vreg_i16()
        sng = vreg_i16()
        m1 = vreg_i16()
        m2 = vreg_i16()
        c1 = vreg_i16()
        c2 = vreg_i16()
        pu = vreg_i16()
        ck = vreg_i16()
        vv = vreg_i16()
        nmsg = vreg_i16()
        vset_i16(off_v, offset)
        vset_i16(sat_v, sat_val)
        vset_i16(neg_sat_v, np.int16(-sat_val))
        for c in range(n_chk):
            k_start, k_stop = chk_offsets[c], chk_offsets[c + 1]
            deg = k_stop - k_start
            vset_i16(sng, np.int16(0))
            vset_i16(m1, np.int16(32767))
            vset_i16(m2, np.int16(32767))
            for idx in range(deg):
                k = k_start + idx
                u = chk_to_var[k]
                vload_i16(pu, post, u)
                vload_i16(ck, c2v, k)
                vssub_i16(vv, pu, ck)
                vstore_i16(contrib, idx, vv)
                voms_in_i16(vv, sng, m1, m2)
            voms_cst_i16(c1, c2, m1, m2, off_v)
            for idx in range(deg):
                k = k_start + idx
                u = chk_to_var[k]
                vload_i16(vv, contrib, idx)
                voms_out_i16(nmsg, vv, sng, m1, c1, c2, sat_v, neg_sat_v)
                vstore_i16(c2v, k, nmsg)
                vsadd_sat_val_i16(pu, vv, nmsg, sat_v, neg_sat_v)
                vstore_i16(post, u, pu)

    @njit(nogil=True, fastmath=True)
    def _vcheck_syndrome_i16(chk_offsets, chk_to_var, post) -> bool:
        n_chk = chk_offsets.shape[0] - 1
        s = vreg_i16()
        pv = vreg_i16()
        for c in range(n_chk):
            vset_i16(s, np.int16(0))
            for k in range(chk_offsets[c], chk_offsets[c + 1]):
                vload_i16(pv, post, chk_to_var[k])
                vsynd_xor_i16(s, pv)
            if vsynd_any_neg_i16(s):
                return False
        return True

    @njit(nogil=True, fastmath=True)
    def kernel_i16(
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
        Y_N_wave: np.ndarray,
        V_K_wave: np.ndarray,
    ) -> int:
        K = info_bits_pos.shape[0]
        n_var = var_offsets.shape[0] - 1
        target_in = post if is_layered else Y_N_buf
        for w in range(W):
            for v in range(n_var):
                target_in[v, w] = Y_N_wave[w, v]

        z_v = vreg_i16()
        vset_i16(z_v, np.int16(0))
        for e in range(c2v.shape[0]):
            vstore_i16(c2v, e, z_v)

        cur_depth = 0
        valid = True
        ite_done = 0
        for ite in range(1, n_ite + 1):
            if not is_layered:
                _vupdate_var_i16(var_offsets, Y_N_buf, c2v, v2c, post, True)
                _vstep_flooding_oms_i16(chk_offsets, transpose, offset, sat_val, v2c, c2v)
            else:
                _vstep_layered_oms_i16(chk_offsets, chk_to_var, offset, sat_val, post, c2v, contrib)

            if enable_syndrome and (is_layered or ite != n_ite):
                if not is_layered:
                    _vupdate_var_i16(var_offsets, Y_N_buf, c2v, v2c, post, False)
                synd_ok = _vcheck_syndrome_i16(chk_offsets, chk_to_var, post)
                cur_depth = ((cur_depth + 1) % syndrome_depth) if synd_ok else 0
                valid = synd_ok and (cur_depth == 0)
                if valid:
                    ite_done = ite
                    break
            if ite == n_ite:
                ite_done = n_ite

        if not is_layered and ite_done == n_ite:
            _vupdate_var_i16(var_offsets, Y_N_buf, c2v, v2c, post, False)

        for w in range(W):
            for i in range(K):
                V_K_wave[w, i] = np.int16(1) if post[info_bits_pos[i], w] < np.int16(0) else np.int16(0)
        return 0 if valid else 1

    return kernel_i16


decode_siho_opt_inter_f32_w4 = _make_f32_simd_kernel(4)
decode_siho_opt_inter_f32_w8 = _make_f32_simd_kernel(8)
decode_siho_opt_inter_f32_w16 = _make_f32_simd_kernel(16)

decode_siho_opt_inter_i16_w8 = _make_i16_simd_kernel(8)
decode_siho_opt_inter_i16_w16 = _make_i16_simd_kernel(16)
decode_siho_opt_inter_i16_w32 = _make_i16_simd_kernel(32)


def decode_siho_opt_inter_f32(
    var_offsets, chk_offsets, transpose, chk_to_var, info_bits_pos,
    is_layered, is_oms, offset, n_ite, enable_syndrome, syndrome_depth,
    Y_N_buf, post, c2v, v2c, vals, contrib,
    sng_w, min1_w, min2_w, cst1_w, cst2_w, prd_w, nz_w, synd_w,
    Y_N_wave, V_K_wave,
) -> int:
    W = Y_N_wave.shape[0]
    if W == 8:
        return int(decode_siho_opt_inter_f32_w8(
            var_offsets, chk_offsets, transpose, chk_to_var, info_bits_pos,
            is_layered, is_oms, offset, n_ite, enable_syndrome, syndrome_depth,
            Y_N_buf, post, c2v, v2c, vals, contrib, Y_N_wave, V_K_wave,
        ))
    if W == 4:
        return int(decode_siho_opt_inter_f32_w4(
            var_offsets, chk_offsets, transpose, chk_to_var, info_bits_pos,
            is_layered, is_oms, offset, n_ite, enable_syndrome, syndrome_depth,
            Y_N_buf, post, c2v, v2c, vals, contrib, Y_N_wave, V_K_wave,
        ))
    if W == 16:
        return int(decode_siho_opt_inter_f32_w16(
            var_offsets, chk_offsets, transpose, chk_to_var, info_bits_pos,
            is_layered, is_oms, offset, n_ite, enable_syndrome, syndrome_depth,
            Y_N_buf, post, c2v, v2c, vals, contrib, Y_N_wave, V_K_wave,
        ))
    import ldpc_bp_std
    return int(ldpc_bp_std.decode_siho_std_f32(
        var_offsets, chk_offsets, transpose, chk_to_var, info_bits_pos,
        is_layered, is_oms, offset, n_ite, enable_syndrome, syndrome_depth,
        Y_N_buf, post, c2v, v2c, vals, contrib,
        sng_w, min1_w, min2_w, cst1_w, cst2_w, prd_w, nz_w, synd_w,
        Y_N_wave, V_K_wave,
    ))


def decode_siho_opt_inter_i16(
    var_offsets, chk_offsets, transpose, chk_to_var, info_bits_pos,
    is_layered, offset, sat_val, n_ite, enable_syndrome, syndrome_depth,
    Y_N_buf, post, c2v, v2c, contrib,
    sng_w, min1_w, min2_w, cst1_w, cst2_w, synd_w,
    Y_N_wave, V_K_wave,
) -> int:
    W = Y_N_wave.shape[0]
    if W == 16:
        return int(decode_siho_opt_inter_i16_w16(
            var_offsets, chk_offsets, transpose, chk_to_var, info_bits_pos,
            is_layered, offset, sat_val, n_ite, enable_syndrome, syndrome_depth,
            Y_N_buf, post, c2v, v2c, contrib, Y_N_wave, V_K_wave,
        ))
    if W == 8:
        return int(decode_siho_opt_inter_i16_w8(
            var_offsets, chk_offsets, transpose, chk_to_var, info_bits_pos,
            is_layered, offset, sat_val, n_ite, enable_syndrome, syndrome_depth,
            Y_N_buf, post, c2v, v2c, contrib, Y_N_wave, V_K_wave,
        ))
    if W == 32:
        return int(decode_siho_opt_inter_i16_w32(
            var_offsets, chk_offsets, transpose, chk_to_var, info_bits_pos,
            is_layered, offset, sat_val, n_ite, enable_syndrome, syndrome_depth,
            Y_N_buf, post, c2v, v2c, contrib, Y_N_wave, V_K_wave,
        ))
    import ldpc_bp_std
    return int(ldpc_bp_std.decode_siho_std_i16(
        var_offsets, chk_offsets, transpose, chk_to_var, info_bits_pos,
        is_layered, offset, sat_val, n_ite, enable_syndrome, syndrome_depth,
        Y_N_buf, post, c2v, v2c, contrib,
        sng_w, min1_w, min2_w, cst1_w, cst2_w, synd_w,
        Y_N_wave, V_K_wave,
    ))


# Ahead-of-execution compilation for Scalar and Inter-frame SIMD kernels (mirroring Julia `precompile(...)`)
_I32_1D = types.int32[::1]
_F32_1D = types.float32[::1]
_F32_2D = types.float32[:, ::1]
_I16_1D = types.int16[::1]
_I16_2D = types.int16[:, ::1]

decode_siho_opt_scalar_f32.compile((
    _I32_1D, _I32_1D, _I32_1D, _I32_1D, _I32_1D,
    types.boolean, types.boolean, types.float32,
    types.int64, types.boolean, types.int64,
    _F32_1D, _F32_1D, _F32_1D, _F32_1D, _F32_1D,
    _F32_2D, types.int32[:, ::1],
))

decode_siho_opt_scalar_i16.compile((
    _I32_1D, _I32_1D, _I32_1D, _I32_1D, _I32_1D,
    types.boolean, types.int16, types.int16,
    types.int64, types.boolean, types.int64,
    _I16_1D, _I16_1D, _I16_1D, _I16_1D,
    _I16_2D, _I16_2D,
))

_SIG_INTER_F32 = (
    _I32_1D, _I32_1D, _I32_1D, _I32_1D, _I32_1D,
    types.boolean, types.boolean, types.float32,
    types.int64, types.boolean, types.int64,
    _F32_2D, _F32_2D, _F32_2D, _F32_2D, _F32_2D, _F32_2D,
    _F32_2D, types.int32[:, ::1],
)
for _fn_f32 in (decode_siho_opt_inter_f32_w4, decode_siho_opt_inter_f32_w8):
    _fn_f32.compile(_SIG_INTER_F32)

_SIG_INTER_I16 = (
    _I32_1D, _I32_1D, _I32_1D, _I32_1D, _I32_1D,
    types.boolean, types.int16, types.int16,
    types.int64, types.boolean, types.int64,
    _I16_2D, _I16_2D, _I16_2D, _I16_2D, _I16_2D,
    _I16_2D, _I16_2D,
)
for _fn_i16 in (decode_siho_opt_inter_i16_w8, decode_siho_opt_inter_i16_w16):
    _fn_i16.compile(_SIG_INTER_I16)
