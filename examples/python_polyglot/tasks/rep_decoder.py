"""Pedagogical Soft-Input Hard-Output (SIHO) Repetition Decoder in Python/Numba (`Stateful_core`).

Mirrors `examples/pipeline_julia/src/rep_decoder.jl` with a progressive optimization ladder:
  - v0 (`PY_V0`): Naive idiomatic Numba (`@njit(nogil=True)`) — allocates `ext = np.zeros(K, dtype=np.float32)`
                  and slices `Y_N` on the heap every frame (1 frame/call, `single_wave = False`).
  - v1 (`PY_V1`): Zero-allocation scalar loops with concrete types (`float32`/`int32`, 1 frame/call,
                  `single_wave = False`, no `fastmath` / SIMD tiling).
  - v2 (`PY_V2`): Register-tiled SIMD (`@njit(nogil=True, fastmath=True)` + `llvmlite` 16-lane `<16 x float>`
                  SIMD vector load/add/compare-store intrinsics) + explicit ahead-of-execution compilation
                  (1 frame/call, `single_wave = False`).
  - v3 (`PY_V3`, alias `PY_V4` / `v4`): Same register-tiled SIMD kernel + `set_single_wave(True)` processing
                  all `n_frames_per_wave` (`F`) frames in a single call.

Both AFF3CT bit layouts are supported:
  - `*_buf`   : default buffered encoding (`[U_K, U_K, ..., U_K]`, contiguous blocks of `K`)
  - `*_nobuf` : unbuffered encoding (`--enc-no-buff`, `[b_1, ..., b_1, b_2, ..., b_2, ...]`)
"""

import numpy as np
from numba import njit, types
from numba.extending import intrinsic
from llvmlite import ir


# ==================================================================================================
# 0. LLVM SIMD 16-lane (`<16 x float>`) intrinsics for register-tiled repetition decoding (v2 & v3/v4)
# ==================================================================================================

def _tuple_to_vec16(builder, tup_val):
    vty = ir.VectorType(ir.FloatType(), 16)
    vec = ir.Constant(vty, ir.Undefined)
    i32 = ir.IntType(32)
    for i in range(16):
        el = builder.extract_value(tup_val, i)
        vec = builder.insert_element(vec, el, ir.Constant(i32, i))
    return vec


def _vec16_to_tuple(builder, vec_val):
    aty = ir.ArrayType(ir.FloatType(), 16)
    res = ir.Constant(aty, ir.Undefined)
    i32 = ir.IntType(32)
    for i in range(16):
        el = builder.extract_element(vec_val, ir.Constant(i32, i))
        res = builder.insert_value(res, el, i)
    return res


@intrinsic
def _vload16_f32(typingctx, arr_2d, row, col):
    sig = types.UniTuple(types.float32, 16)(arr_2d, types.intp, types.intp)

    def codegen(context, builder, signature, args):
        data_ptr = builder.extract_value(args[0], 4)
        ncols = builder.extract_value(args[0], [5, 1])
        idx = builder.add(builder.mul(args[1], ncols), args[2])
        ptr = builder.gep(data_ptr, [idx])
        vty = ir.VectorType(ir.FloatType(), 16)
        vec = builder.load(builder.bitcast(ptr, vty.as_pointer()), align=4)
        return _vec16_to_tuple(builder, vec)

    return sig, codegen


@intrinsic
def _vfadd16(typingctx, a, b):
    sig = a(a, b)

    def codegen(context, builder, signature, args):
        va = _tuple_to_vec16(builder, args[0])
        vb = _tuple_to_vec16(builder, args[1])
        res = builder.fadd(va, vb, flags=["fast"])
        return _vec16_to_tuple(builder, res)

    return sig, codegen


@intrinsic
def _vstore16_hard_decision(typingctx, v_k_2d, row, col, s_vec):
    sig = types.none(v_k_2d, types.intp, types.intp, s_vec)

    def codegen(context, builder, signature, args):
        data_ptr = builder.extract_value(args[0], 4)
        ncols = builder.extract_value(args[0], [5, 1])
        idx = builder.add(builder.mul(args[1], ncols), args[2])
        ptr = builder.gep(data_ptr, [idx])
        vs = _tuple_to_vec16(builder, args[3])
        vzero = ir.Constant(ir.VectorType(ir.FloatType(), 16), [ir.Constant(ir.FloatType(), 0.0)] * 16)
        cmp = builder.fcmp_ordered("<", vs, vzero, flags=["fast"])
        vi32 = builder.zext(cmp, ir.VectorType(ir.IntType(32), 16))
        builder.store(vi32, builder.bitcast(ptr, ir.VectorType(ir.IntType(32), 16).as_pointer()), align=4)
        return context.get_dummy_value()

    return sig, codegen


# ==================================================================================================
# v0: Naive idiomatic implementation (allocates `ext` and slices on the heap every frame)
# ==================================================================================================

@njit(nogil=True)
def decode_rep_v0_buf(sck_Y_N, sck_V_K):
    K = sck_V_K.shape[1]
    n_rep = sck_Y_N.shape[1] // K
    ext = np.zeros(K, dtype=np.float32)
    for r in range(n_rep):
        ext += sck_Y_N[0, (r * K):((r + 1) * K)].copy()
    for i in range(K):
        sck_V_K[0, i] = np.int32(1) if ext[i] < np.float32(0.0) else np.int32(0)
    return np.int32(0)


@njit(nogil=True)
def decode_rep_v0_nobuf(sck_Y_N, sck_V_K):
    K = sck_V_K.shape[1]
    n_rep = sck_Y_N.shape[1] // K
    ext = np.zeros(K, dtype=np.float32)
    for i in range(K):
        off = i * n_rep
        ext[i] = np.sum(sck_Y_N[0, off:(off + n_rep)].copy())
    for i in range(K):
        sck_V_K[0, i] = np.int32(1) if ext[i] < np.float32(0.0) else np.int32(0)
    return np.int32(0)


# ==================================================================================================
# v1: Zero-allocation scalar implementation with concrete types (no `fastmath`, no SIMD tiling)
# ==================================================================================================

@njit(nogil=True)
def decode_rep_v1_buf(sck_Y_N, sck_V_K):
    K = sck_V_K.shape[1]
    n_rep = sck_Y_N.shape[1] // K
    for i in range(K):
        s = np.float32(0.0)
        for r in range(n_rep):
            s += sck_Y_N[0, r * K + i]
        sck_V_K[0, i] = np.int32(1) if s < np.float32(0.0) else np.int32(0)
    return np.int32(0)


@njit(nogil=True)
def decode_rep_v1_nobuf(sck_Y_N, sck_V_K):
    K = sck_V_K.shape[1]
    n_rep = sck_Y_N.shape[1] // K
    for i in range(K):
        off = i * n_rep
        s = np.float32(0.0)
        for r in range(n_rep):
            s += sck_Y_N[0, off + r]
        sck_V_K[0, i] = np.int32(1) if s < np.float32(0.0) else np.int32(0)
    return np.int32(0)


# ==================================================================================================
# Shared register-tiled SIMD helper for v2 and v3/v4
# Tiles `i` by 16 contiguous `Float32` lanes (512 bits: 4 NEON / 2 AVX2 / 1 AVX-512 registers),
# mirroring `rep_decoder.jl` (`@nexprs 16`) and AFF3CT's `Decoder_repetition_fast`.
# ==================================================================================================

@njit(nogil=True, fastmath=True, inline="always")
def _decode_rep_frame_buf_simd(Y_N, V_K, f, K, n_rep):
    i = 0
    while i + 16 <= K:
        s = _vload16_f32(Y_N, f, i)
        for r in range(1, n_rep):
            s = _vfadd16(s, _vload16_f32(Y_N, f, r * K + i))
        _vstore16_hard_decision(V_K, f, i, s)
        i += 16
    while i < K:
        acc = Y_N[f, i]
        for r in range(1, n_rep):
            acc += Y_N[f, r * K + i]
        V_K[f, i] = np.int32(1) if acc < np.float32(0.0) else np.int32(0)
        i += 1


@njit(nogil=True, fastmath=True, inline="always")
def _decode_rep_frame_nobuf_simd(Y_N, V_K, f, K, n_rep):
    for i in range(K):
        off = i * n_rep
        s = np.float32(0.0)
        for r in range(n_rep):
            s += Y_N[f, off + r]
        V_K[f, i] = np.int32(1) if s < np.float32(0.0) else np.int32(0)


# ==================================================================================================
# v2: Register-tiled SIMD kernel + branchless comparison + ahead-of-execution compile (1 frame/call)
# ==================================================================================================

@njit(nogil=True, fastmath=True)
def decode_rep_v2_buf(sck_Y_N, sck_V_K):
    K = sck_V_K.shape[1]
    n_rep = sck_Y_N.shape[1] // K
    _decode_rep_frame_buf_simd(sck_Y_N, sck_V_K, 0, K, n_rep)
    return np.int32(0)


@njit(nogil=True, fastmath=True)
def decode_rep_v2_nobuf(sck_Y_N, sck_V_K):
    K = sck_V_K.shape[1]
    n_rep = sck_Y_N.shape[1] // K
    _decode_rep_frame_nobuf_simd(sck_Y_N, sck_V_K, 0, K, n_rep)
    return np.int32(0)


# ==================================================================================================
# v3 (and v4 alias): Inter-frame wave batching (`set_single_wave(True)`) — processes all `F` frames
# ==================================================================================================

@njit(nogil=True, fastmath=True)
def decode_rep_v3_buf(sck_Y_N, sck_V_K):
    n_fra = sck_Y_N.shape[0]
    K = sck_V_K.shape[1]
    n_rep = sck_Y_N.shape[1] // K
    for f in range(n_fra):
        _decode_rep_frame_buf_simd(sck_Y_N, sck_V_K, f, K, n_rep)
    return np.int32(0)


@njit(nogil=True, fastmath=True)
def decode_rep_v3_nobuf(sck_Y_N, sck_V_K):
    n_fra = sck_Y_N.shape[0]
    K = sck_V_K.shape[1]
    n_rep = sck_Y_N.shape[1] // K
    for f in range(n_fra):
        _decode_rep_frame_nobuf_simd(sck_Y_N, sck_V_K, f, K, n_rep)
    return np.int32(0)


decode_rep_v4_buf = decode_rep_v3_buf
decode_rep_v4_nobuf = decode_rep_v3_nobuf

# Ahead-of-execution compilation for v1, v2, and v3/v4 kernels
_SIG_REP = (types.float32[:, ::1], types.int32[:, ::1])
decode_rep_v1_buf.compile(_SIG_REP)
decode_rep_v1_nobuf.compile(_SIG_REP)
decode_rep_v2_buf.compile(_SIG_REP)
decode_rep_v2_nobuf.compile(_SIG_REP)
decode_rep_v3_buf.compile(_SIG_REP)
decode_rep_v3_nobuf.compile(_SIG_REP)


# ==================================================================================================
# StreamPU `Stateful_core` Module (`DecoderRepetitionPython`)
# ==================================================================================================

import streampu as spu
import py_aff3ct


class DecoderRepetitionPython(spu.Stateful_core):
    """Custom Python repetition decoder implemented as a StreamPU Stateful_core module (NumPy or Numba v0..v3/v4)."""

    def __init__(
        self,
        K: int,
        N: int,
        buffered: bool = True,
        use_numba: bool = False,
        single_wave: bool = False,
        py_ver: str = "v3",
    ):
        super().__init__()
        self.K = K
        self.N = N
        self.rep_count = (N // K) - 1
        self.buffered = buffered
        self.use_numba = use_numba
        self.py_ver = py_ver.lower() if use_numba else "numpy"
        if use_numba:
            if self.py_ver in ("v0", "v1", "v2"):
                single_wave = False
            elif self.py_ver in ("v3", "v4"):
                single_wave = True
        self.single_wave = single_wave
        self.name = f"Decoder_repetition_PY_{self.py_ver}" if use_numba else "Decoder_repetition_PY_NumPy"
        self._kernel = self._resolve_numba_kernel(self.py_ver, buffered) if use_numba else None
        if single_wave:
            py_aff3ct.set_single_wave(self, True)

        task = self.create_task("decode_siho")
        task.replicable = True
        self.create_socket_in(task, "Y_N", N, spu.float32)
        self.create_socket_out(task, "V_K", K, spu.int32)
        if use_numba:
            self.create_codelet(task, self._kernel)
        else:
            self.create_codelet(task, DecoderRepetitionPython._decode_numpy)

    @staticmethod
    def _resolve_numba_kernel(py_ver: str, buffered: bool):
        suffix = "_buf" if buffered else "_nobuf"
        if py_ver == "v0":
            return globals()["decode_rep_v0" + suffix]
        if py_ver == "v1":
            return globals()["decode_rep_v1" + suffix]
        if py_ver == "v2":
            return globals()["decode_rep_v2" + suffix]
        if py_ver in ("v3", "v4"):
            return globals()["decode_rep_v3" + suffix]
        raise ValueError(f"Unknown Python/Numba repetition decoder version: {py_ver}")

    @staticmethod
    def _decode_numpy(mdl, tsk, frame_id):
        Y_N = tsk["Y_N"].wave
        V_K = tsk["V_K"].wave
        K = mdl.K
        n_rep = mdl.rep_count + 1
        f_cnt = Y_N.shape[0]
        if mdl.buffered:
            acc = Y_N.reshape(f_cnt, n_rep, K).sum(axis=1)
        else:
            acc = Y_N.reshape(f_cnt, K, n_rep).sum(axis=2)
        V_K[:] = (acc < 0.0).astype(np.int32)
        return 0

    @staticmethod
    def _decode_numba(mdl, tsk, frame_id):
        return int(mdl._kernel(tsk["Y_N"].wave, tsk["V_K"].wave))

    def clone(self):
        other = type(self).__new__(type(self))
        spu.Stateful_core.__init__(other, self)
        other.K = self.K
        other.N = self.N
        other.rep_count = self.rep_count
        other.buffered = self.buffered
        other.use_numba = self.use_numba
        other.py_ver = self.py_ver
        other.single_wave = self.single_wave
        other._kernel = self._kernel
        return other


