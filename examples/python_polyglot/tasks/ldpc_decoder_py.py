"""Generic Modular LDPC Belief Propagation Decoder in Python (`NumPy` and `Numba`) for `pyStreamPU`.

Implements a `streampu.Stateful_core` module (`DecoderLDPCPython`) supporting the 4-axis
configuration space of the 5G NR / QC-LDPC benchmark:
  - Axis 1 (Schedule)   : `BP_FLOODING` and `BP_HORIZONTAL_LAYERED`
  - Axis 2 (Update Rule): `SPA` (Sum-Product) and `OMS` (Offset Min-Sum)
  - Axis 3 (Precision)  : `f32` (`float32`) and `i16` (`int16` fixed-point saturating arithmetic)
  - Axis 4 (SIMD Mode)  : Scalar (`1` frame per call) and `INTER` (`W = n_frames` frames per wave)

Supports three progressive Numba optimization tiers mirroring Julia (`ldpc_bp_{naive,std,opt}.jl`):
  - `naive` ([`ldpc_bp_naive.py`]): Compact textbook algorithmic specification (~210 lines)
  - `std`   ([`ldpc_bp_std.py`])  : Idiomatic zero-allocation Numba (`@njit(nogil=True, fastmath=True)`, ~350 lines)
  - `opt`   ([`ldpc_bp_opt.py`])  : Hardware-optimized Numba + `llvmlite` SIMD vector intrinsics
plus a pure vectorized `NumPy` reference backend (`use_numba=False`).
"""

import numpy as np
import streampu as spu
import py_aff3ct


# ====================================================================================================
# 1. PURE VECTORIZED NUMPY IMPLEMENTATION (Single-Frame and Inter-Frame `W` via 2D Array Broadcasting)
# ====================================================================================================

def _numpy_decode_layered(mdl, Y_N_wave, V_K_wave):
    """Vectorized NumPy Horizontal-Layered BP decoder (processes all frames in `Y_N_wave` simultaneously)."""
    W = Y_N_wave.shape[0]
    is_i16 = mdl.is_i16
    chk_offsets = mdl.chk_offsets
    chk_to_var = mdl.chk_to_var
    n_chk = len(chk_offsets) - 1
    max_tanh = np.float32(1.0) - np.float32(np.finfo(np.float32).eps)

    var_nodes = Y_N_wave.T.copy()  # shape (N_cw, W)
    messages = np.zeros((mdl.n_branches, W), dtype=var_nodes.dtype)

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
                    contrib.astype(np.int32) + new_msg.astype(np.int32),
                    -int(mdl.sat_val_i16),
                    int(mdl.sat_val_i16),
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
                    tmp - msg_chk_to_var[e_start:e_stop].astype(np.int32), -32767, 32767
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
                    post[v] = np.clip(Y_N_T[v].astype(np.int32) + s, -32767, 32767).astype(np.int16)
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
                post[v] = np.clip(Y_N_T[v].astype(np.int32) + s, -32767, 32767).astype(np.int16)
            else:
                post[v] = Y_N_T[v] + np.sum(msg_chk_to_var[e_start:e_stop], axis=0)

    out_dtype = np.int16 if is_i16 else np.int32
    V_K_wave[:] = (post[mdl.info_bits_pos].T < 0).astype(out_dtype)
    return 0 if valid_synd else 1


# ====================================================================================================
# 2. STREAMPU `Stateful_core` MODULE (`DecoderLDPCPython`)
# ====================================================================================================

class DecoderLDPCPython(spu.Stateful_core):
    """Generic LDPC BP Decoder in Python (`NumPy` or `Numba` tiers `naive`/`std`/`opt`) as a StreamPU `Stateful_core` module."""

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
        py_tier: str = "opt",
    ):
        super().__init__()
        self.dec_type = dec_type
        self.dec_implem = dec_implem
        self.is_layered = dec_type in ("BP_HORIZONTAL_LAYERED", "BP_HL")
        self.is_oms = dec_implem == "OMS"
        self.q_type = "i16" if q_type in ("i16", "int16", "Int16") else "f32"
        self.is_i16 = self.q_type == "i16"
        self.simd_strategy = "INTER" if simd_strategy == "INTER" else ""
        self.single_wave = self.simd_strategy == "INTER"
        self.n_ite = int(n_ite)
        self.offset = float(offset)
        self.enable_syndrome = bool(enable_syndrome)
        self.syndrome_depth = int(syndrome_depth)
        self.use_numba = bool(use_numba)
        self.py_tier = py_tier.lower() if use_numba else "numpy"

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

        backend_tag = f"Numba_{self.py_tier}" if self.use_numba else "NumPy"
        self.name = f"Decoder_LDPC_{self.dec_type}_{self.dec_implem}_{self.q_type}_{backend_tag}"

        self._allocate_workspace(n_frames if self.single_wave else 1)

        if self.use_numba:
            self._kernel = self._resolve_numba_kernel()
        else:
            self._kernel = None

        if self.single_wave:
            py_aff3ct.set_single_wave(self, True)
        self.n_frames = n_frames

        task = self.create_task("decode_siho")
        task.replicable = True
        in_dtype = spu.int16 if self.is_i16 else spu.float32
        out_dtype = spu.int16 if self.is_i16 else spu.int32
        self.create_socket_in(task, "Y_N", self.N_cw, in_dtype)
        self.create_socket_out(task, "V_K", self.K, out_dtype)

        if self.use_numba:
            self.create_codelet(
                task,
                spu.numba_codelet(self._kernel, extra_args=DecoderLDPCPython._get_numba_extra_args),
            )
        else:
            self.create_codelet(task, DecoderLDPCPython._decode_numpy)

    def _resolve_numba_kernel(self):
        if self.is_i16 and not self.is_oms and self.py_tier != "naive":
            raise ValueError("SPA in 16-bit fixed-point (i16) is only supported in naive tier; use OMS for i16.")
        if self.py_tier == "naive":
            from . import ldpc_bp_naive
            return ldpc_bp_naive.decode_siho_naive_i16 if self.is_i16 else ldpc_bp_naive.decode_siho_naive_f32
        if self.py_tier == "std":
            from . import ldpc_bp_std
            return ldpc_bp_std.decode_siho_std_i16 if self.is_i16 else ldpc_bp_std.decode_siho_std_f32
        if self.py_tier in ("opt", "numba"):
            from . import ldpc_bp_opt, ldpc_bp_std
            if self.single_wave:
                W = self._work_W
                if self.is_i16:
                    if W == 8:
                        return ldpc_bp_opt.decode_siho_opt_inter_i16_w8
                    if W == 16:
                        return ldpc_bp_opt.decode_siho_opt_inter_i16_w16
                    if W == 32:
                        return ldpc_bp_opt.decode_siho_opt_inter_i16_w32
                    return ldpc_bp_std.decode_siho_std_i16
                else:
                    if W == 4:
                        return ldpc_bp_opt.decode_siho_opt_inter_f32_w4
                    if W == 8:
                        return ldpc_bp_opt.decode_siho_opt_inter_f32_w8
                    if W == 16:
                        return ldpc_bp_opt.decode_siho_opt_inter_f32_w16
                    return ldpc_bp_std.decode_siho_std_f32
            return ldpc_bp_opt.decode_siho_opt_scalar_i16 if self.is_i16 else ldpc_bp_opt.decode_siho_opt_scalar_f32
        raise ValueError(f"Unknown Python/Numba LDPC tier: {self.py_tier}")

    def _allocate_workspace(self, W: int):
        dtype = np.int16 if self.is_i16 else np.float32
        W = max(1, int(W))
        self._work_W = W
        # 1D buffers (used by `naive` and `opt` Scalar)
        self.msg_chk_to_var = np.zeros(self.n_branches, dtype=dtype)
        self.msg_var_to_chk = np.zeros(self.n_branches, dtype=dtype)
        self.post = np.zeros(self.N_cw, dtype=dtype)
        self.values = np.zeros(self.max_chk_degree, dtype=np.float32)
        self.contrib = np.zeros(self.max_chk_degree, dtype=dtype)
        # 2D `(..., W)` buffers (used by `std` Scalar/INTER and `opt` INTER)
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

    def _get_numba_extra_args(self, W: int):
        if W != self._work_W:
            self._allocate_workspace(W)

        if self.py_tier == "naive":
            if self.is_i16:
                return (
                    self.var_offsets, self.chk_offsets, self.transpose, self.chk_to_var, self.info_bits_pos,
                    self.is_layered, self.is_oms, np.int16(self.offset), self.sat_val_i16,
                    np.int32(self.n_ite), self.enable_syndrome, np.int32(self.syndrome_depth),
                    self.msg_chk_to_var, self.msg_var_to_chk, self.post,
                )
            return (
                self.var_offsets, self.chk_offsets, self.transpose, self.chk_to_var, self.info_bits_pos,
                self.is_layered, self.is_oms, np.float32(self.offset),
                np.int32(self.n_ite), self.enable_syndrome, np.int32(self.syndrome_depth),
                self.msg_chk_to_var, self.msg_var_to_chk, self.post,
            )

        if self.py_tier in ("opt", "numba") and self.single_wave:
            if self.is_i16 and W in (8, 16, 32):
                return (
                    self.var_offsets, self.chk_offsets, self.transpose, self.chk_to_var, self.info_bits_pos,
                    self.is_layered, np.int16(self.offset), self.sat_val_i16,
                    np.int32(self.n_ite), self.enable_syndrome, np.int32(self.syndrome_depth),
                    self.Y_N_inter, self.post_inter, self.msg_chk_to_var_inter, self.msg_var_to_chk_inter,
                    self.contrib_inter,
                )
            if (not self.is_i16) and W in (4, 8, 16):
                return (
                    self.var_offsets, self.chk_offsets, self.transpose, self.chk_to_var, self.info_bits_pos,
                    self.is_layered, self.is_oms, np.float32(self.offset),
                    np.int32(self.n_ite), self.enable_syndrome, np.int32(self.syndrome_depth),
                    self.Y_N_inter, self.post_inter, self.msg_chk_to_var_inter, self.msg_var_to_chk_inter,
                    self.values_inter, self.contrib_inter,
                )

        if self.py_tier == "std" or self.single_wave:
            if self.is_i16:
                return (
                    self.var_offsets, self.chk_offsets, self.transpose, self.chk_to_var, self.info_bits_pos,
                    self.is_layered, np.int16(self.offset), self.sat_val_i16,
                    np.int32(self.n_ite), self.enable_syndrome, np.int32(self.syndrome_depth),
                    self.Y_N_inter, self.post_inter, self.msg_chk_to_var_inter, self.msg_var_to_chk_inter,
                    self.contrib_inter, self.sign_w, self.min1_w, self.min2_w, self.cst1_w, self.cst2_w, self.synd_w,
                )
            return (
                self.var_offsets, self.chk_offsets, self.transpose, self.chk_to_var, self.info_bits_pos,
                self.is_layered, self.is_oms, np.float32(self.offset),
                np.int32(self.n_ite), self.enable_syndrome, np.int32(self.syndrome_depth),
                self.Y_N_inter, self.post_inter, self.msg_chk_to_var_inter, self.msg_var_to_chk_inter,
                self.values_inter, self.contrib_inter,
                self.sign_w, self.min1_w, self.min2_w, self.cst1_w, self.cst2_w, self.prod_w, self.nzeros_w, self.synd_w,
            )

        # `opt` Scalar mode (`W == 1`)
        if self.is_i16:
            return (
                self.var_offsets, self.chk_offsets, self.transpose, self.chk_to_var, self.info_bits_pos,
                self.is_layered, np.int16(self.offset), self.sat_val_i16,
                np.int32(self.n_ite), self.enable_syndrome, np.int32(self.syndrome_depth),
                self.post, self.msg_chk_to_var, self.msg_var_to_chk, self.contrib,
            )
        return (
            self.var_offsets, self.chk_offsets, self.transpose, self.chk_to_var, self.info_bits_pos,
            self.is_layered, self.is_oms, np.float32(self.offset),
            np.int32(self.n_ite), self.enable_syndrome, np.int32(self.syndrome_depth),
            self.post, self.msg_chk_to_var, self.msg_var_to_chk, self.values, self.contrib,
        )

    def _run_numba_wave(self, Y_N_wave, V_K_wave):
        args = self._get_numba_extra_args(Y_N_wave.shape[0])
        return int(self._kernel(*args, Y_N_wave, V_K_wave))

    @staticmethod
    def _decode_numba(mdl, tsk, frame_id):
        return mdl._run_numba_wave(tsk["Y_N"].wave, tsk["V_K"].wave)

    @staticmethod
    def _decode_numpy(mdl, tsk, frame_id):
        if mdl.is_layered:
            return _numpy_decode_layered(mdl, tsk["Y_N"].wave, tsk["V_K"].wave)
        return _numpy_decode_flooding(mdl, tsk["Y_N"].wave, tsk["V_K"].wave)

    def clone(self):
        other = type(self).__new__(type(self))
        spu.Stateful_core.__init__(other, self)
        other.dec_type = self.dec_type
        other.dec_implem = self.dec_implem
        other.is_layered = self.is_layered
        other.is_oms = self.is_oms
        other.q_type = self.q_type
        other.is_i16 = self.is_i16
        other.simd_strategy = self.simd_strategy
        other.single_wave = self.single_wave
        other.n_ite = self.n_ite
        other.offset = self.offset
        other.enable_syndrome = self.enable_syndrome
        other.syndrome_depth = self.syndrome_depth
        other.use_numba = self.use_numba
        other.py_tier = self.py_tier
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
