"""Square Gray QAM demodulator written in Python + Numba, used in the receiver stage of the mixed-language pipeline.

Mirrors AFF3CT's `Modem_generic<..., Constellation_QAM, MAX>::demodulate` (complex constellation):
for each bit b of each received symbol y,

    LLR_b = MAX_{j : bit b of j = 0} ( -|y - c_j|^2 / (2 sigma^2) )  -  MAX_{j : bit b of j = 1} ( ... )

with MAX = max* (exact log-MAP: max(a, b) + log1p(exp(-|a - b|))) or max (max-log approximation), the points c_j
being AFF3CT's own constellation (`py_aff3ct.qam_constellation`), visited in the same order (j = 0 .. M-1) and in
float32, as in the C++ implementation. Attached through `create_codelet`, the `@njit` kernel is compiled into a
C-callable trampoline and invoked by the native worker threads without the Python interpreter or the GIL.
"""

import numpy as np
from numba import njit, types

import py_aff3ct
import streampu as spu

_SIG = (types.float32[:, ::1], types.float32[:, ::1], types.float32[:, ::1])


def _make_kernel(bps: int, logmap: bool):
    re, im = py_aff3ct.qam_constellation(bps)
    c_re = np.asarray(re, dtype=np.float32)
    c_im = np.asarray(im, dtype=np.float32)
    M = c_re.shape[0]
    BPS = bps
    NINF = np.float32(-np.inf)

    if logmap:

        @njit(nogil=True, fastmath=False, inline="always")
        def vmax(a, b):
            m = a if a >= b else b
            return np.float32(m + np.log1p(np.exp(-np.abs(np.float32(a - b)))))

    else:

        @njit(nogil=True, fastmath=False, inline="always")
        def vmax(a, b):
            return a if a >= b else b

    # LOC80-BEGIN PY_DEMOD_QAM
    @njit(nogil=True, fastmath=False)
    def demod_qam(sck_CP, sck_Y_N1, sck_Y_N2):
        d = np.empty(M, dtype=np.float32)
        N = sck_Y_N2.shape[1]
        for f in range(sck_Y_N1.shape[0]):
            s = sck_CP[f, 0]
            inv_s2 = np.float32(1.0) / (np.float32(2.0) * s * s)
            for k in range((N + BPS - 1) // BPS):
                yr = sck_Y_N1[f, 2 * k]
                yi = sck_Y_N1[f, 2 * k + 1]
                for j in range(M):
                    dr = np.float32(yr - c_re[j])
                    di = np.float32(yi - c_im[j])
                    d[j] = -np.float32(np.float32(dr * dr) + np.float32(di * di)) * inv_s2
                for b in range(BPS):
                    n = k * BPS + b
                    if n >= N:
                        break
                    L0 = NINF
                    L1 = NINF
                    for j in range(M):
                        if ((j >> b) & 1) == 0:
                            L0 = vmax(L0, d[j])
                        else:
                            L1 = vmax(L1, d[j])
                    sck_Y_N2[f, n] = L0 - L1
        return np.int32(0)
    # LOC80-END PY_DEMOD_QAM

    demod_qam.compile(_SIG)
    return demod_qam


_KERNELS = {}


class DemodQAMNumba(spu.Stateful_core):
    """Numba QAM demodulator task `demodulate` (sockets CP, Y_N1 [2*ceil(N/bps) floats], Y_N2 [N LLRs])."""

    def __init__(self, N: int, bps: int = 4, logmap: bool = True):
        super().__init__()
        self.N, self.bps, self.logmap = N, bps, logmap
        self.N_mod = 2 * ((N + bps - 1) // bps)
        key = (bps, logmap)
        if key not in _KERNELS:
            _KERNELS[key] = _make_kernel(bps, logmap)
        self.name = f"Modem_QAM{1 << bps}_{'logmap' if logmap else 'maxlog'}_demodulate_Numba"
        py_aff3ct.set_single_wave(self, True)
        t = self.create_task("demodulate")
        t.replicable = True
        self.create_socket_in(t, "CP", 1, spu.float32)
        self.create_socket_in(t, "Y_N1", self.N_mod, spu.float32)
        self.create_socket_out(t, "Y_N2", N, spu.float32)
        self.create_codelet(t, _KERNELS[key])

    def clone(self):
        other = type(self).__new__(type(self))
        spu.Stateful_core.__init__(other, self)
        other.N, other.bps, other.logmap, other.N_mod = self.N, self.bps, self.logmap, self.N_mod
        return other

