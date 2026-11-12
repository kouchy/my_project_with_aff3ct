"""BPSK demodulator written in Python + Numba, used as a stage-1 task of the mixed-language pipeline.

Computes the channel LLRs `Y_N2 = Y_N1 * 2 / sigma^2` exactly like AFF3CT's `Modem_BPSK_fast::demodulate`,
with the noise standard deviation read from the `CP` input socket (one value per frame). Attached through
`create_codelet`, the `@njit` kernel is compiled into a C-callable trampoline and invoked by the native worker
threads without the Python interpreter or the GIL.
"""

import numpy as np
from numba import njit, types

import py_aff3ct
import streampu as spu


@njit(nogil=True, fastmath=False)
def demod_bpsk(sck_CP, sck_Y_N1, sck_Y_N2):
    for f in range(sck_Y_N1.shape[0]):
        s = sck_CP[f, 0]
        fac = np.float32(2.0) / np.float32(s * s)
        for i in range(sck_Y_N1.shape[1]):
            sck_Y_N2[f, i] = sck_Y_N1[f, i] * fac
    return np.int32(0)


demod_bpsk.compile((types.float32[:, ::1], types.float32[:, ::1], types.float32[:, ::1]))


class DemodBPSKNumba(spu.Stateful_core):
    def __init__(self, N: int):
        super().__init__()
        self.N = N
        self.name = "Modem_BPSK_demodulate_Numba"
        py_aff3ct.set_single_wave(self, True)
        t = self.create_task("demodulate")
        t.replicable = True
        self.create_socket_in(t, "CP", 1, spu.float32)
        self.create_socket_in(t, "Y_N1", N, spu.float32)
        self.create_socket_out(t, "Y_N2", N, spu.float32)
        self.create_codelet(t, demod_bpsk)

    def clone(self):
        other = type(self).__new__(type(self))
        spu.Stateful_core.__init__(other, self)
        other.N = self.N
        return other



def build_demod_bpsk_numba(N: int) -> DemodBPSKNumba:
    """Build the Numba BPSK demodulator of `N` LLRs per frame."""
    return DemodBPSKNumba(N)
