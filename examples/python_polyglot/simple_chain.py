#!/usr/bin/env python3
"""Polyglot Simple Chain Micro-Benchmark (C++ / Julia / Python Data Plane orchestrated from Python).

Unifies the four StreamPU simple_chain micro-benchmarks:
  - `simple_chain.cpp`           (App_v1: `increment::in` -> `increment::out`, C++ `Incrementer`)
  - `simple_chain_fwd.cpp`       (App_v2: `incrementf::fwd` in-place, C++ `Incrementer`)
  - `simple_chain_julia.cpp`     (App_v1: `increment::in` -> `increment::out`, Julia `Stateless_Julia`)
  - `simple_chain_fwd_julia.cpp` (App_v2: `incrementf::fwd` in-place, Julia `Stateless_Julia`)

and adds three Python incrementer backends (`spu.Stateful_core`):
  - `PY_NUMBA` : Numba `@njit(nogil=True, fastmath=True)` ahead-of-time compiled kernel
  - `PY_NUMPY` : Vectorized in-place NumPy kernel (`np.add(..., out=...)`)
  - `PY`       : Pure native Python `for` loop over a `memoryview`

Supports both homogeneous chains (e.g. `--implem JL`) and mixed/heterogeneous chains across the
6 stages `Inc0..Inc5` (e.g. `--implem CPP,JL,PY_NUMBA,CPP,JL,PY_NUMBA`), as well as an automated
multi-size / multi-backend sweep mode (`--sweep` and `--csv`).
"""

import argparse
import csv
import os
import sys
import time
from typing import List, Tuple

import numpy as np
from numba import njit, types

import streampu as spu

HERE = os.path.dirname(os.path.abspath(__file__))


# ==================================================================================================
# 1. Numba Ahead-of-Time Compiled Kernels (App_v1 `in`/`out` and App_v2 `fwd`)
# ==================================================================================================

@njit(nogil=True, fastmath=True)
def _numba_increment(w_in, w_out):
    n_rows, n_cols = w_in.shape
    for r in range(n_rows):
        for c in range(n_cols):
            w_out[r, c] = w_in[r, c] + np.uint8(1)
    return np.int32(0)


@njit(nogil=True, fastmath=True)
def _numba_incrementf(w_fwd):
    n_rows, n_cols = w_fwd.shape
    for r in range(n_rows):
        for c in range(n_cols):
            w_fwd[r, c] = w_fwd[r, c] + np.uint8(1)
    return np.int32(0)


_SIG_U8_2D = types.uint8[:, ::1]
_SIG_U8_2D_RO = types.Array(types.uint8, 2, "C", readonly=True)
_numba_increment.compile((_SIG_U8_2D_RO, _SIG_U8_2D))
_numba_increment.compile((_SIG_U8_2D, _SIG_U8_2D))
_numba_incrementf.compile((_SIG_U8_2D,))


# ==================================================================================================
# 2. Inline Julia Codelets (matching `simple_chain_julia.cpp` & `simple_chain_fwd_julia.cpp`)
# ==================================================================================================

JULIA_INCREMENT_CODE = """
function increment(sck_in, sck_out, cst_wait_time_ns, rnt_frame_id, rnt_n_frames_per_wave)
    if cst_wait_time_ns > 0
        start_incr = time_ns()
    end

    for n in eachindex(sck_in)
        sck_out[n] = sck_in[n] + one(eltype(sck_in))
    end

    if cst_wait_time_ns != 0
        stop_incr = time_ns()
        elapse_incr = stop_incr - start_incr

        while elapse_incr < cst_wait_time_ns
            elapse_incr = time_ns() - start_incr
        end
    end

    return 0
end
"""

JULIA_INCREMENTF_CODE = """
function incrementf(sck_fwd, cst_wait_time_ns, rnt_frame_id, rnt_n_frames_per_wave)
    if cst_wait_time_ns > 0
        start_incr = time_ns()
    end

    for n in eachindex(sck_fwd)
        sck_fwd[n] = sck_fwd[n] + one(eltype(sck_fwd))
    end

    if cst_wait_time_ns != 0
        stop_incr = time_ns()
        elapse_incr = stop_incr - start_incr

        while elapse_incr < cst_wait_time_ns
            elapse_incr = time_ns() - start_incr
        end
    end

    return 0
end
"""


def resolve_default_codelets_jl() -> str:
    candidates = [
        os.path.join(HERE, "..", "..", "..", "tests", "julia", "codelets.jl"),
        os.path.join(HERE, "..", "..", "lib", "aff3ct", "lib", "streampu", "tests", "julia", "codelets.jl"),
        os.path.join(HERE, "..", "tests", "julia", "codelets.jl"),
    ]
    for path in candidates:
        norm = os.path.normpath(path)
        if os.path.isfile(norm):
            return norm
    return os.path.normpath(candidates[0])


# ==================================================================================================
# 3. Python Incrementer Module (`spu.Stateful_core`: `PY`, `PY_NUMPY`, `PY_NUMBA`)
# ==================================================================================================

_U8_ONE = np.uint8(1)
_NP_ADD = np.add


class IncrementerPython(spu.Stateful_core):
    """Python Incrementer module supporting App_v1 (`in`/`out`) and App_v2 (`fwd`) in 3 flavours:
      - `PY`       : pure Python loop over `memoryview`
      - `PY_NUMPY` : vectorized in-place `np.add`
      - `PY_NUMBA` : ahead-of-time compiled Numba `@njit(nogil=True, fastmath=True)`
    """

    def __init__(self, data_length: int, ns: int = 0, mode: str = "PY_NUMBA", fwd: bool = False):
        super().__init__()
        self.data_length = data_length
        self.ns = int(ns)
        self.mode = mode.upper()
        self.fwd = bool(fwd)

        if self.fwd:
            task = self.create_task("incrementf")
            task.replicable = True
            self.create_socket_fwd(task, "fwd", data_length, spu.uint8)
            if self.mode == "PY_NUMBA":
                self.create_codelet(task, _numba_incrementf if self.ns == 0 else IncrementerPython._incrementf_numba)
            elif self.mode == "PY_NUMPY":
                self.create_codelet(
                    task, IncrementerPython._incrementf_numpy_fast if self.ns == 0 else IncrementerPython._incrementf_numpy
                )
            elif self.mode in ("PY", "PY_NATIVE"):
                self.create_codelet(
                    task, IncrementerPython._incrementf_py_fast if self.ns == 0 else IncrementerPython._incrementf_py
                )
            else:
                raise ValueError(f"Unsupported Python incrementer mode: {self.mode}")
        else:
            task = self.create_task("increment")
            task.replicable = True
            self.create_socket_in(task, "in", data_length, spu.uint8)
            self.create_socket_out(task, "out", data_length, spu.uint8)
            if self.mode == "PY_NUMBA":
                self.create_codelet(task, _numba_increment if self.ns == 0 else IncrementerPython._increment_numba)
            elif self.mode == "PY_NUMPY":
                self.create_codelet(
                    task, IncrementerPython._increment_numpy_fast if self.ns == 0 else IncrementerPython._increment_numpy
                )
            elif self.mode in ("PY", "PY_NATIVE"):
                self.create_codelet(
                    task, IncrementerPython._increment_py_fast if self.ns == 0 else IncrementerPython._increment_py
                )
            else:
                raise ValueError(f"Unsupported Python incrementer mode: {self.mode}")

    @staticmethod
    def _busy_wait(start_ns: int, wait_ns: int):
        while (time.perf_counter_ns() - start_ns) < wait_ns:
            pass

    @staticmethod
    def _increment_numba(mdl, tsk, frame_id):
        wait_ns = mdl.ns
        if wait_ns > 0:
            t0 = time.perf_counter_ns()
        res = int(_numba_increment(tsk["in"].wave, tsk["out"].wave))
        if wait_ns > 0:
            IncrementerPython._busy_wait(t0, wait_ns)
        return res

    @staticmethod
    def _incrementf_numba(mdl, tsk, frame_id):
        wait_ns = mdl.ns
        if wait_ns > 0:
            t0 = time.perf_counter_ns()
        res = int(_numba_incrementf(tsk["fwd"].wave))
        if wait_ns > 0:
            IncrementerPython._busy_wait(t0, wait_ns)
        return res

    @staticmethod
    def _increment_numpy_fast(mdl, tsk, frame_id):
        _NP_ADD(tsk["in"].wave, _U8_ONE, tsk["out"].wave)
        return 0

    @staticmethod
    def _incrementf_numpy_fast(mdl, tsk, frame_id):
        w_fwd = tsk["fwd"].wave
        _NP_ADD(w_fwd, _U8_ONE, w_fwd)
        return 0

    @staticmethod
    def _increment_numpy(mdl, tsk, frame_id):
        wait_ns = mdl.ns
        if wait_ns > 0:
            t0 = time.perf_counter_ns()
        _NP_ADD(tsk["in"].wave, _U8_ONE, tsk["out"].wave)
        if wait_ns > 0:
            IncrementerPython._busy_wait(t0, wait_ns)
        return 0

    @staticmethod
    def _incrementf_numpy(mdl, tsk, frame_id):
        wait_ns = mdl.ns
        if wait_ns > 0:
            t0 = time.perf_counter_ns()
        w_fwd = tsk["fwd"].wave
        _NP_ADD(w_fwd, _U8_ONE, w_fwd)
        if wait_ns > 0:
            IncrementerPython._busy_wait(t0, wait_ns)
        return 0

    @staticmethod
    def _increment_py_fast(mdl, tsk, frame_id):
        mv_in = tsk["in"].memoryview
        mv_out = tsk["out"].memoryview
        for i in range(len(mv_in)):
            mv_out[i] = (mv_in[i] + 1) & 0xFF
        return 0

    @staticmethod
    def _incrementf_py_fast(mdl, tsk, frame_id):
        mv_fwd = tsk["fwd"].memoryview
        for i in range(len(mv_fwd)):
            mv_fwd[i] = (mv_fwd[i] + 1) & 0xFF
        return 0

    @staticmethod
    def _increment_py(mdl, tsk, frame_id):
        wait_ns = mdl.ns
        if wait_ns > 0:
            t0 = time.perf_counter_ns()
        mv_in = tsk["in"].memoryview
        mv_out = tsk["out"].memoryview
        for i in range(len(mv_in)):
            mv_out[i] = (mv_in[i] + 1) & 0xFF
        if wait_ns > 0:
            IncrementerPython._busy_wait(t0, wait_ns)
        return 0

    @staticmethod
    def _incrementf_py(mdl, tsk, frame_id):
        wait_ns = mdl.ns
        if wait_ns > 0:
            t0 = time.perf_counter_ns()
        mv_fwd = tsk["fwd"].memoryview
        for i in range(len(mv_fwd)):
            mv_fwd[i] = (mv_fwd[i] + 1) & 0xFF
        if wait_ns > 0:
            IncrementerPython._busy_wait(t0, wait_ns)
        return 0

    def clone(self):
        other = type(self).__new__(type(self))
        spu.Stateful_core.__init__(other, self)
        other.data_length = self.data_length
        other.ns = self.ns
        other.mode = self.mode
        other.fwd = self.fwd
        return other


# ==================================================================================================
# 4. Incrementer Stage Factory (C++ / Julia / Python)
# ==================================================================================================

VALID_IMPLEMS = ("CPP", "JL", "JL_UNSAFE", "JL_SAFE", "PY", "PY_NATIVE", "PY_NUMPY", "PY_NUMBA")


def parse_implem_list(implem_spec: str, n_inc: int, default_jl_safe: bool) -> List[str]:
    tokens = [tok.strip().upper() for tok in implem_spec.split(",") if tok.strip()]
    if not tokens:
        raise ValueError("Empty --implem specification.")
    normalized = []
    for tok in tokens:
        if tok not in VALID_IMPLEMS:
            raise ValueError(f"Unknown incrementer backend '{tok}'. Valid choices: {', '.join(VALID_IMPLEMS)}")
        if tok == "JL" and default_jl_safe:
            normalized.append("JL_SAFE")
        elif tok == "JL_UNSAFE":
            normalized.append("JL")
        elif tok == "PY_NATIVE":
            normalized.append("PY")
        else:
            normalized.append(tok)
    return [normalized[i % len(normalized)] for i in range(n_inc)]


_JULIA_EVAL_CACHE = set()


def _ensure_julia_codelets_evaluated(jl_extern: bool, jl_path: str) -> None:
    if not spu.julia.available():
        raise RuntimeError("pyStreamPU was not compiled with Julia support (-DSPU_LINK_JULIA=ON).")
    spu.julia.init(256)
    key = ("file", jl_path) if jl_extern else ("inline", "")
    if key not in _JULIA_EVAL_CACHE:
        if jl_extern:
            spu.julia.eval_file(jl_path)
        else:
            spu.julia.eval(JULIA_INCREMENT_CODE)
            spu.julia.eval(JULIA_INCREMENTF_CODE)
        _JULIA_EVAL_CACHE.add(key)


def build_incrementer(
    stage_idx: int,
    backend: str,
    data_length: int,
    sleep_time_us: int,
    fwd: bool,
    jl_extern: bool,
    jl_path: str,
):
    sleep_ns = int(sleep_time_us) * 1000
    task_name = "incrementf" if fwd else "increment"

    if backend == "CPP":
        inc = spu.incrementer(data_length, ns=sleep_ns, dtype=spu.uint8)
    elif backend in ("JL", "JL_SAFE"):
        _ensure_julia_codelets_evaluated(jl_extern, jl_path)
        jl_safe = backend == "JL_SAFE"
        inc = spu.Stateless_Julia(jl_safe=jl_safe)
        t = inc.create_task(task_name)
        t.replicable = True
        if fwd:
            inc.create_socket_fwd(t, "fwd", data_length, spu.uint8)
        else:
            inc.create_socket_in(t, "in", data_length, spu.uint8)
            inc.create_socket_out(t, "out", data_length, spu.uint8)
        inc.create_constant(t, np.uint32(sleep_ns), spu.uint32)
        inc.create_codelet(t, "", task_name)
    elif backend in ("PY", "PY_NUMPY", "PY_NUMBA"):
        inc = IncrementerPython(data_length, ns=sleep_ns, mode=backend, fwd=fwd)
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    inc.name = f"Inc{stage_idx}"
    return inc


# ==================================================================================================
# 5. Chain Construction & Execution Engine
# ==================================================================================================

def run_chain_once(
    data_length: int,
    n_exec: int,
    n_threads: int,
    n_inter_frames: int,
    sleep_time_us: int,
    no_copy_mode: bool,
    print_stats: bool,
    step_by_step: bool,
    debug: bool,
    verbose: bool,
    fwd: bool,
    implems: List[str],
    jl_extern: bool,
    jl_path: str,
    dot_filepath: str = "",
    quiet: bool = False,
) -> Tuple[bool, float, float, float]:
    """Build and execute the simple_chain once.

    Returns:
        (tests_passed, warmup_ms, elapsed_ms, avg_inc_ns)
    """
    n_inc = len(implems)
    task_name = "incrementf" if fwd else "increment"

    initializer = spu.initializer(data_length, dtype=spu.uint8)
    finalizer = spu.finalizer(data_length, dtype=spu.uint8)
    incs = [
        build_incrementer(s, implems[s], data_length, sleep_time_us, fwd, jl_extern, jl_path)
        for s in range(n_inc)
    ]

    # Bind sockets
    if fwd:
        incs[0]["incrementf::fwd"] = initializer["initialize::out"]
        for s in range(n_inc - 1):
            incs[s + 1]["incrementf::fwd"] = incs[s]["incrementf::fwd"]
        finalizer["finalize::in"] = incs[n_inc - 1]["incrementf::fwd"]
    else:
        incs[0]["increment::in"] = initializer["initialize::out"]
        for s in range(n_inc - 1):
            incs[s + 1]["increment::in"] = incs[s]["increment::out"]
        finalizer["finalize::in"] = incs[n_inc - 1]["increment::out"]

    sequence_chain = spu.Sequence(initializer["initialize"], n_threads)
    sequence_chain.n_frames = n_inter_frames
    sequence_chain.no_copy_mode = no_copy_mode

    # Warmup (1 execution per thread to trigger Julia / Numba JIT compilation and proxy allocation)
    t_start_warmup = time.perf_counter()
    sequence_chain.exec_n_times(n_threads)
    warmup_ms = (time.perf_counter() - t_start_warmup) * 1000.0
    if not quiet:
        print(f"# Warmup (= JIT compilation) time: {warmup_ms:.3f} ms")

    # Initialize per-thread / per-frame input buffers
    for tid, cur_initializer in enumerate(sequence_chain.get_cloned_modules(initializer)):
        for f, arr in enumerate(cur_initializer.data_init):
            init_val = (tid * n_inter_frames + f) % (256 - n_inc)
            arr[:] = init_val

    if dot_filepath:
        sequence_chain.export_dot(dot_filepath)

    # Configure tasks (reset stats after warmup, set debug/stats/fast)
    for lt in sequence_chain.get_tasks_per_threads():
        for tsk in lt:
            tsk.reset()
            tsk.debug = debug
            tsk.set_debug_limit(16)
            tsk.stats = print_stats
            tsk.fast = True

    if verbose and not quiet:
        print("\nHelper information:")
        print("-------------------")
        spu.help(initializer)
        for inc in incs:
            spu.help(inc)
        spu.help(finalizer)

    # Timed execution
    t_start = time.perf_counter()
    if not step_by_step:
        sequence_chain.exec_n_times(n_exec)
    else:
        n_steps = max(1, n_exec // n_threads)
        for _ in range(n_steps):
            for tid in range(n_threads):
                while sequence_chain.exec_step(tid) is not None:
                    pass
    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    if not quiet:
        print(f"Sequence elapsed time: {elapsed_ms:.4f} ms")
        chain_sleep_ns = n_inc * sleep_time_us * 1000
        theoretical_ms = (chain_sleep_ns * n_exec * n_inter_frames) / 1e6 / n_threads
        print(f"Sequence theoretical time: {theoretical_ms:.4f} ms")

    # Verification of the sequence execution across all threads and frames
    tests_passed = True
    for tid, cur_finalizer in enumerate(sequence_chain.get_cloned_modules(finalizer)):
        final_frames = cur_finalizer.final_data
        for f in range(n_inter_frames):
            expected = (n_inc + ((tid * n_inter_frames + f) % (256 - n_inc))) % 256
            arr = np.asarray(final_frames[f], dtype=np.uint8)
            if not np.all(arr == expected):
                bad_indices = np.flatnonzero(arr != expected)
                for d in bad_indices[:5]:
                    if not quiet:
                        print(f"# expected = {expected} - obtained = {int(arr[d])} (d = {int(d)}, tid = {tid})")
                tests_passed = False

    if not quiet:
        if tests_passed:
            print("# \033[1;32mTests passed!\033[0m")
        else:
            print("# \033[1;31mTests failed :-(\033[0m")

    # Compute average duration per Incrementer task (in ns)
    avg_inc_ns = 0.0
    if print_stats:
        inc_names = {f"Inc{s}" for s in range(n_inc)}
        inc_total_s = 0.0
        inc_total_calls = 0
        for lt in sequence_chain.get_tasks_per_types():
            if lt and lt[0].module.name in inc_names and lt[0].name == task_name:
                inc_total_s += sum(t.total_duration for t in lt)
                inc_total_calls += sum(t.n_calls for t in lt)
        if inc_total_calls > 0:
            avg_inc_ns = (inc_total_s / inc_total_calls) * 1e9
        if not quiet:
            print("#")
            sequence_chain.show_stats(ordered=True, display_thr=False)
    else:
        # Estimate per-task time when task timers are disabled
        avg_inc_ns = (elapsed_ms * 1e6) / (n_exec * (n_inc + 2))

    # Unbind sockets cleanly (matching C++ binaries)
    sequence_chain.n_frames = 1
    if fwd:
        incs[0]["incrementf::fwd"].unbind(initializer["initialize::out"])
        for s in range(n_inc - 1):
            incs[s + 1]["incrementf::fwd"].unbind(incs[s]["incrementf::fwd"])
        finalizer["finalize::in"].unbind(incs[n_inc - 1]["incrementf::fwd"])
    else:
        incs[0]["increment::in"].unbind(initializer["initialize::out"])
        for s in range(n_inc - 1):
            incs[s + 1]["increment::in"].unbind(incs[s]["increment::out"])
        finalizer["finalize::in"].unbind(incs[n_inc - 1]["increment::out"])

    return tests_passed, warmup_ms, elapsed_ms, avg_inc_ns


# ==================================================================================================
# 6. Automated Sweep Mode (`--sweep`)
# ==================================================================================================

def choose_sweep_n_exec(backend: str, data_length: int, base_n_exec: int) -> int:
    """Adapt n_exec in sweep mode so slow combinations (e.g. pure Python at large N) finish quickly."""
    if backend == "PY":
        if data_length >= 65536:
            return max(5, min(base_n_exec, 20))
        if data_length >= 8192:
            return max(20, min(base_n_exec, 100))
        if data_length >= 1024:
            return max(100, min(base_n_exec, 500))
        return max(500, min(base_n_exec, 2000))
    if backend == "PY_NUMPY":
        if data_length >= 524288:
            return max(200, min(base_n_exec, 1000))
        if data_length >= 65536:
            return max(1000, min(base_n_exec, 5000))
        return max(2000, min(base_n_exec, 20000))
    # CPP, JL, JL_SAFE, PY_NUMBA
    if data_length >= 524288:
        return max(500, min(base_n_exec, 2000))
    if data_length >= 65536:
        return max(2000, min(base_n_exec, 10000))
    return base_n_exec


def format_latency(ns: float) -> str:
    if ns < 1000.0:
        return f"{ns:7.1f} ns"
    if ns < 1e6:
        return f"{ns / 1e3:7.2f} us"
    return f"{ns / 1e6:7.2f} ms"


def run_sweep(args) -> int:
    sizes = [int(x.strip()) for x in args.sweep_sizes.split(",") if x.strip()]
    backends = [x.strip().upper() for x in args.sweep_implems.split(",") if x.strip()]
    for b in backends:
        if b not in VALID_IMPLEMS:
            raise ValueError(f"Unknown backend '{b}' in --sweep-implems.")

    print("################################################################################")
    print("# Polyglot Simple Chain Sweep: C++ vs Julia vs Python (Numba / NumPy / Native) #")
    print("################################################################################")
    print(f"# Threads (-t)      : {args.n_threads}")
    print(f"# Inter-frames (-f) : {args.n_inter_frames}")
    print(f"# Incrementers      : {args.n_inc}")
    print(f"# Sizes N (bytes)   : {', '.join(str(s) for s in sizes)}")
    print(f"# Backends          : {', '.join(backends)}")
    metric_desc = (
        "Average latency per Incrementer task (with -p task timers enabled)"
        if args.print_stats
        else "Average latency per task = elapsed_time / (n_exec * (n_inc + 2)) (no timer overhead, pass -p for task timers)"
    )
    print(f"# Metric            : {metric_desc}")
    print("#")

    rows_csv = []
    all_passed = True

    for fwd_flag, app_label in [(False, "App_v1 (in -> out)"), (True, "App_v2 (fwd in-place)")]:
        print(f"\n=== {app_label} ===")
        col_headers = ["N (bytes)"] + backends
        header_line = f"| {col_headers[0]:>10s} | " + " | ".join(f"{b:>14s}" for b in backends) + " |"
        sep_line = "| " + "-" * 10 + " | " + " | ".join("-" * 14 for _ in backends) + " |"
        print(header_line)
        print(sep_line)

        for d in sizes:
            cells = [f"{d:10d}"]
            cpp_ref_ns = None
            for b in backends:
                n_exec_eff = choose_sweep_n_exec(b, d, args.n_exec)
                implems = parse_implem_list(b, args.n_inc, default_jl_safe=(b == "JL_SAFE"))
                passed, warmup_ms, elapsed_ms, avg_inc_ns = run_chain_once(
                    data_length=d,
                    n_exec=n_exec_eff,
                    n_threads=args.n_threads,
                    n_inter_frames=args.n_inter_frames,
                    sleep_time_us=args.sleep_time,
                    no_copy_mode=not args.copy_mode,
                    print_stats=args.print_stats,
                    step_by_step=False,
                    debug=False,
                    verbose=False,
                    fwd=fwd_flag,
                    implems=implems,
                    jl_extern=args.jl_extern,
                    jl_path=args.jl_path,
                    dot_filepath="",
                    quiet=True,
                )
                all_passed = all_passed and passed
                if b == "CPP":
                    cpp_ref_ns = avg_inc_ns
                cell_str = format_latency(avg_inc_ns) + ("" if passed else " !")
                cells.append(f"{cell_str:>14s}")
                rows_csv.append(
                    {
                        "topology": "App_v2" if fwd_flag else "App_v1",
                        "backend": b,
                        "data_length": d,
                        "n_threads": args.n_threads,
                        "n_inter_frames": args.n_inter_frames,
                        "n_exec": n_exec_eff,
                        "avg_task_ns": f"{avg_inc_ns:.3f}",
                        "overhead_vs_cpp_ns": f"{(avg_inc_ns - cpp_ref_ns):.3f}" if cpp_ref_ns is not None else "",
                        "ratio_vs_cpp": f"{(avg_inc_ns / cpp_ref_ns):.3f}" if cpp_ref_ns and cpp_ref_ns > 0 else "",
                        "warmup_ms": f"{warmup_ms:.3f}",
                        "elapsed_ms": f"{elapsed_ms:.3f}",
                        "passed": passed,
                    }
                )
            print("| " + " | ".join(cells) + " |")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "topology",
                    "backend",
                    "data_length",
                    "n_threads",
                    "n_inter_frames",
                    "n_exec",
                    "avg_task_ns",
                    "overhead_vs_cpp_ns",
                    "ratio_vs_cpp",
                    "warmup_ms",
                    "elapsed_ms",
                    "passed",
                ],
            )
            writer.writeheader()
            writer.writerows(rows_csv)
        print(f"\n# CSV results written to: {args.csv}")

    return 0 if all_passed else 1


# ==================================================================================================
# 7. CLI Argument Parser & Entry Point
# ==================================================================================================

def parse_args():
    default_jl_path = resolve_default_codelets_jl()
    p = argparse.ArgumentParser(
        description="Polyglot Simple Chain Micro-Benchmark (C++ / Julia / Python Numba / NumPy / Native)"
    )
    p.add_argument(
        "-t",
        "--n-threads",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of threads to run in parallel [default: hardware_concurrency]",
    )
    p.add_argument(
        "-f",
        "--n-inter-frames",
        type=int,
        default=1,
        help="Number of frames to process in one task [1]",
    )
    p.add_argument(
        "-s",
        "--sleep-time",
        type=int,
        default=0,
        help="Sleep time duration in one task (microseconds) [0]",
    )
    p.add_argument(
        "-d",
        "--data-length",
        type=int,
        default=2048,
        help="Size of data to process in one task (in bytes) [2048]",
    )
    p.add_argument(
        "-e",
        "--n-exec",
        type=int,
        default=100000,
        help="Number of sequence executions [100000]",
    )
    p.add_argument(
        "-o",
        "--dot-filepath",
        type=str,
        default="",
        help="Path to dot output file [empty]",
    )
    p.add_argument(
        "-c",
        "--copy-mode",
        action="store_true",
        help="Enable to copy data in sequence (performance will be reduced) [False]",
    )
    p.add_argument(
        "-b",
        "--step-by-step",
        action="store_true",
        help="Enable step-by-step sequence execution (performance will be reduced) [False]",
    )
    p.add_argument(
        "-p",
        "--print-stats",
        action="store_true",
        help="Enable to print per task statistics (performance will be reduced) [False]",
    )
    p.add_argument(
        "-J",
        "--jl-safe",
        action="store_true",
        help="Enable Julia safe mode (throw exceptions to C++/Python) [False]",
    )
    p.add_argument(
        "-E",
        "--jl-extern",
        action="store_true",
        help="Evaluate the Julia code from an external file [False]",
    )
    p.add_argument(
        "--jl-path",
        type=str,
        default=default_jl_path,
        help=f"Path to external Julia codelets file [{default_jl_path}]",
    )
    p.add_argument(
        "-F",
        "--fwd",
        action="store_true",
        help="Use forward socket in-place task 'incrementf' (App_v2) instead of 'increment' (App_v1) [False]",
    )
    p.add_argument(
        "-i",
        "--implem",
        type=str,
        default="CPP",
        help=(
            "Incrementer backend(s): single value or comma-separated list across Inc0..Inc5 "
            "(choices: CPP, JL, JL_SAFE, PY_NUMBA, PY_NUMPY, PY) [CPP]"
        ),
    )
    p.add_argument(
        "--n-inc",
        type=int,
        default=6,
        help="Number of chained Incrementer modules [6]",
    )
    p.add_argument(
        "-g",
        "--debug",
        action="store_true",
        help="Enable task debug mode (print socket data) [False]",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose mode [False]",
    )
    p.add_argument(
        "--sweep",
        action="store_true",
        help="Run automated benchmark sweep across N, App_v1/App_v2, and backends",
    )
    p.add_argument(
        "--sweep-sizes",
        type=str,
        default="1,64,1024,8192,65536,524288",
        help="Comma-separated data sizes N (in bytes) for --sweep [1,64,1024,8192,65536,524288]",
    )
    p.add_argument(
        "--sweep-implems",
        type=str,
        default="CPP,JL,JL_SAFE,PY_NUMBA,PY_NUMPY,PY",
        help="Comma-separated backends for --sweep [CPP,JL,JL_SAFE,PY_NUMBA,PY_NUMPY,PY]",
    )
    p.add_argument(
        "--csv",
        type=str,
        default="",
        help="Export sweep benchmark table to CSV file [empty]",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.sweep:
        return run_sweep(args)

    implems = parse_implem_list(args.implem, args.n_inc, default_jl_safe=args.jl_safe)
    no_copy_mode = not args.copy_mode
    title = (
        "# Micro-benchmark: Simple chain forward (polyglot) #"
        if args.fwd
        else "# Micro-benchmark: Simple chain (polyglot) #"
    )
    border = "#" * len(title)

    print(border)
    print(title)
    print(border)
    print("#")
    print("# Command line arguments:")
    print(f"#   - n_threads      = {args.n_threads}")
    print(f"#   - n_inter_frames = {args.n_inter_frames}")
    print(f"#   - sleep_time_us  = {args.sleep_time}")
    print(f"#   - data_length    = {args.data_length}")
    print(f"#   - n_exec         = {args.n_exec}")
    print(f"#   - dot_filepath   = {'[empty]' if not args.dot_filepath else args.dot_filepath}")
    print(f"#   - no_copy_mode   = {'true' if no_copy_mode else 'false'}")
    print(f"#   - print_stats    = {'true' if args.print_stats else 'false'}")
    print(f"#   - step_by_step   = {'true' if args.step_by_step else 'false'}")
    print(f"#   - fwd (App_v2)   = {'true' if args.fwd else 'false'}")
    print(f"#   - implems        = {', '.join(implems)}")
    print(f"#   - jl_safe        = {'true' if args.jl_safe or ('JL_SAFE' in implems) else 'false'}")
    print(f"#   - jl_extern      = {'true' if args.jl_extern else 'false'}")
    print(f"#   - debug          = {'true' if args.debug else 'false'}")
    print(f"#   - verbose        = {'true' if args.verbose else 'false'}")
    print("#")

    tests_passed, _, _, _ = run_chain_once(
        data_length=args.data_length,
        n_exec=args.n_exec,
        n_threads=args.n_threads,
        n_inter_frames=args.n_inter_frames,
        sleep_time_us=args.sleep_time,
        no_copy_mode=no_copy_mode,
        print_stats=args.print_stats,
        step_by_step=args.step_by_step,
        debug=args.debug,
        verbose=args.verbose,
        fwd=args.fwd,
        implems=implems,
        jl_extern=args.jl_extern,
        jl_path=args.jl_path,
        dot_filepath=args.dot_filepath,
        quiet=False,
    )
    return 0 if tests_passed else 1


if __name__ == "__main__":
    sys.exit(main())
