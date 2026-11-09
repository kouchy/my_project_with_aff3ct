#!/usr/bin/env python3
"""Polyglot Repetition Code Pipeline / Sequence Benchmark (Python Control Plane + C++ / Julia / Python Data Plane).

Assembles the exact Repetition Code digital communication chain of `examples/pipeline_julia/`
entirely from Python using `pyStreamPU` (`streampu`) and `py_aff3ct`, and supports:
  - C++ AFF3CT decoders: `--dec-implem FAST` (SIMD MIPP) or `--dec-implem STD` (scalar)
  - Julia Stateless_Julia decoders: `--dec-jl-ver v0|v1|v2|v3|v4` (from `../pipeline_julia/src/rep_decoder.jl`)
  - Python Stateful_core decoders: `--dec-implem PY_NUMPY` or `--dec-implem PY_NUMBA`
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(HERE, "build")
if os.path.isdir(BUILD_DIR) and BUILD_DIR not in sys.path:
    sys.path.insert(0, BUILD_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import streampu as spu
import py_aff3ct
from tasks.rep_decoder import DecoderRepetitionPython


def build_decoder_rep_julia(
    K: int,
    N_cw: int,
    n_frames: int,
    buffered: bool,
    jl_ver: str,
    jl_filepath: str,
    force_jl_safe: bool,
) -> spu.Stateless_Julia:
    if not spu.julia.available():
        raise RuntimeError("pyStreamPU was not compiled with Julia support (-DSPU_LINK_JULIA=ON).")

    if jl_ver == "v0":
        fn_prefix, jl_safe, single_wave = "decode_rep_v0", True, False
    elif jl_ver == "v1":
        fn_prefix, jl_safe, single_wave = "decode_rep_v1", True, False
    elif jl_ver == "v2":
        fn_prefix, jl_safe, single_wave = "decode_rep_v2", True, False
    elif jl_ver == "v3":
        fn_prefix, jl_safe, single_wave = "decode_rep_v2", False, False
    elif jl_ver == "v4":
        fn_prefix, jl_safe, single_wave = "decode_rep_v4", force_jl_safe, True
    else:
        raise ValueError(f"Unknown Julia repetition decoder version: {jl_ver}")

    fn_name = fn_prefix + ("_buf" if buffered else "_nobuf")
    dec = spu.Stateless_Julia(jl_safe=jl_safe)
    dec.name = f"Decoder_repetition_JL_{jl_ver}"
    if single_wave:
        py_aff3ct.set_single_wave(dec, True)
    dec.n_frames = n_frames

    t = dec.create_task("decode_siho")
    t.replicable = True
    dec.create_socket_in(t, "Y_N", N_cw, spu.float32)
    dec.create_socket_out(t, "V_K", K, spu.int32)
    dec.create_codelet_file(t, jl_filepath, fn_name)
    dec.eval()
    return dec


def parse_args():
    default_jl_path = os.path.normpath(os.path.join(HERE, "..", "pipeline_julia", "src", "rep_decoder.jl"))
    p = argparse.ArgumentParser(description="Polyglot Repetition Code Pipeline / Sequence Benchmark (pyStreamPU)")
    p.add_argument("-K", "--info-bits", type=int, default=128, help="Information bits K (default: 128)")
    p.add_argument("-N", "--cw-size", type=int, default=512, help="Codeword size N (default: 512)")
    p.add_argument("-F", "--inter-fra", type=int, default=1, help="Inter-frame level F (default: 1)")
    p.add_argument("-t", "--threads", type=int, default=os.cpu_count() or 1, help="Number of worker threads")
    p.add_argument("-m", "--ebn0-min", type=float, default=0.0, help="Min Eb/N0 in dB")
    p.add_argument("-M", "--ebn0-max", type=float, default=6.0, help="Max Eb/N0 in dB")
    p.add_argument("-s", "--ebn0-step", type=float, default=1.0, help="Eb/N0 step in dB")
    p.add_argument("-e", "--max-fe", type=int, default=100, help="Max frame errors per SNR point")
    p.add_argument("--mnt-max-fra", type=int, default=0, help="Max analyzed frames per SNR point (0 = unlimited)")
    p.add_argument(
        "--dec-implem",
        type=str,
        default="FAST",
        choices=[
            "FAST",
            "STD",
            "JL_V0",
            "JL_V1",
            "JL_V2",
            "JL_V3",
            "JL_V4",
            "PY_V0",
            "PY_V1",
            "PY_V2",
            "PY_V3",
            "PY_V4",
            "PY_NUMPY",
            "PY_NUMPY_SW",
            "PY_NUMBA",
            "PY_NUMBA_SW",
        ],
        help="Decoder implementation",
    )
    p.add_argument("--dec-jl", action="store_true", help="Use Julia Stateless_Julia repetition decoder")
    p.add_argument("--dec-jl-ver", type=str, default=None, choices=["v0", "v1", "v2", "v3", "v4"], help="Julia decoder tier")
    p.add_argument("--dec-py-ver", type=str, default=None, choices=["v0", "v1", "v2", "v3", "v4"], help="Python/Numba decoder tier")
    p.add_argument("--dec-jl-safe", action="store_true", help="Force jl_safe=True on v4")
    p.add_argument("--dec-jl-path", type=str, default=default_jl_path, help="Path to rep_decoder.jl")
    p.add_argument("--enc-no-buff", action="store_true", help="Disable buffered encoding")
    p.add_argument("--sim-sequence", action="store_true", help="Run a replicated Sequence instead of 3-stage Pipeline")
    p.add_argument("--sim-interstage-buf", type=int, default=0, help="Pipeline inter-stage buffer size")
    p.add_argument("--sim-active-waiting", action="store_true", help="Enable active waiting on inter-stage buffers")
    p.add_argument("--sim-stats", action="store_true", help="Enable per-task statistics")
    p.add_argument("-v", "--sim-verbose", action="store_true", help="Print # Exec_stats line per SNR")
    p.add_argument("--sim-dot", type=str, default="", help="Export Graphviz .dot file")
    return p.parse_args()


def main():
    args = parse_args()
    K, N_cw = args.info_bits, args.cw_size
    n_frames = args.inter_fra
    n_threads = max(1, args.threads)
    buffered = not args.enc_no_buff
    R = float(K) / float(N_cw)

    jl_dec = args.dec_jl
    jl_ver = args.dec_jl_ver or "v4"
    if args.dec_implem.startswith("JL_V"):
        jl_dec = True
        jl_ver = "v" + args.dec_implem[-1]

    # Instantiate C++ / Julia / Python modules
    source = py_aff3ct.Source_random_fast(K)
    encoder = py_aff3ct.Encoder_repetition_sys(K, N_cw, buffered)
    modem = py_aff3ct.Modem_BPSK_fast(N_cw)
    channel = py_aff3ct.Channel_AWGN_LLR(N_cw, "FAST")
    monitor = py_aff3ct.Monitor_BFER(K, args.max_fe, args.mnt_max_fra)
    sink = spu.sink_no(K, dtype=spu.int32)

    if jl_dec:
        spu.julia.init()
        decoder = build_decoder_rep_julia(K, N_cw, n_frames, buffered, jl_ver, args.dec_jl_path, args.dec_jl_safe)
        dec_desc = f"Julia Stateless_Julia ({jl_ver})"
    elif args.dec_implem == "FAST":
        decoder = py_aff3ct.Decoder_repetition_fast(K, N_cw, buffered)
        dec_desc = "C++ AFF3CT (FAST SIMD)"
    elif args.dec_implem == "STD":
        decoder = py_aff3ct.Decoder_repetition_std(K, N_cw, buffered)
        dec_desc = "C++ AFF3CT (STD scalar)"
    elif args.dec_implem in ("PY_NUMPY", "PY_NUMPY_SW"):
        decoder = DecoderRepetitionPython(K, N_cw, buffered, use_numba=False, single_wave=(args.dec_implem == "PY_NUMPY_SW"))
        dec_desc = f"Python Stateful_core ({args.dec_implem})"
    elif args.dec_implem in ("PY_V0", "PY_V1", "PY_V2", "PY_V3", "PY_V4") or args.dec_py_ver is not None:
        py_ver = args.dec_py_ver or ("v" + args.dec_implem[-1])
        decoder = DecoderRepetitionPython(K, N_cw, buffered, use_numba=True, py_ver=py_ver)
        dec_desc = f"Python Stateful_core (Numba {py_ver})"
    elif args.dec_implem in ("PY_NUMBA", "PY_NUMBA_SW"):
        py_ver = "v3" if args.dec_implem == "PY_NUMBA_SW" else "v2"
        decoder = DecoderRepetitionPython(K, N_cw, buffered, use_numba=True, py_ver=py_ver)
        dec_desc = f"Python Stateful_core ({args.dec_implem} / Numba {py_ver})"
    else:
        raise ValueError(f"Unsupported decoder implementation: {args.dec_implem}")

    for mdl in (source, encoder, modem, channel, decoder, monitor, sink):
        mdl.n_frames = n_frames

    # Bind task sockets in Python (Control Plane)
    sigma_buf = py_aff3ct.Sigma_buffer(n_frames, 1.0)
    encoder["encode::U_K"].bind(source["generate::out_data"])
    modem["modulate::X_N1"].bind(encoder["encode::X_N"])
    channel["add_noise::X_N"].bind(modem["modulate::X_N2"])
    sigma_buf.bind(channel["add_noise::CP"])
    modem["demodulate::Y_N1"].bind(channel["add_noise::Y_N"])
    sigma_buf.bind(modem["demodulate::CP"])
    decoder["decode_siho::Y_N"].bind(modem["demodulate::Y_N2"])
    monitor["check_errors::U"].bind(source["generate::out_data"])
    monitor["check_errors::V"].bind(decoder["decode_siho::V_K"])
    sink["send_count::in_count"].bind(source["generate::out_count"])
    sink["send_count::in_data"].bind(decoder["decode_siho::V_K"])

    eff_buf = args.sim_interstage_buf if args.sim_interstage_buf > 0 else max(4, 2 * n_threads)

    print("#----------------------------------------------------------")
    print("# Polyglot Repetition Chain (Assembled in Python via pyStreamPU)")
    print(f"# Runtime        : {'Sequence (replicated)' if args.sim_sequence else 'Pipeline (3 stages)'}")
    print(f"# Threads (t)    : {n_threads}")
    print(f"# Inter-frame (F): {n_frames}")
    print(f"# Code (N, K)    : ({N_cw}, {K}), R = {R:.4f}")
    print(f"# Decoder backend: {dec_desc}")
    print("#----------------------------------------------------------")

    if args.sim_sequence:
        runtime = spu.Sequence(source["generate"], n_threads)
        cloned_monitors = runtime.get_cloned_modules(monitor)
        monitor_red = py_aff3ct.Monitor_BFER_reduction(cloned_monitors)
        monitor_red.set_reduce_frequency_ms(1)
        if args.sim_dot:
            runtime.export_dot(args.sim_dot)
    else:
        runtime = spu.Pipeline(
            [source["generate"]],
            [monitor["check_errors"], sink["send_count"]],
            [
                [[source["generate"]], [source["generate"]], []],
                [[encoder["encode"]], [decoder["decode_siho"]], []],
                [[monitor["check_errors"], sink["send_count"]], [], []],
            ],
            [1, n_threads, 1],
            [eff_buf, eff_buf],
            [args.sim_active_waiting, args.sim_active_waiting],
        )
        monitor_red = None
        if args.sim_dot:
            runtime.export_dot(args.sim_dot)

    if args.sim_sequence:
        py_aff3ct.configure_sequence_tasks(runtime, args.sim_stats)
    else:
        py_aff3ct.configure_pipeline_tasks(runtime, args.sim_stats)

    print("#  Eb/N0 |   Es/N0 |        FRA |         BE |         FE |        BER |        FER |  SIM_CTHR")
    print("#   (dB) |    (dB) |            |            |            |            |            |    (Mb/s)")

    ebn0 = args.ebn0_min
    while ebn0 <= args.ebn0_max + 1e-5:
        if args.sim_sequence:
            py_aff3ct.seed_sequence_modules(runtime)
        else:
            py_aff3ct.seed_pipeline_modules(runtime)

        esn0 = py_aff3ct.ebn0_to_esn0(ebn0, R, 1)
        sigma = py_aff3ct.esn0_to_sigma(esn0, 1)
        sigma_buf.set_sigma(sigma)

        if args.sim_sequence:
            elapsed_us = py_aff3ct.exec_sequence(runtime, monitor_red)
            n_fra = monitor_red.get_n_analyzed_fra()
            n_be = monitor_red.get_n_be()
            n_fe = monitor_red.get_n_fe()
            ber = monitor_red.get_ber()
            fer = monitor_red.get_fer()
        else:
            elapsed_us = py_aff3ct.exec_pipeline(runtime, monitor, source, args.mnt_max_fra, n_frames)
            n_fra = monitor.get_n_analyzed_fra()
            n_be = monitor.get_n_be()
            n_fe = monitor.get_n_fe()
            ber = monitor.get_ber()
            fer = monitor.get_fer()

        info_thr_mbps = (float(n_fra) * float(K)) / elapsed_us if elapsed_us > 0.0 else 0.0
        coded_thr_mbps = (float(n_fra) * float(N_cw)) / elapsed_us if elapsed_us > 0.0 else 0.0

        print(
            f"  {ebn0:6.2f} |  {esn0:6.2f} | {n_fra:10d} | {n_be:10d} | {n_fe:10d} | "
            f"{ber:10.2e} | {fer:10.2e} | {info_thr_mbps:9.3f}"
        )

        if args.sim_verbose:
            print(
                f"# Exec_stats: ebn0={ebn0:.2f} fra={n_fra} elapsed_us={elapsed_us:.3f} "
                f"info_thr_mbps={info_thr_mbps:.6f} coded_thr_mbps={coded_thr_mbps:.6f}"
            )

        if args.sim_sequence:
            monitor_red.reset()
            py_aff3ct.reset_sequence_modules(runtime)
        else:
            py_aff3ct.reset_pipeline_modules(runtime)

        ebn0 += args.ebn0_step

    if args.sim_stats:
        print("#")
        runtime.show_stats()

    print("#\n# End of the simulation")
    del monitor_red
    del runtime


if __name__ == "__main__":
    main()
