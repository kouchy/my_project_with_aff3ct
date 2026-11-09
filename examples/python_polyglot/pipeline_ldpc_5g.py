#!/usr/bin/env python3
"""Polyglot 5G NR QC-LDPC Pipeline / Sequence Benchmark (Python Control Plane + C++ / Julia / Python Data Plane).

Assembles the exact 5G NR UL-SCH QC-LDPC digital communication chain of `examples/ldpc_5g_julia/`
entirely from Python using `pyStreamPU` (`streampu`) and `py_aff3ct`, and supports the generic
4-axis LDPC decoder family across C++, Julia, and Python (Numba / NumPy):
  - Axis 1 (Schedule)   : `-D BP_FLOODING` or `-D BP_HORIZONTAL_LAYERED`
  - Axis 2 (Update Rule): `--dec-implem SPA` or `--dec-implem OMS` (plus `_JL`, `_PY_NUMBA`, `_PY_NUMPY` suffixes)
  - Axis 3 (Precision)  : `-p 32` (`float32`) or `-p 16` (`int16` fixed-point with `Quantizer_pow2_fast_i16`)
  - Axis 4 (SIMD Mode)  : Scalar (default) or `--dec-simd INTER` (inter-frame SIMD across W = F frames per wave)
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
from tasks.ldpc_decoder_py import DecoderLDPCPython


def build_decoder_ldpc_julia(
    code: py_aff3ct.LDPC_Code,
    n_frames: int,
    sched: str,
    rule: str,
    dtype_str: str,
    simd_strategy: str,
    n_ite: int,
    offset: float,
    enable_syndrome: bool,
    syndrome_depth: int,
    jl_filepath: str,
    jl_safe: bool,
) -> spu.Stateful_Julia:
    if not spu.julia.available():
        raise RuntimeError("pyStreamPU was not compiled with Julia support (-DSPU_LINK_JULIA=ON).")

    dec = spu.Stateful_Julia(jl_safe=jl_safe)
    dec.name = f"Decoder_LDPC_{sched}_{rule}_{dtype_str}_Julia"
    if simd_strategy == "INTER":
        py_aff3ct.set_single_wave(dec, True)

    # 1. Shared read-only state across all worker threads (LDPC_Graph)
    graph_ctor = code.get_julia_graph_ctor(
        n_ite,
        enable_syndrome,
        syndrome_depth,
        sched,
        rule,
        dtype_str,
        offset,
    )
    dec.create_state_file(graph_ctor, jl_filepath, shared=True)

    # 2. Per-thread mutable workspace cloned via Base.deepcopy for each worker thread (LDPC_Workspace)
    work_ctor = code.get_julia_workspace_ctor(
        simd_strategy,
        dtype_str,
        n_frames if simd_strategy == "INTER" else 1,
    )
    dec.create_state(work_ctor, shared=False)

    t = dec.create_task("decode_siho")
    t.replicable = True
    spu_dtype = spu.int16 if dtype_str == "i16" else spu.float32
    dec.create_socket_in(t, "Y_N", code.N_cw, spu_dtype)
    dec.create_socket_out(t, "V_K", code.K, spu_dtype if dtype_str == "i16" else spu.int32)
    dec.create_codelet(t, function_name="decode_siho_julia")
    dec.n_frames = n_frames
    dec.eval()
    return dec


def parse_args():
    jl_src_dir = os.path.normpath(os.path.join(HERE, "..", "ldpc_5g_julia", "src"))
    default_jl_path = os.path.join(jl_src_dir, "ldpc_bp_opt.jl")
    p = argparse.ArgumentParser(description="Polyglot 5G NR QC-LDPC Pipeline / Sequence Benchmark (pyStreamPU)")
    p.add_argument("-K", "--info-bits", type=int, default=512, help="Information bits K (default: 512)")
    p.add_argument("-N", "--cw-size", type=int, default=1024, help="Transmitted codeword size N (default: 1024)")
    p.add_argument("--enc-type", type=str, default="LDPC_5G", choices=["LDPC_5G", "LDPC_QC"], help="LDPC encoder type")
    p.add_argument("--dec-h-path", type=str, default="", help="Path to QC parity-check matrix (.qc)")
    p.add_argument("--dec-h-reorder", type=str, default="NONE", choices=["NONE", "ASC", "DSC"], help="H matrix reordering")
    p.add_argument(
        "-D",
        "--dec-type",
        type=str,
        default="BP_FLOODING",
        choices=["BP_FLOODING", "BP_HORIZONTAL_LAYERED", "BP_VERTICAL_LAYERED"],
        help="LDPC decoder schedule",
    )
    p.add_argument(
        "--dec-implem",
        type=str,
        default="SPA",
        choices=[
            "SPA",
            "LSPA",
            "MS",
            "OMS",
            "NMS",
            "AMS",
            "SPA_JL",
            "OMS_JL",
            "SPA_JL_NAIVE",
            "OMS_JL_NAIVE",
            "SPA_JL_STD",
            "OMS_JL_STD",
            "SPA_JL_OPT",
            "OMS_JL_OPT",
            "SPA_PY_NAIVE",
            "OMS_PY_NAIVE",
            "SPA_PY_STD",
            "OMS_PY_STD",
            "SPA_PY_OPT",
            "OMS_PY_OPT",
            "SPA_PY_NUMBA",
            "OMS_PY_NUMBA",
            "SPA_PY_NUMPY",
            "OMS_PY_NUMPY",
        ],
        help="LDPC check-node update rule (and optional backend suffix)",
    )
    p.add_argument("--dec-simd", type=str, default="", choices=["", "INTER"], help="SIMD strategy ('INTER' for inter-frame)")
    p.add_argument("-p", "--sim-prec", type=int, default=32, choices=[16, 32], help="Simulation precision in bits: 32 (f32) or 16 (i16)")
    p.add_argument("--qnt-bits", type=int, default=8, help="Quantizer total bit width for 16-bit fixed-point (default: 8)")
    p.add_argument("--qnt-dec", type=int, default=2, help="Quantizer fractional bits for 16-bit fixed-point (default: 2)")
    p.add_argument("-i", "--dec-ite", type=int, default=20, help="Max BP iterations (default: 20)")
    p.add_argument("--dec-off", type=float, default=None, help="OMS offset factor (default: 0.15 for f32, 1.0 for i16)")
    p.add_argument("--dec-norm", type=float, default=1.0, help="NMS normalization factor")
    p.add_argument("--dec-no-synd", action="store_true", help="Disable syndrome early stopping")
    p.add_argument("--dec-synd-depth", type=int, default=1, help="Syndrome stopping depth (default: 1)")
    p.add_argument("--dec-jl", action="store_true", help="Use Julia Stateful_Julia LDPC decoder")
    p.add_argument("--dec-jl-safe", action="store_true", help="Enable Julia exception checking on every call")
    p.add_argument("--dec-jl-path", type=str, default="", help=f"Path to Julia LDPC decoder script (default: {default_jl_path})")
    p.add_argument(
        "--dec-py",
        type=str,
        default="",
        choices=["", "NAIVE", "STD", "OPT", "NUMBA", "NUMPY"],
        help="Use Python Stateful_core LDPC decoder (NAIVE, STD, OPT/NUMBA, or NUMPY)",
    )
    p.add_argument("-F", "--inter-fra", type=int, default=1, help="Inter-frame level F (default: 1)")
    p.add_argument("-t", "--threads", type=int, default=os.cpu_count() or 1, help="Number of worker threads")
    p.add_argument("-m", "--ebn0-min", type=float, default=0.0, help="Min Eb/N0 in dB")
    p.add_argument("-M", "--ebn0-max", type=float, default=5.0, help="Max Eb/N0 in dB")
    p.add_argument("-s", "--ebn0-step", type=float, default=0.5, help="Eb/N0 step in dB")
    p.add_argument("-e", "--max-fe", type=int, default=100, help="Max frame errors per SNR point")
    p.add_argument("--mnt-max-fra", type=int, default=0, help="Max analyzed frames per SNR point (0 = unlimited)")
    p.add_argument("--sim-sequence", action="store_true", help="Run a replicated Sequence instead of 3-stage Pipeline")
    p.add_argument("--sim-interstage-buf", type=int, default=0, help="Pipeline inter-stage buffer size")
    p.add_argument("--sim-active-waiting", action="store_true", help="Enable active waiting on inter-stage buffers")
    p.add_argument("--sim-stats", action="store_true", help="Enable per-task statistics")
    p.add_argument("-v", "--sim-verbose", action="store_true", help="Print # Exec_stats line per SNR")
    p.add_argument("--sim-dot", type=str, default="", help="Export Graphviz .dot file")
    return p.parse_args()


def main():
    args = parse_args()
    n_frames = args.inter_fra
    n_threads = max(1, args.threads)
    enable_syndrome = not args.dec_no_synd
    is_i16 = args.sim_prec == 16
    dtype_str = "i16" if is_i16 else "f32"

    # Resolve backend and base rule from --dec-implem / --dec-jl / --dec-py
    implem = args.dec_implem
    jl_dec = args.dec_jl
    py_mode = args.dec_py
    jl_src_dir = os.path.normpath(os.path.join(HERE, "..", "ldpc_5g_julia", "src"))
    default_jl_file = "ldpc_bp_opt.jl"

    if implem.endswith("_JL_NAIVE"):
        jl_dec = True
        base_rule = implem[:-9]
        default_jl_file = "ldpc_bp_naive.jl"
    elif implem.endswith("_JL_STD"):
        jl_dec = True
        base_rule = implem[:-7]
        default_jl_file = "ldpc_bp_std.jl"
    elif implem.endswith("_JL_OPT"):
        jl_dec = True
        base_rule = implem[:-7]
        default_jl_file = "ldpc_bp_opt.jl"
    elif implem.endswith("_JL"):
        jl_dec = True
        base_rule = implem[:-3]
        default_jl_file = "ldpc_bp_opt.jl"
    elif implem.endswith("_PY_NAIVE"):
        py_mode = "NAIVE"
        base_rule = implem[:-9]
    elif implem.endswith("_PY_STD"):
        py_mode = "STD"
        base_rule = implem[:-7]
    elif implem.endswith("_PY_OPT"):
        py_mode = "OPT"
        base_rule = implem[:-7]
    elif implem.endswith("_PY_NUMBA"):
        py_mode = "OPT"
        base_rule = implem[:-9]
    elif implem.endswith("_PY_NUMPY"):
        py_mode = "NUMPY"
        base_rule = implem[:-9]
    else:
        base_rule = implem

    if not args.dec_jl_path:
        args.dec_jl_path = os.path.join(jl_src_dir, default_jl_file)

    if args.dec_off is not None:
        offset = args.dec_off
    else:
        if base_rule == "OMS":
            offset = 1.0 if is_i16 else 0.15
        else:
            offset = 0.0

    # Build LDPC code metadata and parity-check matrix H
    code = py_aff3ct.LDPC_Code(
        args.info_bits,
        args.cw_size,
        args.enc_type,
        args.dec_h_path,
        args.dec_h_reorder,
    )
    K, N_cw, N = code.K, code.N_cw, code.N
    R = float(K) / float(N)

    # Instantiate C++ / Julia / Python modules
    if is_i16:
        source = py_aff3ct.Source_random_fast_i16(K)
        encoder = code.build_encoder_i16()
        puncturer = code.build_puncturer_i16()
        modem = py_aff3ct.Modem_BPSK_fast_i16(N)
        channel = py_aff3ct.Channel_AWGN_LLR(N, "FAST")
        quantizer = code.build_quantizer_i16(args.qnt_dec, args.qnt_bits)
        monitor = py_aff3ct.Monitor_BFER_i16(K, args.max_fe, args.mnt_max_fra)
        sink = spu.sink_no(K, dtype=spu.int16)
    else:
        source = py_aff3ct.Source_random_fast(K)
        encoder = code.build_encoder()
        puncturer = code.build_puncturer()
        modem = py_aff3ct.Modem_BPSK_fast(N)
        channel = py_aff3ct.Channel_AWGN_LLR(N, "FAST")
        quantizer = None
        monitor = py_aff3ct.Monitor_BFER(K, args.max_fe, args.mnt_max_fra)
        sink = spu.sink_no(K, dtype=spu.int32)

    simd_tag = f" {args.dec_simd}" if args.dec_simd else ""
    if jl_dec:
        spu.julia.init()
        decoder = build_decoder_ldpc_julia(
            code,
            n_frames,
            args.dec_type,
            base_rule,
            dtype_str,
            args.dec_simd,
            args.dec_ite,
            offset,
            enable_syndrome,
            args.dec_synd_depth,
            args.dec_jl_path,
            args.dec_jl_safe,
        )
        dec_desc = f"Julia Stateful_Julia ({args.dec_type} {base_rule} {dtype_str}{simd_tag}, i={args.dec_ite}, {'safe' if args.dec_jl_safe else 'unsafe'})"
    elif py_mode in ("NAIVE", "STD", "OPT", "NUMBA", "NUMPY"):
        py_tier = "opt" if py_mode == "NUMBA" else py_mode.lower()
        decoder = DecoderLDPCPython(
            code=code,
            dec_type=args.dec_type,
            dec_implem=base_rule,
            q_type=dtype_str,
            simd_strategy=args.dec_simd,
            n_ite=args.dec_ite,
            offset=offset,
            enable_syndrome=enable_syndrome,
            syndrome_depth=args.dec_synd_depth,
            n_frames=n_frames,
            use_numba=(py_mode != "NUMPY"),
            py_tier=py_tier,
        )
        dec_desc = f"Python Stateful_core ({py_mode} {args.dec_type} {base_rule} {dtype_str}{simd_tag}, i={args.dec_ite})"
    else:
        if is_i16:
            decoder = code.build_decoder_i16(
                args.dec_type,
                base_rule,
                args.dec_ite,
                args.dec_simd,
                offset,
                args.dec_norm,
                enable_syndrome,
                args.dec_synd_depth,
            )
        else:
            decoder = code.build_decoder(
                args.dec_type,
                base_rule,
                args.dec_ite,
                args.dec_simd,
                offset,
                args.dec_norm,
                enable_syndrome,
                args.dec_synd_depth,
            )
        dec_desc = f"C++ AFF3CT ({args.dec_type} {base_rule} {dtype_str}{simd_tag}, i={args.dec_ite})"

    all_modules = [source, encoder, puncturer, modem, channel, decoder, monitor, sink]
    if quantizer is not None:
        all_modules.append(quantizer)
    for mdl in all_modules:
        mdl.n_frames = n_frames

    # Bind task sockets in Python (Control Plane)
    sigma_buf = py_aff3ct.Sigma_buffer(n_frames, 1.0)
    encoder["encode::U_K"].bind(source["generate::out_data"])
    puncturer["puncture::X_N1"].bind(encoder["encode::X_N"])
    modem["modulate::X_N1"].bind(puncturer["puncture::X_N2"])
    channel["add_noise::X_N"].bind(modem["modulate::X_N2"])
    sigma_buf.bind(channel["add_noise::CP"])
    modem["demodulate::Y_N1"].bind(channel["add_noise::Y_N"])
    sigma_buf.bind(modem["demodulate::CP"])
    if is_i16:
        quantizer["process::Y_N1"].bind(modem["demodulate::Y_N2"])
        puncturer["depuncture::Y_N1"].bind(quantizer["process::Y_N2"])
    else:
        puncturer["depuncture::Y_N1"].bind(modem["demodulate::Y_N2"])
    decoder["decode_siho::Y_N"].bind(puncturer["depuncture::Y_N2"])
    monitor["check_errors::U"].bind(source["generate::out_data"])
    monitor["check_errors::V"].bind(decoder["decode_siho::V_K"])
    sink["send_count::in_count"].bind(source["generate::out_count"])
    sink["send_count::in_data"].bind(decoder["decode_siho::V_K"])

    eff_buf = args.sim_interstage_buf if args.sim_interstage_buf > 0 else max(4, 2 * n_threads)

    print("#----------------------------------------------------------")
    print("# Polyglot 5G NR QC-LDPC Chain (Assembled in Python via pyStreamPU)")
    print(f"# Runtime        : {'Sequence (replicated)' if args.sim_sequence else 'Pipeline (3 stages)'}")
    print(f"# Precision (p)  : {args.sim_prec}-bit ({dtype_str})")
    print(f"# Threads (t)    : {n_threads}")
    print(f"# Inter-frame (F): {n_frames}")
    print(f"# Code (N_cw,N,K): ({N_cw}, {N}, {K}), R = {R:.4f}")
    print(f"# QC Matrix      : {code.get_qc_path()} ({code.n_rows}x{code.n_cols}, edges={code.n_connections})")
    print(f"# Decoder backend: {dec_desc}")
    print("#----------------------------------------------------------")

    if args.sim_sequence:
        runtime = spu.Sequence(source["generate"], n_threads)
        cloned_monitors = runtime.get_cloned_modules(monitor)
        if is_i16:
            monitor_red = py_aff3ct.Monitor_BFER_reduction_i16(cloned_monitors)
        else:
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
            if is_i16:
                elapsed_us = py_aff3ct.exec_sequence_i16(runtime, monitor_red)
            else:
                elapsed_us = py_aff3ct.exec_sequence(runtime, monitor_red)
            n_fra = monitor_red.get_n_analyzed_fra()
            n_be = monitor_red.get_n_be()
            n_fe = monitor_red.get_n_fe()
            ber = monitor_red.get_ber()
            fer = monitor_red.get_fer()
        else:
            if is_i16:
                elapsed_us = py_aff3ct.exec_pipeline_i16(runtime, monitor, source, args.mnt_max_fra, n_frames)
            else:
                elapsed_us = py_aff3ct.exec_pipeline(runtime, monitor, source, args.mnt_max_fra, n_frames)
            n_fra = monitor.get_n_analyzed_fra()
            n_be = monitor.get_be() if hasattr(monitor, "get_be") else monitor.get_n_be()
            n_fe = monitor.get_n_fe()
            ber = monitor.get_ber()
            fer = monitor.get_fer()

        info_thr_mbps = (float(n_fra) * float(K)) / elapsed_us if elapsed_us > 0.0 else 0.0
        coded_thr_mbps = (float(n_fra) * float(N)) / elapsed_us if elapsed_us > 0.0 else 0.0

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
