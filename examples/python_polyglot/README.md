# Polyglot Python + Julia + C++ Examples (`python_polyglot`)

This directory contains three polyglot benchmarks and digital communication chains orchestrated from **Python** via [`pyStreamPU`](../../lib/pystreampu) that seamlessly combine:
- **C++ modules** from **StreamPU** and **AFF3CT** (exposed through `streampu` and a lightweight `pybind11` wrapper `py_aff3ct`),
- **Julia modules** (`Stateless_Julia` and `Stateful_Julia`) executed in-process via the native `libjulia` C API (`unsafe_wrap` zero-copy array views),
- **Python modules** (`Stateful_core` with ahead-of-time compiled **Numba** `@njit(nogil=True, fastmath=True)` + `llvmlite` SIMD intrinsics, vectorized **NumPy**, or pure native **Python**),

all sharing a **single `StreamPU` runtime instance** (`aff3ct_install/`) and communicating through zero-copy pointer handoff (`sck::status::forward`).

## Overview of the Three Scripts

| Script | Description | Required C++ Extensions |
| :--- | :--- | :--- |
| [`simple_chain.py`](simple_chain.py) | **Runtime & Polyglot Kernel Overhead Micro-Benchmark**<br>Chains `Initializer<uint8_t>` $\rightarrow$ $6 \times$ `Incrementer` (`Inc0..Inc5`) $\rightarrow$ `Finalizer<uint8_t>` in `App_v1` (`in` $\rightarrow$ `out`) or `App_v2` (`-F` in-place `fwd`) mode across homogeneous or mixed C++ / Julia / Numba / NumPy / Python stages. | `streampu` (`pyStreamPU`) only |
| [`pipeline_repetition.py`](pipeline_repetition.py) | **Repetition Code BFER Communication Chain**<br>Full AWGN BPSK chain (`Source` $\rightarrow$ `Encoder` $\rightarrow$ `Modem` $\rightarrow$ `Channel` $\rightarrow$ `Decoder` $\rightarrow$ `Monitor_BFER`) in 3-stage `Pipeline` or replicated `Sequence` mode, comparing C++ (`STD`, `FAST`), Julia (`v0`..`v4`), Python Numba (`v0`..`v3`), and NumPy decoders. | `streampu` + `py_aff3ct` |
| [`pipeline_ldpc_5g.py`](pipeline_ldpc_5g.py) | **5G NR UL-SCH QC-LDPC BFER Communication Chain**<br>Full 5G NR QC-LDPC chain (`Source` $\rightarrow$ `Encoder_LDPC` $\rightarrow$ `Puncturer` $\rightarrow$ `Modem` $\rightarrow$ `Channel` $\rightarrow$ [`Quantizer`] $\rightarrow$ `Depuncturer` $\rightarrow$ `Decoder_LDPC` $\rightarrow$ `Monitor_BFER`) across a **4-axis decoder matrix** (`BP_FLOODING` / `BP_HORIZONTAL_LAYERED` $\times$ `SPA` / `OMS` $\times$ `f32` / `i16` $\times$ Scalar / `INTER` SIMD) and **3 optimization tiers** (`NAIVE` $\rightarrow$ `STD` $\rightarrow$ `OPT`) in Julia and Python/Numba. | `streampu` + `py_aff3ct` |

---

## 1. Prerequisites

Ensure the following tools are installed and available in your `PATH`:
- A C++17 compiler (`clang++` or `g++`), `cmake` ($\ge 3.15$), and `git`
- **Julia** ($\ge 1.10$, `julia` executable on `PATH`) — no external Julia packages are required (all kernels use only Julia `Base`, `Base.Cartesian`, and `Core.Intrinsics.llvmcall`)
- **Python** ($\ge 3.10$) with `venv` support

Initialize all Git submodules from the root of `my_project_with_aff3ct`, and synchronize `lib/aff3ct/lib/streampu` onto the newer `StreamPU` version shipped in `lib/pystreampu/lib/streampu`:

```bash
# From the root of my_project_with_aff3ct:
git submodule update --init --recursive

# Align AFF3CT's StreamPU submodule with the newer StreamPU version in pyStreamPU:
git -C lib/aff3ct/lib/streampu fetch ../../pystreampu/lib/streampu
git -C lib/aff3ct/lib/streampu checkout $(git -C lib/pystreampu/lib/streampu rev-parse HEAD)
```

> [!IMPORTANT]
> **Single Shared `StreamPU` ABI Rule:**
> Because `pyStreamPU` (`streampu`) and `py_aff3ct` exchange C++ pointers (`spu::module::Module`, `spu::runtime::Sequence`, `spu::runtime::Pipeline`, `spu::runtime::Task`, `spu::runtime::Socket`) inside the same Python process, **both extensions must be compiled against the exact same `StreamPU` headers and library installed in `aff3ct_install/`** (`INCLUDE_SPU_LIB=OFF`).
> The commands above ensure that `lib/aff3ct` builds and installs the exact `StreamPU` commit expected by `lib/pystreampu`. You can verify at any time that both point to the same commit with:
> ```bash
> git -C lib/pystreampu/lib/streampu rev-parse HEAD
> git -C lib/aff3ct/lib/streampu rev-parse HEAD
> ```

---

## 2. Complete Build Procedure (Step-by-Step)

### Step 1: Compile and Install `AFF3CT` & `StreamPU` (with Julia Support)

Compile `AFF3CT` with both static and shared libraries enabled for `AFF3CT` and `StreamPU`, position-independent code (`-DCMAKE_POSITION_INDEPENDENT_CODE="ON"`), and `libjulia` support (`-DSPU_LINK_JULIA="ON"`). Installing into `aff3ct_install/` exports `aff3ct-config.cmake`, `streampu-config.cmake`, and `cpptrace-config.cmake`:

```bash
# From the root of my_project_with_aff3ct:
cd lib/aff3ct
mkdir -p build && cd build
cmake .. -G"Unix Makefiles" \
         -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_CXX_FLAGS="-O3 -Wall -funroll-loops -march=native" \
         -DCMAKE_POSITION_INDEPENDENT_CODE="ON" \
         -DAFF3CT_COMPILE_EXE="OFF" \
         -DAFF3CT_COMPILE_STATIC_LIB="ON" \
         -DAFF3CT_COMPILE_SHARED_LIB="ON" \
         -DSPU_COMPILE_STATIC_LIB="ON" \
         -DSPU_COMPILE_SHARED_LIB="ON" \
         -DSPU_LINK_JULIA="ON"
cmake --build . -j 8
cmake --install . --prefix ../../../aff3ct_install
ln -sfn ../../../aff3ct_install/lib/cmake/aff3ct-* ../../../aff3ct_install/lib/cmake/aff3ct
cd ../../..
```

### Step 2: Create the Python Virtual Environment and Install `pyStreamPU`

Create a Python virtual environment in `lib/pystreampu/venv-julia`, install the Python build/runtime dependencies (`numpy`, `numba`, `pybind11`, `scikit-build-core`), and compile `lib/pystreampu` with `INCLUDE_SPU_LIB=OFF` so that it links against the `StreamPU` and `cpptrace` libraries installed in `aff3ct_install/` during Step 1:

```bash
# From the root of my_project_with_aff3ct:
cd lib/pystreampu
python3 -m venv venv-julia
source venv-julia/bin/activate
pip install --upgrade pip
pip install numpy numba pybind11 scikit-build-core

# Clean any previous CMake build directory to avoid cached standalone builds:
rm -rf build
export CMAKE_BUILD_PARALLEL_LEVEL=8
AFF3CT_INST="$(cd ../../aff3ct_install && pwd)"

pip install --no-build-isolation -v . \
    -C build-dir=build \
    -C cmake.define.INCLUDE_SPU_LIB=OFF \
    -C cmake.define.SPU_LINK_JULIA=ON \
    -C cmake.define.StreamPU_DIR="${AFF3CT_INST}/lib/cmake/streampu" \
    -C cmake.define.cpptrace_DIR="${AFF3CT_INST}/lib/cmake/cpptrace"
cd ../..
```

### Step 3: Compile the `py_aff3ct` Wrapper for `pipeline_repetition.py` and `pipeline_ldpc_5g.py`

With `venv-julia` still active (so CMake picks up the exact same Python interpreter and `pybind11`), compile the `py_aff3ct` extension module in `examples/python_polyglot/build/`:

```bash
# From the root of my_project_with_aff3ct (with venv-julia active):
cd examples/python_polyglot
mkdir -p build && cd build
AFF3CT_INST="$(cd ../../../aff3ct_install && pwd)"

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-O3 -Wall -funroll-loops -march=native" \
    -DAFF3CT_DIR="${AFF3CT_INST}/lib/cmake/aff3ct" \
    -DStreamPU_DIR="${AFF3CT_INST}/lib/cmake/streampu" \
    -Dcpptrace_DIR="${AFF3CT_INST}/lib/cmake/cpptrace"
cmake --build . -j 8
cd ..
```

### Step 4: Verify the Installation

From `examples/python_polyglot/` (with `venv-julia` active), verify that `streampu`, Julia support (`streampu._ext.julia`), and `build/py_aff3ct*.so` load cleanly:

```bash
python -c "import sys; sys.path.insert(0, 'build'); import streampu as spu, py_aff3ct; print('Julia enabled:', hasattr(spu._ext, 'julia')); print('All extensions loaded successfully!')"
```

---

## 3. Detailed Guide & Usage Examples for Each Script

Before running any script from `examples/python_polyglot/`, make sure the virtual environment is active:

```bash
cd examples/python_polyglot
source ../../lib/pystreampu/venv-julia/bin/activate
```

---

### 3.1. `simple_chain.py` — Polyglot Kernel & Runtime Overhead Micro-Benchmark

[`simple_chain.py`](simple_chain.py) unifies the four StreamPU `simple_chain` micro-benchmarks (`simple_chain.cpp`, `simple_chain_fwd.cpp`, `simple_chain_julia.cpp`, `simple_chain_fwd_julia.cpp`) into a single polyglot Python script and extends them with Numba, NumPy, and pure Python tasks.

#### Architecture & Topologies
The chain consists of 8 tasks:
$$\text{Initializer}\langle\text{uint8\_t}\rangle \longrightarrow \text{Inc}_0 \longrightarrow \text{Inc}_1 \longrightarrow \text{Inc}_2 \longrightarrow \text{Inc}_3 \longrightarrow \text{Inc}_4 \longrightarrow \text{Inc}_5 \longrightarrow \text{Finalizer}\langle\text{uint8\_t}\rangle$$

Two socket topologies are supported:
- **`App_v1` (default)**: Each incrementer executes task `increment(in -> out)` with separate input and output sockets.
- **`App_v2` (`-F` / `--fwd`)**: Each incrementer executes task `incrementf(fwd)` with a single in-place forward socket.

#### Supported Backends (`-i` / `--implem`)
You can specify a **single backend** (applied homogeneously to all 6 incrementers `Inc0..Inc5`) or a **comma-separated list of 6 backends** (one per stage `Inc0..Inc5` to build a heterogeneous polyglot chain):

| Backend Token | Module Type | Description |
| :--- | :--- | :--- |
| `CPP` | `spu.incrementer` | Native C++ compiled `Incrementer<uint8_t>` |
| `JL` | `spu.Stateless_Julia` | Julia `@inbounds @simd` kernel with zero-copy `unsafe_wrap` (`jl_safe=False`) |
| `JL_SAFE` | `spu.Stateless_Julia` | Julia `@inbounds @simd` kernel with isolated Julia array copy (`jl_safe=True`, or pass `-J`) |
| `PY_NUMBA` | `spu.Stateful_core` | Python task calling an ahead-of-time compiled Numba `@njit(nogil=True, fastmath=True)` kernel |
| `PY_NUMPY` | `spu.Stateful_core` | Python task calling vectorized in-place `np.add(a, 1, out=b, casting="unsafe")` |
| `PY` | `spu.Stateful_core` | Pure Python `for` loop over zero-copy `memoryview` buffers |

#### Key CLI Options
- `-t, --threads <int>`: Number of threads in `Sequence` (default: `1`)
- `-f, --inter-frames <int>`: Inter-frame level $F$ / frames per wave (default: `1`)
- `-d, --data-length <int>`: Number of `uint8_t` elements per frame $N$ (default: `2048`)
- `-e, --n-exec <int>`: Number of sequence executions (default: `100000`)
- `-s, --sleep-time <int>`: Artificial sleep duration per task in $\mu\text{s}$ (default: `5`; set `-s 0` to measure pure dispatch/kernel overhead)
- `-F, --fwd`: Use `App_v2` (`incrementf` with `fwd` sockets) instead of `App_v1` (`increment` with `in`/`out` sockets)
- `-c, --copy-mode`: Disable StreamPU zero-copy mode (`set_no_copy_mode(False)`)
- `-p, --print-stats`: Display detailed per-task StreamPU latency and throughput table
- `-b, --step-by-step`: Execute the sequence task-by-task (`exec_step()`) instead of `exec_seq()`
- `-o, --dot-filepath <path>`: Export the `Sequence` topology to a Graphviz `.dot` file
- `--sweep`: Run an automated benchmark sweep across multiple frame sizes $N$, topologies (`App_v1` & `App_v2`), and backends

#### Examples for `simple_chain.py`

```bash
# 1. Measure pure runtime + kernel dispatch overhead (-s 0, N = 1 byte) across C++, Julia, and Python Numba:
python simple_chain.py -t 1 -d 1 -e 100000 -s 0 -p --implem CPP
python simple_chain.py -t 1 -d 1 -e 100000 -s 0 -p --implem JL
python simple_chain.py -t 1 -d 1 -e 20000  -s 0 -p --implem PY_NUMBA

# 2. Compare App_v1 (in -> out) vs. App_v2 (-F, in-place fwd) at N = 4096 bytes with Julia:
python simple_chain.py -t 1 -d 4096 -e 100000 -s 0 -p --implem JL
python simple_chain.py -t 1 -d 4096 -e 100000 -s 0 -p -F --implem JL

# 3. Compare zero-copy mode (default) vs. copy mode (-c) in App_v2:
python simple_chain.py -t 1 -d 65536 -e 20000 -s 0 -p -F --implem CPP
python simple_chain.py -t 1 -d 65536 -e 20000 -s 0 -p -F -c --implem CPP

# 4. Heterogeneous polyglot chain: mix C++, Julia, Numba, NumPy, and pure Python in the SAME Sequence!
#    Inc0=CPP -> Inc1=JL -> Inc2=PY_NUMBA -> Inc3=PY_NUMPY -> Inc4=PY -> Inc5=CPP
python simple_chain.py -t 1 -d 64 -e 5000 -s 0 -p --implem CPP,JL,PY_NUMBA,PY_NUMPY,PY,CPP

# 5. Multi-threaded scaling (-t 4) with inter-frame batching (-f 8) and external Julia codelets (-E):
python simple_chain.py -t 4 -f 8 -d 2048 -e 50000 -s 0 -p -E --implem JL

# 6. Export Graphviz DOT topology and inspect frame contents in debug mode (-g):
python simple_chain.py -t 1 -d 16 -e 1 -s 0 -g -o simple_chain.dot --implem JL

# 7. Automated multi-size / multi-backend sweep with CSV export:
python simple_chain.py --sweep -t 1 -e 50000 \
    --sweep-sizes 1,64,1024,16384,262144 \
    --sweep-implems CPP,JL,JL_SAFE,PY_NUMBA,PY_NUMPY \
    --csv simple_chain_sweep.csv
```

---

### 3.2. `pipeline_repetition.py` — Polyglot Repetition Code BFER Simulation

[`pipeline_repetition.py`](pipeline_repetition.py) is the Python-orchestrated counterpart of [`examples/pipeline_julia`](../pipeline_julia/README.md). It simulates a complete **Repetition Code** $(N, K)$ BPSK AWGN communication chain and showcases a **5-step progressive optimization ladder** in both **Julia** ([`../pipeline_julia/src/rep_decoder.jl`](../pipeline_julia/src/rep_decoder.jl)) and **Python/Numba** ([`tasks/rep_decoder.py`](tasks/rep_decoder.py)).

#### Architecture & Execution Modes
$$\text{Source\_random\_fast} \longrightarrow \text{Encoder\_repetition\_sys} \longrightarrow \text{Modem\_BPSK\_fast} \longrightarrow \text{Channel\_AWGN\_LLR} \longrightarrow \text{Decoder} \longrightarrow \text{Monitor\_BFER}$$

Two StreamPU execution runtimes are available:
- **3-Stage Multi-Threaded `Pipeline` (default)**:
  - **Stage 0** (1 thread): `Source_random_fast::generate`
  - **Stage 1** (`-t` threads, replicated): `Encoder::encode` $\rightarrow$ `Modem::modulate` $\rightarrow$ `Channel::add_noise` $\rightarrow$ `Decoder::decode_siho`
  - **Stage 2** (1 thread): `Monitor_BFER::check_errors` $\rightarrow$ `sink_no`
- **Single-Stage Replicated `Sequence` (`--sim-sequence`)**:
  - The entire chain (`Source` $\rightarrow \dots \rightarrow$ `Monitor_BFER`) is cloned across `-t` threads without inter-stage ring buffers.

#### Supported Decoder Implementations (`--dec-implem` / `--dec-jl-ver` / `--dec-py-ver`)

| Backend | CLI Flag(s) | Single-Wave | Description |
| :--- | :--- | :---: | :--- |
| **C++ Scalar** | `--dec-implem STD` | No | AFF3CT reference scalar C++ repetition decoder |
| **C++ SIMD** | `--dec-implem FAST` | No | AFF3CT `mipp` intra-frame SIMD C++ repetition decoder |
| **Julia `v0`** | `--dec-jl --dec-jl-ver v0` (or `JL_V0`) | No | Naive 2D `reshape` view + heap allocation (`zeros(Float32, K)`) every frame |
| **Julia `v1`** | `--dec-jl --dec-jl-ver v1` (or `JL_V1`) | No | Zero-allocation scalar loops (`@inbounds`) |
| **Julia `v2`** | `--dec-jl --dec-jl-ver v2` (or `JL_V2`) | No | Auto-vectorized `@simd ivdep` with safe memory copy (`jl_safe=True`) |
| **Julia `v3`** | `--dec-jl --dec-jl-ver v3` (or `JL_V3`) | No | Zero-copy `unsafe_wrap` + 4-accumulator register-tiled `@simd ivdep` |
| **Julia `v4`** | `--dec-jl --dec-jl-ver v4` (or `JL_V4`) | **Yes** | Same register-tiled SIMD kernel + `set_single_wave(True)` across all $F$ frames |
| **Python Numba `v0`** | `--dec-implem PY_V0` (or `--dec-py-ver v0`) | No | Naive `@njit(nogil=True)` allocating `np.zeros(K)` and slicing on the heap every frame |
| **Python Numba `v1`** | `--dec-implem PY_V1` (or `--dec-py-ver v1`) | No | Zero-allocation scalar `@njit(nogil=True)` loops |
| **Python Numba `v2`** | `--dec-implem PY_V2` (or `PY_NUMBA`) | No | Register-tiled 16-lane `<16 x float>` `llvmlite` SIMD intrinsics (1 frame/call) |
| **Python Numba `v3`** | `--dec-implem PY_V3` (or `PY_V4` / `PY_NUMBA_SW`) | **Yes** | Register-tiled 16-lane LLVM SIMD + `set_single_wave(True)` across all $F$ frames |
| **Python NumPy** | `--dec-implem PY_NUMPY` / `PY_NUMPY_SW` | No / **Yes** | Vectorized 3D NumPy array broadcasting (`reshape(W, n_rep, K).sum(axis=1)`) |

#### Examples for `pipeline_repetition.py`

```bash
# 1. Compare C++ SIMD (FAST), Julia SIMD (v4), Python Numba SIMD (PY_V3), and Python NumPy in Sequence mode:
python pipeline_repetition.py --sim-sequence -K 2048 -N 8192 -F 8 -t 4 -e 1000 -m 0.0 -M 4.0 -s 1.0 --dec-implem FAST
python pipeline_repetition.py --sim-sequence -K 2048 -N 8192 -F 8 -t 4 -e 1000 -m 0.0 -M 4.0 -s 1.0 --dec-implem JL_V4
python pipeline_repetition.py --sim-sequence -K 2048 -N 8192 -F 8 -t 4 -e 1000 -m 0.0 -M 4.0 -s 1.0 --dec-implem PY_V3
python pipeline_repetition.py --sim-sequence -K 2048 -N 8192 -F 8 -t 1 -e 1000 -m 0.0 -M 4.0 -s 1.0 --dec-implem PY_NUMPY_SW

# 2. Walk the 5-step Julia optimization ladder (v0 -> v1 -> v2 -> v3 -> v4) with per-task statistics (--sim-stats):
for ver in v0 v1 v2 v3 v4; do
    python pipeline_repetition.py --sim-sequence -K 512 -N 2048 -F 8 -t 4 -e 500 -m 2.0 -M 2.0 --dec-jl --dec-jl-ver $ver --sim-stats
done

# 3. Walk the 4-step Python/Numba optimization ladder (PY_V0 -> PY_V1 -> PY_V2 -> PY_V3):
for ver in v0 v1 v2 v3; do
    python pipeline_repetition.py --sim-sequence -K 512 -N 2048 -F 8 -t 4 -e 500 -m 2.0 -M 2.0 --dec-py-ver $ver --sim-stats
done

# 4. Run in 3-stage multi-threaded Pipeline mode (default) with custom inter-stage buffer size and stats:
python pipeline_repetition.py -K 2048 -N 8192 -F 8 -t 4 -e 1000 -m 0.0 -M 4.0 -s 1.0 \
    --dec-implem JL_V4 --sim-interstage-buf 16 --sim-stats

# 5. Export the 3-stage Pipeline topology to a Graphviz DOT file:
python pipeline_repetition.py -K 32 -N 128 -F 1 -t 4 -e 50 -m 2.0 -M 2.0 --dec-implem JL_V4 --sim-dot rep_pipeline.dot
```

---

### 3.3. `pipeline_ldpc_5g.py` — Polyglot 5G NR UL-SCH QC-LDPC BFER Simulation

[`pipeline_ldpc_5g.py`](pipeline_ldpc_5g.py) is the Python-orchestrated counterpart of [`examples/ldpc_5g_julia`](../ldpc_5g_julia/README.md). It implements a complete **5G NR UL-SCH Quasi-Cyclic LDPC (QC-LDPC)** rate-matched communication chain and supports a **4-axis decoder configuration matrix** across C++ (`AFF3CT`), Julia (`Stateful_Julia`), Python Numba (`Stateful_core`), and Python NumPy.

#### Architecture
$$\text{Source} \longrightarrow \text{Encoder\_LDPC\_from\_QC} \longrightarrow \text{Puncturer\_LDPC} \longrightarrow \text{Modem\_BPSK} \longrightarrow \text{Channel\_AWGN} \longrightarrow [\text{Quantizer\_i16}] \longrightarrow \text{Depuncturer} \longrightarrow \text{Decoder\_LDPC} \longrightarrow \text{Monitor\_BFER}$$

- By default, loads the 5G NR Base Graph 2 matrix `../../lib/aff3ct/conf/dec/LDPC/5G/NR_2_0_64.qc` ($N_{\text{cw}} = 3328$, $M = 2688$, $12608$ edges) punctured/rate-matched to $(N, K)$ (default $K = 512, N = 1024, R = 0.5$). You can pass any QC file via `--dec-h-path <path>`.

#### The 4-Axis LDPC Decoder Configuration Matrix

1. **Axis 1 — Message-Passing Schedule (`-D` / `--dec-type`)**:
   - `BP_FLOODING` (`BP-F`): Two-phase flooding schedule (all Check-to-Variable updates, then all Variable-to-Check updates).
   - `BP_HORIZONTAL_LAYERED` (`BP-HL`): Single-phase horizontal-layered schedule updating posterior LLRs in-place check-node by check-node (converges in ~half the iterations of flooding).
   - `BP_VERTICAL_LAYERED` (`BP-VL`): Vertical-layered schedule (supported by C++ `AFF3CT`).

2. **Axis 2 — Check-Node Update Rule & Language Backend (`--dec-implem`)**:
   - **Update Rules**:
     - **Sum-Product Algorithm (`SPA`)**: Exact box-plus update via $\tanh / \text{atanh}$ products.
     - **Offset Min-Sum (`OMS`)**: Hardware-friendly $\min_1 / \min_2$ approximation with offset (`--dec-off`, default `0.15` for `f32`, `1` for `i16`).
   - **Backends & Progressive Optimization Tiers**:

| Language Backend | Default (= `OPT` Tier) | Tier 1: `NAIVE` (Textbook Spec) | Tier 2: `STD` (Idiomatic Zero-Alloc) | Tier 3: `OPT` (Hardware LLVM SIMD) |
| :--- | :--- | :--- | :--- | :--- |
| **C++ (`AFF3CT`)** | `SPA`, `OMS` (plus `MS`, `NMS`, `LSPA`, `AMS`) | — | — | `SPA`, `OMS` |
| **Julia (`Stateful_Julia`)** | `SPA_JL`, `OMS_JL` | `SPA_JL_NAIVE`, `OMS_JL_NAIVE`<br>([`ldpc_bp_naive.jl`](../ldpc_5g_julia/src/ldpc_bp_naive.jl)) | `SPA_JL_STD`, `OMS_JL_STD`<br>([`ldpc_bp_std.jl`](../ldpc_5g_julia/src/ldpc_bp_std.jl)) | `SPA_JL_OPT`, `OMS_JL_OPT`<br>([`ldpc_bp_opt.jl`](../ldpc_5g_julia/src/ldpc_bp_opt.jl)) |
| **Python Numba (`@njit`)** | `SPA_PY_NUMBA`, `OMS_PY_NUMBA` | `SPA_PY_NAIVE`, `OMS_PY_NAIVE`<br>([`tasks/ldpc_bp_naive.py`](tasks/ldpc_bp_naive.py)) | `SPA_PY_STD`, `OMS_PY_STD`<br>([`tasks/ldpc_bp_std.py`](tasks/ldpc_bp_std.py)) | `SPA_PY_OPT`, `OMS_PY_OPT`<br>([`tasks/ldpc_bp_opt.py`](tasks/ldpc_bp_opt.py)) |
| **Python NumPy** | `SPA_PY_NUMPY`, `OMS_PY_NUMPY` | — | — | — |

3. **Axis 3 — Numeric Precision (`-p` / `--sim-prec`)**:
   - `-p 32` (default): 32-bit floating-point (`float` / `Float32` / `np.float32`).
   - `-p 16`: 16-bit fixed-point (`int16_t` / `Int16` / `np.int16`). Automatically inserts `Quantizer_pow2_fast_i16` (`--qnt-bits 8`, `--qnt-dec 2`) before depuncturing and uses saturating `int16` arithmetic (`[-sat_val, +sat_val]`, hardware `vqaddq_s16` / `vqsubq_s16` / `llvm.sadd.sat.v16i16` in `OPT`).

4. **Axis 4 — Vectorization Strategy (`--dec-simd`)**:
   - **Scalar (default, `-F 1`)**: Decodes 1 codeword per kernel call (with intra-node SIMD in `OPT`).
   - **Inter-Frame SIMD (`--dec-simd INTER`)**: Decodes a wave of $W = F$ codewords in lockstep SIMD vectors (`-F 8` for `-p 32`, `-F 16` for `-p 16`).

#### Examples for `pipeline_ldpc_5g.py`

```bash
# 1. BP-Flooding + SPA (32-bit float, Scalar, F=1) across C++, Julia, and Python Numba:
python pipeline_ldpc_5g.py --sim-sequence -K 512 -N 1024 -i 10 -F 1 -t 4 -e 50 -m 1.5 -M 2.5 -s 0.5 -D BP_FLOODING -p 32 --dec-implem SPA
python pipeline_ldpc_5g.py --sim-sequence -K 512 -N 1024 -i 10 -F 1 -t 4 -e 50 -m 1.5 -M 2.5 -s 0.5 -D BP_FLOODING -p 32 --dec-implem SPA_JL
python pipeline_ldpc_5g.py --sim-sequence -K 512 -N 1024 -i 10 -F 1 -t 4 -e 50 -m 1.5 -M 2.5 -s 0.5 -D BP_FLOODING -p 32 --dec-implem SPA_PY_NUMBA

# 2. BP-Horizontal-Layered + OMS (32-bit float, Inter-Frame SIMD W=8) across C++, Julia, and Python Numba:
python pipeline_ldpc_5g.py --sim-sequence -K 512 -N 1024 -i 10 -F 8 --dec-simd INTER -t 4 -e 100 -m 1.5 -M 2.5 -s 0.5 -D BP_HORIZONTAL_LAYERED -p 32 --dec-implem OMS
python pipeline_ldpc_5g.py --sim-sequence -K 512 -N 1024 -i 10 -F 8 --dec-simd INTER -t 4 -e 100 -m 1.5 -M 2.5 -s 0.5 -D BP_HORIZONTAL_LAYERED -p 32 --dec-implem OMS_JL
python pipeline_ldpc_5g.py --sim-sequence -K 512 -N 1024 -i 10 -F 8 --dec-simd INTER -t 4 -e 100 -m 1.5 -M 2.5 -s 0.5 -D BP_HORIZONTAL_LAYERED -p 32 --dec-implem OMS_PY_NUMBA

# 3. BP-Horizontal-Layered + OMS (16-bit fixed-point Int16, Inter-Frame SIMD W=16) across C++, Julia, and Python Numba:
python pipeline_ldpc_5g.py --sim-sequence -K 512 -N 1024 -i 10 -F 16 --dec-simd INTER -t 4 -e 100 -m 1.5 -M 2.5 -s 0.5 -D BP_HORIZONTAL_LAYERED -p 16 --dec-implem OMS
python pipeline_ldpc_5g.py --sim-sequence -K 512 -N 1024 -i 10 -F 16 --dec-simd INTER -t 4 -e 100 -m 1.5 -M 2.5 -s 0.5 -D BP_HORIZONTAL_LAYERED -p 16 --dec-implem OMS_JL
python pipeline_ldpc_5g.py --sim-sequence -K 512 -N 1024 -i 10 -F 16 --dec-simd INTER -t 4 -e 100 -m 1.5 -M 2.5 -s 0.5 -D BP_HORIZONTAL_LAYERED -p 16 --dec-implem OMS_PY_NUMBA

# 4. Compare the 3 progressive optimization tiers (NAIVE -> STD -> OPT) in Julia and Python/Numba:
for imp in OMS_JL_NAIVE OMS_JL_STD OMS_JL_OPT OMS_PY_NAIVE OMS_PY_STD OMS_PY_OPT; do
    python pipeline_ldpc_5g.py --sim-sequence -K 512 -N 1024 -i 10 -F 16 --dec-simd INTER -t 4 -e 100 \
        -m 2.0 -M 2.0 -D BP_HORIZONTAL_LAYERED -p 16 --dec-implem $imp --sim-stats
done

# 5. Multi-threaded 3-stage Pipeline execution with per-stage task statistics:
python pipeline_ldpc_5g.py -K 512 -N 1024 -i 10 -F 16 --dec-simd INTER -t 4 -e 100 \
    -m 1.5 -M 2.5 -s 0.5 -D BP_HORIZONTAL_LAYERED -p 16 --dec-implem OMS_JL --sim-stats
```
