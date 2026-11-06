# Polyglot Python + Julia + C++ Examples (`python_polyglot`)

This directory contains polyglot communication chain examples orchestrated from **Python** via [`pyStreamPU`](../../lib/pystreampu) that mix:
- **C++ modules** from **AFF3CT** (exposed through a lightweight `pybind11` module `py_aff3ct`),
- **Julia modules** (`Stateless_Julia` / `Stateful_Julia`) executed in-process via the native `libjulia` C API (`unsafe_wrap` zero-copy views),
- **Python modules** (`Stateful_core` with NumPy or Numba `@njit`),

all sharing a **single `StreamPU` installation** (`aff3ct_install/`) and communicating through zero-copy pointer handoff (`sck::status::forward`).

---

## 1. Prerequisites

Ensure the following tools are installed and available in your `PATH`:
- A C++17 compiler (`clang++` or `g++`), `cmake` (>= 3.15), and `git`
- **Julia** (>= 1.10) — no third-party Julia packages are required (the kernels only use Julia `Base` and `Base.Cartesian`)
- **Python** (>= 3.10) with `venv` support

Initialize the Git submodules from the root of `my_project_with_aff3ct`:

```bash
git submodule update --init --recursive
```

---

## 2. Step 1: Compile and Install AFF3CT & StreamPU (Single Shared Instance)

Compile `AFF3CT` with **both** `AFF3CT` and `StreamPU` libraries enabled (`-DSPU_COMPILE_STATIC_LIB="ON" -DSPU_COMPILE_SHARED_LIB="ON"`) and linked against `libjulia` (`-DSPU_LINK_JULIA="ON"`). Installing into `aff3ct_install/` exports both `aff3ct-config.cmake` and `streampu-config.cmake` so that `AFF3CT`, `pyStreamPU`, and `py_aff3ct` all link against the **exact same `StreamPU` library**:

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

---

## 3. Step 2: Create the Python Virtual Environment and Install `pyStreamPU`

Create a Python virtual environment, install dependencies (`numpy`, `numba`, `pybind11`, `scikit-build-core`), and compile `lib/pystreampu` with `INCLUDE_SPU_LIB=OFF` so that it reuses the `StreamPU` and `cpptrace` libraries installed in `aff3ct_install/`:

```bash
# From the root of my_project_with_aff3ct:
cd lib/pystreampu
python3 -m venv venv-julia
source venv-julia/bin/activate
pip install --upgrade pip
pip install numpy numba pybind11 scikit-build-core

export CMAKE_BUILD_PARALLEL_LEVEL=8
AFF3CT_INST="$(cd ../../aff3ct_install && pwd)"

pip install --no-build-isolation -v . \
    -C cmake.define.INCLUDE_SPU_LIB=OFF \
    -C cmake.define.StreamPU_DIR="${AFF3CT_INST}/lib/cmake/streampu" \
    -C cmake.define.cpptrace_DIR="${AFF3CT_INST}/lib/cmake/cpptrace"
cd ../..
```

---

## 4. Step 3: Compile the `py_aff3ct` Wrapper for the Examples

With the `venv-julia` virtual environment still active, build the `py_aff3ct` extension module in `examples/python_polyglot/build/`:

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

---

## 5. Step 4: Run the Polyglot Python Scripts

Make sure your virtual environment (`lib/pystreampu/venv-julia`) is active before running the scripts from `examples/python_polyglot/`:

```bash
source ../../lib/pystreampu/venv-julia/bin/activate
```

### 5.1. Repetition Code Chain (`pipeline_julia.py`)

Equivalent to `examples/pipeline_julia`, orchestrated from Python. It supports:
- **C++ decoders**: `--dec-implem FAST` (SIMD `mipp`), `--dec-implem STD` (scalar C++)
- **Julia decoders** (`Stateless_Julia` in `../pipeline_julia/src/rep_decoder.jl`): `--dec-jl --dec-jl-ver v0|v1|v2|v3|v4` (or `--dec-implem JL_V0..JL_V4`)
- **Python decoders** (`Stateful_core`): `--dec-implem PY_NUMPY` / `PY_NUMPY_SW` (vectorized NumPy), `--dec-implem PY_NUMBA` / `PY_NUMBA_SW` (`@njit` LLVM JIT)
- **Execution modes**: multi-threaded `Pipeline` (default) or multi-threaded `Sequence` (`--sim-sequence`)

```bash
# Julia register-tiled SIMD single-wave decoder (v4) in 3-stage Pipeline mode (4 threads in stage 1):
python pipeline_julia.py -K 32 -N 128 -F 1 -t 4 -e 100 -m 0.0 -M 10.0 -s 1.0 --dec-jl --dec-jl-ver v4

# Julia register-tiled SIMD single-wave decoder (v4) in single-stage Sequence mode (4 threads, inter-frame = 8):
python pipeline_julia.py --sim-sequence -K 2048 -N 8192 -F 8 -t 4 -e 1000 -m 0.0 -M 5.0 -s 1.0 --dec-jl --dec-jl-ver v4

# C++ SIMD decoder (FAST) in Sequence mode:
python pipeline_julia.py --sim-sequence -K 2048 -N 8192 -F 8 -t 4 -e 1000 -m 0.0 -M 5.0 -s 1.0 --dec-implem FAST

# Python NumPy decoder in Sequence mode (1 thread due to Python GIL):
python pipeline_julia.py --sim-sequence -K 2048 -N 8192 -F 8 -t 1 -e 1000 -m 0.0 -M 5.0 -s 1.0 --dec-implem PY_NUMPY_SW

# Python Numba @njit decoder in Sequence mode:
python pipeline_julia.py --sim-sequence -K 2048 -N 8192 -F 8 -t 4 -e 1000 -m 0.0 -M 5.0 -s 1.0 --dec-implem PY_NUMBA_SW
```

### 5.2. 5G NR / QC-LDPC Chain (`ldpc_5g_julia.py`)

Equivalent to `examples/ldpc_5g_julia`, orchestrated from Python. It supports:
- **Code families**: `--enc-type LDPC_5G` (3GPP 5G NR Base Graph 1 & 2) or `--enc-type LDPC_QC` (Quasi-Cyclic LDPC from `.qc` file via `--dec-h-path`)
- **Decoders**:
  - `--dec-implem SPA_JL` (or `--dec-jl`): `Stateful_Julia` Sum-Product Algorithm in `../ldpc_5g_julia/src/ldpc_bp_flooding_spa.jl`
  - `-D BP_FLOODING --dec-implem SPA`: AFF3CT C++ Flooding SPA
  - `-D BP_FLOODING --dec-implem MS`: AFF3CT C++ Flooding Min-Sum
  - `-D BP_HORIZONTAL_LAYERED --dec-implem SPA`: AFF3CT C++ Layered SPA
  - `-D BP_HORIZONTAL_LAYERED --dec-implem MS`: AFF3CT C++ Layered Min-Sum
  - `-D BP_HORIZONTAL_LAYERED --dec-implem NMS`: AFF3CT C++ Layered Normalized Min-Sum
  - `-D BP_HORIZONTAL_LAYERED --dec-implem OMS`: AFF3CT C++ Layered Offset Min-Sum
  - `-D BP_FLOODING --dec-implem AMS`: AFF3CT C++ Flooding Approximate Min-Star
- **Execution modes**: multi-threaded `Pipeline` (default) or multi-threaded `Sequence` (`--sim-sequence`)

```bash
# Julia SIMD SPA decoder (--dec-implem SPA_JL) on 5G NR LDPC (K=512, N=1024, i=10) in Sequence mode:
python ldpc_5g_julia.py --sim-sequence --enc-type LDPC_5G -K 512 -N 1024 -i 10 -F 8 -t 4 -e 100 -m 1.0 -M 2.5 -s 0.5 --dec-implem SPA_JL

# C++ AFF3CT SPA decoder (--dec-implem SPA) on the exact same 5G NR LDPC code in Sequence mode:
python ldpc_5g_julia.py --sim-sequence --enc-type LDPC_5G -K 512 -N 1024 -i 10 -F 8 -t 4 -e 100 -m 1.0 -M 2.5 -s 0.5 --dec-implem SPA

# Julia SIMD SPA decoder in 3-stage Pipeline mode:
python ldpc_5g_julia.py --enc-type LDPC_5G -K 512 -N 1024 -i 10 -F 1 -t 4 -e 100 -m 1.0 -M 2.5 -s 0.5 --dec-implem SPA_JL
```

