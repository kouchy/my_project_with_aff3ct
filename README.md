# Using AFF3CT as a library for your codes

[![pipeline status](https://gitlab.com/aff3ct/my_project_with_aff3ct/badges/master/pipeline.svg)](https://gitlab.com/aff3ct/my_project_with_aff3ct/pipelines)

This repository contains some simple code examples. It helps to understand how 
to use the AFF3CT library in your code. The first step is to compile AFF3CT into 
a library.

Get the AFF3CT library:

```bash
git submodule update --init --recursive
```

Compile the library on Linux/MacOS/MinGW/WSL:

```bash
cd lib/aff3ct
mkdir build
cd build
cmake .. -G"Unix Makefiles" -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_FLAGS="-Wall -funroll-loops -march=native" \
           -DAFF3CT_COMPILE_EXE="OFF" -DAFF3CT_COMPILE_STATIC_LIB="ON" -DAFF3CT_COMPILE_SHARED_LIB="ON"
cmake --build . -j 10
cmake --install . --prefix ../../../aff3ct_install
ln -sfn ../../../aff3ct_install/lib/cmake/aff3ct-* ../../../aff3ct_install/lib/cmake/aff3ct
```

Now the AFF3CT library has been built in the `lib/aff3ct/build` folder.

The source codes of the examples are in the `examples/` folder.
You can go in this folder to see the next steps.

> [!note]
> Those examples are documented 
> [here](https://aff3ct.readthedocs.io/en/latest/user/library/examples.html).

> [!note]
> If you want to compile Julia examples (or the polyglot Python + Julia + C++
> examples in [`examples/python_polyglot/`](examples/python_polyglot/README.md)),
> AFF3CT must be built with the `StreamPU` version shipped in
> `lib/pystreampu/lib/streampu`, which is newer than the one recorded in the
> AFF3CT submodule. The AFF3CT submodule itself is **not** updated: only its
> `lib/streampu` working copy is moved to the pyStreamPU commit:
> ```bash
> git submodule update --init --recursive
> ./scripts/use_pystreampu_streampu.sh          # align lib/aff3ct/lib/streampu on lib/pystreampu/lib/streampu
> ./scripts/use_pystreampu_streampu.sh --check  # verify (also useful before every rebuild)
> ```
> `git status` then shows `lib/aff3ct` as modified (new commit in its nested
> `lib/streampu`): this is expected, do **not** commit `lib/aff3ct` for that.
> Any later `git submodule update --recursive` restores the old StreamPU commit:
> run the script again and rebuild AFF3CT, pyStreamPU and `py_aff3ct`.
>
> Then explicitly link StreamPU with `libjulia` (`-DSPU_LINK_JULIA="ON"`) and
> `hwloc` (`-DSPU_LINK_HWLOC="ON"`, required by the thread pinning options:
> without it, pinning is silently a no-op), and install the StreamPU library
> alongside AFF3CT so that both `AFF3CT` and `pyStreamPU` share the exact same
> `StreamPU` instance:
> ```bash
> cmake .. -G"Unix Makefiles" -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_FLAGS="-Wall -funroll-loops -march=native" \
>           -DCMAKE_POSITION_INDEPENDENT_CODE="ON" \
>           -DAFF3CT_COMPILE_EXE="OFF" -DAFF3CT_COMPILE_STATIC_LIB="ON" -DAFF3CT_COMPILE_SHARED_LIB="ON" \
>           -DSPU_COMPILE_STATIC_LIB="ON" -DSPU_COMPILE_SHARED_LIB="ON" -DSPU_LINK_JULIA="ON" -DSPU_LINK_HWLOC="ON"
> ```
> See [`examples/python_polyglot/README.md`](examples/python_polyglot/README.md) for the complete instructions to build `lib/pystreampu` (`INCLUDE_SPU_LIB=OFF`) and run the Python polyglot benchmarks.

> [!warning]
> If you are encountering some problems with
> [`cpptrace`](https://github.com/jeremy-rifkin/cpptrace) (typically when you 
> try to compile an example), you can simply disable it by adding 
> `-DSPU_STACKTRACE="OFF"` in the previous `cmake` command line as follow:
> ```bash
> cmake .. -G"Unix Makefiles" -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_FLAGS="-Wall -funroll-loops -march=native" \
>           -DAFF3CT_COMPILE_EXE="OFF" -DAFF3CT_COMPILE_STATIC_LIB="ON" -DAFF3CT_COMPILE_SHARED_LIB="ON" \
>           -DSPU_STACKTRACE="OFF"
> ```
