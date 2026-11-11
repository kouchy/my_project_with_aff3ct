#!/bin/bash
# Make AFF3CT build the StreamPU version shipped with pyStreamPU (lib/pystreampu/lib/streampu), without updating
# (nor committing) the AFF3CT submodule itself.
#
#   ./scripts/use_pystreampu_streampu.sh          # align lib/aff3ct/lib/streampu on lib/pystreampu/lib/streampu
#   ./scripts/use_pystreampu_streampu.sh --check  # only check that both point to the same commit (exit 1 otherwise)
#
# pyStreamPU (`streampu`) and `py_aff3ct` exchange C++ objects in the same Python process: both must be built
# against the exact same StreamPU (installed in aff3ct_install/ by the AFF3CT build, INCLUDE_SPU_LIB=OFF).
#
# Notes:
# - this is a local checkout in lib/aff3ct/lib/streampu (detached HEAD): `git status` of this repository then shows
#   `lib/aff3ct` as modified ("m", new commits in a nested submodule). Do NOT commit `lib/aff3ct` for that.
# - `git submodule update --recursive` restores the commit recorded by AFF3CT: run this script again afterwards,
#   then rebuild/reinstall AFF3CT (aff3ct_install/), pyStreamPU and py_aff3ct.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SRC=lib/pystreampu/lib/streampu
DST=lib/aff3ct/lib/streampu
want="$(git -C "${SRC}" rev-parse HEAD)"
have="$(git -C "${DST}" rev-parse HEAD)"

if [[ "${1:-}" == "--check" ]]; then
    if [[ "${want}" == "${have}" ]]; then
        echo "OK: AFF3CT and pyStreamPU use the same StreamPU (${want:0:9})"
        exit 0
    fi
    echo "MISMATCH: pyStreamPU StreamPU ${want:0:9} != AFF3CT StreamPU ${have:0:9} (run $0)" >&2
    exit 1
fi

if [[ "${want}" == "${have}" ]]; then
    echo "Already aligned: ${DST} is at ${want:0:9}"
    exit 0
fi
if [[ -n "$(git -C "${DST}" status --porcelain --untracked-files=no)" ]]; then
    echo "error: ${DST} has local modifications, commit/stash them first:" >&2
    git -C "${DST}" status --short --untracked-files=no >&2
    exit 1
fi
echo "Aligning ${DST}: ${have:0:9} -> ${want:0:9} (from ${SRC})"
git -C "${DST}" fetch --quiet "$(cd "${SRC}" && pwd)" "${want}"
git -C "${DST}" checkout --quiet "${want}"
git -C "${DST}" submodule update --init --recursive --quiet
echo "Done: ${DST} is at $(git -C "${DST}" log --oneline -1)"
echo "Rebuild and reinstall AFF3CT (aff3ct_install/), then pyStreamPU and py_aff3ct (examples/python_polyglot/README.md)."

