#!/usr/bin/env bash
set -euo pipefail
dev=0
role=draft
backend=default
gpu=()
for argument in "$@"; do
  case "$argument" in
    --dev) dev=1 ;;
    --target) role=target ;;
    --megatron) backend=megatron ;;
    --gpu) gpu=(--gpu) ;;
    --help)
      echo 'Usage: bash tools/install.sh [--target] [--megatron] [--dev] [--gpu]'
      echo 'For an already activated isolated venv. New users: bash setup.sh'
      exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$argument" >&2; exit 2 ;;
  esac
done
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
bash tools/run_in_env.sh true
command -v git >/dev/null || { echo 'Install Git before setup.' >&2; exit 2; }
python - <<'CHECK'
import ctypes
import platform
name, version = platform.libc_ver()
if name != 'glibc' or tuple(map(int, version.split('.')[:2])) < (2, 34):
    raise SystemExit('Pinned SGLang wheels require glibc >= 2.34; use a newer Linux base image.')
if platform.machine() != 'x86_64':
    raise SystemExit('The pinned runtime requires Linux x86-64.')
for name in ('libibverbs.so.1', 'libmlx5.so.1'):
    try:
        ctypes.CDLL(name)
    except OSError as error:
        raise SystemExit(f'{name}: {error}. Install rdma-core runtime libraries '
                         '(Ubuntu: libibverbs1 libmlx5-1), or use the container.') from error
CHECK
if ! printf '#include <bit>\nint main() { return std::bit_cast<unsigned>(1.0f) == 0; }\n' |
  "${CXX:-c++}" -std=c++20 -x c++ -fsyntax-only -; then
  echo 'SGLang JIT needs a C++20 compiler and matching standard library.' >&2
  echo 'Install a modern C++ toolchain (Ubuntu: g++), or set CXX to one; see installation-advanced.md.' >&2
  exit 2
fi
manifest=requirements/draft.txt
check=()
if [[ $role == target ]]; then
  manifest=requirements/sglang-v0.5.21.txt
  check=(--target)
fi
extras="."
(( dev == 0 )) || extras=".[dev]"
if [[ $backend == megatron ]]; then
  python -c 'import sys; sys.exit("Megatron requires Python >=3.12" if sys.version_info[:2] < (3,12) else 0)'
  manifest=requirements/megatron-v0.20.0.txt
  check+=(--backend megatron)
  # Build TE against the actual execution Torch, including its native headers.
  # An unconstrained isolated build can select a different Torch ABI.
  python -m pip install -r requirements/megatron-build.txt
  aspec_nvidia=$(python -c 'import pathlib,sysconfig; print(pathlib.Path(sysconfig.get_paths()["purelib"]) / "nvidia")')
  # The transport also needs CUDA 12's runtime. Including every NVIDIA wheel
  # would put its headers ahead of CUDA 13 and break TE's C++ compilation.
  CUDA_HOME="$aspec_nvidia/cu13" \
    NVTE_CUDA_INCLUDE_PATH="$aspec_nvidia/cu13/include" \
    CPLUS_INCLUDE_PATH="$aspec_nvidia/cudnn/include:$aspec_nvidia/nccl/include:${CPLUS_INCLUDE_PATH:-}" \
    NVTE_BUILD_USE_NVIDIA_WHEELS=0 MAX_JOBS="${MAX_JOBS:-4}" \
    python -m pip install --no-build-isolation -r "$manifest" -e "$extras"
  printf 'megatron\n' > "$VIRTUAL_ENV/.aspec-backend"
else
  python -m pip install -r "$manifest" -e "$extras"
fi
printf '%s\n' "$role" > "$VIRTUAL_ENV/.aspec-role"
if [[ $role == target && $backend != megatron ]]; then
  aspec patch-sglang
  aspec patch-sglang --check
fi
aspec doctor "${check[@]}" "${gpu[@]}"
