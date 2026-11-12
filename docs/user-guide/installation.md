# Installation

Use Linux x86-64, Python >=3.11 and an isolated virtual environment. Training requires NVIDIA
GPUs, a CUDA-compatible driver and a CUDA development toolkit with `nvcc`.
See [native prerequisites](installation-advanced.md#requirements) for the compiler, glibc and RDMA
library requirements, or use the [container](containers.md).

## Install from PyPI

```bash
python -m venv .venv
source .venv/bin/activate
pip install ling-aspec
```

This installs ASpec and its training dependencies. The Python package and command are named
`aspec`. Keep the unrelated PyPI package `aspec` out of this environment.

## Target nodes

On nodes that run the Target model:

```bash
aspec patch-sglang
```

The command installs the compatible SGLang runtime if it is missing, then applies the training
patch. Runtime versions and compatibility overrides are shipped with ASpec; no manual dependency
selection is needed. See [advanced installation](installation-advanced.md#dedicated-environment)
for the override details.

Recheck the patch after reinstalling SGLang:

```bash
aspec patch-sglang --check
```

Applying the patch again verifies an already patched installation. Conflicting changes or an
unsupported SGLang version are rejected. `--check` never installs dependencies or modifies files.

## Check the environment

Optionally run `aspec doctor` to check the training environment, or `aspec doctor --target` to
also require the Target runtime and patch. Add `--gpu` to check CUDA and native libraries on an
allocated GPU. These checks do not run training.

## First run

Quickstart requires two NVLink-connected GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 aspec quickstart ling-tiny --output outputs/ling-tiny-001
# Or:
CUDA_VISIBLE_DEVICES=0,1 aspec quickstart qwen3-8b --output outputs/qwen3-8b-001
```

Add `--target /absolute/path/to/model` to use a local checkpoint.
Continue with [Quickstart](quickstart.md) or choose a [training recipe](recipes.md).

## Install from source

From a source checkout:

```bash
bash setup.sh
source .venv/bin/activate
```

Setup installs the runtime, applies the patch and runs doctor. Contributors can use
`bash setup.sh --dev`. Run `bash setup.sh --help` for environment and interpreter options.

## Install a release wheel

Replace `ling-aspec` in the PyPI install command with the path to the downloaded wheel.
Then prepare Target nodes and optionally check the environment as above. The installed commands,
recipes and Quickstart data work outside the source checkout.

See [advanced installation](installation-advanced.md) for native-library paths and manual
dependency management, or [troubleshooting](troubleshooting.md) if a check fails.
