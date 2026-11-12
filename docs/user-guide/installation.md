# Installation

Use Linux x86-64, Python >=3.11 and an isolated virtual environment. Training requires NVIDIA
GPUs and a CUDA 13-compatible driver. The Target package supplies its matching CUDA compiler;
the host still needs the C++20 compiler and native libraries listed below.
See [native prerequisites](installation-advanced.md#requirements) for the compiler, glibc and RDMA
library requirements, or use the [container](containers.md).

## Install from PyPI

```bash
python -m venv .venv
source .venv/bin/activate
pip install ling-aspec
```

This installs ASpec and its Draft training dependencies. The Python package and command are named
`aspec`. Keep the unrelated PyPI package `aspec` out of this environment.

## Target nodes

In a separate virtual environment on nodes that run the Target model:

```bash
python -m venv .venv-target
source .venv-target/bin/activate
pip install ling-aspec-target
aspec patch-sglang
```

Target dependencies follow SGLang's requirements; Draft acceleration packages stay in the Draft
venv. The patch command only changes the supported SGLang source files, never dependencies.

In the Draft environment, point training at the Target interpreter:

```bash
export ASPEC_TARGET_PYTHON=/absolute/path/to/.venv-target/bin/python
```

Alternatively set `inference.target_python` in the training config, or pass `--target-python`
to Quickstart. Target and Draft use the same Python minor version and Ray version. On multiple Target nodes,
the configured absolute path must exist on each node.

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

Setup installs the Draft environment. Run `bash setup.sh --target` to prepare `.venv-target`,
then set `ASPEC_TARGET_PYTHON` as above. Each setup runs the appropriate doctor. Contributors can use
`bash setup.sh --dev`. Run `bash setup.sh --help` for environment and interpreter options.

## Install a release wheel

Use `pip install --find-links /path/to/release ling-aspec` in Draft, and
`pip install --find-links /path/to/release ling-aspec-target` in Target. Each role installs
the same `ling-aspec-runtime` wheel, which owns the `aspec` code and CLI. Then prepare Target nodes and optionally check the environment as above. The installed commands,
recipes and Quickstart data work outside the source checkout.

See [advanced installation](installation-advanced.md) for native-library paths and manual
dependency management, or [troubleshooting](troubleshooting.md) if a check fails.
