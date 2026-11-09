# Share a Target across Drafts

Shared mode captures compatible Target features once and trains independent Draft experiments.
Each experiment owns its weights, optimizer, LR, RNG, evaluation, training progress, checkpoint and
tracking output. Shared execution supports one node or a static homogeneous multi-node cluster.

The [comparison and independent recovery walkthrough](../../examples/qwen3-8b-shared-target/README.md)
derives two learning-rate experiments from the shared-target recipe, records separate
TensorBoard curves and resumes from unequal checkpoints while one Draft is already complete.

Use this workflow to compare Draft algorithms, structures or hyperparameters for the same Target.
Algorithm selection chooses what each experiment trains; shared execution lets several such
experiments run concurrently with compatible Target capture generated once. Each keeps its own
training state and progress rather than participating in one Draft's data-parallel optimizer step.

| Example | Purpose | Starting GPU budget |
| --- | --- | --- |
| `configs/recipes/qwen3_8b_4k_multi_draft_nvlink.yaml` | Two independent DSpark experiments | 4 Target + 4 Draft |
| `configs/recipes/qwen3_8b_4k_multinode_multi_draft_nvlink.yaml` | Two independent cross-node DSpark experiments | 2 nodes, each 4 Target + 2 GPUs per Draft |
| `configs/examples/qwen3_8b_multi_algorithm_feature_views.yaml` | Independent DSpark/EAGLE3/DFlash feature views and packing/CP | 1 Target + 7 Draft |
| `configs/examples/ling3_flash_multi_structure_dspark.yaml` | Dense, Qwen MoE and Ling MoE Drafts | TP2 Target + 3 Draft |
| `configs/examples/ling3_flash_swa_compare_dspark.yaml` | Full context versus SWA | TP2 Target + 2 Draft |

Supply training and held-out evaluation data and inspect each Draft's Plan before starting.
The single-node Qwen 4K recipe uses `Qwen/Qwen3-8B` by default; set `ASPEC_TARGET_MODEL`
to use a local copy, and provide `ASPEC_TRAIN_DATA` and `ASPEC_EVAL_DATA`.
The multi-node recipe requires local model and data replicas, as described below.
The Ling example uses bundled sample conversations:

```bash
aspec --config configs/examples/ling3_flash_multi_structure_dspark.yaml --dry-run \
  model.target_model_path=/path/to/Ling-3.0-flash
RAY_ADDRESS=local aspec --config configs/examples/ling3_flash_multi_structure_dspark.yaml \
  model.target_model_path=/path/to/Ling-3.0-flash
aspec --report outputs/ling3-flash-multi-structure
aspec --report outputs/ling3-flash-multi-structure/dense
```

`experiments` entries override Draft-specific model/training/logging settings. Feature sharing checks
actual Target, normalization and dtype, not architecture names. Capture can take a compatible layer
union while each consumer reads its ordered view. A receiver may need a compact tensor copy;
shared capture is not a blanket zero-copy claim. Native MTP's post-norm feature cannot substitute
for a regular auxiliary layer.

The Qwen algorithm example covers both shared capture and independent feature views. To compare
identical views, select the full-layer EAGLE3 and DFlash model configs in its experiment entries and
set DFlash's target-layer count to five. Adjust per-child packing and CP in those same entries;
each child Plan still validates its own execution requirements.

## Progress, resources and failure

Experiments progress independently without a step barrier. Slow experiments limit feature production
to keep queues bounded. When an experiment finishes, is cancelled or fails, its trainer processes
stop reading before shared features are released. Other experiments can continue, but a failed or
cancelled experiment makes the session exit nonzero.
Reports distinguish shared source counts and each experiment's committed work. Do not add child
consumption counts and label the sum unique Target production.

Feature and transport status is finalized when the shared session ends. Check the session and
experiment reports after all experiments finish.

Use each Draft's report to inspect training progress, evaluation and resource use. Its throughput
depends on shared Target supply and queue pressure. Root wall time and GPU budget describe the
whole session. Compare experiments using the same held-out evaluation metric rather than their
algorithm-specific training losses.

Set `training.num_train_steps` for each Draft's update limit. Shared mode supplies documents until
those limits are reached. `training.source_work_budget` is available only for single-Draft runs.

## Resume and tracking

Copy the original saved `config.yaml` outside the run directory and edit each existing
`experiments[].training.load_path` to that child's complete checkpoint. Keep the full experiment
list, model/data/ordering, LR horizon and execution semantics. Current dotlist overrides do not
support patching an individual list index; replacing the list with partial entries can lose fields.

Set both `training.continual_training=false` and `training.continue_source_cursor=false`
for every child, keep the full `lr_total_steps`, choose a fresh root output
directory, and set `num_train_steps` to the intended total frontier. Children may resume from different
committed frontiers. One child's progress never licenses skipping another child's work.

## Static multi-node sharing

Prepare each node and start a task-owned cluster using [multi-node setup](multinode.md).
The multi-Draft recipe uses the same portable model/data/output environment variables and
independent local filesystems. Run it through the normal configuration entry point:

```bash
aspec --config configs/recipes/qwen3_8b_4k_multinode_multi_draft_nvlink.yaml --dry-run
aspec --config configs/recipes/qwen3_8b_4k_multinode_multi_draft_nvlink.yaml
```

All experiments use the same ordered training nodes and Target allocation. Each experiment's
`training_num_gpus_per_node` reserves its own GPUs on every node. GPU budgets add across experiments;
the Target allocation is shared. Each Draft synchronizes gradients within its own cross-node group.
Model-parallel groups and required NVLink feature reads stay node-local.

Each Draft can enable `training.dflash_packing` independently. Packed rows contain complete documents
from the same producer node, with document boundaries and local positions preserved. The Target still
captures one original document per request (`inference_batch_size=1`); packing happens in each Draft’s
controller. Supply follows each Draft’s remaining optimizer steps and node-local queue space.

Multi-node sharing requires TransferEngine and `feature_affinity=node_local_required`.
The normal launch preflights each child and the aggregate allocation. Use `--preflight-only`
with a fresh output directory for a separate check.
The `multinode-shared` recipe launcher also provides a complete training, recovery and export workflow.

### Start two Drafts and resume from different checkpoints

Run the following from the coordinator's installed checkout after completing
[node preparation and cluster startup](multinode.md#prepare-each-node-independently).
Keep the owned cluster running across both segments. Set the input environment from
[multi-node configuration](multinode.md#configure-preflight-and-train), including
`ASPEC_NODE_0_IP`, `ASPEC_NODE_1_IP`, `ASPEC_TARGET_MODEL`, `ASPEC_DRAFT_CONFIG`,
`ASPEC_TRAIN_DATA`, `ASPEC_EVAL_DATA`, `ASPEC_CACHE_DIR` and an explicit `RAY_ADDRESS`.
Model, data and Draft JSON replicas use the same absolute logical paths on both nodes.
Reserve eight GPUs per node: four Target GPUs, two for Draft A and two for Draft B.

This example gives A an initial limit of 32 optimizer steps and B a limit of 33, with a full
LR horizon of 64 from the beginning. The recipe's accumulation of 64 remains in effect:
with DP4 and micro-batch 1, each Draft's global batch is 256 packed rows. The number of original
documents per row varies with sequence length. Supply enough usable supervised documents for
complete optimizer batches. The source repeats the dataset as needed to finish the configured
optimizer steps. Without an explicit step limit, the horizon is derived from `num_epochs` and
the dataset size divided by the global batch; with packing, this is not a fixed number of
original-document passes.

Create a complete first-segment configuration without editing the recipe:

```bash
export ASPEC_OUTPUT_DIR=/data/aspec/runs/compare-first
python - <<'PY'
from pathlib import Path
from omegaconf import OmegaConf

config = OmegaConf.load("configs/recipes/qwen3_8b_4k_multinode_multi_draft_nvlink.yaml")
config.training.num_train_steps = 33
config.training.lr_total_steps = 64
config.training.continual_training = False
config.training.continue_source_cursor = False
config.training.save_interval = 16
config.dataset.eval_interval = 16
config.logging.report_to = "tensorboard"
limits = {"draft_a": 32, "draft_b": 33}
for experiment in config.experiments:
    experiment.training.num_train_steps = limits[experiment.name]
path = Path("run-configs/compare-first.yaml")
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("x") as stream:
    OmegaConf.save(config, stream, resolve=True)
PY
bash tools/run_in_env.sh aspec --config run-configs/compare-first.yaml --dry-run
bash tools/run_in_env.sh aspec --config run-configs/compare-first.yaml
aspec --report /data/aspec/runs/compare-first
```

Wait for both experiments and the shared session to finish successfully. On the coordinator,
`compare-first/draft_a/checkpoints` and `compare-first/draft_b/checkpoints` then hold the independently
published complete checkpoints. Derive the resumed configuration from the session's saved
`config.yaml`, preserving every experiment entry:

```bash
python - <<'PY'
from pathlib import Path
from omegaconf import OmegaConf

first = Path("/data/aspec/runs/compare-first")
resumed = Path("/data/aspec/runs/compare-resumed")
config = OmegaConf.load(first / "config.yaml")
assert config.training.lr_total_steps == 64
config.output_dir = str(resumed)
config.training.num_train_steps = 64
config.training.continual_training = False
config.training.continue_source_cursor = False
for experiment in config.experiments:
    experiment.training.load_path = str(first / experiment.name / "checkpoints")
    experiment.training.num_train_steps = 64
    experiment.training.continual_training = False
    experiment.training.continue_source_cursor = False
with Path("run-configs/compare-resumed.yaml").open("x") as stream:
    OmegaConf.save(config, stream, resolve=True)
PY
bash tools/run_in_env.sh aspec --config run-configs/compare-resumed.yaml --dry-run
bash tools/run_in_env.sh aspec --config run-configs/compare-resumed.yaml
aspec --report /data/aspec/runs/compare-resumed
aspec --report /data/aspec/runs/compare-resumed/draft_a
aspec --report /data/aspec/runs/compare-resumed/draft_b
```

The new YAML contains its own absolute output directory. Both run directories and generated YAML
filenames must be fresh. A resumes at 32 and performs 32 more updates; B resumes at 33 and performs
31 more. Their restored LR schedules retain the original horizon of 64. The source can apply
backpressure while supplying work needed by the less advanced reader; each reader preserves its
own committed progress. Use the root report for session status and each Draft's report for its
training and evaluation results. Each Draft has its own checkpoint to
[export and evaluate](evaluation.md#export); the session root is not a Draft checkpoint.

Before releasing shared features, the runtime verifies that the stopped Draft's trainer processes
have exited on every node. If a node becomes unreachable, keep the driver and source running until
those processes have been stopped.

TensorBoard and W&B use separate runs/writers per experiment. See
[training and tracking](training.md#tensorboard-and-wb). Online weight synchronization, elastic membership
and isolation between untrusted tenants are unsupported.
