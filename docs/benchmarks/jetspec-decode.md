# JetSpec bounded decode compatibility

This is a bounded decode integration check, not a throughput benchmark, convergence
result or serving-quality claim. It verifies that an ASpec-exported JetSpec asset is
loaded by official SGLang 0.5.21, that the causal head is active, and that exporting
the published checkpoint through ASpec preserves the decode result.

## Setup

- Host: one H100 PCIe 80GB, `sglang==0.5.21`, torch 2.13.0+cu130.
- Target: `Qwen/Qwen3-8B` (`/public/models/Qwen3-8B`).
- Data: 16 prompts from the held-out validation split.
- Decode: greedy, 32 new tokens, two complete repeats, normal radix cache.
- Draft: JetSpec block size 16, five Target feature layers `[1, 9, 17, 25, 33]`.
- Export: `aspec export` default SGLang layout with top-level `is_causal=true` and
  `dflash_config.causal_head=true`.

SGLang 0.5.21 reads the top-level `is_causal` flag for DFlash full-attention layers;
`dflash_config.causal_head` alone leaves those layers bidirectional.

## Direct published asset

The "loaded directly" row did **not** use the stock published config. SGLang 0.5.21
ignores `dflash_config.causal_head`, so the direct copy also needs top-level
`is_causal=true`:

```bash
cp -r /models/jetspec-qwen3-8b /tmp/jetspec-direct
python - <<'PY'
import hashlib, json
from pathlib import Path

root = Path("/tmp/jetspec-direct")
config = json.loads((root / "config.json").read_text())
config["is_causal"] = True
(root / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

manifest = {
    "schema_version": "aspec.serving-export/v1",
    "storage_format": "sglang",
    "algorithm": "jetspec",
    "model_sha256": sha(root / "model.safetensors"),
    "config_sha256": sha(root / "config.json"),
    "draft_structure": {"loader_contract": {
        "runtime": "sglang", "validated_version": "0.5.21",
        "speculative_algorithm": "DFLASH", "validation": "required_per_artifact",
    }},
}
(root / "export_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
)
PY
```

Then run `aspec evaluate` on `/tmp/jetspec-direct`. The stock config alone is a
bidirectional baseline and is not comparable to the causal ASpec export matching on
these prompts; verifier equality alone does not prove the draft proposals were equal.

## Results

| Draft | Acceptance length | Verifications | Median speedup | Greedy equal | Repeats stable |
| --- | ---: | ---: | ---: | --- | --- |
| ASpec 8-step checkpoint | 1.032 | 992 | 0.795x | yes | yes |
| Published `JetSpec/jetspec-qwen3-8b` loaded directly | 2.723 | 376 | 1.823x | yes | yes |
| Published checkpoint exported through ASpec | 2.723 | 376 | 1.823x | yes | yes |

Acceptance length follows `aspec evaluate`: completion tokens divided by real
verifier calls, including Target tokens. Each repeat produced 512 completion tokens;
the two repeats produced 1024 tokens total, which is the numerator used above.

Both published-checkpoint rows used top-level `is_causal=true`. The ASpec-exported
published checkpoint matched the directly loaded checkpoint's acceptance length,
verification count and speedup in this bounded setup. That checks
config/weight/layout compatibility; it does not establish convergence or a trained
quality comparison.

Timing is fixed-order: the evaluator completes all Target-only repeats before all
speculative repeats. Two repeats do not counterbalance that order, so the speedup
column is a bounded decode observation with an uncontrolled order-dependent
limitation, not an unbiased throughput result.

The 8-step ASpec checkpoint was trained only as a functional smoke test on synthetic
data. Its lower acceptance and slower speculative decode are expected and are not
evidence against the method.

## Reproduction

```bash
aspec export --input-dir outputs/jetspec/checkpoints/iter_0000009 \
  --output-dir outputs/jetspec-sglang

CUDA_VISIBLE_DEVICES=0 aspec evaluate \
  --target /models/Qwen3-8B --draft outputs/jetspec-sglang \
  --data held-out.jsonl --output outputs/jetspec-decode.json \
  --samples 16 --max-new-tokens 32 --repeats 2
```

The command above applies to an ASpec FSDP training checkpoint. For the published
draft head, use the reproducible validation harness:

```bash
python -m tools.validation.jetspec_published_export \
  --checkpoint /models/jetspec-qwen3-8b \
  --target-model /models/Qwen3-8B \
  --output outputs/jetspec-official-sglang
```

The harness loads the published tensors through `load_initial_draft_weights`, writes
the minimal distributed-checkpoint layout the converter expects, and calls
`convert_checkpoint(..., artifact_format="sglang")`. The published checkpoint omits
`embed_tokens`, so `--target-model` supplies the borrowed embedding; converting only
the config is not sufficient.
