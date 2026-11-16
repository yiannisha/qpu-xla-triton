# Reuse and experiment contracts

Copy `profiles.json` to the ignored `artifact/local-profiles.json`. Each stage
has a unique `id` and an `argv` list; arguments are passed directly without a
shell. `{python}`, `{root}`, and `{output}` are supplied by the artifact tool.
Other placeholders name environment variables. Stages are sequential and
stop on the first nonzero exit, preserving the partial campaign record.

For a small new FP32 matmul experiment, replace the stages of the copied
`smoke` profile with:

```json
{
  "id": "matmul-128",
  "argv": [
    "{python}", "examples/benchmark_qpu_xla_matrix.py",
    "--size", "128", "--warmup", "2", "--repeat", "7",
    "--output", "{output}/matmul-128.json"
  ]
}
```

Set that profile's `hardware` field to `true`, export
`OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4`, and plan it with:

```sh
uv run --locked python scripts/artifact.py run smoke \
  --config artifact/local-profiles.json --output build/new-matmul
```

The same complete example is supplied as `artifact/profiles-custom-example.json`:

```sh
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 uv run --locked python scripts/artifact.py \
  run smoke --config artifact/profiles-custom-example.json \
  --output build/custom-example --execute
```

Add `--execute` after checking the plan. This is a new experiment, not a
reproduction of the archived dimensions. Use a controlled runner for retained
performance evidence; the small example does not establish a new automatic
placement record. Other shapes, row/output fractions, thread counts and models
can be varied through the existing runner arguments. Keep the INT32 signed
24-bit input restriction and explicit padding/tile contracts.

The artifact command emits versioned `artifact-environment-v1`,
`artifact-verification-v1`, `artifact-smoke-v1`, and `artifact-assets-v1` JSON.
Each includes `ok`; failure returns exit 2. Smoke reports `max_abs_error` by
dtype and `hardware`, so a fake run cannot be mistaken for a QPU run.
Profile execution writes `campaign.json` containing exact argument arrays,
environment, exit status and elapsed time per executed stage. Plan-only output
has `executed: false` and performs no experiment.
Controlled stages also include their outer kernel/retention decision. A failed
control gate sets `completed: false` and `failed_stage` in the campaign record,
even if the child program exits zero. Inner numerical success does not override
an outer kernel-error rejection.

`analyze` writes the established rebuttal summary structures, the full
projection grid, recomputed inexpensive raw-output numerical metrics, and
separate retained report views. Bootstrap designs remain as recorded:
independent process pairs/blocks where applicable, and within-process sample
variation for single-process diagnostics. Do not pool them into one interval.

For a new QPU operator, use the runtime contracts and candidate interface in
`QPU-XLA.md`, add a NumPy/Torch numerical oracle and an explicit hardware
differential check, then benchmark against a competitive same-contract CPU
control. A single stage win is insufficient to promote a model path. Keep
new experimental candidates opt-in until full-boundary evidence passes.
