# Native GEMM MMU diagnosis and correction

Two GPU read-boundary errors were found and corrected in the tiled native
Q4_0/Q8_0 GEMM used by the FFN. Numerical output checks alone missed them:
unused speculative loads and unused bytes of a word can fault without changing
the consumed arithmetic results.

| Diagnostic variant | Compute dispatches | Observed MMU faults | Numerical cases passed |
|---|---:|---:|---:|
| Original kernel | 840 | 1080 | 6/6 |
| Final K16 prefetch corrected | 840 | 234 | 6/6 |
| Same kernel, with allocation inventory | 840 | 234 | 6/6 |
| Prefetch and FP16 reads corrected | 840 | 0 | 6/6 |

All four diagnostics enabled repeated MMU logging and captured private V3D
trace instances without lost events. Rendering continued concurrently; the
allocation inventory links the remaining three fault addresses directly to
this runtime's buffer ends. The traces and original failures are preserved in
[evidence/kernel-bounds-fix/](evidence/kernel-bounds-fix/).

The double-buffered GEMM stream requested another K16 chunk after the last
useful chunk. It now re-reads that final valid chunk. FP16 scales occupy two
bytes, while their TMU reads fetch four: the final unaligned read could cross
a buffer boundary. Reads now use the containing aligned word and select the
requested half. Tile sizes, numerical representations, accumulation order and
ownership remain unchanged. Generated binaries, headers and source identities
were refreshed; historical calibration identities do not transfer to them.

The allocation trace identifies the remaining addresses `0x2bc9000`,
`0x2ded000` and `0x3875000` exactly at the ends of 147456-byte scale allocations.
The correction removes those faults as well as the speculative-read faults.
Six one-/two-block numerical boundary cases cover FP16, word-scale and
column-scale variants. The full model preflight covers M=129/257 and all 35
layers, with fixed probability/TV/KL gates. The subsequent full protocol also
passes with zero observed faults and repeated logging enabled.

## Fault visibility is a required control

The installed kernel is `6.12.75+rpt-rpi-2712`, built from Raspberry Pi commit
`89050b1059997d38d55462b323b099a6436dc10d`. Its
[IRQ implementation](https://github.com/raspberrypi/linux/blob/89050b1059997d38d55462b323b099a6436dc10d/drivers/gpu/drm/v3d/v3d_irq.c#L173-L191)
logs only its first MMU error unless `debug_mmu` is enabled. Its older client
name decoding also cannot establish the originating process. An empty later
log with that parameter disabled is insufficient evidence of fault-free work.

`run_rebuttal_diagnostics.py` now enables and verifies repeated logging for the
command, rejects GPU errors immediately, drains the kernel stream at process
exit, and restores the original parameter. Missing or ineffective control is
an error. It also retains the governor/swap controls and restores host state.
Earlier records retain their original retention fields; they do not satisfy
this newly established observability requirement retroactively.

For a separate diagnostic, outside timing measurements:

```sh
QPU_LLAMA_TRACE_BUFFERS=1 uv run --locked python scripts/trace_artifact_v3d.py \
  --output build/v3d-diagnostic --timeout 180 -- \
  "$BUILD_ROOT/llama-qpu-runtime/qpu_ffn_island_model_smoke" "$GEMMA_MODEL"
```

This records submission/IRQ events, kernel logs, allocation addresses/sizes,
monotonic timing and per-CPU loss counters. It removes only its own trace
instance; global tracing settings and the original MMU setting are preserved.
The allocation tracing flag is off by default. Traced durations are diagnostic
and must not be presented as paper performance measurements.

The corrected runs establish bounded functionality on the tested shapes.
They do not establish sustained reliability or resolve every historical
SmolVLA/depthwise hang. Those separate failures and quality limitations remain
in the artifact; packed INT16 stays quarantined.
