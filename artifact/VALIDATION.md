# Local preparation and reproduction checks

Author checks on Raspberry Pi 5, 2026-11-16/17 UTC. The repository and revised
manuscript are prepared for independent evaluation. No independent review,
publisher badge, sustained-reliability claim or public deposit is implied.
The author subsequently selected GPL-2.0-only for newer author-owned code/docs; existing
component notices remain in effect. This licensing-only revision rebuilds the
package and manuscript without rerunning numerical/performance campaigns.

## Current corrected build

The native preparer now selects Release for upstream and all integration
variants, records the actual isolated source checkout, and builds every target
used in HOWTO.md. Two tiled Q4 GEMM out-of-bounds reads were fixed: a final
unused K16 prefetch and an unaligned 32-bit read of the last FP16 scale.
Generated programs and source identities were refreshed. Historical calibration
identities were preserved; no automatic placement was promoted.

The installed kernel suppresses repeated MMU messages by default. Controlled
runs now enable and verify `debug_mmu=Y`, monitor the kernel, abort on errors,
and restore the original setting. A private trace instance and optional native
buffer inventory isolated both bugs without changing global tracing or killing
desktop clients. See [MMU-INVESTIGATION.md](MMU-INVESTIGATION.md).

| Check | Outcome |
|---|---|
| Focused portable controls | 24 passed in 2.02 s; log in `validation/final-controls.log` |
| Hardware boundary regression | Six cases: one/two K16 blocks across FP16, word and column scales; all pass with zero kernel errors and verified repeated logging |
| Six-case full-vocabulary diagnostic | 840 compute dispatches, zero MMU faults after both fixes; original/tail-only variants produced 1080/234 faults under the same observable logging |
| Complete paper protocol | 20 prompt processes and 15 cached-request pairs; numerical, dispatch, context, greedy-output and environment gates pass; zero kernel errors |
| Prepared producer/down | M=65/129/257 exact Q8 values/scales; maximum down error 9.54e-6; complete producer/down remains slower than CPU |
| Archived-data agreement | 117 default printed-value checks; 123 with optional pinned COCO annotations, zero mismatches; all 39 tables/five figures indexed, not every supplemental cell recomputed |
| Model-output analysis | Classification metrics, SmolVLA errors, static memory plans and optional COCO AP recomputed; expensive quality confidence intervals remain preserved views |
| Manuscript | Supplied source/assets integrated; acmart 2.19 build yields a two-page AE before References; no AE overfull boxes; original PDF unchanged |

### Fresh full protocol

| Boundary | Historical ratio [95% interval] | Corrected fresh ratio [95% interval] | Interpretation |
|---|---|---|---|
| Prompt M=129 | 1.0342 [1.0090, 1.1104] | 1.0738 [0.9835, 1.0883] | Positive median; fresh interval spans parity |
| Prompt M=257 | 1.0069 [0.9651, 1.0168] | 1.0310 [1.0196, 1.1300] | Positive in this author run |
| Cached request wall | 1.0379 [1.0241, 1.0520] | 1.0431 [1.0231, 1.0504] | Supports the historical modest benefit |

Thirteen of 15 cached pairs favor the candidate; both losses remain included.
No additional campaign was run to tighten the M=129 interval. This does not
establish an automatic-placement policy or independent Results Validated award.
Prompt and cached stages cost 262.8 s and 600.7 s including preflights and
process setup: about 14.4 minutes total. Peak cached-process RSS was
4,405,002,240 bytes. Governors, swap and repeated-MMU logging were restored.
Full raw records, trace diagnostics, printed-value outputs, producer results
and controls are in [evidence/kernel-bounds-fix/](evidence/kernel-bounds-fix/).

## Preserved earlier attempts

- `evidence/clean-build/`: initial native preparation took 443.7 s (7.4 min).
  Upstream checkout/build occupied 275.7 MB and integration outputs 13.3 MB,
  excluding separately shared Git objects. Pi smoke errors were zero; the
  customized 21-case example had maximum error 9.54e-6. A preflight MMU event
  rejected the entire reduced prompt campaign despite numerical passes.
  Its separately executed two-pair cached stage reported no new errors, but
  repeated logging was not verified then; quiet logs are not proof of no fault.
- `evidence/manuscript-validation/`: initial full 20-process/15-pair run had
  an unoptimized integration unlike the historical Release plugin. Its
  M129/cached intervals span parity and are retained unchanged. Its timing is
  not a matched-build reproduction; repeated MMU logging was also unverified.
  See `build-mismatch.json` and the summary's build qualification.
- `evidence/kernel-bounds-fix/interrupted-release-*`: a subsequent Release
  campaign was stopped when the logging blind spot was discovered. It is not
  retained performance evidence. The original stop reason and partial records
  remain included.

The corrected campaign follows explicit build, observability and bounds fixes;
it does not erase these failures or reinterpret earlier retention fields.
Historical SmolVLA hangs, rejected depthwise runs, missing optimized CPU W8A8
control and task-quality regressions remain limitations.

## Package and manuscript checks

The explicit release inventory excludes models, arbitrary benchmark_runs,
local configuration and build caches. The archive includes content-based
version metadata, a complete per-file manifest and checksums. Its sidecars
record the final archive hash and checks from a separate Git-free extracted
tree: locked install, verification, portable smoke, default paper analysis and
import/version identity. Identical inputs must produce identical archive bytes.
Changed Python files pass Ruff lint/format checks. The broader repository
formatter still flags 27 unchanged files; unrelated formatting was left out of
scope. This is not a claim that the entire existing CI suite passes.
The source/evidence payload is roughly 210 MB; allow several GB for dependencies
on ordinary disk. An earlier install into a 4 GiB RAM-backed /tmp failed while
copying Torch; ordinary-disk extraction passed. Cached installation costs do
not represent cold downloads. Models/datasets and fresh quality campaigns add
separate costs. Docker is supplied but unbuilt on this host.

`manuscript-revision.json` records original/revised source hashes and the local
draft PDF identity. The actual AE occupies pages 13–14; References start on 15.
Both supplied and revised source have 12 main-body pages under the same local
XeTeX/acmart toolchain. The original pdfTeX PDF has different pagination; exact
pagination equivalence is not claimed. `manuscript-ae.patch` exposes all source
and bibliography edits. No historical result-table values were rewritten.
