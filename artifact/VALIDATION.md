# Local preparation checks

Local checks on Raspberry Pi 5, 2026-11-16. Structured results, source-file
fingerprints and the focused test log are in [validation/](validation/).
No independent reviewer, full fresh performance/quality campaign, container
build, or publisher badge is implied. The license decision and public deposit
remain deferred by the author; exact manuscript figure/table alignment remains
pending.

| Check | Observed outcome |
|---|---|
| Focused portable tests | 20 passed: artifact integrity, archive regeneration, explicit diagnostic selection, isolated build selection, failed control-gate propagation, existing FFN aggregation and process guards |
| Evidence verification | 41 October package files, 37 original provenance records, original selected model outputs plus clean-build follow-through records; release inventory also checked |
| Offline analysis | Corrected September headline medians and bootstrap intervals reproduce exactly; October TODO3–9 summary equals the retained summary; 112 projection cases with 10 supported wins |
| Raw quality-output checks | Recomputed MMLU 36.2%/36.3%, ResNet FP32 agreement, W8A8 regression, and SmolVLA numerical errors agree with retained reports; expensive CI/AP calculations are preserved as report views rather than rerun |
| Pi Python basic test | Forced QPU FP32 and INT32 matmul and public widened INT16 convolution: all max absolute errors zero |
| Native basic checks | Newly compiled stride regression and joined synthetic FFN graph passed, along with fresh Pi Python smoke; historical binaries remain preserved |
| Native preparation | Complete clean pinned checkout, patches 0001–0005, llama-bench/server/perplexity, three native integration variants and two isolated diagnostic CPU libraries all built successfully |
| ImageNet preparation | All 5000 retained sample identities reconstructed from the pinned public Parquet, with original encoded bytes and content hashes |
| MTP preparation | Exact retained 294912-byte weight slice extracted from hashed GGUF; 1024-byte activation identity also verified |
| Acquisition identities | Pinned model/source manifests retained; MMLU upstream file metadata and downloaded YOLO checkpoint match the retained SHA256 identities |
| CPU/hardware selection | Legacy driver/DRM tests are deselected under `not hardware`; non-Pi collection avoids importing the Pi-only register extension |
| Extracted release | Locked fresh environment, compiled Python C extension, checksum verification, portable smoke and offline analysis all passed from a separate extracted tree with no `.git` directory; imports point into that tree and installed package version matches its content identifier |
| Customized workload | Documented size-128 example executed all 21 matrix/attention configurations; maximum absolute error 9.54e-6 |
| Bounded prompt workflow | Six full-vocabulary cases passed; 560 timed/warmup dispatches with zero fallback and valid contracts. A new MMU event during real-model preflight rejects the entire campaign's latency, despite its successful process exit |
| Bounded cached-request workflow | Separately executed remaining stage: two CPU/QPU pairs, identical greedy outputs, all context/execution contracts passing, zero kernel errors and retained environment gates. This is exercisability evidence, not independent performance confirmation |

The detailed follow-through summary, raw records and complete native build log
are in [evidence/clean-build/](evidence/clean-build/). The prompt's inner numerical
record reports successful execution; **its outer control rejects latency**.
Do not interpret the inner `retention` field as overriding the kernel monitor.
The profile command now includes the outer control decision directly in its
campaign summary and rejects a failed gate even if a child exits zero.

The complete clean native preparation took **443.7 s (7.4 min)** with four build
jobs. Its checkout plus upstream build occupied 275.7 MB, and the integration/
profile outputs another 13.3 MB. Upstream acquisition used a local shared Git
object store; cold network clone/download time and separate Git-object storage
are not included. The bounded prompt and cached-request stages took about
126.3 s and 130.7 s respectively, including mandatory preflights and process
setup. The largest observed model process peak RSS was 4.41 GB; 8 GiB RAM leaves
headroom for the controlled no-swap environment. These are setup/workflow cost
observations, not new inference-speed claims.

The extracted-tree install took about 2.4 s with cached dependencies and
hardlinks on ordinary disk. Its verification took about 0.7 s, portable smoke
1.7 s and full offline analysis 4.6 s. These are preparation checks, not cold
download costs, inference latency, or portable timing guarantees. Each native
basic binary completed in under a second. Full fresh paper/quality campaign
costs were not measured here.

The first extracted install failed while copying Torch into a 4 GiB RAM-backed
`/tmp`; its log was retained under `build/artifact-preparation/`. Only that
task-created failed extraction was removed. Retrying on ordinary disk passed.
Allow several GB for dependency installation, distinct from the roughly
210 MB unpacked source/evidence archive.

The final local archive and its exact identity are written to
`dist/qpu-xla-artifact.tar.gz` and its `.sha256` sidecar. Within an extracted
release, `artifact/RELEASE.json` gives the complete content identity, original
source-base commit, per-file hashes and modes; `SHA256SUMS` covers the packaged
files. Do not place a package's own hash into its input documentation and then
silently rebuild it under the same identity.

The optional Docker recipe is supplied but unbuilt: Docker is not installed
on this preparation host. Native setup is the exercised path. Full statistical
paper confirmation and long quality campaigns were not repeated. The new
bounded runs do not validate a badge or establish sustained driver reliability.
