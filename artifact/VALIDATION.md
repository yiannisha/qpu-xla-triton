# Local preparation checks

Local checks on Raspberry Pi 5, 2026-11-16. Structured results, source-file
fingerprints and the focused test log are in [validation/](validation/).
No independent reviewer, fresh long quality campaign, container build, or
publisher badge is implied. Licensing and public deposit remain deferred.
The supplied manuscript is now mapped; its paper source still needs appendix
and qualification edits. The full fresh FFN protocol completed, with numerical
passes but **without confirming the historical statistically positive gains**.

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
basic binary completed in under a second. The full fresh FFN costs are recorded below; long fresh task-quality campaign
costs remain additional.

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
on this preparation host. Native setup is the exercised path. Long quality
campaigns were not repeated. Neither bounded nor full author runs award a badge
or establish sustained driver reliability.

## Official-manuscript follow-through

The 26-page author-supplied PDF is identified by SHA256 in `manuscript.json`.
All **39 tables and five figures** are mapped. The offline paper checker passes
**111 printed-value checks**, including main timing, classification quality,
selected operator values and both empirical figures. The complete FP32 operator
grid is exported with slower cases included; static estimates, preserved quality
intervals and COCO AP remain labeled report views. This is archived-data
agreement, not a new experiment or independent validation.

The focused tests still pass: **20 passed in 1.37 s**, recorded in
`validation/manuscript-tests.log`. The preparer now also builds `ggml-qpu` and
`qpu_ggml_geglu_bench`, required by Table 2's standalone command. Incremental
preparation from the existing clean build took 11.9 s. Its three prepared
producer/down cases passed exact Q8 bytes and numerical checks, with maximum
down-projection error 9.54e-6 and clean kernel retention. Complete producer+down
ratios were 0.225/0.350/0.289 at M=65/129/257, preserving the conclusion that
the sequence is slower than CPU. These are prepared, not full-FFN timings.

The full declared paper profile ran once after manuscript alignment: **20 prompt
processes** (five independent CPU/candidate processes at each M=129/257) and
**15 cached-request pairs**. Both outer kernel monitors were clean, all inner
retention gates passed, and all candidate dispatch/context contracts and greedy
outputs passed. Prompt and cached-request stages took **296.9 s and 647.6 s**
including preflights and process setup, about 15.7 minutes total. Peak observed
cached-request RSS was 4,405,198,848 bytes. Host governors and zram were restored.

| Boundary | Historical ratio [95% interval] | Fresh ratio [95% interval] | Interpretation |
|---|---|---|---|
| Prompt M=129 | 1.0342 [1.0090, 1.1104] | 1.0167 [0.9843, 1.0567] | Fresh interval spans parity; historical positive gain not confirmed |
| Prompt M=257 | 1.0069 [0.9651, 1.0168] | 1.0032 [0.9667, 1.0255] | Still inconclusive |
| Cached request wall | 1.0379 [1.0241, 1.0520] | 0.9919 [0.9726, 1.0170] | Fresh interval spans parity and does not overlap the historical interval |

The small historical performance gains have therefore **not** been freshly
confirmed. Do not mark Results Validated ready or claim independent reproduction
on this evidence. Investigating the discrepancy or revising the claim needs an
explicitly recorded follow-up, preserving this attempt and its sample design.
The successful functional run also does not erase the earlier reduced attempt's
MMU event. Its cause remains unresolved; a client inventory found desktop V3D
clients as well as this runtime. See `MMU-INVESTIGATION.md`.

All full-run raw records, controls, exact fresh intervals, printed-value checks,
CSV outputs and figure PDFs are retained in
[evidence/manuscript-validation/](evidence/manuscript-validation/). The supplied
PDF remains unchanged. `APPENDIX.tex` replaces its blank p.12 template, and
`MANUSCRIPT-EDITS.md` gives concrete qualification/rebuttal text. Integration into
the actual paper source and the two-page fit check remain author actions.
