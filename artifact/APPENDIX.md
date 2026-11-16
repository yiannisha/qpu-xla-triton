# Artifact appendix source notes

The insertable draft is [APPENDIX.tex](APPENDIX.tex). Its content is based on
verified repository evidence and explicitly records deferred publication and
licensing. Insert it into the conference template before References, verify
the two-page limit in the actual manuscript, and add final figure/table
locations once that manuscript is available. CGO permits author identification.

Template: https://www.asplos-conference.org/asplos2027/ae-artifact-appendix/

| Field | Content / action |
|---|---|
| Paper title | Practical: Building and Evaluating a Native CPU-QPU Runtime for Edge AI Inference on Raspberry Pi 5 |
| Authors and final figure/table IDs | Pending authoritative manuscript metadata |
| Requested badge categories | Available; Evaluated Functional/Reusable if independent audit passes; Results Validated if independent results agree |
| Access | Local content-identified release; public archival link/DOI intentionally deferred |
| Programs | VideoCore VII Python assembler/driver, QPU-XLA runtime, pinned native llama.cpp integration and experiment scripts |
| Inventory | `artifact/release-files.json`, `artifact/claims.json`, original and October rebuttal raw records and model-output evidence |
| Hardware | Raspberry Pi 5, 64-bit Linux, historical model machine 8 GiB RAM, active cooling/reliable power; CPU-only smoke/analysis on ordinary Linux hosts |
| Software | Python >=3.12; exact locked Python dependencies; C/C++ compiler, CMake, Unix Makefiles; llama.cpp base `91d2fc387529940230555abd297a8b5e99737d3f` plus patches 0001–0005 |
| Models/datasets | See assets.json for exact identities; external acquisition instructions in ASSETS.md |
| Installation and basic test | `uv sync --locked --group dev`; `uv run --locked python scripts/artifact.py smoke --output build/smoke` |
| Hardware basic test | Native preparation then `artifact.py smoke --hardware --native-build build/llama-qpu-runtime --output build/pi-smoke` |
| Workflow | `artifact.py analyze` for archived data; explicit `paper`, `rebuttal`, `quality` profiles for new experiments |
| Expected outputs | Checksummed JSON records, max numerical errors, regenerated summaries; claim boundaries and failures in LIMITATIONS.md |
| Resource/time estimates | Clean native setup 7.4 min; bounded main stages about 2.1 min each; roughly 290 MB checkout/build outputs excluding shared Git objects; peak model RSS 4.41 GB. Cold downloads and full statistical/quality campaigns remain additional |
| Allowable variation | Fixed numerical gates and experiment-specific uncertainty; independent results must preserve the main claims. No blanket timing tolerance is asserted |
| Licenses | Existing mixed per-file notices retained; missing newer-code license declaration requires author decision |
| Paper tables/figures | Populate null manuscript locations in claims.json from the final accepted paper |

Before submission, verify the artifact corresponds to the final manuscript,
including all rebuttal corrections. Distinguish historical headline confirmation,
instrumented diagnostics, exploratory screens and rejected attempts. Do not
claim an anonymous upload, manuscript replacement, independently validated
badge, optimized W8A8 CPU control, or robot-success study that has not occurred.
