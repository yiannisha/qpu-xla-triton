# Artifact review and clarification log

Maintain this alongside release versions. Preserve the submitted archive;
corrections create a new content-identified package. Publication is deferred.

| Date | Version/evidence ID | Issue | Action and verification | State |
|---|---|---|---|---|
| 2026-11-16 | Local artifact preparation | Repository lacked a consolidated reviewer workflow | Inventory, evidence/provenance checks, original/rebuttal analysis, profiles, portable/Pi smoke and local release packaging added; focused checks and extracted install pass; see VALIDATION.md | Local preparation validated |
| 2026-11-16 | Clean-build follow-through | Native preparation and full FFN invocation had not been exercised from clean sources | Full native build passed in 443.7 s; fresh smoke/custom example and all model numerical checks pass; raw evidence in `evidence/clean-build/` | Exercised and recorded |
| 2026-11-16 | Prompt kernel event | New PTB MMU event occurred during real-model preflight | Whole prompt campaign latency rejected despite numerical/execution passes; no gate relaxed or retry-until-success | Initial failure preserved; later bounds diagnosis below |
| 2026-11-16 | Cached-request check | First campaign stopped before its remaining stage | Execute only the unexecuted agentic stage in a new directory; both pairs pass with clean kernel retention | Bounded exercisability; no new performance claim |
| 2026-11-16 | Manuscript alignment | Exact manuscript and figure/table IDs unavailable | Author supplied the exact PDF; all 39 tables/five figures and stable claims mapped in `manuscript.json`; printed-value regeneration added | Mapped; source integration subsequently completed below |
| 2026-11-16 | License declaration | Several newer files have no explicit notice | Author explicitly requested leaving the license decision pending; preserve mixed existing notices | Deferred by author |
| 2026-11-16 | Archival publication | Deposit intentionally excluded from this work | No DOI or publisher badge claimed | Deferred by author |
| 2026-11-16/17 UTC | Official manuscript follow-through | Exact result map and full protocol now available | 39 tables/five figures mapped; 111 printed-value checks and 20 tests pass; full numerical/execution controls pass but fresh positive gains not confirmed | Initial unoptimized/logging-limited attempt preserved; later corrections below |
| 2026-11-17 UTC | Build/source correction | Integration lacked Release and cached runner recorded an unrelated checkout | Explicit Release for all projects; actual --llama-root recorded; build regression passes | Fixed; old attempt retained |
| 2026-11-17 UTC | MMU diagnosis | Kernel suppresses repeats; tiled GEMM overreads tail and FP16 scale | Verified repeated logging; private trace and allocation boundaries isolate both reads; corrected six-case preflight: 0 faults/840 dispatches; six boundary tests pass | Corrected within tested scope |
| 2026-11-17 UTC | Full corrected reproduction | Need matched, observable full protocol | 20 prompt processes + 15 cached pairs pass; cached 1.0431 [1.0231,1.0504], M129 prompt interval spans parity; 117 default/123 optional AP checks pass | Author reproduction; no automatic promotion |
| 2026-11-17 UTC | Manuscript source integration | Author supplied main.tex, bibliography and five figures | AE and qualifications integrated; acmart 2.19 draft has two AE pages; source patch and original/revised hashes retained | Local review draft, no submission |
| 2026-11-17 UTC | GPL licensing | Author resolved the newer-code license decision as GPL-2.0-only | Root GPL license, mixed-component notices and package metadata updated; prior GPL/MIT notices preserved; regenerated manuscript and local bundle | GPL-2.0-only selected for author-owned software/docs; public deposit remains separate |
| — | Independent evaluator | No external review performed by these preparation tools | Record evaluator environment, commands, issues, outcomes and tolerances here when available | Pending independent evaluation |

For each reviewer issue, record the original failure/output, affected claim,
minimal patch, validation command/result, previous release hash and replacement
release hash. Keep numerical gates and evidence exclusions unchanged unless a
separately explained study revision warrants a new protocol. Do not replace
rejected runs or silently promote exploratory screens.

CGO R2 clarification is December 10–12, 2026, with notification December 18.
Use the conference review channel for technical clarification; retain a
repository record of proposed artifact/appendix updates before camera ready.
No messages or submissions are sent by the artifact tools.
