# Component notices and external assets

The author explicitly left the newer-code license decision pending on
2026-11-16. This repository has mixed per-file notices. Preserve them and
`LICENSE.GPL2`; do not label the entire artifact MIT.

| Component | Existing notice or status |
|---|---|
| `src/_videocore7/driver.py`, `src/_videocore7/drm_v3d.py` | GPL version 2 or later, with Broadcom and Idein copyright notices |
| Assembler and other files carrying the permission grant | MIT-style permission text in the individual files; preserve their copyright notices |
| Newer `src/qpu_xla` and integration files without notices | No blanket license is inferred; author license decision is pending |
| Artifact tools and documentation without notices | Included for evaluation; project-wide license declaration is pending author confirmation |
| Retained measurements, numerical outputs, sample manifests | Author-created evidence, with upstream dataset/model provenance retained; no new blanket data license asserted |
| `llama.cpp`, LeRobot, PyTorch and other dependencies | Acquired separately under their upstream terms; not relicensed by this package |
| Models and image/robot datasets | Acquired separately; consult their pinned model/dataset cards and access terms |

Only a 1024-byte production activation is retained for the MTP diagnostic.
Its Q4_0 weights are extracted from a separately acquired hashed GGUF, rather
than redistributed here. Model-quality payloads contain numerical outputs,
not model weights, decoded images, or source robot videos.

Before public deposit, authors must confirm distribution rights and declare
the intended licenses for files currently lacking notices. This document
records the current state and does not grant rights on behalf of third parties.
