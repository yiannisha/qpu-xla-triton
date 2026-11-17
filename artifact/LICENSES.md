# Component notices and external assets

On 2026-11-17 UTC the author selected **GPL-2.0-only** for their newer code and
associated technical documentation previously lacking a notice. Copyright
(c) 2026 Yiannis Hadjiyianni and contributors. The full license is in root
[LICENSE](../LICENSE).

This grant covers author contributions in `src/qpu_xla`, the native integration,
examples, scripts, tests and artifact tooling/documentation, except where a
file has an existing notice or is third-party-derived material. Existing
copyright notices and per-file terms take precedence and remain unchanged.
The Python distribution declares `GPL-2.0-only AND GPL-2.0-or-later AND MIT`
to identify the licenses present, not alternative licenses for every file.

| Component | License or distribution boundary |
|---|---|
| Newer author-owned code and associated technical documentation without another notice | GPL-2.0-only, in root `LICENSE` |
| `src/_videocore7/driver.py`, `drm_v3d.py`, `v3d.py` | Existing GPL version 2 or later; Broadcom/Idein notices and `LICENSE.GPL2` preserved |
| Assembler and other files carrying an MIT-style permission grant | Existing per-file permission grant and original copyright notices preserved |
| Third-party-derived code and bundled patches | Existing upstream terms and notices preserved; no third-party contribution is relicensed |
| Retained measurements, numerical outputs and sample manifests | Distributed for reproduction with source provenance; the software license does not relicense upstream data or model-derived material |
| Manuscript source, bibliography and figure PDFs | Paper materials; the software GPL grant does not assign a new publication license |
| `llama.cpp`, LeRobot, PyTorch and other dependencies | Acquired separately under their upstream terms |
| Models and image/robot datasets | Acquired separately under their pinned model/dataset terms |

Only a 1024-byte production activation is retained for the MTP diagnostic.
Its Q4_0 weights are extracted from a separately acquired hashed GGUF, rather
than redistributed here. Model-quality payloads contain numerical outputs,
not model weights, decoded images, or source robot videos.

For a mixed-license Zenodo record, declare GPL-2.0-only, GPL-2.0-or-later and
MIT with this component scope rather than assigning one blanket license to
every file. External assets retain their own terms. Public deposit remains a
separate author action; selecting GPL does not itself publish the artifact.
