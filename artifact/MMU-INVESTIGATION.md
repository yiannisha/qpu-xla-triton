# Retained MMU event investigation

The reduced clean-build prompt campaign recorded:

```text
2026-11-16T17:38:42,333593-05:00 v3d 1002000000.v3d: MMU error from client PTB (1) at 0x2bc9000, pte invalid
```

Its timestamp falls inside the real-model full-vocabulary preflight
(22:37:59–22:39:19 UTC), after the synthetic joined-graph test finished.
All numerical cases passed, but the outer monitor correctly rejected the
entire campaign's latency. Numerical correctness does not annul a kernel error.
The original process, preflight and kernel records remain in
`evidence/clean-build/`.

A bounded source review checked the native submission/wait and failure-injection
paths. Each submission includes its program, uniforms and data BO handles.
The runtime performs the real BO wait before returning an injected wait error,
poisons the context after a wait error, and prevents another submission on that
context. There is no evidence here that the event came from failure injection,
nor a confirmed source-level fix. No native kernel, allocator or synchronization
code was changed speculatively.

The read-only DRM client inventory in
`evidence/manuscript-validation/gpu-clients.json` also found `labwc` and
`xdg-desktop-portal` clients during the subsequent full campaign. This is an
observed possible source of GPU contention, not an attribution of the earlier
fault. A global V3D message does not identify its originating process from a
wall-clock timestamp. The compositor was not stopped and no display service
was reconfigured. The [kernel driver overview](https://docs.kernel.org/gpu/v3d.html)
describes both rendering and compute pipelines behind V3D.

The subsequent full protocol is a separately declared manuscript-validation
campaign, using the paper's actual sample counts rather than repeating the
reduced stage until it passes. Both campaigns remain in the package; fresh
results and retention are recorded in `VALIDATION.md`. Clean completion of a
later campaign does not prove the fault fixed or establish sustained stability.

Further causal isolation needs a dedicated headless evaluation session and
per-submission/BO diagnostics correlated with kernel events, with every attempt
retained. That is unresolved runtime/driver work. Kernel-error rejection must
remain enabled for both author and independent reviewer runs.
