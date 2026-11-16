# Reviewer hardware and access plan

The artifact needs a Raspberry Pi 5 for QPU execution. Portable smoke and
retained-data analysis run without it. This document prepares the access
arrangement; it does not claim that a remote reviewer account has been created
or that an independent evaluator has used the machine.

## Reviewer-owned Pi

Use a 64-bit Linux Pi 5 with active cooling and reliable power. The historical
model campaigns used 8 GiB RAM. Follow `README.md` from a fresh extracted
release, using ordinary disk rather than a small RAM-backed `/tmp`.

1. Run the environment check and portable smoke.
2. Acquire the exact inputs listed in `assets.json`, verifying their hashes.
3. Clone the pinned upstream source and run the native builder into a new
   `--build-root`. Do not reuse the authors' compiled objects.
4. Run Pi/native smoke, then the bounded `profiles-functional.json` workflow.
5. Run the full `paper` profile for independent empirical validation, recording
   every process, rejected attempt and uncertainty interval. The bounded run
   establishes exercisability and must not replace the full confirmation.

A failed control gate stops a profile. Preserve that campaign and its raw
kernel log. An unexecuted stage may be exercised separately with `--only` and
a new output directory; this is not permission to retry a rejected stage
until it passes. The clean-build example's prompt latency was rejected for
an MMU event, while the separately executed cached-request stage passed. The
artifact preserves both outcomes and does not claim global driver reliability.

## Author-provided dedicated Pi

Before AE submission, coordinate the hardware requirement with the AE chairs
through the conference channel. Authors must provide the actual endpoint,
availability and credentials separately; none are invented or embedded here.

Reserve the machine exclusively for one evaluator/campaign at a time. Provide
a dedicated account, enough ordinary-disk space, the pinned assets under their
applicable access terms, and a clean workspace. Keep the original release
read-only and write results to a new directory. Preserve the environment
snapshot and shell commands with returned results. Do not share another
researcher's live checkout or access unrelated files.

Basic QPU execution requires the host's `video`/`render` device-group access.
The current driver uses the V3D card path as well as the render node described
in the guide. Controlled timing also uses existing noninteractive lab access
for CPU governor writes, live kernel-log capture, swap changes, and the
`dev-zram0.swap` systemd unit. An administrator should review these exact
runner operations; the artifact does not install sudo rules, create accounts,
change SSH configuration or grant broad root access.

The runners restore normal state after ordinary completion/failure. A board
hang or power loss needs manual recovery; checkpointed resumes are not proof
of driver-hang recovery. Record each failed attempt and check machine state
before a new campaign. Never silently retry unstable cases until one succeeds.

## Draft request to the AE chairs

Subject: Pi 5 hardware access for “Practical: Building and Evaluating a Native
CPU-QPU Runtime for Edge AI Inference on Raspberry Pi 5”

Our artifact requires a Raspberry Pi 5 with VideoCore VII, 64-bit Linux,
active cooling and reliable power; the model experiments used 8 GiB RAM.
Portable smoke and archived-result analysis are also available. Could you
confirm whether suitable reviewer hardware is available, or whether we should
arrange access to a dedicated author-provided Pi? The artifact guide includes
the complete native build, input hashes, bounded basic workflow, full main
experiments, environment controls and known limitations. We can coordinate
the actual endpoint and access details through the review channel.

This draft has not been sent. Consult the current
[CGO artifact page](https://2027.cgo.org/track/cgo-2027-artifact-evaluation)
for chair contact details and submission procedure.

## Evaluator return record

Return the release content hash, host environment, native build record, exact
asset hashes, command plans, `campaign.json`, raw process/dispatch/kernel logs,
numerical outcomes, retained/rejected statuses and derived results. Identify
which claim IDs were checked and whether each qualitative conclusion survives
the observed variation. Add clarification issues and replacement release
identities to `CLARIFICATIONS.md`. Independent badge decisions belong to the
evaluators/publisher, not these author-run preparation commands.
