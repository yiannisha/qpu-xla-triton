# Artifact appendix integration

[APPENDIX.tex](APPENDIX.tex) is integrated into the supplied [main.tex](../main.tex)
before References. It fills the existing template fields and follows the
[current linked template](https://www.asplos-conference.org/asplos2027/ae-artifact-appendix/).
The supplement continues at Appendix B, avoiding duplicate Appendix A numbering.
Supplied figures and bibliography are included; original PDF and figure bytes
are preserved. The original PDF's page/table mapping remains in `manuscript.json`.

The local Tectonic 0.17.0 / acmart 2.19 build verifies **two AE pages (13–14)**,
References beginning on 15, and no AE overfull boxes. The revised review PDF is
`dist/QPUs_CGO2027-artifact-draft.pdf`; this generated draft is separate from the
supplied official PDF. `manuscript-revision.json` records source/tool identities,
hashes and pagination; `manuscript-ae.patch` exposes the paper/bibliography edits.
A normal acmart 2.19 installation can compile main.tex with the root figures,
main.bib and artifact/APPENDIX.tex. XeTeX and pdfTeX may paginate differently.

The appendix preserves deferred licensing/deposit, numerical/stability limits
and the distinction between author checks and independently awarded badges.
No historical table values were changed. Add an archival link only when deposit
is authorized; no DOI or manuscript submission has been performed.
