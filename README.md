# Athena-FKA

> **This repository is a curated export of the code supporting *Factored Knowledge Architecture* (doi:10.5281/zenodo.22059102). The exact archived state backing each paper is its Zenodo deposit, built from `paper-v1.2` at SHA `2ebcb4409ec35c3189cfc769f3e7d7b661def029`. This export has fresh history and is not that commit.**

It is the code supporting *Factored Knowledge Architecture*. Athena consumes this repository as a **pinned dependency** and never edits it; the sibling repository's papers cite this one.

## What is here, and what is not

This is a **curated** export, not a mirror. It contains the code, the measurement artifacts those papers' numbers are derived from, and the tests that red-verify them. It does **not** contain the working repository's internal records.

`MANIFEST.json` maps every file to its content hash **and** to its git blob id at the tagged commit, so the curation is verifiable file-by-file against the Zenodo archive. **Files that diverge are listed there with the reason.**

## Runnable vs inspectable

| | |
|---|---|
| `python docs/paper/figures/make_figures.py` | **runnable** — regenerates all eight figures from the artifacts here |
| `python -m pytest tests/` | **runnable** — needs `pytest-timeout`, because `pyproject.toml` sets `--timeout=120` and travels with the suite so the export runs under the SAME configuration as the source. Without the plugin pytest refuses the argument before collecting anything |
| `requirements-rocm.txt` | the **resolved numerical environment**. A different torch build is a different numerical environment, and a figure inherited from one does not transfer to another |

## The other repository

https://github.com/Vexillon-ai/Athena — the sibling export.

## Licence

MIT. See `LICENSE`.
