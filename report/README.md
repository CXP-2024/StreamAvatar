# StreamAvatar Report

This folder contains the final course-project style report, using a local NeurIPS preprint template.

Files:

- `StreamAvatar_report.tex`: main report.
- `ref.bib`: bibliography.
- `neurips_2024.sty`: local template style file.
- `figures/`: copied experiment figures and qualitative snapshots.
- `build.sh`: LaTeX build script.

Build:

```bash
cd /mnt/pfs/group-jt/changxun.pan/runs/test/float_playground/DyStream
./report/build.sh
```

The build script uses `report/.texenv/bin/tectonic` when that local binary is available, then falls back to a system `tectonic`, and finally to a standard `pdflatex`/`bibtex` sequence.
