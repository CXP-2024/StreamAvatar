#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ -x ".texenv/bin/tectonic" ]; then
  XDG_CACHE_HOME="${PWD}/.tectonic_cache" .texenv/bin/tectonic StreamAvatar_report.tex
elif command -v tectonic >/dev/null 2>&1; then
  tectonic StreamAvatar_report.tex
else
  pdflatex -interaction=nonstopmode StreamAvatar_report.tex
  bibtex StreamAvatar_report
  pdflatex -interaction=nonstopmode StreamAvatar_report.tex
  pdflatex -interaction=nonstopmode StreamAvatar_report.tex
fi
