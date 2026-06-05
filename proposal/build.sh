#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ -x "../report/.texenv/bin/tectonic" ]; then
  XDG_CACHE_HOME="${PWD}/.tectonic_cache" ../report/.texenv/bin/tectonic StreamAvatar_proposal.tex
elif command -v tectonic >/dev/null 2>&1; then
  tectonic StreamAvatar_proposal.tex
else
  pdflatex -interaction=nonstopmode StreamAvatar_proposal.tex
  bibtex StreamAvatar_proposal
  pdflatex -interaction=nonstopmode StreamAvatar_proposal.tex
  pdflatex -interaction=nonstopmode StreamAvatar_proposal.tex
fi
