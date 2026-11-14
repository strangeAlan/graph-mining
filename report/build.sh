#!/bin/bash
# Build a report document with pdfLaTeX (TinyTeX, same toolchain as cv-3Drecon).
# Usage: bash report/build.sh [report]   -> report/build/<name>.pdf
set -e
cd "$(dirname "$0")"
export PATH="$HOME/.local/opt/TinyTeX/bin/x86_64-linux:$PATH"
name="${1:-report}"
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build "$name.tex" > "build/$name.latexmk.log" 2>&1 \
    || { tail -40 "build/$name.latexmk.log"; exit 1; }
grep -E "^(LaTeX|Package).*Warning|Overfull|Undefined" "build/$name.log" | sort | uniq -c | head -20 || true
grep -o "Output written on .*" "build/$name.log" || true
