#!/usr/bin/env bash
# Idempotent install of docsearch on an Apple Silicon Mac (native, Apple MPS).
# Usage: deploy/macos/install.sh [--allow-cpu] [--no-weights]
set -euo pipefail

die() { echo "install.sh: $*" >&2; exit 1; }

[ "$(uname -s)" = "Darwin" ] || die "this script is for macOS (Darwin); found $(uname -s)"
[ "$(uname -m)" = "arm64" ] || die "requires Apple Silicon (arm64); found $(uname -m)"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:$PATH"

ALLOW_CPU=0
NO_WEIGHTS=0
for arg in "$@"; do
    case "$arg" in
        --allow-cpu) ALLOW_CPU=1 ;;
        --no-weights) NO_WEIGHTS=1 ;;
        -h|--help) sed -n '2,3p' "$0"; exit 0 ;;
        *) die "unknown argument: $arg" ;;
    esac
done

command -v uv >/dev/null 2>&1 || die "uv not found. Install it: brew install uv  (or https://docs.astral.sh/uv/)"
command -v brew >/dev/null 2>&1 || die "Homebrew not found. Install it from https://brew.sh, then re-run"

TORCH_VERSION="$(sed -n 's/^ARG TORCH_VERSION=\([^[:space:]]*\).*/\1/p' Dockerfile | head -n 1)"
[ -n "$TORCH_VERSION" ] || die "could not find 'ARG TORCH_VERSION=' in Dockerfile"

if ! command -v pdftoppm >/dev/null 2>&1; then
    echo "Installing poppler via Homebrew (brew may upgrade poppler's dependency formulae too)."
    HOMEBREW_NO_AUTO_UPDATE=1 HOMEBREW_NO_ENV_HINTS=1 brew install poppler
fi

if [ ! -x .venv/bin/python ]; then
    uv venv --python 3.12 .venv
fi

uv pip install --python .venv/bin/python "torch==${TORCH_VERSION}"
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python --no-deps -e .

if .venv/bin/python -c 'import sys, torch; sys.exit(0 if torch.backends.mps.is_available() else 1)'; then
    echo "torch MPS available: yes"
elif [ "$ALLOW_CPU" = 1 ]; then
    echo "torch MPS available: no (continuing: --allow-cpu)" >&2
else
    die "torch.backends.mps.is_available() is False; the service would run on CPU. Re-run with --allow-cpu to accept that"
fi

if [ "$NO_WEIGHTS" = 0 ]; then
    HF_HUB_DISABLE_XET=1 .venv/bin/python -m docsearch prefetch
fi

echo "install complete. Foreground: DOCSEARCH_BIND=<tailnet addr> DOCSEARCH_PORT=8010 deploy/macos/run.sh"
