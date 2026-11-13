#!/usr/bin/env bash
# Foreground docsearch web server on macOS. Env: DOCSEARCH_BIND, DOCSEARCH_PORT,
# DOCSEARCH_ALLOWED_HOSTS, DOCSEARCH_WORK. Run deploy/macos/install.sh first.
set -euo pipefail

die() { echo "run.sh: $*" >&2; exit 1; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:$PATH"

BIND="${DOCSEARCH_BIND:-127.0.0.1}"
PORT="${DOCSEARCH_PORT:-8000}"
ALLOWED="${DOCSEARCH_ALLOWED_HOSTS:-}"
WORK="${DOCSEARCH_WORK:-$HOME/ocr-docsearch/work}"

case "$BIND" in
    0.0.0.0|::|'[::]')
        die "refusing DOCSEARCH_BIND=$BIND: it would expose the unauthenticated service on every interface. Bind to 127.0.0.1 or the tailnet address (e.g. 100.76.130.63)." ;;
esac
case "$PORT" in
    ''|*[!0-9]*) die "DOCSEARCH_PORT must be a number, got '$PORT'" ;;
esac

case "$BIND" in
    127.*|localhost|::1|'[::1]') ;;
    *) [ -n "$ALLOWED" ] || ALLOWED="$BIND" ;;
esac

[ -x .venv/bin/python ] || die "no .venv found in $ROOT; run deploy/macos/install.sh first"

mkdir -p "$WORK"
chmod 700 "$WORK"

export DOCSEARCH_WORK="$WORK"
export DOCSEARCH_ALLOWED_HOSTS="$ALLOWED"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false

exec .venv/bin/python -m docsearch serve --host "$BIND" --port "$PORT"
