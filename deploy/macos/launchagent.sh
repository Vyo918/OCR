#!/usr/bin/env bash
# Manage the per-user launchd agent com.docsearch.web (auto-start docsearch web on login).
# Usage:
#   launchagent.sh install --bind ADDR [--port N] [--allowed-hosts LIST] [--work DIR]
#   launchagent.sh uninstall | status | restart
set -euo pipefail

die() { echo "launchagent.sh: $*" >&2; exit 1; }

LABEL="com.docsearch.web"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEMPLATE="$ROOT/deploy/macos/com.docsearch.web.plist.template"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOG_DIR="$HOME/Library/Logs/docsearch"
TARGET="gui/$(id -u)/$LABEL"

usage() { sed -n '2,6p' "$0"; exit "${1:-0}"; }

check_bind() {
    case "$1" in
        0.0.0.0|::|'[::]')
            die "refusing --bind $1: it would expose the unauthenticated service on every interface. Use 127.0.0.1 or the tailnet address." ;;
        '') die "--bind is required (tailnet address, e.g. 100.76.130.63)" ;;
    esac
}

is_loaded() { launchctl print "$TARGET" >/dev/null 2>&1; }

do_install() {
    local bind="" port="8000" allowed="" work="$HOME/ocr-docsearch/work"
    while [ $# -gt 0 ]; do
        case "$1" in
            --bind) bind="${2:-}"; shift 2 ;;
            --port) port="${2:-}"; shift 2 ;;
            --allowed-hosts) allowed="${2:-}"; shift 2 ;;
            --work) work="${2:-}"; shift 2 ;;
            *) die "unknown install argument: $1" ;;
        esac
    done
    check_bind "$bind"
    case "$port" in ''|*[!0-9]*) die "--port must be a number, got '$port'" ;; esac
    [ -x "$ROOT/.venv/bin/python" ] || die "no .venv in $ROOT; run deploy/macos/install.sh first"
    [ -f "$TEMPLATE" ] || die "missing template $TEMPLATE"

    mkdir -p "$LOG_DIR" "$(dirname "$PLIST")"

    # Render first: if this fails, the running service is left untouched.
    local tmp
    tmp="$(mktemp "$PLIST.XXXXXX")"
    ROOT="$ROOT" BIND="$bind" PORT="$port" ALLOWED_HOSTS="$allowed" WORK="$work" \
        PATH_VALUE="/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:/usr/bin:/bin" \
        HOME_VALUE="$HOME" LOG_DIR="$LOG_DIR" \
        python3 - "$TEMPLATE" "$tmp" <<'PY' || { rm -f "$tmp"; die "failed to render plist"; }
import html, os, plistlib, re, sys
template, out = sys.argv[1], sys.argv[2]
values = {
    "ROOT": os.environ["ROOT"], "BIND": os.environ["BIND"], "PORT": os.environ["PORT"],
    "ALLOWED_HOSTS": os.environ["ALLOWED_HOSTS"], "WORK": os.environ["WORK"],
    "PATH": os.environ["PATH_VALUE"], "HOME": os.environ["HOME_VALUE"], "LOG_DIR": os.environ["LOG_DIR"],
}
text = open(template, encoding="utf-8").read()
def sub(m):
    if m.group(1) not in values:
        raise SystemExit(f"unknown placeholder @{m.group(1)}@ in template")
    return html.escape(values[m.group(1)], quote=True)
rendered = re.sub(r"@([A-Z_]+)@", sub, text)
plistlib.loads(rendered.encode("utf-8"))  # proves well-formed XML plist
with open(out, "w", encoding="utf-8") as fh:
    fh.write(rendered)
PY
    chmod 644 "$tmp"
    mv "$tmp" "$PLIST"

    if is_loaded; then launchctl bootout "$TARGET"; fi
    launchctl bootstrap "gui/$(id -u)" "$PLIST" || die "launchctl bootstrap failed (a logged-in GUI session is required)"
    launchctl kickstart -k "$TARGET"
    echo "installed $PLIST (bind $bind:$port, work $work)"
    echo "logs: $LOG_DIR"
}

do_uninstall() {
    if is_loaded; then launchctl bootout "$TARGET"; fi
    rm -f "$PLIST"
    echo "uninstalled $LABEL"
}

do_status() {
    if ! is_loaded; then
        echo "$LABEL: not loaded"
    else
        launchctl print "$TARGET" | grep -E '^[[:space:]]*(state|pid|last exit code|last exit status) = ' || true
    fi
    for log in web.out.log web.err.log; do
        if [ -f "$LOG_DIR/$log" ]; then
            echo "--- last 20 lines of $LOG_DIR/$log"
            tail -n 20 "$LOG_DIR/$log"
        fi
    done
}

do_restart() {
    is_loaded || die "$LABEL is not loaded; run install first"
    launchctl kickstart -k "$TARGET"
    echo "restarted $LABEL"
}

[ $# -ge 1 ] || usage 1
cmd="$1"; shift
case "$cmd" in
    install) do_install "$@" ;;
    uninstall) do_uninstall ;;
    status) do_status ;;
    restart) do_restart ;;
    -h|--help|help) usage 0 ;;
    *) usage 1 ;;
esac
