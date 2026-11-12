#!/usr/bin/env bash
# Thin wrapper around `docker compose run` for the docsearch image.
#
#   ./docsearch.sh prefetch                                  once: download model weights into the models volume
#   ./docsearch.sh ask -f drawing.pdf "消防泵浦幾HP?"        file + question in one step (-f = host path; its folder is mounted read-only)
#   ./docsearch.sh keyword -f drawing.pdf 初沉池              keyword search with sheet position + crops
#   ./docsearch.sh build                                     OCR + index every PDF in $DOCS_DIR
#   ./docsearch.sh ask "question" | chat | bench             work on what `build` indexed
# Env: DOCS_DIR (folder of PDFs for `build`; default: the folder you run this from)
#      WORK_DIR (OCR json, index, crops, reports; default: ./work next to your files)
#      MODELS_DIR (model weights; default a docker volume, e.g. ~/.cache/huggingface to reuse an existing HF cache)
set -euo pipefail
orig="$PWD"
cd "$(dirname "$0")"
export DOCSEARCH_UID="$(id -u)" DOCSEARCH_GID="$(id -g)"
export DOCS_DIR="${DOCS_DIR:-$orig}" WORK_DIR="${WORK_DIR:-$orig/work}"
mkdir -p "$WORK_DIR"        # created by us, not by docker (which would make it root-owned)

args=(); vols=(); n=0; prev=""
for a in "$@"; do
  cur="$a"
  if [[ "$prev" == "-f" || "$prev" == "--file" ]]; then
    abs="$(cd "$orig" && realpath -- "$a")"
    [[ -e "$abs" ]] || { echo "no such file or folder: $a" >&2; exit 2; }
    if [[ -d "$abs" ]]; then
      vols+=(-v "$abs:/files/$n:ro"); a="/files/$n"
    else
      vols+=(-v "$(dirname "$abs"):/files/$n:ro"); a="/files/$n/$(basename "$abs")"
    fi
    n=$((n + 1))
  fi
  args+=("$a"); prev="$cur"
done

[ -t 0 ] || T="-T"            # no TTY when stdin is piped
exec docker compose run --rm ${T:-} ${vols[@]+"${vols[@]}"} docsearch ${args[@]+"${args[@]}"}
