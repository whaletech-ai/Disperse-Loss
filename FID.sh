#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <folder_a> <folder_b> [--batch-size N] [--device cpu|cuda] [--num-workers N] [--out FILE]"
  exit 1
fi

folder_a="$1"
folder_b="$2"
shift 2

python fid_folders.py "$folder_a" "$folder_b" "$@"
