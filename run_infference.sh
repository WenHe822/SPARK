#!/usr/bin/env bash
set -euo pipefail

echo "run_infference.sh is deprecated; use run_inference.sh instead." >&2
exec "$(dirname "$0")/run_inference.sh" "$@"
