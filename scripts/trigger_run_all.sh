#!/usr/bin/env bash
set -euo pipefail

if [ -z "${BACKEND_URL:-}" ]; then
  echo "ERROR: BACKEND_URL is not set"
  exit 1
fi

echo "Triggering run_all on: ${BACKEND_URL}"
curl -fsS -X POST "${BACKEND_URL%/}/admin/run_all"
echo ""
echo "OK"
