#!/usr/bin/env bash
# Fail if any workflow still has a floating action tag.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
wf_dir="$root/.github/workflows"

if [[ ! -d "$wf_dir" ]]; then
  echo "missing $wf_dir"
  exit 1
fi

shopt -s nullglob
files=("$wf_dir"/*.yml "$wf_dir"/*.yaml)
if [[ ${#files[@]} -eq 0 ]]; then
  echo "no workflow files in $wf_dir"
  exit 1
fi

# uses: org/action@<exactly 40 hex> then space, comment, or end of line.
# A SHA prefix plus suffix (@<40-hex>-v4) must fail.
ok_re='uses: [-A-Za-z0-9_./]+@[0-9a-f]{40}([[:space:]]|#|$)'
unpinned="$(grep -E 'uses:' "${files[@]}" | grep -vE "$ok_re" || true)"
if [[ -n "$unpinned" ]]; then
  echo "unpinned actions:"
  printf '%s\n' "$unpinned"
  exit 1
fi

ci_files=()
[[ -f "$wf_dir/ci.yml" ]] && ci_files+=("$wf_dir/ci.yml")
[[ -f "$wf_dir/ci.yaml" ]] && ci_files+=("$wf_dir/ci.yaml")
if [[ ${#ci_files[@]} -eq 0 ]]; then
  echo "missing ci.yml"
  exit 1
fi

# Require a real run: step. A comment that names the script must not pass.
if ! grep -Eq '^[[:space:]]+run:[[:space:]]+bash[[:space:]]+scripts/test-workflow-pins\.sh[[:space:]]*$' "${ci_files[@]}"; then
  echo "ci.yml does not invoke test-workflow-pins.sh"
  exit 1
fi

echo "workflow pins ok"
