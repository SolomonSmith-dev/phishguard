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

# Match the uses: value, not a SHA mentioned in a comment.
unpinned=""
while IFS= read -r raw; do
  trimmed="${raw#"${raw%%[![:space:]]*}"}"
  [[ "$trimmed" == \#* ]] && continue
  line="${raw%%#*}"
  [[ "$line" == *uses:* ]] || continue
  rest="${line#*uses:}"
  rest="${rest#"${rest%%[![:space:]]*}"}"
  action="${rest%%[[:space:]]*}"
  ref="${action##*@}"
  if [[ "$action" != *@* || ! "$ref" =~ ^[0-9a-f]{40}$ ]]; then
    unpinned+="$raw"$'\n'
  fi
done < <(grep -E 'uses:' "${files[@]}" || true)
if [[ -n "$unpinned" ]]; then
  echo "unpinned actions:"
  printf '%s' "$unpinned"
  exit 1
fi

ci_files=()
[[ -f "$wf_dir/ci.yml" ]] && ci_files+=("$wf_dir/ci.yml")
[[ -f "$wf_dir/ci.yaml" ]] && ci_files+=("$wf_dir/ci.yaml")
if [[ ${#ci_files[@]} -eq 0 ]]; then
  echo "missing ci.yml"
  exit 1
fi

# Require a pins step, not a comment or a string inside a block scalar.
invoke_ok=0
for f in "${ci_files[@]}"; do
  if awk '
    {
      sub(/#.*/, "")
      if ($0 ~ /^[[:space:]]+- name: Test workflow pins[[:space:]]*$/) { expect=1; next }
      if (expect) {
        if ($0 ~ /^[[:space:]]*$/) next
        if ($0 ~ /^[[:space:]]+run:[[:space:]]+bash[[:space:]]+scripts\/test-workflow-pins\.sh[[:space:]]*$/) { found=1; exit }
        expect=0
      }
    }
    END { exit found ? 0 : 1 }
  ' "$f"; then
    invoke_ok=1
  fi
done
if [[ "$invoke_ok" -ne 1 ]]; then
  echo "ci.yml does not invoke test-workflow-pins.sh"
  exit 1
fi

echo "workflow pins ok"
