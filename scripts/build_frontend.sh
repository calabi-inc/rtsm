#!/usr/bin/env bash
# Rebuild the demo frontend and refresh the PACKAGED copy under rtsm/static/.
#
# `vite build` writes to demo/dist/ (gitignored); rtsm/static/ is the tracked
# copy that ships in the wheel, and rtsm/utils/static_dir.py prefers a valid
# demo/dist/ when one exists (dev build wins). This script keeps the two in
# step: build, replace the hashed bundle pair, copy index.html, show the diff.
#
# Type errors do not block a build (vite does not type-check); they are
# printed as a warning so they stay visible.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== type check (advisory)"
( cd demo && npx tsc --noEmit ) || echo "WARN: tsc reported errors (pre-existing ones are known); continuing"

echo "== vite build"
( cd demo && npm run build )

echo "== refresh rtsm/static"
# Remove tracked hashed assets the new build no longer produces, copy the new
# ones, and leave a file untouched when its content did not change (vite
# writes LF; the checkout is CRLF under core.autocrlf, so a blind copy would
# show an EOL-only modification).
for old in rtsm/static/assets/index-*.js rtsm/static/assets/index-*.css; do
  [ -e "$old" ] || continue
  [ -e "demo/dist/assets/$(basename "$old")" ] || rm -f "$old"
done
for new in demo/dist/index.html demo/dist/assets/index-*; do
  dst="rtsm/static/${new#demo/dist/}"
  if [ -e "$dst" ] && git diff --quiet --no-index --ignore-cr-at-eol -- "$dst" "$new" 2>/dev/null; then
    continue   # identical modulo line endings
  fi
  cp "$new" "$dst"
done
git status --short rtsm/static
echo "done: commit what git status lists (index.html and the hashed assets whose content changed)"
