#!/usr/bin/env bash

set -e

NOTEBOOK="lets-go-baby.ipynb"

echo "Watching changes for: $NOTEBOOK and its paired files"
echo "Press Ctrl+C to stop."

while true; do
  inotifywait -e close_write "${NOTEBOOK}" "${NOTEBOOK%.ipynb}.py" "${NOTEBOOK%.ipynb}.md"
  echo "Change detected. Syncing formats..."
  jupytext --sync "$NOTEBOOK"
done

